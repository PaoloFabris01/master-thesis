# %%
# %%
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from sklearn.metrics import accuracy_score

import numpy as np

import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist

import matplotlib.pyplot as plt

import os
import glob

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io as skimageIO

from tqdm import tqdm

import random


# carico dataset

print('Loading data...')
directory = 'data_all_image_mask_patch_3_edgeweight.pt'
data = torch.load(directory, weights_only=False)

directory_test = 'data_all_image_mask_test_patch_3_n_px_balanced_edgeweight.pt'
data_test = torch.load(directory_test, weights_only=False)





# Divido il dataset in train, validation e test set. 

from sklearn.model_selection import train_test_split
import random


print('DataLoader creation...')
# Prima divisione: separa train+val da test (80% train+val, 20% test)
train_set, val_set = train_test_split(
    data, 
    test_size=0.2, 
    random_state=42  # per riproducibilità
)

test_set = data_test

print(f"Train set: {len(train_set)} samples")
print(f"Validation set: {len(val_set)} samples") 
print(f"Test set: {len(test_set)} samples")

#Creo Dataloader per train, val e test
batch_size = 16

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)




# MODELLO FINALE
# Modello (ChebConv K=3 + Attention head=2 (concatenazione) + edge weight + 8 features + dropout=0.5 + layer 3)
#usato sia per:
# - dataset con spx features con slic per ogni tempo
# - dataset con spx features con slic solo primo tempo


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv


class ChebConv_Temp_Att_combined_edgeweight(nn.Module):
    def __init__(self, data, hidden_channels, F=8, T=12, num_heads=2, K=3):
        super().__init__()
        torch.manual_seed(123)
        self.T = T
        self.F = F
        self.attn = nn.MultiheadAttention(embed_dim=F, num_heads=num_heads, batch_first=True)
        
        # Ora il primo layer riceve F*T*2 features (originali + attention)
        self.conv1 = ChebConv(F * T * 2, hidden_channels, K)  # 192 invece di 96
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, data.num_classes, K)

    def forward(self, x, edge_index, edge_weight):
        # x shape: [num_nodes, F × T] = [N, 72]
        num_nodes = x.size(0)
        x_original = x.clone()  # Salva le features originali
        
        # Step 1: reshape to [num_nodes, T, F]
        x = x.view(num_nodes, self.F, self.T).permute(0, 2, 1)  # [N, T, F]
        
        # Step 2: apply multihead attention over time
        x_attn, _ = self.attn(x, x, x)  # [N, T, F]
        
        # Step 3: flatten back to [N, T × F]
        x_attn_flat = x_attn.reshape(num_nodes, self.T * self.F)  # [N, 96]
        #Now contains temporally-aware features that understand relationships across time
#Original features [0:72]: Raw temporal data
#Attention features [72:144]: Temporally-aware representations

        # Step 4: concatenate original and attention features
        x_combined = torch.cat([x_original, x_attn_flat], dim=1)  # [N, 192]
        
        # Step 5: apply GCN
        x = self.conv1(x_combined, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        return x


k = 3
num_heads = 2
hidden_channels = 128
model = ChebConv_Temp_Att_combined_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads)
print(model)




'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv


class ChebConv_edgeweight(nn.Module):
    def __init__(self, data, hidden_channels, F=8, T=12, K=3):
        super().__init__()
        torch.manual_seed(123)
        self.T = T
        self.F = F
        
        self.conv1 = ChebConv(F * T, hidden_channels, K)  # 96 features
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, data.num_classes, K)

    def forward(self, x, edge_index, edge_weight):
        # Step 5: apply Chebconv
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        return x


k = 3
hidden_channels = 128
model = ChebConv_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k)
print(model)

'''


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv


class ChebConv_Temp_Att_combined(nn.Module):
    def __init__(self, data, hidden_channels, F=8, T=12, num_heads=2, K=3):
        super().__init__()
        torch.manual_seed(123)
        self.T = T
        self.F = F
        self.attn = nn.MultiheadAttention(embed_dim=F, num_heads=num_heads, batch_first=True)
        
        # Ora il primo layer riceve F*T*2 features (originali + attention)
        self.conv1 = ChebConv(F * T * 2, hidden_channels, K)  # 192 invece di 96
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, data.num_classes, K)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, F × T] = [N, 72]
        num_nodes = x.size(0)
        x_original = x.clone()  # Salva le features originali
        
        # Step 1: reshape to [num_nodes, T, F]
        x = x.view(num_nodes, self.F, self.T).permute(0, 2, 1)  # [N, T, F]
        
        # Step 2: apply multihead attention over time
        x_attn, _ = self.attn(x, x, x)  # [N, T, F]
        
        # Step 3: flatten back to [N, T × F]
        x_attn_flat = x_attn.reshape(num_nodes, self.T * self.F)  # [N, 96]
        #Now contains temporally-aware features that understand relationships across time
#Original features [0:72]: Raw temporal data
#Attention features [72:144]: Temporally-aware representations

        # Step 4: concatenate original and attention features
        x_combined = torch.cat([x_original, x_attn_flat], dim=1)  # [N, 192]
        
        # Step 5: apply GCN
        x = self.conv1(x_combined, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


k = 3
num_heads = 2
hidden_channels = 128
model = ChebConv_Temp_Att_combined(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads)
print(model)
'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv


class GCNConv_Temp_Att_combined_edgeweight(nn.Module):
    def __init__(self, data, hidden_channels, F=8, T=12, num_heads=2):
        super().__init__()
        torch.manual_seed(123)
        self.T = T
        self.F = F
        self.attn = nn.MultiheadAttention(embed_dim=F, num_heads=num_heads, batch_first=True)
        
        # Ora il primo layer riceve F*T*2 features (originali + attention)
        self.conv1 = GCNConv(F * T * 2, hidden_channels)  # 192 invece di 96
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index, edge_weight):
        # x shape: [num_nodes, F × T] = [N, 72]
        num_nodes = x.size(0)
        x_original = x.clone()  # Salva le features originali
        
        # Step 1: reshape to [num_nodes, T, F]
        x = x.view(num_nodes, self.F, self.T).permute(0, 2, 1)  # [N, T, F]
        
        # Step 2: apply multihead attention over time
        x_attn, _ = self.attn(x, x, x)  # [N, T, F]
        
        # Step 3: flatten back to [N, T × F]
        x_attn_flat = x_attn.reshape(num_nodes, self.T * self.F)  # [N, 96]
        #Now contains temporally-aware features that understand relationships across time
#Original features [0:72]: Raw temporal data
#Attention features [72:144]: Temporally-aware representations

        # Step 4: concatenate original and attention features
        x_combined = torch.cat([x_original, x_attn_flat], dim=1)  # [N, 192]
        
        # Step 5: apply GCN
        x = self.conv1(x_combined, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        return x


num_heads = 2
hidden_channels = 128
model = GCNConv_Temp_Att_combined_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, num_heads=num_heads)
print(model)

'''

'''
#WITH LINEAR CLASSIFIER

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv


class ChebConv_Temp_Att_LinearClassifier_edgeweight(nn.Module):
    def __init__(self, data, hidden_channels, F=8, T=12, num_heads=2, K=3):
        super().__init__()
        torch.manual_seed(123)
        self.T = T
        self.F = F
        self.attn = nn.MultiheadAttention(embed_dim=F, num_heads=num_heads, batch_first=True)
        
        # Ora il primo layer riceve F*T*2 features (originali + attention)
        self.conv1 = ChebConv(F * T * 2, hidden_channels, K)  # 192 invece di 96
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, data.num_classes)
        )

    def forward(self, x, edge_index, edge_weight):
        # x shape: [num_nodes, F × T] = [N, 72]
        num_nodes = x.size(0)
        x_original = x.clone()  # Salva le features originali
        
        # Step 1: reshape to [num_nodes, T, F]
        x = x.view(num_nodes, self.F, self.T).permute(0, 2, 1)  # [N, T, F]
        
        # Step 2: apply multihead attention over time
        x_attn, _ = self.attn(x, x, x)  # [N, T, F]
        
        # Step 3: flatten back to [N, T × F]
        x_attn_flat = x_attn.reshape(num_nodes, self.T * self.F)  # [N, 96]
        #Now contains temporally-aware features that understand relationships across time
#Original features [0:72]: Raw temporal data
#Attention features [72:144]: Temporally-aware representations

        # Step 4: concatenate original and attention features
        x_combined = torch.cat([x_original, x_attn_flat], dim=1)  # [N, 192]
        
        # Encoder
        x1 = F.relu(self.conv1(x_combined, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        
        # Skip connection
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = x3 + x1  # residual connection
        
        # Classifier
        return self.classifier(x3)


k = 3
num_heads = 2
hidden_channels = 128
model = ChebConv_Temp_Att_LinearClassifier_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads)
print(model)

'''

'''
#CON TRANSFORMER
Cosa fa il Transformer Encoder
Input:
Dopo il reshape, ogni nodo ha una sequenza temporale di vettori di feature:[num_nodes, T, F] (es: [N, 12, 8]).

Transformer Encoder:
Il Transformer Encoder processa questa sequenza temporale per ogni nodo, modellando le dipendenze tra i diversi tempi (ad esempio, come cambia la feature nel tempo).

Output:
L'output ha la stessa forma dell'input: [num_nodes, T, F].
Ogni vettore temporale è stato "arricchito" con informazioni dagli altri tempi tramite self-attention.

A cosa serve dim_feedforward
Definizione:
dim_feedforward è la dimensione interna del feedforward network all'interno di ogni layer del Transformer.

Funzionamento:
Ogni layer del Transformer Encoder, per ogni tempo, applica:

Self-attention (per combinare informazioni tra tempi)
Feedforward network:
È una MLP a due layer:
Primo layer: da F a dim_feedforward (es: da 8 a 128)
Attivazione (ReLU)
Secondo layer: da dim_feedforward a F (es: da 128 a 8)
Add & Norm (residual connection + normalizzazione)
Perché si usa?
Serve per aumentare la capacità espressiva del layer:
anche se l'input e l'output restano di dimensione F, all'interno il layer può rappresentare relazioni più complesse grazie a uno spazio intermedio più grande (dim_feedforward).

In sintesi
Il transformer arricchisce ogni vettore temporale con informazioni dagli altri tempi.
dim_feedforward=128 non cambia la dimensione dell'output (che resta F),
ma rende il layer più potente perché la MLP interna lavora in uno spazio più ampio.
Se vuoi che l'output abbia dimensione 128, devi aggiungere un layer lineare dopo il flatten, oppure impostare d_model=128 e proiettare le feature iniziali!
'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ChebConv_Temp_Transformer_edgeweight(nn.Module):
    def __init__(self, data, hidden_channels, F=8, T=12, num_heads=2, K=3, num_layers=2, dim_feedforward=128):
        super().__init__()
        torch.manual_seed(123)
        self.T = T
        self.F = F

        # Transformer encoder layer and encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=F, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Ora il primo layer riceve F*T*2 features (originali + attention)
        self.conv1 = ChebConv(F * T * 2, hidden_channels, K)  # 192 invece di 96
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, data.num_classes, K)

    def forward(self, x, edge_index, edge_weight):
        # x shape: [num_nodes, F × T] = [N, 72]
        num_nodes = x.size(0)
        x_original = x.clone()  # Salva le features originali
        
        # Step 1: reshape to [num_nodes, T, F]
        x = x.view(num_nodes, self.F, self.T).permute(0, 2, 1)  # [N, T, F]
        
        # Step 2: apply transformer encoder over time
        x_trans = self.transformer_encoder(x)  # [N, T, F]
        
        # Step 3: flatten back to [N, T × F]
        x_trans_flat = x_trans.reshape(num_nodes, self.T * self.F)  # [N, 96]

        # Step 4: concatenate original and attention features
        x_combined = torch.cat([x_original, x_trans_flat], dim=1)  # [N, 192]
        
        # Step 5: apply GCN
        x = self.conv1(x_combined, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        return x


k = 3
num_heads = 2
hidden_channels = 128
num_layers = 2
dim_feedforward = 128
model = ChebConv_Temp_Transformer_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward)
print(model)
'''




# Count the number of parameters in the model
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Number of model parameters: ", pytorch_total_params)





from sklearn.metrics import accuracy_score
import numpy as np

# Updated accuracy function using sklearn
def accuracy(output, labels):
    # Seleziona solo i nodi con etichetta valida
    mask = labels != -1
    if mask.sum().item() == 0:
        return None
    
    preds = output.argmax(dim=1)
    
    # numpy arrays and filter valid labels
    y_true = labels[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    
    #sklearn accuracy_score
    acc = accuracy_score(y_true, y_pred)
    return acc



#############################################################################
import torch
import torch.nn as nn
import torch.optim as optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Usa il primo grafo del train_set per inizializzare il modello
#model = ChebConv_Temp_Att_combined_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads).to(device)
#model = ChebConv_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k)
#model = ChebConv_Temp_Att_combined(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads)
#model = GCNConv_Temp_Att_combined_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, num_heads=num_heads).to(device)
#model = ChebConv_Temp_Att_LinearClassifier_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads).to(device)
model = ChebConv_Temp_Transformer_edgeweight(train_set[0], hidden_channels=hidden_channels, F=8, T=12, K=k, num_heads=num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
lr = 1e-4
print(f'Learning rate = {lr}')
weight_decay = 1e-3
print(f'Weight decay = {weight_decay}')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



# Funzione di training per grafo
def train(data):
    model.train()
    data = data.to(device)
    data.x = data.x.float()  # Assicura che x sia in float32
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        data.edge_weight = data.edge_weight.float()
    # Skip graphs with only -1 labels
    if (data.y != -1).sum().item() == 0:
        return None
    optimizer.zero_grad()
    #con edge_weight
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        out = model(data.x, data.edge_index, data.edge_weight)
    #senza edge_weight
    else:
        out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    
    loss.backward()
    optimizer.step()
    return loss.item()



from sklearn.metrics import accuracy_score

def evaluate(data):
    model.eval()
    data = data.to(device)
    data.x = data.x.float()  # Assicura che x sia in float32
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        data.edge_weight = data.edge_weight.float()
    if (data.y != -1).sum().item() == 0:
        return None, None, None
    
    with torch.no_grad():
        #con edge_weight
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            out = model(data.x, data.edge_index, data.edge_weight)
        #senza edge_weight
        else:
            out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
    
    # Return predictions and labels for sklearn accuracy calculation
    mask = data.y != -1
    preds = out.argmax(dim=1)[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    
    return loss.item(), preds, labels



##################################################################
import time

print('Training...')
train_start_time = time.time()

train_losses_history = []
val_losses_history = []
val_accuracies_history = []

# Early stopping parameters
patience = 50
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
#min_delta = 0.001  # Minimum change to qualify as an improvement

epoch = 501
div = epoch // 10

for epoch in tqdm(range(1, epoch)):
    train_losses = []
    for batch in train_loader:
        loss = train(batch)
        if loss is not None:
            train_losses.append(loss)

    val_losses = []
    all_val_preds = []
    all_val_labels = []
    
    for batch in val_loader:
        val_loss, preds, labels = evaluate(batch)
        if val_loss is not None:
            val_losses.append(val_loss)
            all_val_preds.extend(preds)
            all_val_labels.extend(labels)

    # Calculate epoch metrics
    epoch_train_loss = sum(train_losses)/len(train_losses) if train_losses else 0
    epoch_val_loss = sum(val_losses)/len(val_losses) if val_losses else 0
    epoch_val_acc = accuracy_score(all_val_labels, all_val_preds) if all_val_labels else 0

    train_losses_history.append(epoch_train_loss)
    val_losses_history.append(epoch_val_loss)
    val_accuracies_history.append(epoch_val_acc)

    # Early stopping logic
    if epoch_val_loss < best_val_loss: #- min_delta:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # Save the best model state
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if epoch % div == 0:
        
        # evaluate on test set
        test_losses = []
        all_test_preds = []
        all_test_labels = []
        for batch in test_loader:
            loss, preds, labels = evaluate(batch)
            if loss is not None:
                test_losses.append(loss)
                all_test_preds.extend(preds)
                all_test_labels.extend(labels)
        # Calculate final test metrics
        final_test_loss = sum(test_losses)/len(test_losses) if test_losses else 0
        final_test_acc = accuracy_score(all_test_labels, all_test_preds) if all_test_labels else 0


        print(f"Epoch {epoch:03d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f} | "
            f"Test Loss: {final_test_loss:.4f} | "
            f"Test Acc: {final_test_acc:.4f} | "
            f"Patience: {patience_counter}/{patience}")
        
    # Check for early stopping
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        break

train_end_time = time.time()
training_time = train_end_time - train_start_time

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model state for final evaluation")



#######################################################
print('Testing...')
test_start_time = time.time()

test_losses = []
all_test_preds = []
all_test_labels = []

for batch in tqdm(test_loader):
    loss, preds, labels = evaluate(batch)
    if loss is not None:
        test_losses.append(loss)
        all_test_preds.extend(preds)
        all_test_labels.extend(labels)

# Calculate final test metrics
final_test_loss = sum(test_losses)/len(test_losses) if test_losses else 0
final_test_acc = accuracy_score(all_test_labels, all_test_preds) if all_test_labels else 0

print(f"Test Loss: {final_test_loss:.4f}")
print(f"Test Acc: {final_test_acc:.4f}")

# Store final results for saving
test_losses_history = [final_test_loss]
test_accuracies_history = [final_test_acc]

test_end_time = time.time()
testing_time = test_end_time - test_start_time



#############################################

#save the model
checkpoint = {
    'model_description': 'connectivity=8, stacking attention e input features (8 features) - dati: "data_all_image_mask_patch_3_n_px_balanced_edgeweight.pt", modello: Cheb_att_edgeweight_TestPatch_3 (pixel_superpixel)',
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses_history,
    'val_losses': val_losses_history,
    'val_accuracies': val_accuracies_history,
    'test_loss': test_losses_history,
    'test_accuracy': test_accuracies_history,
    'epoch': epoch,
    'batch_size': batch_size,
    'hidden_channels': hidden_channels,
    'layers': 3,
    'num_heads': num_heads,
    'K': k,
    'num_layers_transformer': num_layers,
    'dim_feedforward_transformer': dim_feedforward,
    'learning_rate': lr,
    'weight_decay': weight_decay,
    'early_stopping_patience': patience,
    'total_params': pytorch_total_params,
    'training_time_seconds': training_time,
    'testing_time_seconds': testing_time, 
    'grafi_train': len(train_set),
    'grafi_val': len(val_set),
    'grafi_test': len(test_set)
}

#save the model
model_filename = 'model_Cheb_att_edgeweight_TestPatch_3.pth'
torch.save(checkpoint, model_filename)
print(f"Model saved at {model_filename}")





