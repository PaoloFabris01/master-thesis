# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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


print('Loading data...')
# Load data
train_path = 'normalized_ts_array.pt'
full_img_array = torch.load(train_path, weights_only=False)
print(f'Train data shape: {full_img_array.shape}')

# Load the ground truth image
gt_a = np.load('gt.npy')
gt = gt_a[1, 0]
print(f'Ground truth train shape: {gt.shape}')

#tre patches 200*200 come test set
coords = torch.load('patch_coords_tre.pt', weights_only=False) # list of the three (x_start, y_start) of the three patches
test_mask = torch.load('test_mask_tre.pt', weights_only=False)
train_mask = torch.load('train_mask_tre.pt', weights_only=False)


#########################################################################
y_0, x_0 = coords[0]
y_1, x_1 = coords[1]
y_2, x_2 = coords[2]
h, w = 200, 200 # 350x350 is the size of the test image patch
print(f"Best patch found at: {coords} with size {h}x{w}")
test_img_array_0 = full_img_array[:, :, y_0:y_0+h, x_0:x_0+w]
test_img_array_1 = full_img_array[:, :, y_1:y_1+h, x_1:x_1+w]
test_img_array_2 = full_img_array[:, :, y_2:y_2+h, x_2:x_2+w]
#print(f'Test image shape: {test_img_array.shape}')
gt_test_0 = gt[y_0:y_0+h, x_0:x_0+w]
gt_test_1 = gt[y_1:y_1+h, x_1:x_1+w]
gt_test_2 = gt[y_2:y_2+h, x_2:x_2+w]
#print(f'Ground truth test shape: {gt_test_0.shape}')

# Visualizza
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Test patch")
plt.imshow(test_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Train area")
plt.imshow(train_mask, cmap="gray")
plt.show()



##############################################################################
#creo funzione per creare spx diversi per ogni tempo e calcolare le features dei spx per ogni tempo

def apply_slic_to_each_time(full_img_array, mask = train_mask, mask_t = test_mask,  coordinates = coords):
    """
    Applica SLIC a ogni immagine in un array di immagini e restituisce le etichette dei superpixel.
    
    :param img_array: Array di immagini (tensor) con dimensioni [num_images, height, width, channels].
    :param n_segments: Numero di segmenti (superpixel) da creare.
    :param compactness: Compattezza dei superpixel.
    :param sigma: Sigma per il filtro gaussiano.
    :return: Lista di etichette dei superpixel per ogni immagine.
    """
    labels_list = []
    labels_list_test_0 = []
    labels_list_test_1 = []
    labels_list_test_2 = []
    labels_list_test_full = []
    print("Applying SLIC to each image in the array...")
    for img in tqdm(range(len(full_img_array))):
        img_train = full_img_array[img, 0:3, :, :]
        img_train_t = img_train.transpose(1, 2, 0)
        labels_full = slic(img_train_t, n_segments=30000, compactness=2, sigma=0, start_label=0)
        # 4) zero‐out or mark as −1 everything outside your true mask
        labels = labels_full.copy()
        labels[~mask] = -1
        
        labels_list.append(labels)

        
        #y, x = coordinates
        y_0, x_0 = coordinates[0]
        y_1, x_1 = coordinates[1]
        y_2, x_2 = coordinates[2]
        h, w = 200, 200 # 350x350 is the size of the test image patch
        # img: H×W×C, train_mask: H×W boolean
        labels_test = slic(
            img_train_t,
            n_segments=2600,
            compactness=2,
            sigma=0,
            channel_axis=2,
            mask=mask_t
        )
        labels_test_0 = labels_test[y_0:y_0+h, x_0:x_0+w]
        labels_test_1 = labels_test[y_1:y_1+h, x_1:x_1+w]
        labels_test_2 = labels_test[y_2:y_2+h, x_2:x_2+w]
        labels_list_test_0.append(labels_test_0)
        labels_list_test_1.append(labels_test_1)
        labels_list_test_2.append(labels_test_2)
    
    
    return labels_list, labels_list_test_0, labels_list_test_1, labels_list_test_2



def pixel_temporal_feature_stack_4d(img_4d,         # (T,B,H,W)  e.g. (12,10,H,W)
                                    segments_3d,    # (T,H,W)    same T
                                    nir_idx=6,      # indices of NIR, Red, Green, SWIR in the B dimension
                                    red_idx=2,
                                    green_idx=1,
                                    swir_idx=7,
                                    eps: float = 1e-8):
    """
    For every pixel build a vector
      [NDVI_t0, NDWI_t0, EVI2_t0,  NDVI_t1, NDWI_t1, EVI2_t1, ...]
    using superpixel means at each time step.

    Returns
    -------
    torch.Tensor  shape (T*3, H, W)
    """
    assert img_4d.shape[0] == segments_3d.shape[0], "time dimension mismatch"
    T, B, H, W = img_4d.shape
    feature_stack = np.zeros((T * 8, H, W), dtype=np.float32)

    for t in (range(T)):
        print(f"Computing features for time {t+1} of {T} ...")
        img  = img_4d[t]        # (B,H,W)
        segs = segments_3d[t]   # (H,W)

        nir   = img[nir_idx]
        red   = img[red_idx]
        green = img[green_idx]
        swir = img[swir_idx]

        ndvi_full = (nir - red)   / (nir + red   + eps)
        ndwi_full = (green - nir) / (green + nir + eps)
        evi2_full = 2.4 * (nir - red) / (nir + red + 1 + eps)
        ndmi_full = (nir - swir) / (nir + swir + eps)

        ndvi_img = np.zeros_like(ndvi_full, dtype=np.float32)
        ndwi_img = np.zeros_like(ndwi_full, dtype=np.float32)
        evi2_img = np.zeros_like(evi2_full, dtype=np.float32)
        ndmi_img = np.zeros_like(ndmi_full, dtype=np.float32)

        for seg_id in tqdm(np.unique(segs)):
            mask = (segs == seg_id)
            if not mask.any():
                continue
            ndvi_img[mask] = ndvi_full[mask].mean()
            ndwi_img[mask] = ndwi_full[mask].mean()
            evi2_img[mask] = evi2_full[mask].mean()
            ndmi_img[mask] = ndmi_full[mask].mean()

        #features superpixel
        feature_stack[t*8 + 0] = ndvi_img
        feature_stack[t*8 + 1] = ndwi_img
        feature_stack[t*8 + 2] = evi2_img
        feature_stack[t*8 + 3] = ndmi_img
        #features pixel
        feature_stack[t*8 + 4] = ndvi_full
        feature_stack[t*8 + 5] = ndwi_full
        feature_stack[t*8 + 6] = evi2_full
        feature_stack[t*8 + 7] = ndmi_full

    return torch.from_numpy(feature_stack)          # (T*3, H, W)        # (T*3, H, W)



from scipy.sparse import csr_matrix
import networkx as nx

def create_pixel_adjacency_graph(segments, segment_id, connectivity=4):
    """
    Create adjacency graph for pixels within a specific segment
    
    Args:
        segments: 2D array with segment labels
        segment_id: ID of the segment to process
        connectivity: 4 or 8 (4-connected or 8-connected neighbors)
    
    Returns:
        adjacency_matrix: sparse adjacency matrix
        pixel_coords: list of (row, col) coordinates for each pixel
    """
    # Get mask for the specific segment
    mask = (segments == segment_id)    
    
    # Get coordinates of pixels in this segment
    pixel_coords = np.argwhere(mask)  # Returns [(row, col), ...]
    num_pixels = len(pixel_coords)
    
    if num_pixels == 0:
        return None, None
    
    # Create a mapping from coordinates to indices
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(pixel_coords)}
    
    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                   (0, 1), (1, -1), (1, 0), (1, 1)]  # all 8 neighbors
    if connectivity == 16:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), # 8 neighbors
                   (-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)] # sparse second 8 neighbors
    
    # Build adjacency list
    edges = []
    for idx, (row, col) in enumerate(pixel_coords):
        for dr, dc in offsets:
            neighbor_row, neighbor_col = row + dr, col + dc
            neighbor_coord = (neighbor_row, neighbor_col)
            
            # Check if neighbor is within the same segment
            if neighbor_coord in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_coord]
                edges.append((idx, neighbor_idx))
    
    # Create sparse adjacency matrix
    if edges:
        rows, cols = zip(*edges)
        adjacency_matrix = csr_matrix((np.ones(len(edges)), (rows, cols)), 
                                    shape=(num_pixels, num_pixels))
        # Make it symmetric (undirected graph)
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        adjacency_matrix.data = np.clip(adjacency_matrix.data, 0, 1)
    else:
        adjacency_matrix = csr_matrix((num_pixels, num_pixels))
    
    return adjacency_matrix, pixel_coords




from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from torch_geometric.utils import add_self_loops

def adjacency_to_pyg_data(adjacency_matrix, pixel_coords, gt_image, feature_tensor):
    """
    Convert adjacency matrix to PyTorch Geometric Data object
    
    Args:
        adjacency_matrix: sparse adjacency matrix from previous function
        pixel_coords: list of (row, col) coordinates for each pixel
        gt_image: ground truth image to get pixel labels
        feature_tensor: torch tensor of shape (72, 1113, 1113) containing features for each pixel
    
    Returns:
        PyTorch Geometric Data object
    """
    num_nodes = len(pixel_coords)
    
    # Convert sparse adjacency matrix to COO format for edge extraction
    if adjacency_matrix.nnz > 0:  # Check if there are any edges
        coo_adj = coo_matrix(adjacency_matrix)
        edge_index = torch.tensor(np.vstack([coo_adj.row, coo_adj.col]), dtype=torch.long)
    else:
        # No edges - create empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Extract features for each pixel in this segment
    node_features = []
    for row, col in pixel_coords:
        # Extract all 72 features for this pixel
        pixel_feature = feature_tensor[:, row, col]  # Shape: (72,)
        node_features.append(pixel_feature)
    
    # Convert to tensor
    x = torch.stack(node_features)  # Shape: (num_nodes, 72)
    
    # Get node labels from ground truth
    y = torch.tensor([int(gt_image[coord[0], coord[1]]) for coord in pixel_coords], dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.pos = torch.tensor(pixel_coords, dtype=torch.float)  # Save pixel coordinates for future use
    
    # Put to -1 invalid nodes (background index 0 and general index 255)
    mask = (data.y == 0) | (data.y == 255)
    data.y[mask] = 0
    # Adjust valid labels to start from 0
    data.y = data.y - 1
    # Assign number of classes to each data object
    data.num_classes = 17
    
    # Add self loops
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    
    return data




def create_all_segments_pyg_data(train_img_array, test_img_array_0, test_img_array_1, test_img_array_2, gt, gt_test_0, gt_test_1, gt_test_2):
    """
    Create PyG Data objects for all segments
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    #slic
    all_labels_train, all_labels_test_0, all_labels_test_1, all_labels_test_2 = apply_slic_to_each_time(full_img_array=train_img_array)
    #features
    segments = np.array(all_labels_train)  # Convert list to numpy array
    segments_test_0 = np.array(all_labels_test_0)  # Convert list to numpy array
    segments_test_1 = np.array(all_labels_test_1)  # Convert list to numpy array
    segments_test_2 = np.array(all_labels_test_2)  # Convert list to numpy array
    # img_4d          : (12, 10, H, W)
    # segments_3d     : (12, H, W)   – one SLIC label map per time 
    pixel_features = pixel_temporal_feature_stack_4d(train_img_array, segments)
    pixel_features_test_0 = pixel_temporal_feature_stack_4d(test_img_array_0, segments_test_0)
    pixel_features_test_1 = pixel_temporal_feature_stack_4d(test_img_array_1, segments_test_1)
    pixel_features_test_2 = pixel_temporal_feature_stack_4d(test_img_array_2, segments_test_2)
    print(f"Shape features train: {pixel_features.shape}")        # torch.Size([36, H, W])   (= 12×3)
    print(f"Shape features test: {pixel_features_test_0.shape}")   # torch.Size([36, H, W])   (= 12×3)
    #graphs
    data_list_train = []
    for segment_id in tqdm(np.unique(segments[0])):
        # salto subito il valore -1
        if segment_id < 0:
            continue
        # Create adjacency graph for the segment using the first time segmentation
        adjacency_matrix, pixel_coords = create_pixel_adjacency_graph(segments[0], segment_id=segment_id, connectivity=8)
    #data objects
        data = adjacency_to_pyg_data(adjacency_matrix, pixel_coords, gt, pixel_features)
        if data is not None:
            data_list_train.append(data)    
    data_list_test = []
    for segment_id in tqdm(np.unique(segments_test_0[0])):
        if segment_id < 0:
            continue
        # Create adjacency graph for the segment using the first time segmentation
        adjacency_matrix, pixel_coords = create_pixel_adjacency_graph(segments_test_0[0], segment_id=segment_id, connectivity=8)
        #data objects
        data = adjacency_to_pyg_data(adjacency_matrix, pixel_coords, gt_test_0, pixel_features_test_0)
        if data is not None:
            data_list_test.append(data)
    for segment_id in tqdm(np.unique(segments_test_1[0])):
        if segment_id < 0:
            continue
        # Create adjacency graph for the segment using the first time segmentation
        adjacency_matrix, pixel_coords = create_pixel_adjacency_graph(segments_test_1[0], segment_id=segment_id, connectivity=8)
        #data objects
        data = adjacency_to_pyg_data(adjacency_matrix, pixel_coords, gt_test_1, pixel_features_test_1)
        if data is not None:
            data_list_test.append(data)
    for segment_id in tqdm(np.unique(segments_test_2[0])):
        if segment_id < 0:
            continue
        # Create adjacency graph for the segment using the first time segmentation
        adjacency_matrix, pixel_coords = create_pixel_adjacency_graph(segments_test_2[0], segment_id=segment_id, connectivity=8)
        #data objects
        data = adjacency_to_pyg_data(adjacency_matrix, pixel_coords, gt_test_2, pixel_features_test_2)
        if data is not None:
            data_list_test.append(data)
    print(f"Number of segments train: {len(data_list_train)}")
    print(f"Number of segments test: {len(data_list_test)}")
    return data_list_train, data_list_test



#####################################################################################
# Create PyG Data objects for all segments
print('START')
data_list_train, data_list_test = create_all_segments_pyg_data(full_img_array, test_img_array_0, test_img_array_1, test_img_array_2, gt, gt_test_0, gt_test_1, gt_test_2)


########################################################################################
# Save the lists of graphs
nome_file = 'data_all_image_mask_patch_3_n_px_balanced.pt'
nome_file_test = 'data_all_image_mask_test_patch_3_n_px_balanced.pt'
print(f'Salvo liste grafi in: {nome_file}, {nome_file_test}')
torch.save(data_list_train, nome_file)
torch.save(data_list_test, nome_file_test)



#Edgeweight
import torch.nn.functional as F
from torch_geometric.data import Data

def add_edge_weights_cosine_normalized(data):
    x = data.x  # [num_nodes, num_features]
    edge_index = data.edge_index  # [2, num_edges]
    
    src, dst = edge_index  # [num_edges], [num_edges]
    
    x_src = x[src]  # [num_edges, num_features]
    x_dst = x[dst]  # [num_edges, num_features]
    
    # Cosine similarity in range [-1, 1]
    cosine_sim = F.cosine_similarity(x_src, x_dst, dim=1)  # [num_edges]

    # Normalize to range [0, 1]
    cosine_sim_norm = (cosine_sim + 1.0) / 2.0

    # Add edge_weight to the data object
    data.edge_weight = cosine_sim_norm.float()

    return data


data_list_train_edgeweight = [add_edge_weights_cosine_normalized(data) for data in data_list_train]
data_list_test_edgeweight = [add_edge_weights_cosine_normalized(data) for data in data_list_test]


##############################################################################################
nome_file = 'data_all_image_mask_patch_3_n_px_balanced_edgeweight.pt'
nome_file_test = 'data_all_image_mask_test_patch_3_n_px_balanced_edgeweight.pt'
print(f'Salvo liste in: {nome_file}, {nome_file_test}')
torch.save(data_list_train_edgeweight, nome_file)
torch.save(data_list_test_edgeweight, nome_file_test)



