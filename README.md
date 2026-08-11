# Agricultural Field Segmentation and Crop Type Mapping in Satellite Image Time Series Using Graph Neural Networks

This repository contains the code and trained model developed as part of my Master's thesis on **agricultural field segmentation and crop type mapping from Sentinel-2 satellite image time series using Graph Neural Networks (GNNs)**.

The framework was developed **from scratch** and is designed to exploit both the **spatial structure of agricultural fields** and their **temporal evolution** across a satellite image time series.

> **Repository status:** This repository is primarily intended for **research archiving and demonstration purposes**. The complete original dataset, intermediate preprocessed data, and some validation scripts are not included due to data size, storage constraints, and other practical considerations.

## Repository Contents

The main files included in this repository are:

### `slic_on_each_time_patch_3.py`

Contains the **preprocessing and graph construction pipeline**. In particular, it includes the processing step based on **SLIC (Simple Linear Iterative Clustering)** segmentation, applied to the satellite image patches at each time step.

The resulting segments are used to construct the graph representation of the agricultural scene, which is subsequently provided as input to the GNN model.

### `model_slic_on_each_time_patch_3.py`

Contains the **Graph Neural Network architecture and training pipeline** developed for the thesis.

The model operates on the graph representation generated during preprocessing and incorporates spatial and temporal information from the satellite image time series.

### `model_Cheb_att_edgeweight_TestPatch_3.pth`

Contains the **trained model weights** corresponding to the final model configuration used in the experiments.

The model is based on **Chebyshev graph convolutions (ChebConv)** and incorporates **attention mechanisms and edge weights** to model relationships between neighbouring image segments.



Abstract

Modern agriculture is increasingly a data-intensive domain, in which satellite Earth observation supports
sustainable and timely decision-making. Satellite Image Time Series (SITS), and in particular
the Sentinel-2 mission of the Copernicus programme, provide dense multispectral and multitemporal
observations that capture the phenological evolution of crops throughout the growing season. Exploiting
this information for automatic field segmentation and crop-type mapping is, however, challenging,
since SITS data are high-dimensional and require models able to jointly capture spatial, spectral and
temporal dependencies.

Deep learning has substantially advanced SITS analysis, with recurrent, convolutional, and
transformer-based architectures achieving strong results on benchmark datasets. More recently,
Graph Neural Networks (GNNs) have emerged as an attractive alternative, since they can represent
an image as a graph. Nodes correspond to spatial units and edges encode neighbourhood relations.
In this manner has been possible to overcome the rigid, regular-grid assumption of standard convolutions.
Most existing GNN-based approaches for SITS, however, build graphs at the superpixel level,
aggregating pixels to keep the graph tractable. While computationally convenient, this choice discards
fine spatial detail that is often crucial for delineating agricultural field boundaries and capturing
within-field heterogeneity.

This work proposes a GNN-based framework for pixel-level segmentation and crop-type classification
from Sentinel-2 SITS that preserves full spatial resolution while still benefiting from local
structural coherence. Rather than classifying superpixels, the method builds a graph of individual
pixels inside each superpixel. Thus, it uses superpixels only as local aggregation units for feature computation,
and explicitly consider temporal features into the node attributes so that the network can
learn how spectral signatures evolve across the growing season cycle. The approach combines an attention
mechanism, which weights temporal features, with Chebyshev graph convolutions (ChebConv),
which aggregate spatial information hierarchically within a K-hop neighbourhood. The method was
evaluated on a year-long series of monthly Sentinel-2 images acquired over an agricultural region in
Austria, achieving 87.58% overall accuracy and 78.21% weighted mIoU, and it consistently outperformed
a set of ablated and alternative configurations.

The full thesis manuscript is available here: https://thesis.unipd.it/handle/20.500.12608/102107

