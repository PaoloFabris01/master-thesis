# Agricultural Field Segmentation and Crop Type Mapping in Satellite Image Time Series Data by applying Graph Neural Network
Abstract

Modern agriculture is increasingly a data-intensive domain, in which satellite Earth observation supports
sustainable and timely decision-making. Satellite Image Time Series (SITS), and in particular
the Sentinel-2 mission of the Copernicus programme, provide dense multispectral and multitemporal
observations that capture the phenological evolution of crops throughout the growing season. Exploiting
this information for automatic field segmentation and crop-type mapping is, however, challenging,
since SITS data are high-dimensional and require models able to jointly capture spatial, spectral and
temporal dependencies.
Deep learning has substantially advanced SITS analysis, with recurrent, convolutional, and
transformer-based architectures achieving strong results on benchmark datasets.1, 2 More recently,
Graph Neural Networks (GNNs) have emerged as an attractive alternative, since they can represent
an image as a graph. Nodes correspond to spatial units and edges encode neighbourhood relations.
In this manner has been possible to overcome the rigid, regular-grid assumption of standard convolutions.
Most existing GNN-based approaches for SITS, however, build graphs at the superpixel level,
aggregating pixels to keep the graph tractable.3, 4 While computationally convenient, this choice discards
fine spatial detail that is often crucial for delineating agricultural field boundaries and capturing
within-field heterogeneity.
This work proposes a GNN-based framework for pixel-level segmentation and crop-type classification
from Sentinel-2 SITS that preserves full spatial resolution while still benefiting from local
structural coherence. Rather than classifying superpixels, the method builds a graph of individual
pixels inside each superpixel. Thus, it uses superpixels only as local aggregation units for feature computation,
and explicitly consider temporal features into the node attributes so that the network can
learn how spectral signatures evolve across the growing season cycle. The approach combines an attention
mechanism, which weights temporal features, with Chebyshev graph convolutions (ChebConv),5
which aggregate spatial information hierarchically within a K-hop neighbourhood. The method was
evaluated on a year-long series of monthly Sentinel-2 images acquired over an agricultural region in
Austria, achieving 87.58% overall accuracy and 78.21% weighted mIoU, and it consistently outperformed
a set of ablated and alternative configurations.
