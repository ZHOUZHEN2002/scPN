# scPN: Simultaneous Inference of Pseudotime and Gene Interaction Networks

## Overview

**scPN** is a framework that simultaneously infers pseudotime and gene-gene interaction networks from scRNA-seq data. The framework integrates clustering, piecewise linear modeling, and an iterative EM-style algorithm to recover both temporal dynamics and regulatory relationships among genes.

## Figure 1. The Framework of scPN

![Figure 1](/pipeline_revised.png)

**Figure 1.** *The framework of scPN, which can simultaneously obtain the temporal dynamics and the gene-gene interaction matrix.*

- **(a)** Raw dataset of gene expression. The single-cell gene expression matrix is typically of size \( T 	imes N \), where \( T \) represents the number of cells and \( N \) denotes the number of genes. This matrix is often sparse.
- **(b)** The preprocessing procedure of scPN. It includes normalization, gene selection, imputation, clustering, piecewise linear network modeling, and initialization of the gene-gene interaction matrix using prior knowledge.
- **(c–d)** Constructing individual piecewise networks after clustering. scPN clusters cells using the Leiden algorithm and constructs distinct piecewise gene regulatory networks for each cluster, corresponding to different time intervals.
- **(e)** scPN algorithm. The iterative algorithm, similar to the Expectation-Maximization (EM) algorithm, alternates between inferring pseudotime via a TSP-based approach and estimating the interaction matrix using regression.
- **(f)** Output of scPN. Outputs include single-cell pseudotime, velocity fields, and a gene-gene interaction matrix. Further downstream analysis can be conducted on the learned regulatory networks.

## 🔧 Requirements

To use **scPN**, you need to install the following Python packages:

```bash
pip install scanpy scvelo numpy torch matplotlib
```

### Example

```python
import scPN

data_path = 'path/to/your/adata.h5ad'
adata, adata_100, target_idx = scPN.preprocess(data_path)

# Set the root cluster (leiden label of initial cells)
cluster_root = '16'

adata = scPN.get_adata_all_pseudotime(adata, adata_100, cluster_root, target_idx)

# Visualize pseudotime on UMAP
scPN.pl_pseudotime(adata)


## 🚀 Usage

To run the demo, simply run the cells in Test&Contrast one by one.

Ensure your working directory contains the input data.

## 📫 Contact

For questions or suggestions, feel free to open an issue or contact the authors.
