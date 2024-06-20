# STAIG
STAIG: Spatial Transcriptomics Analysis via Image-Aided Graph Contrastive Learning for Domain Deciphering and Alignment-Free Integration

## Overview
We unveil Spatial Transcriptomics and Image-based Graph learning (STAIG), a deep learning framework for advanced spatial region identification in spatial transcriptomics. STAIG integrates gene expression, spatial coordinates, and histological images using graph contrastive learning, excelling in feature extraction and enhancing analysis on datasets with or without histological images. STAIG was specifically engineered to counter batch effects and to allow for the integration of tissue slices without pre-alignment. It is compatible with multiple platforms, including 10x Visium, Slide-seq, Stereo-seq, and STARmap.

## System Requirements

### Hardware Requirements

- Memory: 16GB or higher for efficient processing.
- GPU: NVIDIA GPU (A100/3090) is highly recommended for accelerating training times.

### Operating System Requirements

- Linux: Ubuntu 16.04 or newer.


## Environment Setup

This section details the steps to set up the project environment using Anaconda.

### Prerequisites

- R with Mclust, nabor package
- Python 3.10.11
- pytorch==2.2.2

### Cloning the Repository and Preparing the Environment

Actual installation time depends on network conditions and takes about 15 minutes.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/y-itao/STAIG.git
   cd STAIG
   ```
   **or download the code**:
   ```bash
   wget https://github.com/y-itao/STAIG/archive/refs/heads/main.zip
   unzip main.zip
   cd /home/.../STAIG-main  ### your own path
   ```
2. **Create and Activate the Environment**:
   ```bash
   conda create -n staig_env python=3.10
   conda activate staig_env
   
   ## step1 Installing PyTorch 
   # For GPU (CUDA 12.1)
   conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

   ## step2 Installing Pyg
   conda install pyg -c pyg
      
   ## step3 Download other dependencies
   pip install -r requirements.txt
   ```
## Usage

Note that we conducted experiments with the A100/3090 on Linux. 

Before running, please download the compressed folder of the `Dataset` from [Google drive](https://drive.google.com/file/d/1dN9pLRyVXFXLxopt2FD82NDrwFqUjliu/view?usp=sharing) and decompress it in `./`, After decompression, the dir structure under `./` will be: 

```bash
/home/.../STAIG
|-- Dataset
|   |-- 9.Mouse_Brain_Merge_Anterior_Posterior_Section_1
|   |   `-- filtered_feature_bc_matrix.h5ad
|   |-- DLPFC
|   |   |-- 151507
|   |   |-- 151508
|   |   |-- 151673
|   |   |-- 151675
|   |   `-- 151676
|   |-- Slide-seqV2
|   |   `-- v2.h5ad
|   |-- merfish
|   |   |-- mouse1.AUD_TEA_VIS.242.unexpand_cluster3.h5ad
|   |   `-- mouse2.AUD_TEA_VIS.242.unexpand_cluster3.h5ad
|   |-- stereo-seq
|   |   `-- Mouse_Olfactory
|   `-- visium
|       |-- Human_Breast_Cancer
|       |-- Mouse_Brain_Anterior
|       `-- Mouse_Brain_Posterior
|-- example
|   |-- SOME EXAMPLES.ipynb

|-- requirements.txt
|-- staig
|   |-- __init__.py
|   |-- adata_processing.py
|   |-- batchKL.R
|   |-- calLISI.R
|   |-- metrics.py
|   |-- net.py
|   |-- staig.py
|   `-- utils.py
|-- train_img_config.yaml
`-- train_no_img_config.yaml
```

### Extract image features by BYOL

We assume the root path for the dataset is `./Dataset`, and the path for the DLPFC dataset's 151673 slice is `./Dataset/DLPFC/151673`.

The example of extract image features is under `./examples`.


2.	`Image_feature_extraction-151673`: The process and display of image feature extraction in DLPFC for #151673.


Change the work_dir to `./` when loading packages. Then, clik and run.


We assume the root path for the dataset is `./Dataset`, and the path for the DLPFC dataset's 151673 slice is `./Dataset/DLPFC/151673`.


### Spatial Domain Identification on the 10x Visium Platform (Case #151673)

The example is under `./examples`. Change the work_dir to `./` when loading packages. Then, clik and run.




1. **Image mode**: `Spatial_clustering-151673-img`: Displays the clustering results based on image similarity for #151673, with an ARI of 0.68.


### Alignment-Free Integration (10x Visium)

The example is under `./examples`. Change the work_dir to `./` when loading packages. Then, clik and run.

1. **vertical integration**

`Integration_vertical-07087576`: Displays the integration based on slices from different sources in the DLPFC, for Figure 5c in the paper.


`Integration_vertical-7576`: Displays the integration of adjacent DLPFC slices #151675 and #151676, with an ARI of 0.64, corresponding to Figure 5a in the paper.


2. **horizontal integration （mouse brain）**

`Integration_horizontal`: Displays the results of horizontal integration, corresponding to Figure 5b in the paper.

### Alignment-Free Integration (MERFISH)

`Integration_vertical-merfish`: Displays the integration results from the MERFISH platform for two mice on the same site, corresponding to Figure 5d in the paper.

### Alignment-Free Integration (Cross platform)


`Integration_cross-mini`: Displays the cross-platform integration of Stereo-seq and Slide_seqV2. Due to platform GPU memory limitations, this is a sampled version with 7000 random points from each dataset. 


`Integration_cross`: Displays the cross-platform integration of Stereo-seq and Slide_seqV2. 

## Compared tools

Tools that are compared include: 

* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [Seurat](https://satijalab.org/seurat/)
* [SEDR](https://github.com/JinmiaoChenLab/SEDR/)
* [DeepST](https://github.com/JiangBioLab/DeepST)
* [GraphST](https://github.com/JinmiaoChenLab/GraphST)
* [ConST](https://github.com/ys-zong/conST)
* [STAGATE](https://github.com/zhanglabtools/STAGATE)

## Download data

The data we used for training can be downloaded from [here]. We also provide datasets processed by STAIG, which can be downloaded from [here](https://drive.google.com/file/d/1wxmRnhjXxH3eV52dvv5d7gH_YM9Ist14/view?usp=sharing) (in `h5ad` format, where `obs['domain']` is the clustering result, `obsm['emb']` is the low-dimensional feature, and `obsm['img_emb']` is the image feature after dimensionality reduction).

## License

This project is covered under the Apache 2.0 License.

## Acknowledgments
Parts of our work are based on code from [GraphST](https://github.com/JinmiaoChenLab/GraphST)，[GCA](https://github.com/CRIPAC-DIG/GCA)，[GDCL](https://github.com/hzhao98/GDCL)以及[NCLA](https://github.com/shenxiaocam/NCLA).We are very grateful for their contributions. Thank WANG Tao for his assistance.

