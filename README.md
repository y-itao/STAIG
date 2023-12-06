# STAIG
STAIG: Spatial Transcriptomics Analysis via Image-Aided Graph Contrastive Learning for Domain Deciphering and Alignment-Free Integration
## Overview
We unveil Spatial Transcriptomics and Image-based Graph learning (STAIG), a deep learning framework for advanced spatial region identification in spatial transcriptomics. STAIG integrates gene expression, spatial coordinates, and histological images using graph contrastive learning, excelling in feature extraction and enhancing analysis on datasets with or without histological images. STAIG was specifically engineered to counter batch effects and to allow for the integration of tissue slices without pre-alignment. It is compatible with multiple platforms, including 10x Visium, Slide-seq, Stereo-seq, and STARmap.

## Environment Setup

This section details the steps to set up the project environment using Anaconda.

### Prerequisites

- R with Mclust package
- Python 3.10.11
- pytorch==1.13.1

### Cloning the Repository and Preparing the Environment

1. **Clone the Repository**:
   ```bash
   git clone your-git-repo-url
   cd xxx
   ```
   **or download the code**:
   ```bash
   wget xxxx/main.zip
   unzip main.zip
   cd /home/.../STAIG-main  ### your own path
   ```
2. **Create and Activate the Environment**:
   ```bash
   conda create -n staig_env python=3.10
   conda activate staig_env
   
   ## step1 Installing PyTorch 
   # For GPU (CUDA 11.7)
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   # For CPU
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch

   ## step2 Installing Pyg
   conda install pyg -c pyg
      
   ## step3 Download other dependencies
   pip install -r requirements.txt
   ```
## Usage
Note that we conducted experiments with the A100 on Linux. 
### Extract image features by BYOL
We assume the root path for the dataset is `./Dataset`, and the path for the DLPFC dataset's 151673 slice is `./Dataset/DLPFC/151673`.

1. **Image Cropping Based on Coordinates**
   The cropped images are stored in `./Dataset/DLPFC/151673/clip_image`.
   Input variables: `dataset` is the dataset name, `slide` is the slice name, `patch_size` is the size of the cropped image (default is 512), `label` indicates whether the image includes class labels.
   ```bash
   python image_step1_clip.py --dataset DLPFC --slide 151673 --patch_size 512 --label True
   ```
2. **Further Processing with Gaussian Blur and Bandpass Filters**
   Processed images are stored in `./Dataset/DLPFC/151673/clip_image_filter.`
   Input variables: `dataset` is the dataset name, `slide` is the slice name, `lower` and `upper` are the lower and upper frequency limits for the bandpass filter, typically around half the value of `patch_size` (default is `245-275`).
   ```bash
   python image_step2_filter.py --dataset DLPFC --slide 151673 --lower 245 --upper 275
   ```
3. **Feature Extraction using BYOL**
   Extracted image features are stored in `./Dataset/DLPFC/151673/embeddings.npy`.
   Input variables: `dataset` is the dataset name, `slide` is the slice name, `epoch_num` is the number of iterations.
   ```bash
   python image_step3_byol.py --dataset DLPFC --slide 151673 --epoch_num 200
   ```

4. **Displaying Image Features with KNN**
   Results are outputted to `./figures.`
   Input variables: `dataset` is the dataset name, `slide` is the slice name, `n_clusters` is the number of clusters (default is 20), `label` indicates whether the image includes class labels.
   ```bash
   python image_step4_show.py --dataset DLPFC --slide 151673 --n_clusters 20 --label True
   ```
### Spatial Domain Identification on the 10x Visium Platform (Case #151673)
The primary script for execution is `train_with_image.py`, with specific hyperparameters controllable through an external `train_img_config.yaml`.
`train_with_image.py` accepts four arguments: `--dataset` for the dataset name (default is DLPFC), `--slide` for the slide name (default is 151673), `--label` indicating if class labels exist (default is true), and `--config` for other hyperparameters in the yaml configuration file, by default `train_img_config.yaml`.
```bash
python example_train_with_image.py --dataset DLPFC  --slide 151673  --label True --config train_img_config.yaml
```
The clustering result is displayed in `./figures`.
The trained model parameters are available in `./example_model/151673/model.pt`.

### Spatial Domain Identification on Other Platforms
The main script for execution is `train_without_image.py`, with specific hyperparameters controllable through an external `train_no_img_config.yaml`.
The dataset is pre-converted into `h5ad` format and pre-processed.
`train_without_image.py` accepts four arguments: `--dataset` for the dataset name, `--slide` for the slide name, and `--config` for other hyperparameters in the yaml configuration file, by default `train_no_img_config.yaml`.

```bash
python example_train_without_image.py 
```
The clustering result is displayed in `./figures`.

### Alignment-Free Integration (10x Visium)
Before running, please download the compressed folder of the `Dataset` from [xxx] and decompress it in `./`, After decompression, the file structure under `./` will be: xxx.


1. **vertical integration**
`example_integration_vertical.py` takes four parameters: `--dataset` for the dataset name (default is DLPFC), `--slide` for the slide name (default is `integration_vertical`), `--label` indicating if class labels exist (default is true), `--config` for other hyperparameters in the yaml configuration file, default is `train_img_config.yaml`, and `--filelist` for the names of slides to integrate, separated by spaces. The result is shown in `./figures`.

```bash
python example_integration_vertical.py  --filelist 151675 151676
```
2. **horizontal integration （mouse brain）**
```bash
python example_integration_horizontal.py  --filelist Mouse_Brain_Anterior Mouse_Brain_Posterior
```
3. **partial integration**
```bash
python example_integration_partial.py  --filelist WS_PLA_S9101764 WS_PLA_S9101765 WS_PLA_S9101767
```

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

The data we used for training can be downloaded from [here]. We also provide datasets processed by STAIG, which can be downloaded from [here] (in `h5ad` format, where `obs['domain']` is the clustering result, `obsm['emb']` is the low-dimensional feature, and `obsm['img_emb']` is the image feature after dimensionality reduction).

## Acknowledgments
Parts of our work are based on code from [GraphST](https://github.com/JinmiaoChenLab/GraphST)，[GCA](https://github.com/CRIPAC-DIG/GCA)，[GDCL](https://github.com/hzhao98/GDCL)以及[NCLA](https://github.com/shenxiaocam/NCLA).We are very grateful for their contributions. Thank WANG Tao for his assistance.

