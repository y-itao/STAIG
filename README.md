# STAIG
STAIG: Spatial Transcriptomics Analysis via Image-Aided Graph Contrastive Learning for Domain Deciphering and Alignment-Free Integration
## Overview
We unveil Spatial Transcriptomics and Image-based Graph learning (STAIG), a deep learning framework for advanced spatial region identification in spatial transcriptomics. STAIG integrates gene expression, spatial coordinates, and histological images using graph contrastive learning, excelling in feature extraction and enhancing analysis on datasets with or without histological images. STAIG was specifically engineered to counter batch effects and to allow for the integration of tissue slices without pre-alignment. It is compatible with multiple platforms, including 10x Visium, Slide-seq, Stereo-seq, and STARmap.

## Environment Setup

This section details the steps to set up the project environment using Anaconda.

### Prerequisites

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
   # For GPU:
   conda install pyg -c pyg
   
   # For CPU (Mac os wtih M1,2,3):
   
   ## step3 Download other dependencies
   pip install -r requirements.txt
   ```
## Usage
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
### 10x visium平台的Spatial Domain 识别（以#151673为例）
主要运行的脚本为train_with_image.py, 具体的超参可以由外置的train_img_config.yaml控制。
train_with_image.py由四个，'--dataset'是数据集名字(默认为DLPFC)，'--slide'切片名字（默认为151673）,'--label'是否存在类标（默认为true），'--config'其他超参的yaml配置文件
```bash
python train_with_image.py --dataset DLPFC  --slide 151673  --label True --config train_img_config.yaml
```
the cluster result is shown in `./figures`
注意，事例结果从我们在linux的A100显卡的实验下获得。训练完成的model参数提供在‘./example_model/151673/model.pt’。
### 其他平台的spatial domain 识别
主要运行的脚本为train_without_image.py, 具体的超参可以由外置的train_no_img_config.yaml控制。
数据集事先转化为h5ad形式，并且已经做好前置处理。
train_with_image.py由四个，'--dataset'是数据集名字，'--slide'切片名字，，'--config'其他超参的yaml配置文件
```bash
python train_without_image.py 
```
the cluster result is shown in `./figures`
### alignment free integration （10x visium）
1. **vertical integration**



