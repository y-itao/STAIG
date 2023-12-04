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



