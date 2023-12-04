import argparse
import os
import random
import yaml
from yaml import SafeLoader
import torch
from adata_processing import LoadSingle10xAdata,LoadBatchAdata
import numpy as np
from staig import STAIG
import scanpy as sc

if __name__ == '__main__':
    os.chdir('/root/STAIG/')
    # 数据集路径
    file_fold = '/root/GLST/Dataset'
    fl=['merfish_1_-0.24.h5ad','merfish_1_-0.19.h5ad','merfish_1_-0.14.h5ad']
    data = LoadBatchAdata(dataset_path='/root/STGCA/Data/merfish',n_neighbors=5,n_top_genes=161,file_list=fl).run()
    data.write('merfish.h5ad',compression='gzip')
    print(data)