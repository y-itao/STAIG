import argparse
import os
import random
import yaml
from yaml import SafeLoader
import torch
from staig.adata_processing import LoadSingle10xAdata,LoadBatch10xAdata
import numpy as np
from staig.staig import STAIG
import pandas as pd

if __name__ == '__main__':

    # 数据集路径
    file_fold = './Dataset'

    # 加载预训练参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='visium')
    parser.add_argument('--slide', type=str, default='integration_horizontal')
    parser.add_argument('--config', type=str, default='train_img_config.yaml')
    parser.add_argument('--label', type=bool, default=False)
    parser.add_argument('--filelist', nargs='+', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[str(args.slide)]
    slide_path = os.path.join(file_fold, args.dataset)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(12345)

    data = LoadBatch10xAdata(dataset_path=slide_path,file_list = args.filelist,n_neighbors=config['num_neigh'],n_top_genes=config['num_gene'],image_emb=False, label = args.label).run()

    staig = STAIG(args=args,config=config,single=True)    
    staig.adata = data
    staig.train()
    staig.eva()
    staig.cluster(args.label)
    staig.draw_umap()
    staig.draw_horizontal()