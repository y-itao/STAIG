import argparse
import os
import random
import yaml
from yaml import SafeLoader
import torch
from staig.adata_processing import LoadSingle10xAdata,LoadSingleAdata
import numpy as np
from staig.staig import STAIG
import pandas as pd
if __name__ == '__main__':

    # 数据集路径
    file_fold = './Dataset'

    # 加载预训练参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stereo-seq')
    parser.add_argument('--slide', type=str, default='Mouse_Olfactory')
    parser.add_argument('--config', type=str, default='train_no_img_config.yaml')
    parser.add_argument('--label', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=39)
    parser.add_argument('--num_epochs', type=int, default=340)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[str(args.slide)]
    slide_path = os.path.join(file_fold, args.dataset, args.slide,'filtered_feature_bc_matrix_norm.h5ad')

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(12345)
    # config['tau'] = args.tau
    # config['num_epochs']=args.num_epochs

    data = LoadSingleAdata(path=slide_path,n_neighbors=config['num_neigh'],image_emb=False, label = args.label).run()

    staig = STAIG(args=args,config=config,single=True)    
    staig.adata = data
    staig.train()
    staig.eva()
    staig.cluster(args.label)
    staig.draw_single_spatial()