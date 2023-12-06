import argparse
import os
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch_geometric.nn import GCNConv
from adata_processing import LoadSingle10xAdata,LoadBatch10xAdata
import numpy as np
from utils import clustering
from sklearn import metrics
import scanpy as sc
from sklearn.cluster import KMeans
from net import Encoder, MVmodel, SVmodel, drop_feature, dropout_adj, random_dropout_adj
from scipy.special import softmax
import pickle
import datetime
import matplotlib.pyplot as plt


def generate_pseudo_labels(img_emb, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    pseudo_labels = kmeans.labels_
    return torch.tensor(pseudo_labels)


def adj_to_edge_index(adj):
    row, col = torch.where(adj != 0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def convert_edge_probabilities(adj_matrix, edge_prob_matrix):
    row, col = torch.where(adj_matrix != 0)
    edge_probs = edge_prob_matrix[row, col]
    return edge_probs



class STAIG:

    def __init__(self,args,config, single: bool = False):
        self.args = args
        self.single = single

        self.learning_rate = config['learning_rate']
        self.num_hidden = config['num_hidden']
        self.num_proj_hidden = config['num_proj_hidden']
        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
        self.base_model = ({'GCNConv': GCNConv})[config['base_model']]
        self.num_layers = config['num_layers']
        self.drop_feature_rate_1 = config['drop_feature_rate_1']
        self.drop_feature_rate_2 = config['drop_feature_rate_2']
        self.tau = config['tau']
        self.num_epochs = config['num_epochs']
        self.weight_decay = config['weight_decay']
        self.num_clusters = config['num_clusters']
        self.num_gene = config['num_gene']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.radius = 15
        self.tool = 'mclust' # mclust, leiden, and louvain

        self.encoder = Encoder(self.num_gene, self.num_hidden, self.activation, base_model=self.base_model, k=self.num_layers).to(self.device)
        self.adata = None

        if single:
            self.model = SVmodel(self.encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device)
            self.drop_edge_rate_1 = config['drop_edge_rate_1']
            self.drop_edge_rate_2 = config['drop_edge_rate_2']
        else:
            self.model = MVmodel(self.encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device)
        # print('start para')
        # self.export_model_parameters("initial_model_parameterszzz.csv")
        # print('done para')

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def export_model_parameters(self, file_name):
        # 初始化一个用于存储所有数据的列表
        data = []

        # 获取排序后的模型参数名称
        sorted_names = sorted(self.model.state_dict().keys())

        # 遍历排序后的参数名称，提取参数
        for name in sorted_names:
            param = self.model.state_dict()[name]
            # 转换参数为一维数组
            param_data = param.cpu().numpy().flatten()

            # 将参数数据和层名称添加到列表
            for value in param_data:
                data.append([name, value])

        # 一次性将数据转换为 DataFrame
        parameters_df = pd.DataFrame(data, columns=['Layer', 'Value'])

        # 保存到 CSV 文件
        parameters_df.to_csv(file_name, index=False)



    def train(self):
        if self.adata is None:
            raise ValueError("adata not load!")
        if self.single is False:
            features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        
            graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device)
            edge_probabilities = torch.FloatTensor(self.adata.obsm['edge_probabilities'].copy()).to(self.device)
           
            pseudo_labels = generate_pseudo_labels(self.adata.obsm['img_emb'], 40).to(self.device)
            edge_index = adj_to_edge_index(graph_neigh)
            edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)

            start = datetime.datetime.now()
            prev = start

            for epoch in range(1, self.num_epochs + 1):

                self.model.train()
                self.optimizer.zero_grad()
                edge_index_1 = dropout_adj(edge_index, edge_probs, force_undirected=True)[0]
                edge_index_2 = dropout_adj(edge_index, edge_probs, force_undirected=True)[0]
                x_1 = drop_feature(features_matrix, self.drop_feature_rate_1)
                x_2 = drop_feature(features_matrix, self.drop_feature_rate_2)
                z1 = self.model(x_1, edge_index_1)
                z2 = self.model(x_2, edge_index_2)
                loss = self.model.contrastive_loss_bias(z1, z2, graph_neigh, pseudo_labels)
                loss.backward()
                self.optimizer.step()
                if epoch == 5:
                    self.export_model_parameters(f"model_parameters_after_epoch_{epoch}.csv")
                now = datetime.datetime.now()
                print('(T) | Epoch={:03d}, loss={:.4f}, this epoch {}, total {}'.format(epoch, loss.item(), now - prev, now - start))
                prev = now
            torch.save(self.model.state_dict(), 'model.pt')

        if self.single:
            features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)

            graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device)
            edge_index = adj_to_edge_index(graph_neigh)

            start = datetime.datetime.now()
            prev = start

            for epoch in range(1, self.num_epochs + 1):

                self.model.train()
                self.optimizer.zero_grad()
                edge_index_1 = random_dropout_adj(edge_index, p=self.drop_edge_rate_1, force_undirected=True)[0]
                edge_index_2 = random_dropout_adj(edge_index, p=self.drop_edge_rate_2, force_undirected=True)[0]
                x_1 = drop_feature(features_matrix, self.drop_feature_rate_1)
                x_2 = drop_feature(features_matrix, self.drop_feature_rate_2)
                z1 = self.model(x_1, edge_index_1)
                z2 = self.model(x_2, edge_index_2)
                loss = self.model.contrastive_loss(z1, z2, graph_neigh)
                loss.backward()
                self.optimizer.step()
                now = datetime.datetime.now()
                print('(T) | Epoch={:03d}, loss={:.4f}, this epoch {}, total {}'.format(epoch, loss.item(), now - prev, now - start))
                prev = now
            torch.save(self.model.state_dict(), 'model.pt')

    def eva(self):
        print("=== load ===")
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()
        features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device)
        edge_index = adj_to_edge_index(graph_neigh)
        self.adata.obsm['emb'] = self.model(features_matrix, edge_index).detach().cpu().numpy()

    def cluster(self,label=True):
        if self.tool == 'mclust':
            clustering(self.adata, self.num_clusters, radius=self.radius, method=self.tool,
                       refinement=True)  # For DLPFC dataset, we use optional refinement step.
        elif self.tool in ['leiden', 'louvain']:
            clustering(self.adata, self.num_clusters, radius=self.radius, method=self.tool, start=0.01, end=0.27, increment=0.005,
                       refinement=True)

        if label:
            print('calculate metric ARI')
            # calculate metric ARI
            ARI = metrics.adjusted_rand_score(self.adata.obs['domain'], self.adata.obs['ground_truth'])
            self.adata.uns['ari'] = ARI

            print('ARI:', ARI)

    def draw_spatial(self,p=''):
        sc.pl.spatial(self.adata,
                      img_key='hires',
                      size=1.6,
                      color=["domain"],
                      show=True, save=p+str(self.args.slide)+'.png')

    def draw_single_spatial(self):
        sc.pl.embedding(self.adata, basis="spatial", color="domain",
                        save=str(self.args.slide)+'.png')

    def draw_umap(self):
        sc.pp.neighbors(self.adata, use_rep='norm_emb')
        sc.tl.umap(self.adata)
        sc.pl.umap(self.adata, color='domain', show=False, save=str(self.args.slide)+ '_label.png')
        sc.pl.umap(self.adata, color='batch', show=False, save=str(self.args.slide)+ '_batch.png')
        
    def draw_horizontal(self):
        adata_batch_0 = self.adata[self.adata.obs['batch'] == '0', :]
        sc.pl.embedding(adata_batch_0, basis="spatial", color="domain",
                        save='0.png')

        adata_batch_1 = self.adata[self.adata.obs['batch'] == '1', :]
        sc.pl.embedding(adata_batch_1, basis="spatial", color="domain",
                        save='1.png')


