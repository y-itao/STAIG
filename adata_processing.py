import scanpy as sc
import ot
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.special import softmax
from anndata import AnnData
from scipy.linalg import block_diag


class LoadSingle10xAdata:
    def __init__(self, path: str, n_top_genes: int = 3000, n_neighbors: int = 3, image_emb: bool = False, label: bool = True, filter_na: bool = True):
        self.path = path
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na


    def load_data(self):
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()

    def preprocess(self):
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        sc.pp.scale(self.adata, zero_center=False, max_value=10)

    def construct_interaction(self):
        position = self.adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)

        self.adata.obsm['graph_neigh'] = adj

    def generate_gene_expr(self):
        adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        self.adata.obsm['feat'] = feat

    def load_label(self):
        df_meta = pd.read_csv(os.path.join(self.path, 'truth.txt'), sep='\t', header=None)
        df_meta_layer = df_meta[1]

        self.adata.obs['ground_truth'] = df_meta_layer.values
        # filter out NA nodes
        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]

    def load_image_emb(self):
        data = np.load(os.path.join(self.path, 'embeddings.npy'))
        data = data.reshape(data.shape[0], -1)
        scaler = StandardScaler()
        embedding = scaler.fit_transform(data)
        pca = PCA(n_components=128, random_state=42)
        embedding = pca.fit_transform(embedding)
        self.adata.obsm['img_emb'] = embedding

    def calculate_edge_weights(self):
        # 获取现有的邻接矩阵和节点 embedding
        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['img_emb']

        # 计算所有节点之间的欧氏距离
        euclidean_distances = cdist(node_emb, node_emb, metric='euclidean')

        # 计算邻边权重矩阵
        edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)

        # 将邻边权重转换为概率（用 softmax 函数）
        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(edge_weights.shape[0]):
            # edge_probabilities[i] = softmax(-edge_weights[i]) # 注意：这里用负号，使得距离近的节点具有较低的概率被删除。
            non_zero_indices = edge_weights[i] != 0
            non_zero_weights = np.log(edge_weights[i][non_zero_indices] + 1)  # 使用对数函数进行缩放，+1是为了避免对零取对数
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i][non_zero_indices] = softmax_weights
        # 将概率矩阵存储到 adata
        self.adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        if self.label:
            self.load_label()
        self.preprocess()
        self.construct_interaction()
        self.generate_gene_expr()

        if self.image_emb:
            self.load_image_emb()
            self.calculate_edge_weights()

        return self.adata


class LoadBatch10xAdata:
    def __init__(self, dataset_path: str, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
                 image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
        self.dataset_path = dataset_path  # until dataset path (like ./Dataset/DLPFC)
        self.file_list = file_list  # slice name list
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata_list = []
        self.adata_len = []
        self.merged_adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.do_log = do_log

    def construct_interaction(self, input_adata):
        position = input_adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)
        input_adata.obsm['local_graph'] = adj
        return input_adata

    def load_data(self):
        for i in self.file_list:
            print('now load: ' + i)
            load_path = os.path.join(self.dataset_path, i)
            adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
            adata.var_names_make_unique()
            if self.label:
                df_meta = pd.read_csv(os.path.join(load_path, 'truth.txt'), sep='\t', header=None)
                df_meta_layer = df_meta[1]
                adata.obs['ground_truth'] = df_meta_layer.values
                print(i + ' load label done')
                if self.filter_na:
                    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
                    print(i + ' filter NA done')
            if self.image_emb:
                data = np.load(os.path.join(load_path, 'embeddings.npy'))
                data = data.reshape(data.shape[0], -1)
                scaler = StandardScaler()
                embedding = scaler.fit_transform(data)
                pca = PCA(n_components=128, random_state=42)
                embedding = pca.fit_transform(embedding)
                adata.obsm['img_emb'] = embedding
                print(i + ' load img embedding done')
            adata = self.construct_interaction(input_adata=adata)
            print(i + ' build local graph done')
            self.adata_list.append(adata)
            self.adata_len.append(adata.X.shape[0])
            print(i + ' added to list')
        print('load all slices done')

        return self.adata_list

    def concatenate_slices(self):
        adata = AnnData.concatenate(*self.adata_list, join='outer')
        if self.do_log:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=False, max_value=10)
            print('log transform done')

        adata_Vars = adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        adata.obsm['feat'] = feat
        self.merged_adata = adata
        print('merge done')
        return self.merged_adata

    def construct_whole_graph(self):
        matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
        adjacency = block_diag(*matrix_list)
        self.merged_adata.obsm['graph_neigh'] = adjacency
        return self.merged_adata

    def calculate_edge_weights(self):
        # 获取现有的邻接矩阵和节点 embedding
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['img_emb']

        # 计算所有节点之间的欧氏距离
        euclidean_distances = cdist(node_emb, node_emb, metric='euclidean')

        # 计算邻边权重矩阵
        edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)

        # 将邻边权重转换为概率（用 softmax 函数）
        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(edge_weights.shape[0]):
            # edge_probabilities[i] = softmax(-edge_weights[i]) # 注意：这里用负号，使得距离近的节点具有较低的概率被删除。
            non_zero_indices = edge_weights[i] != 0
            non_zero_weights = np.log(edge_weights[i][non_zero_indices] + 1)  # 使用对数函数进行缩放，+1是为了避免对零取对数
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i][non_zero_indices] = softmax_weights
        # 将概率矩阵存储到 adata
        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        self.concatenate_slices()
        self.construct_whole_graph()
        if self.image_emb:
            self.calculate_edge_weights()
        return self.merged_adata
    
class LoadBatchAdata:
    def __init__(self, dataset_path: str, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
                 image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
        self.dataset_path = dataset_path  # until dataset path (like ./Dataset/DLPFC)
        self.file_list = file_list  # slice name list
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata_list = []
        self.adata_len = []
        self.merged_adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.do_log = do_log

    def construct_interaction(self, input_adata):
        position = input_adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1

        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)
        input_adata.obsm['local_graph'] = adj
        return input_adata

    def load_data(self):
        for i in self.file_list:
            print('now load: ' + i)
            load_path = os.path.join(self.dataset_path, i)
            adata = sc.read_h5ad(load_path)
            adata.var_names_make_unique()

            adata = self.construct_interaction(input_adata=adata)
            adata.var['new_column'] = 1
            print(adata)
            print(i + ' build local graph done')
            self.adata_list.append(adata)
            self.adata_len.append(adata.X.shape[0])
            print(i + ' added to list')
        print('load all slices done')

        return self.adata_list

    def concatenate_slices(self):
        adata = AnnData.concatenate(*self.adata_list, join='outer')
        if self.do_log:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=False, max_value=10)
            print('log transform done')

        adata_Vars = adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()[:, ]
        else:
            feat = adata_Vars.X[:, ]

        adata.obsm['feat'] = feat
        self.merged_adata = adata
        print('merge done')
        return self.merged_adata

    def construct_whole_graph(self):
        matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
        adjacency = block_diag(*matrix_list)
        self.merged_adata.obsm['graph_neigh'] = adjacency
        return self.merged_adata

    def calculate_edge_weights(self):
        # 获取现有的邻接矩阵和节点 embedding
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['img_emb']

        # 计算所有节点之间的欧氏距离
        euclidean_distances = cdist(node_emb, node_emb, metric='euclidean')

        # 计算邻边权重矩阵
        edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)

        # 将邻边权重转换为概率（用 softmax 函数）
        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(edge_weights.shape[0]):
            # edge_probabilities[i] = softmax(-edge_weights[i]) # 注意：这里用负号，使得距离近的节点具有较低的概率被删除。
            non_zero_indices = edge_weights[i] != 0
            non_zero_weights = np.log(edge_weights[i][non_zero_indices] + 1)  # 使用对数函数进行缩放，+1是为了避免对零取对数
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i][non_zero_indices] = softmax_weights
        # 将概率矩阵存储到 adata
        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        self.concatenate_slices()
        self.construct_whole_graph()
        if self.image_emb:
            self.calculate_edge_weights()
        return self.merged_adata

