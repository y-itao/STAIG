import scanpy as sc
import ot
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist,euclidean,cosine
from scipy.special import softmax
from anndata import AnnData
from scipy.linalg import block_diag
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from tqdm import tqdm
import scGeneClust as gc
import PyWGCNA
import NaiveDE
import SpatialDE

def generate_pseudo_labels(img_emb, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    pseudo_labels = kmeans.labels_
    return pseudo_labels

class LoadSingle10xAdata:
    def __init__(self, path: str, n_top_genes: int = 3000, n_neighbors: int = 3, image_emb: bool = False, label: bool = True, filter_na: bool = True,select = 'default'):
        self.path = path
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.kernel = 'euclidean'
        self.select = 'default'


    def load_data(self):
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()


    def preprocess(self):
        if self.select == 'default':
            sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
        if self.select == 'mvp':
            sc.pp.highly_variable_genes(self.adata, flavor="seurat")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
        if self.select == 'geneclust':
            self.adata.X = self.adata.X.toarray()
            info, selected_genes_ps = gc.scGeneClust(self.adata, n_var_clusters=200, version='fast', return_info=True)
            top_variable_genes = selected_genes_ps.tolist()
            self.adata.var['highly_variable'] = self.adata.var_names.isin(top_variable_genes)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)
        if self.select == 'wgcna':
            pyWGCNA_data = PyWGCNA.WGCNA(name='data', 
                              species='human', 
                              anndata=self.adata, 
                              outputPath='',
                              save=True)
            pyWGCNA_data.preprocess()
            pyWGCNA_data.findModules()
            module_colors = np.unique(pyWGCNA_data.datExpr.var['moduleColors']).tolist()
            all_hub_genes = []
            for module_color in module_colors:
                df_hub_genes = pyWGCNA_data.top_n_hub_genes(moduleName=module_color, n=100)
                gene_names = df_hub_genes.index.tolist()  
                all_hub_genes.extend(gene_names)  
            self.adata.var['highly_variable'] = self.adata.var_names.isin(all_hub_genes)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)

        if self.select == 'spatialde':
            x_coords = self.adata.obsm['spatial'][:, 0]
            y_coords = self.adata.obsm['spatial'][:, 1]
            counts = pd.DataFrame(
                self.adata.X.toarray(),
                index=self.adata.obs_names,
                columns=self.adata.var_names
            )
            counts = counts.T[counts.sum(0) >= 3].T
            self.adata.obs['total_counts'] = np.ravel(self.adata.X.sum(axis=1))
            sample_info = pd.DataFrame({
                'x': x_coords,
                'y': y_coords,
                'total_counts': self.adata.obs['total_counts']
            }, index=self.adata.obs_names)
            norm_expr = NaiveDE.stabilize(counts.T).T
            resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T
            sample_resid_expr = resid_expr.sample(n=1000, axis=1, random_state=1)
            X = sample_info[['x', 'y']].values 
            results = SpatialDE.run(X, resid_expr)
            top_genes_list = results.sort_values('qval')['g'].head(1000).tolist()
            self.adata.var['highly_variable'] = self.adata.var_names.isin(top_genes_list)
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

        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]

    def load_image_emb(self):
        data = np.load(os.path.join(self.path, 'embeddings.npy'))
        data = data.reshape(data.shape[0], -1)
        scaler = StandardScaler()
        embedding = scaler.fit_transform(data)
        pca = PCA(n_components=16, random_state=42)
        embedding = pca.fit_transform(embedding)
        self.adata.obsm['img_emb'] = embedding
        pca_g = PCA(n_components=64, random_state=42)
        self.adata.obsm['feat_pca'] = pca_g.fit_transform(self.adata.obsm['feat'])
        self.adata.obsm['con_feat'] = np.concatenate([self.adata.obsm['feat_pca'], self.adata.obsm['img_emb']], axis=1)
        con_feat = self.adata.obsm['con_feat']

        scaler = StandardScaler()

        con_feat_standardized = scaler.fit_transform(con_feat)

        self.adata.obsm['con_feat'] = con_feat_standardized


    def calculate_edge_weights(self):

        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)  

        
        for i in tqdm(range(num_nodes), desc="Calculating distances"):  
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():  
                non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.adata.obsm['edge_probabilities'] = edge_probabilities


        if self.kernel=='rbf':

            gamma = 0.01 
            similarity_matrix = rbf_kernel(node_emb, gamma=gamma)
            

            edge_weights = np.where(graph_neigh == 1, 1 - similarity_matrix, 0)
            

            edge_probabilities = np.zeros_like(edge_weights)
            for i in range(edge_weights.shape[0]):
                non_zero_indices = edge_weights[i] != 0
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)  
                edge_probabilities[i][non_zero_indices] = softmax_weights


            self.adata.obsm['edge_probabilities'] = edge_probabilities

        if self.kernel=='cosine':

            euclidean_distances = cdist(node_emb, node_emb, metric='cosine')


            edge_weights = np.where(graph_neigh == 1, euclidean_distances, 0)


            edge_probabilities = np.zeros_like(edge_weights)
            for i in range(edge_weights.shape[0]):
                non_zero_indices = edge_weights[i] != 0
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)  
                edge_probabilities[i][non_zero_indices] = softmax_weights

            self.adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):

        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

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
        else:
            self.calculate_edge_weights_gene()

        print('adata load done')

        return self.adata

class LoadSingleAdata:
    def __init__(self, path: str,  n_neighbors: int = 3, image_emb: bool = False, label: bool = False, filter_na: bool = True,n_top_genes: int = 161):
        self.path = path
        self.n_neighbors = n_neighbors
        self.adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.n_top_genes = n_top_genes

    def load_data(self):
        self.adata = sc.read_h5ad(self.path)   
        self.adata.var_names_make_unique()

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

    def preprocess(self):
        sc.pp.highly_variable_genes(self.adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)

    def calculate_edge_weights(self):
        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)  

        for i in tqdm(range(num_nodes), desc="Calculating distances"):  
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():  
                non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):

        graph_neigh = self.adata.obsm['graph_neigh']
        node_emb = self.adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

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
        else:
            self.calculate_edge_weights_gene()

        return self.adata
    
class LoadBatchAdata_cross:
    def __init__(self, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
                 image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
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

            print(i)
            sc.pp.highly_variable_genes(i, flavor="seurat_v3", n_top_genes=5000)
            adata = self.construct_interaction(input_adata=i)

            self.adata_list.append(adata)
            self.adata_len.append(adata.X.shape[0])

        print('load all slices done')

        return self.adata_list

    def concatenate_slices(self):

        highly_variable_genes_set = set(self.adata_list[0].var['highly_variable'][self.adata_list[0].var['highly_variable']].index)


        for adata in self.adata_list[1:]:

            current_set = set(adata.var['highly_variable'][adata.var['highly_variable']].index)
            highly_variable_genes_set = highly_variable_genes_set.intersection(current_set)


        feat_standardized_list = []

        for adata in self.adata_list:
            adata_Vars = adata[:, adata.var.index.isin(highly_variable_genes_set)]
            if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
                feat = adata_Vars.X.toarray()[:]
            else:
                feat = adata_Vars.X[:]
            

            # scaler = StandardScaler()

            # feat_standardized = scaler.fit_transform(feat)
            

            # feat_standardized_list.append(feat_standardized)
            feat_standardized_list.append(feat)


        adata = AnnData.concatenate(*self.adata_list, join='outer')


        merged_feat_standardized = np.concatenate(feat_standardized_list, axis=0)


        adata.obsm['feat'] = merged_feat_standardized

        self.merged_adata = adata
        print(self.merged_adata.obsm['feat'].shape)
        print('merge done')
        return self.merged_adata

    def construct_whole_graph(self):
        matrix_list = [i.obsm['local_graph'] for i in self.adata_list]
        adjacency = block_diag(*matrix_list)
        self.merged_adata.obsm['graph_neigh'] = adjacency

        mask_list = [np.ones_like(i.obsm['local_graph'], dtype=int) for i in self.adata_list]
        mask = block_diag(*mask_list)
        self.merged_adata.obsm['mask_neigh'] = mask

    def calculate_edge_weights(self):
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)  

        for i in tqdm(range(num_nodes), desc="Calculating distances"):  
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():  
                non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):

        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        self.concatenate_slices()
        self.construct_whole_graph()
        if self.image_emb:
            self.calculate_edge_weights()
        else:
            self.calculate_edge_weights_gene()
        return self.merged_adata

class LoadBatch10xAdata:
    def __init__(self, dataset_path: str, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
                 image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
        self.dataset_path = dataset_path  
        self.file_list = file_list  
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
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
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

        highly_variable_genes_set = set(self.adata_list[0].var['highly_variable'][self.adata_list[0].var['highly_variable']].index)


        for adata in self.adata_list[1:]:

            current_set = set(adata.var['highly_variable'][adata.var['highly_variable']].index)
            highly_variable_genes_set = highly_variable_genes_set.intersection(current_set)

        adata = AnnData.concatenate(*self.adata_list, join='outer')

        adata_Vars = adata[:, adata.var.index.isin(highly_variable_genes_set)]
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

        mask_list = [np.ones_like(i.obsm['local_graph'], dtype=int) for i in self.adata_list]
        mask = block_diag(*mask_list)
        self.merged_adata.obsm['mask_neigh'] = mask

    def calculate_edge_weights(self):
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)  

        for i in tqdm(range(num_nodes), desc="Calculating distances"):  
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():  
                non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):

        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        self.concatenate_slices()
        self.construct_whole_graph()
        if self.image_emb:
            self.calculate_edge_weights()
        else:
            self.calculate_edge_weights_gene()
        print('merge adata load done')
        return self.merged_adata

class LoadBatchAdata:
    def __init__(self, dataset_path: str, file_list: list, n_top_genes: int = 3000, n_neighbors: int = 5,
                 image_emb: bool = False, label: bool = True, filter_na: bool = True, do_log:bool=True):
        self.dataset_path = dataset_path  
        self.file_list = file_list  
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

            print(i)
            sc.pp.highly_variable_genes(i, flavor="seurat_v3", n_top_genes=234)
            sc.pp.normalize_total(i, target_sum=1e4)
            sc.pp.log1p(i)
            adata = self.construct_interaction(input_adata=i)
            self.adata_list.append(adata)
            self.adata_len.append(adata.X.shape[0])
        print('load all slices done')

        return self.adata_list

    def concatenate_slices(self):

        highly_variable_genes_set = set(self.adata_list[0].var['highly_variable'][self.adata_list[0].var['highly_variable']].index)


        for adata in self.adata_list[1:]:

            current_set = set(adata.var['highly_variable'][adata.var['highly_variable']].index)
            highly_variable_genes_set = highly_variable_genes_set.intersection(current_set)

        adata = AnnData.concatenate(*self.adata_list, join='outer')

        adata_Vars = adata[:, adata.var.index.isin(highly_variable_genes_set)]
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

        mask_list = [np.ones_like(i.obsm['local_graph'], dtype=int) for i in self.adata_list]
        mask = block_diag(*mask_list)
        self.merged_adata.obsm['mask_neigh'] = mask

    def calculate_edge_weights(self):
        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['img_emb']
        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros_like(graph_neigh)  

        for i in tqdm(range(num_nodes), desc="Calculating distances"):  
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = euclidean(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in tqdm(range(num_nodes), desc="Calculating edge_probabilities"):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():  
                non_zero_weights = np.log(edge_weights[i][non_zero_indices]) 
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def calculate_edge_weights_gene(self):

        graph_neigh = self.merged_adata.obsm['graph_neigh']
        node_emb = self.merged_adata.obsm['feat']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(node_emb)
        pca = PCA(n_components=64, random_state=42)
        embedding = pca.fit_transform(embedding)
        node_emb = embedding

        num_nodes = node_emb.shape[0]
        edge_weights = np.zeros((num_nodes, num_nodes))

        for i in tqdm(range(num_nodes), desc="Calculating distances"):
            for j in range(num_nodes):
                if graph_neigh[i, j] == 1:  
                    edge_weights[i, j] = cosine(node_emb[i], node_emb[j])

        edge_probabilities = np.zeros_like(edge_weights)
        for i in range(num_nodes):
            non_zero_indices = edge_weights[i] != 0
            if non_zero_indices.any():
                non_zero_weights = edge_weights[i][non_zero_indices]
                softmax_weights = softmax(non_zero_weights)
                edge_probabilities[i][non_zero_indices] = softmax_weights

        self.merged_adata.obsm['edge_probabilities'] = edge_probabilities

    def run(self):
        self.load_data()
        self.concatenate_slices()
        self.construct_whole_graph()
        if self.image_emb:
            self.calculate_edge_weights()
        else:
            self.calculate_edge_weights_gene()
        return self.merged_adata
    
