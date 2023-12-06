import argparse
from staig.adata_processing import LoadSingle10xAdata
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import os

def show_emb(dataset, slide, n_clusters, label):
    # Generate dataset path
    path = f"./Dataset/{dataset}/{slide}"

    # Initialize data loader
    loader = LoadSingle10xAdata(path=path, image_emb=False, label=label, filter_na=True)
    loader.load_data()
    if label:
        loader.load_label()
    adata = loader.adata

    # Load embeddings
    data = np.load(os.path.join(path, 'embeddings_1.npy'))
    data = data.reshape(data.shape[0], -1)

    # Standardize and apply PCA
    scaler = StandardScaler()
    embedding = scaler.fit_transform(data)
    pca = PCA(n_components=128, random_state=42)
    embedding = pca.fit_transform(embedding)
    adata.obsm['emb'] = embedding

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(adata.obsm['emb'].copy())
    labels_str = [f'cluster{label}' for label in labels]
    adata.obs['domain'] = labels_str

    # Plot spatial image with clusters
    sc.pl.spatial(adata, img_key=None,size=2, color=['ground_truth', "domain"], show=False, save=slide+'_img_cluster.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings and visualize.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--slide", type=str, required=True, help="Slide name")
    parser.add_argument("--n_clusters", type=int, default=20, help="Number of clusters")
    parser.add_argument("--label", type=bool, default=False, help="Whether to load labels")

    args = parser.parse_args()

    show_emb(args.dataset, args.slide, args.n_clusters, args.label)
