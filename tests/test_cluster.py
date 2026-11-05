import anndata as ad
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from staig import cluster as C
from staig import metrics


def test_gmm_recovers_blobs():
    x, y = make_blobs(n_samples=600, centers=5, n_features=8, cluster_std=0.8, random_state=0)
    lab = C.gmm_labels(x, 5, random_state=0)
    assert metrics.ari(y, lab) > 0.95


def test_refine_labels_majority_vote_on_grid():
    g = 20
    xs, ys = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    lab = (coords[:, 0] >= 10).astype(int).astype(str)
    noisy = lab.copy()
    rng = np.random.default_rng(0)
    flip = rng.choice(len(lab), 20, replace=False)
    noisy[flip] = np.where(noisy[flip] == "0", "1", "0")
    adata = ad.AnnData(X=np.zeros((len(lab), 2), dtype=np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["dom"] = pd.Categorical(noisy)
    C.refine_labels(adata, key="dom", n_neighbors=8, key_added="ref")
    before = metrics.ari(lab, noisy)
    after = metrics.ari(lab, adata.obs["ref"].values)
    assert after > before and after > 0.9


def test_cluster_api_and_selection():
    x, y = make_blobs(n_samples=300, centers=4, n_features=6, cluster_std=0.7, random_state=1)
    adata = ad.AnnData(X=np.zeros((300, 2), dtype=np.float32))
    adata.obsm["X_staig"] = x.astype(np.float32)
    adata.obsm["spatial"] = np.random.default_rng(0).random((300, 2))
    C.cluster(adata, n_clusters=None, method="gmm", n_clusters_range=range(2, 8))
    assert adata.uns["staig"]["cluster"]["n_clusters"] == 4
    assert metrics.ari(y, adata.obs["staig_domain"].values) > 0.95
    C.cluster(adata, n_clusters=4, method="kmeans", key_added="km")
    assert adata.obs["km"].nunique() == 4


def test_consensus_labels():
    x, y = make_blobs(n_samples=200, centers=3, n_features=4, cluster_std=0.6, random_state=2)
    labs = [C.gmm_labels(x, 3, random_state=s) for s in range(3)]
    cons = C.consensus_labels(labs, 3)
    assert metrics.ari(y, cons) > 0.95


def test_batch_kl_and_lisi_behave():
    rng = np.random.default_rng(0)
    mixed = rng.standard_normal((400, 5))
    batch = np.array(["a", "b"] * 200)
    separated = mixed.copy()
    separated[batch == "b"] += 8.0
    assert metrics.batch_kl(mixed, batch, replicates=20) < metrics.batch_kl(separated, batch, replicates=20)
    assert metrics.ilisi(mixed, batch) > metrics.ilisi(separated, batch)
