import anndata as ad
import numpy as np

from staig import graph as G


def _v1_construct_interaction(position, n_neighbors):
    """Port of v1 ``construct_interaction`` (dense)."""
    from scipy.spatial.distance import cdist

    dm = cdist(position, position)
    n = dm.shape[0]
    inter = np.zeros((n, n))
    for i in range(n):
        order = dm[i].argsort()
        for t in range(1, n_neighbors + 1):
            inter[i, order[t]] = 1
    adj = inter + inter.T
    return np.where(adj > 1, 1, adj)


def test_knn_matches_v1_on_random_coordinates():
    rng = np.random.default_rng(0)
    coords = rng.random((300, 2)) * 1000
    a = G._knn_symmetric(coords, 5).toarray()
    b = _v1_construct_interaction(coords, 5)
    assert np.array_equal(a, b)
    assert np.all(np.diag(a) == 0)


def test_block_diagonal_across_batches():
    rng = np.random.default_rng(1)
    coords = rng.random((200, 2))
    adata = ad.AnnData(X=np.zeros((200, 3), dtype=np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["slice"] = ["a"] * 100 + ["b"] * 100
    G.spatial_neighbors(adata, n_neighbors=4, batch_key="slice")
    adj = adata.obsp["spatial_connectivities"].toarray()
    assert adj[:100, 100:].sum() == 0 and adj[100:, :100].sum() == 0
    assert np.array_equal(adj, adj.T)


def test_edge_probabilities_rows_sum_to_one_image_mode():
    rng = np.random.default_rng(2)
    n = 150
    adata = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
    adata.obsm["spatial"] = rng.random((n, 2))
    adata.obsm["image_pcs"] = rng.standard_normal((n, 16)).astype(np.float32)
    G.spatial_neighbors(adata, n_neighbors=5)
    G.edge_removal_probabilities(adata, mode="image")
    p = adata.obsp["staig_edge_prob"]
    assert p.nnz == adata.obsp["spatial_connectivities"].nnz
    rs = np.asarray(p.sum(1)).ravel()
    assert np.allclose(rs, 1.0, atol=1e-5)
    # p_ij proportional to the euclidean distance of the image PCs (Eq. 4)
    c = adata.obsm["image_pcs"]
    i = 3
    cols = p[i].indices
    w = np.linalg.norm(c[i] - c[cols], axis=1)
    assert np.allclose(p[i].data, w / w.sum(), atol=1e-5)


def test_edge_probabilities_fixed_and_gene():
    rng = np.random.default_rng(3)
    n = 120
    adata = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
    adata.obsm["spatial"] = rng.random((n, 2))
    adata.obsm["X_staig_input"] = rng.random((n, 50)).astype(np.float32)
    G.spatial_neighbors(adata, n_neighbors=5)
    G.edge_removal_probabilities(adata, mode="fixed", fixed_drop_prob=0.3)
    assert np.allclose(adata.obsp["staig_edge_prob"].data, 0.3)
    G.edge_removal_probabilities(adata, mode="gene", n_gene_pcs=10)
    p = adata.obsp["staig_edge_prob"]
    assert np.all(p.data >= 0) and np.allclose(np.asarray(p.sum(1)).ravel(), 1.0, atol=1e-5)
