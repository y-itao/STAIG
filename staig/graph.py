"""Spatial graph construction, image PCs, pseudo-labels and edge-removal probabilities.

All graphs are stored as scipy CSR matrices in ``adata.obsp`` (never dense N x N).
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ._utils import fingerprint, logger, to_dense_f32

GRAPH_KEY = "spatial_connectivities"
EDGE_PROB_KEY = "staig_edge_prob"
IMAGE_PCS_KEY = "image_pcs"
IMAGE_FEATURES_KEY = "image_features"
PSEUDO_LABEL_KEY = "staig_pseudo_label"


def _batch_codes(adata: ad.AnnData, batch_key: str | None) -> np.ndarray | None:
    """Integer batch codes, or None without a batch key. Raises for an unknown column or missing values."""
    if batch_key is None:
        return None
    if batch_key not in adata.obs:
        raise KeyError(f"batch_key {batch_key!r} is not a column of adata.obs "
                       f"(available: {list(adata.obs.columns)[:20]})")
    codes = np.asarray(pd.Categorical(adata.obs[batch_key]).codes)
    if (codes < 0).any():
        raise ValueError(f"adata.obs[{batch_key!r}] contains missing values")
    return codes


def _batch_indices(adata: ad.AnnData, batch_key: str | None):
    codes = _batch_codes(adata, batch_key)
    if codes is None:
        return [np.arange(adata.n_obs)]
    return [np.where(codes == c)[0] for c in np.unique(codes)]


def _batch_fingerprint(adata: ad.AnnData, batch_key: str | None) -> str | None:
    codes = _batch_codes(adata, batch_key)
    return None if codes is None else fingerprint(codes)


def obsm_fingerprint(adata: ad.AnnData, key: str) -> str:
    """Fingerprint of ``adata.obsm[key]`` as stored (dtype-exact; sparse matrices are hashed without densifying)."""
    x = adata.obsm[key]
    return fingerprint(x) if sp.issparse(x) else fingerprint(np.asarray(x))


def coords_fingerprint(adata: ad.AnnData, spatial_key: str = "spatial") -> str:
    return fingerprint(np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2])


TIE_WINDOW = 8   # extra neighbours queried per spot; a tie group reaching past the window is completed by radius


def _knn_symmetric(coords: np.ndarray, k: int) -> sp.csr_matrix:
    """Binary symmetric kNN graph without self loops (v1 ``construct_interaction``).

    Each spot keeps its ``k`` nearest other spots; exact distance ties (frequent on the Visium hex
    lattice, where the 6 neighbours are equidistant and k = 5) are broken deterministically in favour
    of the lower spot index, independent of the kd-tree's internal order and of the number of tied
    spots (a tie group that reaches past the ``k + TIE_WINDOW`` nearest candidates is completed with a
    radius query), so the graph is identical on every platform.
    """
    n = coords.shape[0]
    k = int(min(k, n - 1))
    if k < 1:
        return sp.csr_matrix((n, n), dtype=np.float32)
    m = int(min(n, k + 1 + TIE_WINDOW))
    nn = NearestNeighbors(n_neighbors=m, algorithm="kd_tree").fit(coords)
    dist, idx = nn.kneighbors(coords)
    self_mask = idx == np.arange(n)[:, None]
    dk = np.sort(np.where(self_mask, np.inf, dist), axis=1)[:, k - 1]     # distance of the k-th nearest other spot
    tol = 1.0 + 1e-9
    # rows whose tie group at the k-th distance may extend beyond the query window: complete them by radius
    need = (m < n) & (dist[:, -1] <= dk * tol)
    extra_rows, extra_cols = [], []
    for i in np.flatnonzero(need):
        d_i, j_i = (v[0] for v in nn.radius_neighbors(coords[i:i + 1], radius=dk[i] * tol + 1e-12))
        others = j_i != i
        d_i, j_i = d_i[others], j_i[others]
        pick = j_i[np.lexsort((j_i, d_i))][:k]
        extra_rows.append(np.full(pick.size, i))
        extra_cols.append(pick)
    rows = np.repeat(np.arange(n), m)
    cols = idx.ravel()
    d = dist.ravel()
    keep = (rows != cols) & ~need[rows]      # drop self explicitly (robust to duplicate coordinates)
    rows, cols, d = rows[keep], cols[keep], d[keep]
    # per row: sort by distance, then by index; keep exactly k neighbours
    order = np.lexsort((cols, d, rows))
    rows, cols = rows[order], cols[order]
    first = np.searchsorted(rows, np.arange(n))
    counts = np.diff(np.append(first, rows.size))
    sel = np.concatenate([np.arange(f, f + min(k, c)) for f, c in zip(first, counts)])
    rows, cols = rows[sel], cols[sel]
    if extra_rows:
        rows = np.concatenate([rows, *extra_rows])
        cols = np.concatenate([cols, *extra_cols])
    knn = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adj = ((knn + knn.T) > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return sp.csr_matrix(adj)


def spatial_neighbors(
    adata: ad.AnnData,
    n_neighbors: int = 5,
    batch_key: str | None = None,
    spatial_key: str = "spatial",
    key_added: str = GRAPH_KEY,
    copy: bool = False,
) -> ad.AnnData | None:
    """Build the symmetric kNN spatial graph; block-diagonal across ``batch_key`` groups."""
    adata = adata.copy() if copy else adata
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must be an (n_obs, >=2) array")
    coords = coords[:, :2]
    n = adata.n_obs
    blocks = []
    for idx in _batch_indices(adata, batch_key):
        a = _knn_symmetric(coords[idx], n_neighbors).tocoo()
        blocks.append(sp.coo_matrix((a.data, (idx[a.row], idx[a.col])), shape=(n, n)))
    adj = sp.csr_matrix(sum(blocks[1:], blocks[0]) if len(blocks) > 1 else blocks[0], dtype=np.float32)
    adj.sum_duplicates()
    adata.obsp[key_added] = adj
    deg = np.asarray(adj.sum(1)).ravel()
    adata.uns.setdefault("staig", {})["graph"] = dict(n_neighbors=int(n_neighbors), batch_key=batch_key,
                                                       n_edges=int(adj.nnz // 2), mean_degree=float(deg.mean()),
                                                       coords=coords_fingerprint(adata, spatial_key),
                                                       batch=_batch_fingerprint(adata, batch_key))
    logger.info("spatial_neighbors: k=%d, %d undirected edges, mean degree %.2f",
                n_neighbors, adj.nnz // 2, deg.mean())
    return adata if copy else None


def image_pcs(
    adata: ad.AnnData,
    n_comps: int = 16,
    features_key: str = IMAGE_FEATURES_KEY,
    key_added: str = IMAGE_PCS_KEY,
    batch_key: str | None = None,
    random_state: int = 42,
    copy: bool = False,
) -> ad.AnnData | None:
    """StandardScaler + PCA of the raw image features (v1: 16 PCs single slice, 128 per slice
    for integration). PCA is fitted per batch when ``batch_key`` is given."""
    adata = adata.copy() if copy else adata
    feats = to_dense_f32(adata.obsm[features_key]).reshape(adata.n_obs, -1)
    out = np.zeros((adata.n_obs, n_comps), dtype=np.float32)
    for idx in _batch_indices(adata, batch_key):
        x = StandardScaler().fit_transform(feats[idx].astype(np.float64))
        k = int(min(n_comps, x.shape[0] - 1, x.shape[1]))
        out[idx, :k] = PCA(n_components=k, random_state=random_state, svd_solver="full").fit_transform(x)
    adata.obsm[key_added] = out
    adata.uns.setdefault("staig", {})["image_pcs"] = dict(n_comps=int(n_comps), batch_key=batch_key,
                                                           n_features=int(feats.shape[1]),
                                                           features=obsm_fingerprint(adata, features_key),
                                                           batch=_batch_fingerprint(adata, batch_key))
    return adata if copy else None


def pseudo_labels(
    adata: ad.AnnData,
    n_clusters: int = 40,
    key: str = IMAGE_PCS_KEY,
    key_added: str = PSEUDO_LABEL_KEY,
    random_state: int = 0,
    n_init: int = 10,
    copy: bool = False,
) -> ad.AnnData | None:
    """k-means pseudo-labels on image PCs for the debiased negative sampling (v1 ``generate_pseudo_labels``)."""
    adata = adata.copy() if copy else adata
    x = np.asarray(adata.obsm[key], dtype=np.float64)
    n_clusters = int(min(n_clusters, x.shape[0]))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(x)
    adata.obs[key_added] = pd.Categorical(km.labels_.astype(str))
    adata.uns.setdefault("staig", {})["pseudo_labels"] = dict(n_clusters=int(n_clusters), key=key,
                                                            random_state=int(random_state),
                                                            source=obsm_fingerprint(adata, key))
    sizes = np.bincount(km.labels_)
    logger.info("pseudo_labels: %d clusters, size min/median/max = %d/%d/%d",
                n_clusters, sizes.min(), int(np.median(sizes)), sizes.max())
    return adata if copy else None


def _row_softmax_over_edges(adj: sp.csr_matrix, w: np.ndarray, log_input: bool) -> np.ndarray:
    """Row-wise softmax restricted to the non-zero weights of each row (v1 semantics).

    ``w`` holds one weight per stored entry of ``adj`` (CSR order). Entries with w == 0 get p = 0.
    With ``log_input=True`` the softmax is taken over log(w), i.e. p_ij = w_ij / sum_k w_ik.
    """
    p = np.zeros_like(w, dtype=np.float64)
    indptr = adj.indptr
    for i in range(adj.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e <= s:
            continue
        wi = w[s:e]
        nz = wi != 0
        if not nz.any():
            continue
        v = np.log(wi[nz]) if log_input else wi[nz]
        v = v - v.max()
        ev = np.exp(v)
        p[s:e][nz] = ev / ev.sum()
    return p


def edge_removal_probabilities(
    adata: ad.AnnData,
    mode: str = "auto",
    fixed_drop_prob: float = 0.2,
    image_key: str = IMAGE_PCS_KEY,
    features_key: str = "X_staig_input",
    n_gene_pcs: int = 64,
    graph_key: str = GRAPH_KEY,
    key_added: str = EDGE_PROB_KEY,
    batch_key: str | None = None,
    random_state: int = 42,
    image_metric: str = "euclidean",
    copy: bool = False,
) -> ad.AnnData | None:
    """Per-edge removal probabilities stored with the sparsity pattern of the spatial graph.

    * ``"image"``: w_ij = ||c_i - c_j||_2 on image PCs (or the cosine distance with
      ``image_metric="cosine"``), p_ij = w_ij / sum_k A_ik w_ik (paper Eq. 4).
    * ``"gene"``:  cosine distance between spots on ``n_gene_pcs`` PCs of the model input (scaler and
      PCA fitted once on all spots, also for multi-slice data, as v1 ``calculate_edge_weights_gene``),
      p = row-softmax over neighbours.
    * ``"fixed"``: p_ij = ``fixed_drop_prob`` for every edge.
    * ``"auto"``: image if image PCs are available, else gene.

    ``batch_key`` is validated (the graph must already be block-diagonal) but does not change the
    probabilities. ``uns["staig"]["edge_removal"]`` records the parameters and fingerprints of the
    graph and of the source matrix so that ``STAIG.prepare()`` can detect stale probabilities.
    """
    adata = adata.copy() if copy else adata
    if graph_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{graph_key}'] missing; run staig.graph.spatial_neighbors first")
    adj = sp.csr_matrix(adata.obsp[graph_key])
    adj.sort_indices()
    rows = np.repeat(np.arange(adj.shape[0]), np.diff(adj.indptr))
    cols = adj.indices
    if mode == "auto":
        mode = "image" if image_key in adata.obsm else "gene"
    if mode == "image":
        if image_key not in adata.obsm:
            raise KeyError(f"adata.obsm['{image_key}'] missing; run staig.img.extract_image_features "
                           "and staig.graph.image_pcs first, or use mode='gene'/'fixed'")
        c = np.asarray(adata.obsm[image_key], dtype=np.float64)
        if image_metric == "euclidean":
            w = np.linalg.norm(c[rows] - c[cols], axis=1)
        elif image_metric == "cosine":
            cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            w = np.clip(1.0 - np.sum(cn[rows] * cn[cols], axis=1), 0.0, 2.0)
        else:
            raise ValueError("image_metric must be 'euclidean' or 'cosine'")
        p = _row_softmax_over_edges(adj, w, log_input=True)
    elif mode == "gene":
        _batch_codes(adata, batch_key)
        x = to_dense_f32(adata.obsm[features_key]).astype(np.float64)
        xs = StandardScaler().fit_transform(x)
        k = int(min(n_gene_pcs, xs.shape[0] - 1, xs.shape[1]))
        e = PCA(n_components=k, random_state=random_state).fit_transform(xs)
        en = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        w = 1.0 - np.sum(en[rows] * en[cols], axis=1)
        w = np.clip(w, 0.0, 2.0)
        p = _row_softmax_over_edges(adj, w, log_input=False)
    elif mode == "fixed":
        p = np.full(adj.nnz, float(fixed_drop_prob), dtype=np.float64)
    else:
        raise ValueError(f"mode must be image|gene|fixed|auto, got {mode!r}")
    adata.obsp[key_added] = sp.csr_matrix((p.astype(np.float32), cols.copy(), adj.indptr.copy()),
                                          shape=adj.shape)
    if p.size == 0:
        warnings.warn("the spatial graph has no edges (batches of a single spot?); the model will only use the "
                      "cross-view self term", stacklevel=2)
    adata.uns.setdefault("staig", {})["edge_removal"] = dict(mode=mode, mean_prob=float(p.mean()) if p.size else 0.0,
                                                           image_metric=image_metric if mode == "image" else None,
                                                           fixed_drop_prob=float(fixed_drop_prob),
                                                           n_gene_pcs=int(n_gene_pcs),
                                                           graph=fingerprint(adj),
                                                           source=edge_source_fingerprint(adata, mode, image_key,
                                                                                          features_key))
    logger.info("edge_removal_probabilities: mode=%s, mean p=%.3f", mode, p.mean() if p.size else 0.0)
    return adata if copy else None


def edge_source_fingerprint(adata: ad.AnnData, mode: str, image_key: str = IMAGE_PCS_KEY,
                            features_key: str = "X_staig_input") -> str | None:
    """Fingerprint of the matrix the edge-removal probabilities of ``mode`` are derived from."""
    if mode == "image":
        return obsm_fingerprint(adata, image_key) if image_key in adata.obsm else None
    if mode == "gene":
        return obsm_fingerprint(adata, features_key) if features_key in adata.obsm else None
    return None


def check_graph_inputs(adata: ad.AnnData, graph_key: str = GRAPH_KEY, prob_key: str = EDGE_PROB_KEY) -> None:
    adj = adata.obsp.get(graph_key)
    if adj is None:
        raise KeyError("spatial graph missing")
    prob = adata.obsp.get(prob_key)
    if prob is not None and (prob.nnz != adj.nnz):
        warnings.warn("edge probabilities and graph have different sparsity; recomputing is advised",
                      stacklevel=2)
