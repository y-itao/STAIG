"""Clustering of the STAIG embedding, spatial refinement and cluster-number selection.

``method="gmm"`` is a pure-Python stand-in for the R ``mclust`` EEE model used in v1: a Gaussian
mixture with a shared ("tied") covariance, initialised from Ward hierarchical clustering.
``method="mclust"`` calls R through rpy2 when available (identical to v1).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from ._utils import logger
from .graph import _batch_indices
from .train import EMBEDDING_KEY

DOMAIN_KEY = "staig_domain"


# ----------------------------------------------------------------------------------- GMM
WARD_MAX_N = 15000   # Ward initialisation is O(n^2) memory; larger inputs use k-means++ initialisation


def gmm_labels(x: np.ndarray, n_clusters: int, random_state: int = 0, covariance_type: str = "tied",
               init: str = "ward", n_init: int = 1, max_iter: int = 500, reg_covar: float = 1e-6) -> np.ndarray:
    """Gaussian mixture (mclust-EEE-like) labels for ``x`` (n, d)."""
    x = np.asarray(x, dtype=np.float64)
    n_clusters = int(min(n_clusters, x.shape[0]))
    kw = dict(n_components=n_clusters, covariance_type=covariance_type, random_state=random_state,
              max_iter=max_iter, reg_covar=reg_covar, n_init=n_init)
    if init == "ward" and x.shape[0] <= WARD_MAX_N:
        init_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(x)
        means = np.vstack([x[init_labels == k].mean(0) for k in range(n_clusters)])
        weights = np.bincount(init_labels, minlength=n_clusters) / x.shape[0]
        gm = GaussianMixture(means_init=means, weights_init=weights, **kw)
    elif init == "kmeans" or x.shape[0] > WARD_MAX_N:
        gm = GaussianMixture(init_params="k-means++", **kw)
    else:
        raise ValueError("init must be 'ward' or 'kmeans'")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.fit(x)
    return gm.predict(x)


def mclust_labels(x: np.ndarray, n_clusters: int, model_name: str = "EEE", random_state: int = 2023) -> np.ndarray:
    """R mclust (requires rpy2 and the R package 'mclust'); identical to the v1 ``mclust_R``."""
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri
    except ImportError as e:  # pragma: no cover
        raise ImportError("method='mclust' needs rpy2 and R with the 'mclust' package; "
                          "use method='gmm' (pure Python) instead") from e
    numpy2ri.activate()
    try:
        ro.r("suppressMessages(library(mclust))")
        ro.globalenv["staig_emb"] = numpy2ri.numpy2rpy(np.ascontiguousarray(x, dtype=np.float64))
        ro.r(f"set.seed({int(random_state)})")
        res = ro.r(f'Mclust(staig_emb, G={int(n_clusters)}, modelNames="{model_name}", verbose=FALSE)')
        labels = np.asarray(res.rx2("classification")).astype(int) - 1
    finally:
        numpy2ri.deactivate()
    return labels


def _resolution_search(adata_tmp: ad.AnnData, n_clusters: int, method: str, random_state: int,
                       lo: float = 0.01, hi: float = 3.0, max_iter: int = 40) -> tuple[float, np.ndarray]:
    import scanpy as sc

    run = sc.tl.leiden if method == "leiden" else sc.tl.louvain
    kwargs = {"flavor": "igraph", "n_iterations": 2, "directed": False} if method == "leiden" else {}
    best = None
    for _ in range(max_iter):
        res = (lo + hi) / 2
        run(adata_tmp, resolution=res, random_state=random_state, key_added="_tmp", **kwargs)
        k = adata_tmp.obs["_tmp"].nunique()
        if k == n_clusters:
            return res, adata_tmp.obs["_tmp"].cat.codes.values.copy()
        best = (abs(k - n_clusters), res, adata_tmp.obs["_tmp"].cat.codes.values.copy()) \
            if best is None or abs(k - n_clusters) < best[0] else best
        if k < n_clusters:
            lo = res
        else:
            hi = res
    warnings.warn(f"could not reach exactly {n_clusters} clusters with {method}; using resolution "
                  f"{best[1]:.3f} ({n_clusters + best[0]} or {n_clusters - best[0]} clusters)", stacklevel=2)
    return best[1], best[2]


# ---------------------------------------------------------------------------- refinement
def refine_labels(adata: ad.AnnData, key: str = DOMAIN_KEY, n_neighbors: int = 15,
                  spatial_key: str = "spatial", batch_key: str | None = None,
                  key_added: str | None = None) -> np.ndarray:
    """Majority vote of the ``n_neighbors`` nearest spots (self excluded), as v1 ``refine_label``.

    Ties are broken in favour of the label of the nearest tied spot. Vectorised.
    """
    cat = pd.Categorical(adata.obs[key].astype(str).values)
    codes = cat.codes.astype(np.int64)
    n_lab = len(cat.categories)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    new = codes.copy()
    for idx in _batch_indices(adata, batch_key):
        k = int(min(n_neighbors, len(idx) - 1))
        if k < 1:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords[idx])
        _, nb = nn.kneighbors(coords[idx])
        self_col = nb == np.arange(len(idx))[:, None]
        # drop self (first occurrence) and keep k neighbours per row, in distance order
        keep = ~self_col
        rows_with_self = self_col.any(1)
        keep[~rows_with_self, -1] = False          # self not among the k+1 (duplicates): drop the farthest
        nb_k = nb[keep].reshape(len(idx), k)
        lab = codes[idx][nb_k]                       # (m, k)
        m = len(idx)
        rows = np.repeat(np.arange(m), k)
        cols = np.tile(np.arange(k), m)
        counts = np.zeros((m, n_lab), dtype=np.int64)
        np.add.at(counts, (rows, lab.ravel()), 1)
        first = np.full((m, n_lab), k, dtype=np.int64)
        np.minimum.at(first, (rows, lab.ravel()), cols)
        score = counts * (k + 1) + (k - first)
        new[idx] = np.argmax(score, axis=1)
    out = np.asarray(cat.categories)[new].astype(str)
    adata.obs[key_added or key] = pd.Categorical(out)
    return out


# ------------------------------------------------------------------------ smoothing
def smooth_embedding(adata: ad.AnnData, use_rep: str = EMBEDDING_KEY, graph_key: str = "spatial_connectivities",
                     n_steps: int = 1, alpha: float = 0.5, key_added: str | None = None) -> np.ndarray:
    """Spatially smooth the embedding: x_i <- (1 - alpha) x_i + alpha * mean_{j in N(i)} x_j, repeated
    ``n_steps`` times over the spatial kNN graph. Returns the smoothed matrix (stored under
    ``key_added`` when given)."""
    import scipy.sparse as sp

    x = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    adj = sp.csr_matrix(adata.obsp[graph_key])
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    for _ in range(n_steps):
        x = (1 - alpha) * x + alpha * (adj @ x) / deg[:, None]
    if key_added is not None:
        adata.obsm[key_added] = x.astype(np.float32)
    return x


# ------------------------------------------------------------------------------ cluster
def cluster(
    adata: ad.AnnData,
    n_clusters: int | None = None,
    method: str = "gmm",
    use_rep: str = EMBEDDING_KEY,
    key_added: str = DOMAIN_KEY,
    refine: bool = False,
    refine_neighbors: int = 15,
    batch_key: str | None = None,
    random_state: int = 0,
    n_clusters_range: Sequence[int] = range(3, 16),
    select_metric: str = "silhouette",
    copy: bool = False,
) -> ad.AnnData | None:
    """Cluster ``adata.obsm[use_rep]`` into spatial domains.

    ``method``: ``"gmm"`` (default, pure Python, mclust-EEE-like), ``"mclust"`` (R via rpy2),
    ``"kmeans"``, ``"leiden"`` or ``"louvain"`` (resolution searched to match ``n_clusters``).
    When ``n_clusters`` is None it is chosen in ``n_clusters_range`` by the silhouette score,
    as in the paper for datasets without annotations (candidates are clustered with ``method``, or
    with k-means for leiden/louvain; the scores are stored in ``uns["staig"]["n_clusters_selection"]``
    with string keys). ``n_clusters`` must lie in ``[2, n_obs]``; the number of labels actually
    produced is recorded as ``uns["staig"]["cluster"]["n_clusters_found"]``.
    """
    adata = adata.copy() if copy else adata
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm['{use_rep}'] missing; fit a STAIG model first")
    if method not in {"gmm", "mclust", "kmeans", "leiden", "louvain"}:
        raise ValueError(f"unknown method {method!r}")
    x = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    if n_clusters is None:
        sel_method = method if method in {"gmm", "mclust", "kmeans"} else "kmeans"
        n_clusters, scores = select_n_clusters(x, n_clusters_range, method=sel_method, random_state=random_state,
                                               metric=select_metric)
        adata.uns.setdefault("staig", {})["n_clusters_selection"] = {str(k): float(v) for k, v in scores.items()}
        adata.uns["staig"]["n_clusters_selection_method"] = sel_method
        logger.info("cluster: selected n_clusters=%d by %s (%s)", n_clusters, select_metric, sel_method)
    n_clusters = int(n_clusters)
    if not 2 <= n_clusters <= x.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} must be between 2 and n_obs={x.shape[0]}")
    if method == "gmm":
        labels = gmm_labels(x, n_clusters, random_state=random_state)
    elif method == "mclust":
        labels = mclust_labels(x, n_clusters)
    elif method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(x)
    elif method in {"leiden", "louvain"}:
        import scanpy as sc

        tmp = ad.AnnData(obs=adata.obs[[]].copy())
        tmp.obsm["X_rep"] = x.astype(np.float32)
        sc.pp.neighbors(tmp, n_neighbors=20, use_rep="X_rep", random_state=random_state)
        res, labels = _resolution_search(tmp, n_clusters, method, random_state)
        adata.uns.setdefault("staig", {})["resolution"] = float(res)
    labels = _relabel_by_size(labels)
    adata.obs[key_added] = pd.Categorical(labels.astype(str), categories=[str(i) for i in np.unique(labels)])
    if refine:
        refine_labels(adata, key=key_added, n_neighbors=refine_neighbors, batch_key=batch_key)
    found = int(adata.obs[key_added].nunique())
    if found != n_clusters:
        logger.info("cluster: %s produced %d labels for n_clusters=%d", method, found, n_clusters)
    adata.uns.setdefault("staig", {})["cluster"] = dict(method=method, n_clusters=n_clusters, n_clusters_found=found,
                                                        refine=bool(refine), refine_neighbors=int(refine_neighbors))
    return adata if copy else None


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 0..k-1 by decreasing size (stable, cosmetic)."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts, kind="stable")]
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels)


def select_n_clusters(x: np.ndarray, candidates: Sequence[int], method: str = "gmm", random_state: int = 0,
                      metric: str = "silhouette") -> tuple[int, dict[int, float]]:
    """Pick the number of clusters maximising the silhouette score (paper protocol).

    ``method`` must be ``"gmm"``, ``"mclust"`` or ``"kmeans"`` (resolution-based methods cannot be
    asked for a fixed k); ``metric`` must be ``"silhouette"``. Returns ``(best_k, {k: score})``.
    """
    if method not in {"gmm", "mclust", "kmeans"}:
        raise ValueError(f"select_n_clusters supports method 'gmm', 'mclust' or 'kmeans', got {method!r}")
    if metric != "silhouette":
        raise ValueError(f"only metric='silhouette' is implemented, got {metric!r}")
    scores: dict[int, float] = {}
    for k in candidates:
        if k < 2 or k >= x.shape[0]:
            continue
        if method == "gmm":
            lab = gmm_labels(x, k, random_state=random_state)
        elif method == "mclust":
            lab = mclust_labels(x, k)
        else:
            lab = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(x)
        if len(np.unique(lab)) < 2:
            continue
        scores[int(k)] = float(silhouette_score(x, lab))
    if not scores:
        raise ValueError("no valid number of clusters in the candidate range")
    best = max(scores, key=scores.get)
    return best, scores


def consensus_labels(label_sets: Sequence[np.ndarray], n_clusters: int, random_state: int = 0,
                     max_dense: int = 20000) -> np.ndarray:
    """Consensus of several clusterings (e.g. different seeds).

    Up to ``max_dense`` points: average-linkage clustering of the co-association matrix. Larger
    inputs: k-means on the stacked one-hot membership matrix (cluster-based similarity partitioning).
    """
    labs = [pd.Categorical(np.asarray(lab).astype(str)).codes for lab in label_sets]
    n = len(labs[0])
    if n <= max_dense:
        co = np.zeros((n, n), dtype=np.float32)
        for lab in labs:
            co += (lab[:, None] == lab[None, :])
        co /= len(labs)
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        return model.fit_predict(1.0 - co)
    import scipy.sparse as sp

    blocks = []
    for lab in labs:
        k = int(lab.max()) + 1
        blocks.append(sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), lab)), shape=(n, k)))
    h = sp.hstack(blocks).toarray() / np.sqrt(len(labs))
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(h)
