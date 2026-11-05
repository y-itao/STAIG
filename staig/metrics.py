"""Evaluation metrics: ARI/NMI, silhouette/Davies-Bouldin, BatchKL and (i/c)LISI — all in Python."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def _mask_valid(labels_true, labels_pred):
    t = pd.Series(np.asarray(labels_true, dtype=object))
    p = pd.Series(np.asarray(labels_pred, dtype=object))
    ok = t.notna().values & p.notna().values & (t.astype(str) != "nan").values
    return t[ok].astype(str).values, p[ok].astype(str).values


def ari(labels_true, labels_pred) -> float:
    t, p = _mask_valid(labels_true, labels_pred)
    return float(adjusted_rand_score(t, p))


def nmi(labels_true, labels_pred) -> float:
    t, p = _mask_valid(labels_true, labels_pred)
    return float(normalized_mutual_info_score(t, p))


def silhouette(x, labels) -> float:
    return float(silhouette_score(np.asarray(x), np.asarray(labels).astype(str)))


def davies_bouldin(x, labels) -> float:
    return float(davies_bouldin_score(np.asarray(x), np.asarray(labels).astype(str)))


def calinski_harabasz(x, labels) -> float:
    return float(calinski_harabasz_score(np.asarray(x), np.asarray(labels).astype(str)))


def _valid_labels(labels) -> np.ndarray:
    """Boolean mask of non-missing labels (None, NaN and the strings 'nan'/'NA'/'' count as missing)."""
    lab = np.asarray(labels, dtype=object)
    return np.asarray(pd.notna(lab)) & ~np.isin(lab.astype(str), ["nan", "NA", "NaN", ""])


def batch_kl(x, batch, replicates: int = 200, n_cells: int = 100, n_neighbors: int = 100,
             random_state: int = 1) -> float:
    """KL divergence of batch mixing (Li et al., Nat Commun 2020), port of the v1 R function.

    For each of ``replicates`` draws (without replacement) of ``n_cells`` cells, the batch
    composition of each cell's ``k = min(5 * n_batches, n_neighbors)`` nearest neighbours (self
    included) is compared with the global batch proportions with KL (log2). Lower = better mixing.
    Cells with a missing batch label are left out (as ``table(factor(...))`` drops NA in R).
    """
    valid = _valid_labels(batch)
    if not valid.any():
        raise ValueError("batch_kl: no cells with a valid batch label")
    x = np.asarray(x, dtype=np.float64)[valid]
    b = pd.Categorical(np.asarray(batch, dtype=object)[valid].astype(str))
    codes = b.codes
    n_b = len(b.categories)
    global_p = np.bincount(codes, minlength=n_b) / len(codes)
    k = int(min(5 * n_b, n_neighbors, x.shape[0]))
    rng = np.random.RandomState(random_state)
    nn = NearestNeighbors(n_neighbors=k).fit(x)
    kls = []
    for _ in range(replicates):
        sample = rng.choice(x.shape[0], size=min(n_cells, x.shape[0]), replace=False)
        _, idx = nn.kneighbors(x[sample])
        local = np.stack([np.bincount(codes[row], minlength=n_b) for row in idx]) / k
        with np.errstate(divide="ignore", invalid="ignore"):
            term = local * np.log2(local / global_p)
        term = np.where(local > 0, term, 0.0)
        kls.append(float(np.nanmean(term.sum(1))))
    return float(np.mean(kls))


def lisi(x, labels, perplexity: int = 30, n_neighbors: int | None = None) -> np.ndarray:
    """Per-cell Local Inverse Simpson's Index (Korsunsky et al. 2019) for ``labels``.

    Pure-Python port of the reference algorithm (R package ``lisi`` / ``harmonypy.compute_lisi``):
    for every cell the ``3 * perplexity`` nearest neighbours are weighted by a Gaussian kernel whose
    width is tuned by binary search so that the weights have the requested perplexity; the LISI is the
    inverse Simpson's index of the label composition under those weights (1 = one label, up to the
    number of labels for perfect mixing).
    """
    valid = _valid_labels(labels)
    out = np.full(len(valid), np.nan)                         # NaN for cells without a label
    where = np.flatnonzero(valid)
    x = np.asarray(x, dtype=np.float64)[valid]
    lab = pd.Categorical(np.asarray(labels, dtype=object)[valid].astype(str))
    codes = lab.codes
    n_lab = len(lab.categories)
    n = x.shape[0]
    if n < 2:
        return out
    k = int(n_neighbors or perplexity * 3)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    dist, idx = nn.kneighbors(x)
    # drop self by index (with duplicated points self is not necessarily the first neighbour); when self
    # is not among the k+1 returned points (more than k duplicates) drop the farthest instead
    self_col = idx == np.arange(n)[:, None]
    keep = ~self_col
    keep[~self_col.any(1), -1] = False
    dist, idx = dist[keep].reshape(n, k), idx[keep].reshape(n, k)
    log_u = np.log(perplexity)
    tol = 1e-5
    for i in range(n):
        d = dist[i]                                            # reference uses the kNN distances as given
        beta, lo, hi = 1.0, -np.inf, np.inf
        p = None
        for _ in range(50):
            p = np.exp(-d * beta)
            sp_ = p.sum()
            if sp_ == 0:
                h = 0.0
                p = np.zeros_like(d)
            else:
                h = np.log(sp_) + beta * np.sum(d * p) / sp_
                p = p / sp_
            diff = h - log_u
            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        if p is None or not np.any(p):                        # all kernel weights underflowed: undefined
            continue
        comp = np.bincount(codes[idx[i]], weights=p, minlength=n_lab)
        simpson = float(np.sum(comp ** 2))
        out[where[i]] = 1.0 / simpson if simpson > 0 else np.nan
    return out


def ilisi(x, batch, perplexity: int = 30) -> float:
    """Median batch-LISI (higher = better mixing; max = number of batches)."""
    return float(np.nanmedian(lisi(x, batch, perplexity)))


def clisi(x, labels, perplexity: int = 30) -> float:
    """Median cell-type LISI (lower = purer neighbourhoods; min = 1)."""
    return float(np.nanmedian(lisi(x, labels, perplexity)))


def evaluate(adata, domain_key: str = "staig_domain", label_key: str | None = "ground_truth",
             batch_key: str | None = None, use_rep: str = "X_staig", batch_metrics: bool = True) -> dict:
    """Compute every applicable metric for ``adata`` and store them in ``adata.uns['staig']['metrics']``."""
    out: dict[str, float] = {}
    x = np.asarray(adata.obsm[use_rep])
    dom = adata.obs[domain_key].astype(str).values
    if label_key is not None and label_key in adata.obs:
        out["ARI"] = ari(adata.obs[label_key].values, dom)
        out["NMI"] = nmi(adata.obs[label_key].values, dom)
    if 1 < len(np.unique(dom)) < len(dom):                    # silhouette/DB need 2 <= k <= n - 1
        out["SC"] = silhouette(x, dom)
        out["DB"] = davies_bouldin(x, dom)
    if batch_key is not None and batch_key in adata.obs and adata.obs[batch_key].nunique() > 1 and batch_metrics:
        out["BatchKL"] = batch_kl(x, adata.obs[batch_key].values)
        out["iLISI"] = ilisi(x, adata.obs[batch_key].values)
    adata.uns.setdefault("staig", {})["metrics"] = out
    return out
