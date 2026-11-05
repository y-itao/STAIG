"""Plotting helpers (thin wrappers around scanpy with STAIG defaults)."""

from __future__ import annotations

import numpy as np
import scanpy as sc


def spatial(adata, color="staig_domain", img_key="hires", size=1.5, batch_key=None, **kwargs):
    """Spatial scatter of domains. Uses the Visium image when available, otherwise coordinates.

    With ``batch_key`` every slice is drawn in its own panel (coordinates of different slices
    usually overlap).
    """
    if batch_key is not None and batch_key in adata.obs:
        axes = []
        for b in adata.obs[batch_key].cat.categories if hasattr(adata.obs[batch_key], "cat") \
                else np.unique(adata.obs[batch_key]):
            sub = adata[adata.obs[batch_key] == b]
            kw = dict(kwargs)
            kw.setdefault("title", f"{b}")
            axes.append(sc.pl.embedding(sub, basis="spatial", color=color, show=False, **kw))
        return axes
    has_img = "spatial" in adata.uns and any("images" in v for v in adata.uns["spatial"].values())
    if has_img and img_key is not None:
        return sc.pl.spatial(adata, color=color, img_key=img_key, size=size, **kwargs)
    kwargs.setdefault("s", 20)
    return sc.pl.embedding(adata, basis="spatial", color=color, **kwargs)


def umap(adata, color="staig_domain", use_rep="X_staig", recompute=False, **kwargs):
    """UMAP of the STAIG embedding."""
    if recompute or "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)
        sc.tl.umap(adata)
    return sc.pl.umap(adata, color=color, **kwargs)


def loss_curve(adata, ax=None):
    import matplotlib.pyplot as plt

    hist = adata.uns["staig"]["history"]["loss"]
    ax = ax or plt.gca()
    ax.plot(np.arange(1, len(hist) + 1), hist)
    ax.set_xlabel("epoch")
    ax.set_ylabel("neighbour contrastive loss")
    return ax
