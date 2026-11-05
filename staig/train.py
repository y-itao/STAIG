"""Training loop (``STAIG`` class) and the one-call ``run`` pipeline."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

from . import graph as _graph
from ._config import (
    AUTO_N_NEGATIVES,
    CPU_EXACT_MAX_N,
    GPU_AUTO_N_NEGATIVES,
    GPU_EXACT_MAX_N,
    STAIGConfig,
    merge_configs,
)
from ._utils import fingerprint, get_device, logger, set_seed, set_verbosity, to_dense_f32, torch_dtype
from .augment import EdgeSampler, drop_features
from .loss import GraphStructure, neighbor_contrastive_loss
from .model import STAIGNet
from .pp import INPUT_KEY

EMBEDDING_KEY = "X_staig"
SMOOTH_KEY = "X_staig_smooth"


class STAIG:
    """Image-aided graph contrastive model for spatial domain identification and integration.

    Typical use::

        model = staig.STAIG(preset="default", epochs=400)
        model.fit(adata)                 # needs obsm["X_staig_input"], obsp["spatial_connectivities"]
        model.embed(adata)               # -> adata.obsm["X_staig"]
    """

    def __init__(self, config: STAIGConfig | dict | None = None, preset: str = "default", **overrides: Any):
        base = STAIGConfig.from_preset(preset)
        self.config = merge_configs(base, config).update(**overrides)
        self.net: STAIGNet | None = None
        self.history: dict[str, list] = {"loss": []}
        self.timings: dict[str, float] = {}
        self._device = get_device(self.config.device)
        self._dtype = torch_dtype(self.config.dtype)
        self._mask_mode = "none"
        self.batch_key: str | None = None
        self.adata: ad.AnnData | None = None             # set by ``run`` (useful when a path was given)
        self.input_key = INPUT_KEY
        self.graph_key = _graph.GRAPH_KEY
        self.prob_key = _graph.EDGE_PROB_KEY

    # ------------------------------------------------------------------ preparation
    def _resolve_modes(self, adata: ad.AnnData) -> tuple[str, str]:
        cfg = self.config
        multi = self.batch_key is not None and self.batch_key in adata.obs and \
            adata.obs[self.batch_key].nunique() > 1
        has_img = _graph.IMAGE_PCS_KEY in adata.obsm or _graph.IMAGE_FEATURES_KEY in adata.obsm
        edge_mode = cfg.edge_removal
        if edge_mode == "auto":
            edge_mode = "image" if has_img else "gene"
        if cfg.debias == "auto":
            debias = "pseudo_label" if (has_img and not multi) else "none"
        else:
            debias = cfg.debias
        if multi and cfg.within_slice_negatives:
            mask_mode = "batch"
        elif debias == "pseudo_label":
            mask_mode = "pseudo_label"
        else:
            mask_mode = "none"
        return edge_mode, mask_mode

    def prepare(self, adata: ad.AnnData, batch_key: str | None = None) -> ad.AnnData:
        """Make sure graph, image PCs, edge probabilities and pseudo-labels exist and are current.

        Every artefact records the parameters and a content fingerprint of the data it was derived
        from (``uns["staig"][...]``); anything missing, computed with other parameters, from other
        coordinates/features/batches, or for another graph is recomputed.
        """
        cfg = self.config
        if batch_key is not None and batch_key not in adata.obs:
            raise KeyError(f"batch_key {batch_key!r} is not a column of adata.obs "
                           f"(available: {list(adata.obs.columns)[:20]})")
        self.batch_key = batch_key
        if self.input_key not in adata.obsm:
            raise KeyError(f"adata.obsm['{self.input_key}'] missing; run staig.pp.preprocess(adata) first")
        info = adata.uns.setdefault("staig", {})
        bfp = _graph._batch_fingerprint(adata, batch_key)
        g = info.get("graph", {})
        coords_fp = _graph.coords_fingerprint(adata) if "spatial" in adata.obsm else None
        if self.graph_key not in adata.obsp or g.get("n_neighbors") != cfg.n_neighbors or \
                g.get("batch_key") != batch_key or g.get("batch") != bfp or \
                (coords_fp is not None and g.get("coords") != coords_fp):
            _graph.spatial_neighbors(adata, n_neighbors=cfg.n_neighbors, batch_key=batch_key)
        if _graph.IMAGE_FEATURES_KEY in adata.obsm:
            p = info.get("image_pcs", {})
            if _graph.IMAGE_PCS_KEY not in adata.obsm or p.get("n_comps") != cfg.n_image_pcs or \
                    p.get("batch_key") != batch_key or p.get("batch") != bfp or \
                    p.get("features") != _graph.obsm_fingerprint(adata, _graph.IMAGE_FEATURES_KEY):
                _graph.image_pcs(adata, n_comps=cfg.n_image_pcs, batch_key=batch_key)
        edge_mode, mask_mode = self._resolve_modes(adata)
        prev = info.get("edge_removal", {})
        wanted = dict(mode=edge_mode, fixed_drop_prob=cfg.fixed_drop_prob, n_gene_pcs=cfg.n_gene_pcs,
                      graph=fingerprint(adata.obsp[self.graph_key]),
                      source=_graph.edge_source_fingerprint(adata, edge_mode))
        if self.prob_key not in adata.obsp or any(prev.get(k) != v for k, v in wanted.items()):
            _graph.edge_removal_probabilities(adata, mode=edge_mode, fixed_drop_prob=cfg.fixed_drop_prob,
                                              n_gene_pcs=cfg.n_gene_pcs, batch_key=batch_key)
        if mask_mode == "pseudo_label":
            q = info.get("pseudo_labels", {})
            if _graph.PSEUDO_LABEL_KEY not in adata.obs or q.get("requested") != cfg.n_pseudo_labels or \
                    q.get("source") != _graph.obsm_fingerprint(adata, _graph.IMAGE_PCS_KEY):
                _graph.pseudo_labels(adata, n_clusters=cfg.n_pseudo_labels)
                info["pseudo_labels"]["requested"] = int(cfg.n_pseudo_labels)
        self._mask_mode = mask_mode
        return adata

    def _labels_tensor(self, adata: ad.AnnData) -> torch.Tensor | None:
        if self._mask_mode == "pseudo_label":
            codes = pd.Categorical(adata.obs[_graph.PSEUDO_LABEL_KEY]).codes
        elif self._mask_mode == "batch":
            codes = pd.Categorical(adata.obs[self.batch_key]).codes
        else:
            return None
        return torch.as_tensor(np.asarray(codes, dtype=np.int64), device=self._device)

    # ------------------------------------------------------------------ training
    def fit(self, adata: ad.AnnData, batch_key: str | None = None) -> STAIG:
        cfg = self.config
        set_verbosity(cfg.verbose)
        set_seed(cfg.seed, cfg.deterministic)
        self.prepare(adata, batch_key=batch_key)
        dev, dt = self._device, self._dtype
        x_np = to_dense_f32(adata.obsm[self.input_key])
        x = torch.as_tensor(x_np, device=dev, dtype=dt)
        n = x.size(0)
        adj = sp.csr_matrix(adata.obsp[self.graph_key])
        adj.sort_indices()
        graph = GraphStructure(adj.indptr, adj.indices, n, dev)
        sampler = EdgeSampler.from_csr(adj, adata.obsp.get(self.prob_key), n, dev, seed=cfg.seed,
                                       n_samples=cfg.n_graph_samples, fixed_prob=cfg.fixed_drop_prob)
        labels = self._labels_tensor(adata)
        feat_gen = torch.Generator(device="cpu")
        feat_gen.manual_seed(cfg.seed + 1)
        neg_gen = torch.Generator(device="cpu")
        neg_gen.manual_seed(cfg.seed + 2)
        n_neg = cfg.n_negatives
        if n_neg == "auto":
            if dev.type == "cpu":
                n_neg = AUTO_N_NEGATIVES if n > CPU_EXACT_MAX_N else None
            else:
                n_neg = GPU_AUTO_N_NEGATIVES if n > GPU_EXACT_MAX_N else None
        if n_neg is not None and (n_neg == 0 or n_neg >= n):
            n_neg = None
        self._n_negatives = n_neg

        self.net = STAIGNet(x.size(1), cfg.hidden_dim, cfg.proj_dim, cfg.n_layers, cfg.activation).to(dev, dt)
        if cfg.init == "pca":
            from .model import pca_init_first_layer

            pca_init_first_layer(self.net, x, sampler.full_edge_index)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = None
        if cfg.lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        r1, r2 = cfg.drop_feature_rates
        logger.info("fit: n=%d, features=%d, device=%s, dtype=%s, negatives=%s%s, edge_removal=%s, epochs=%d",
                    n, x.size(1), dev, cfg.dtype, self._mask_mode,
                    f" (sampled K={n_neg})" if n_neg else " (exact)",
                    adata.uns["staig"]["edge_removal"]["mode"], cfg.epochs)
        t0 = time.perf_counter()
        self.history = {"loss": []}
        pbar = tqdm(range(1, cfg.epochs + 1), disable=not cfg.verbose, leave=False, desc="STAIG")
        for epoch in pbar:
            self.net.train()
            opt.zero_grad(set_to_none=True)
            ei1, ei2 = sampler.sample(), sampler.sample()
            x1, x2 = drop_features(x, r1, feat_gen), drop_features(x, r2, feat_gen)
            _, h1 = self.net(x1, ei1)
            _, h2 = self.net(x2, ei2)
            loss = neighbor_contrastive_loss(h1, h2, graph, cfg.tau, self._mask_mode, labels,
                                             chunk_size=cfg.loss_chunk_size, n_negatives=n_neg,
                                             generator=neg_gen, variant=cfg.loss_variant)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}; try a larger tau or float64")
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            lv = float(loss.detach())
            self.history["loss"].append(lv)
            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == 1):
                pbar.set_postfix(loss=f"{lv:.4f}")
        self.timings["train"] = time.perf_counter() - t0
        logger.info("fit: done in %.1fs (final loss %.4f)", self.timings["train"], self.history["loss"][-1])
        self.embed(adata)
        adata.uns.setdefault("staig", {})["config"] = self.config.to_dict()
        adata.uns["staig"]["negatives"] = self._mask_mode
        adata.uns["staig"]["n_negatives"] = self._n_negatives if self._n_negatives else "all"
        adata.uns["staig"]["timings"] = dict(self.timings)
        adata.uns["staig"]["history"] = {"loss": np.asarray(self.history["loss"], dtype=np.float32)}
        return self

    @torch.no_grad()
    def embed(self, adata: ad.AnnData, key_added: str = EMBEDDING_KEY) -> np.ndarray:
        """Embedding of every spot on the un-augmented graph (eval mode)."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        x = torch.as_tensor(to_dense_f32(adata.obsm[self.input_key]), device=self._device, dtype=self._dtype)
        adj = sp.triu(sp.csr_matrix(adata.obsp[self.graph_key]), k=1).tocoo()
        r = torch.as_tensor(adj.row.astype(np.int64), device=self._device)
        c = torch.as_tensor(adj.col.astype(np.int64), device=self._device)
        ei = torch.stack([torch.cat([r, c]), torch.cat([c, r])])
        z, h = self.net(x, ei)
        emb = (h if self.config.embedding == "projection" else z).float().cpu().numpy()
        adata.obsm[key_added] = emb
        return emb

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("nothing to save; call fit() first")
        torch.save({"config": self.config.to_dict(), "state_dict": self.net.state_dict(),
                    "in_dim": self.net.encoder.layers[0].weight.shape[0], "mask_mode": self._mask_mode,
                    "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> STAIG:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ck["config"], device=device)
        cfg = obj.config
        obj.net = STAIGNet(ck["in_dim"], cfg.hidden_dim, cfg.proj_dim, cfg.n_layers, cfg.activation).to(obj._dtype)
        obj.net.load_state_dict(ck["state_dict"])                  # same dtype as the checkpoint: no rounding
        obj.net.to(obj._device)
        obj._mask_mode = ck.get("mask_mode", "none")
        obj.history = ck.get("history", {"loss": []})
        return obj


def run(
    adata: ad.AnnData | str | Path | Sequence | dict,
    n_clusters: int | None = None,
    preset: str = "auto",
    config: STAIGConfig | dict | None = None,
    batch_key: str | None = None,
    image: str | None = "auto",
    n_seeds: int | str = "auto",
    cluster_method: str = "gmm",
    smooth: bool = True,
    smooth_steps: int = 1,
    refine: bool | None = None,
    refine_neighbors: int = 15,
    preprocess: bool | dict = True,
    key_added: str = "staig_domain",
    image_kwargs: dict | None = None,
    read_kwargs: dict | None = None,
    **overrides: Any,
) -> STAIG:
    """One-call pipeline: (read) -> (image features) -> preprocess -> graph -> train -> embed -> cluster.

    * ``adata`` may also be a path to platform output (Space Ranger / Visium HD / Xenium / MERSCOPE /
      CosMx / Stereo-seq / Slide-seq / seqFISH folder or file, or ``.h5ad``; see :func:`staig.read`), or a
      list/dict of such paths for alignment-free integration (``batch_key`` defaults to ``"slice"``).
      Reader options go through ``read_kwargs`` (e.g. ``bin_size``, ``level``, ``min_counts``).
    * ``preprocess=True`` applies the platform's defaults (3000 HVGs for sequencing-based data, every
      gene for targeted panels); a dict overrides them.
    * ``preset="auto"`` picks a preset from the data (see :func:`staig.auto_preset`).
    * ``image="auto"`` extracts BYOL features when a full-resolution H&E image can be found next to
      the data (``load_visium_dir``) and no ``obsm["image_features"]`` exist yet; pass a path to use a
      specific image or ``None`` to skip histology.
    * ``n_seeds`` > 1 trains several models (seeds ``seed .. seed+n_seeds-1``), clusters each embedding
      and returns the consensus labelling (co-association clustering): on the DLPFC benchmark this
      gives stable domains from run to run. ``"auto"`` = 3 on a GPU, 1 on CPU.
    * ``smooth=True`` (default) averages every spot's embedding with its spatial neighbours
      (``smooth_steps`` passes) before clustering.
    * ``refine`` (majority vote of nearby spots) defaults to True for single slices, False for
      integration runs; ``n_clusters=None`` selects the number of clusters by silhouette.

    Returns the fitted model of the representative seed (with ``n_seeds > 1``: the seed whose labels
    agree best with the consensus; its embedding is the one stored in ``obsm["X_staig"]``). Results are
    in ``adata``: ``obsm["X_staig"]`` (and ``obsm["X_staig_smooth"]``), ``obs[key_added]``, ``uns["staig"]``;
    when a path was given, the AnnData is available as ``model.adata``.
    """
    from . import cluster as _cluster
    from ._config import auto_preset
    from .io import preprocess_defaults, read
    from .pp import preprocess as _preprocess

    if isinstance(adata, (str, Path)):
        adata = read(adata, **(read_kwargs or {}))
    if isinstance(adata, (list, tuple, dict)):
        adata, batch_key = _combine_inputs(adata, batch_key, preprocess, read_kwargs, image)
        preprocess = False
        image = None
    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"batch_key {batch_key!r} is not a column of adata.obs "
                       f"(available: {list(adata.obs.columns)[:20]})")
    _clear_previous_results(adata, key_added, keep_smooth=bool(smooth) and cluster_method is not None,
                            keep_domains=cluster_method is not None)
    if preprocess is not False and preprocess is not None and INPUT_KEY not in adata.obsm:
        pre = dict(preprocess_defaults(adata))
        pre.update(preprocess if isinstance(preprocess, dict) else {})
        _preprocess(adata, **pre)
    platform = str(adata.uns.get("staig", {}).get("platform", "") or "")
    if image == "auto" and platform not in ("", "visium", "h5ad"):
        image = None                                   # histology patches are defined for Visium spots
    if image is not None and _graph.IMAGE_FEATURES_KEY not in adata.obsm and batch_key is None:
        from .img import extract_image_features
        from .img.patches import resolve_image_path

        path = resolve_image_path(adata) if image == "auto" else image
        if path is not None:
            logger.info("run: extracting histology features from %s", path)
            extract_image_features(adata, image=path, **(image_kwargs or {}))
        elif image != "auto":
            raise FileNotFoundError(image)
    if preset == "auto":
        preset = auto_preset(adata, batch_key)
        logger.info("run: preset=%s", preset)
    base_cfg = merge_configs(STAIGConfig.from_preset(preset), config).update(**overrides)
    if refine is None:
        refine = batch_key is None
    if n_seeds == "auto":
        n_seeds = 3 if get_device(base_cfg.device).type == "cuda" else 1
    label_sets, embeddings, models = [], [], []
    seeds = [base_cfg.seed + i for i in range(max(1, int(n_seeds)))]
    n_steps = max(1, int(smooth_steps))
    for seed in seeds:
        model = STAIG(base_cfg.update(seed=seed))
        model.fit(adata, batch_key=batch_key)
        models.append(model)
        if cluster_method is None:
            continue
        rep = EMBEDDING_KEY
        if smooth:
            _cluster.smooth_embedding(adata, use_rep=EMBEDDING_KEY, key_added=SMOOTH_KEY, n_steps=n_steps)
            rep = SMOOTH_KEY
        _cluster.cluster(adata, n_clusters=n_clusters, method=cluster_method, use_rep=rep, refine=False,
                         key_added=key_added, random_state=seed, batch_key=batch_key)
        if n_clusters is None:
            n_clusters = adata.uns["staig"]["cluster"]["n_clusters"]
        label_sets.append(np.asarray(adata.obs[key_added].astype(str).values))
        embeddings.append(adata.obsm[EMBEDDING_KEY].copy())
    model = models[-1]
    if cluster_method is None:
        model.adata = adata
        return model
    if len(label_sets) > 1:
        cons = _cluster.consensus_labels(label_sets, n_clusters, random_state=base_cfg.seed)
        agreement = [metrics_ari(cons, lab) for lab in label_sets]
        best = int(np.argmax(agreement))
        model = models[best]                                         # representative seed
        adata.obsm[EMBEDDING_KEY] = embeddings[best]
        if smooth:                                                   # keep the smoothed embedding consistent
            _cluster.smooth_embedding(adata, use_rep=EMBEDDING_KEY, key_added=SMOOTH_KEY, n_steps=n_steps)
        adata.obs[key_added] = pd.Categorical(cons.astype(str))
        adata.uns["staig"]["config"] = model.config.to_dict()
        adata.uns["staig"]["history"] = {"loss": np.asarray(model.history["loss"], dtype=np.float32)}
        adata.uns["staig"]["timings"] = dict(model.timings)
        adata.uns["staig"]["consensus"] = dict(seeds=seeds, agreement_ari=[float(a) for a in agreement],
                                              representative_seed=seeds[best])
        logger.info("run: consensus over %d seeds (agreement ARI %.2f-%.2f)", len(seeds), min(agreement),
                    max(agreement))
    if refine:
        _cluster.refine_labels(adata, key=key_added, n_neighbors=refine_neighbors, batch_key=batch_key)
    adata.obs[key_added] = pd.Categorical(adata.obs[key_added].astype(str))
    adata.uns["staig"]["cluster"].update(refine=bool(refine), refine_neighbors=int(refine_neighbors),
                                         consensus=len(label_sets) > 1,
                                         n_clusters_found=int(adata.obs[key_added].nunique()))
    model.adata = adata
    return model


def _combine_inputs(inputs, batch_key: str | None, preprocess, read_kwargs: dict | None,
                    image) -> tuple[ad.AnnData, str]:
    """Read / combine several inputs for alignment-free integration.

    Every input gets a slice name in ``obs[batch_key]`` (dict keys, folder names, or an existing column).
    With ``preprocess`` enabled, the slices are concatenated with ``pp.concat_slices`` using the platform
    defaults shared by all inputs (every gene when all are targeted panels, otherwise the common highly
    variable genes); with ``preprocess=False`` every input must already carry ``obsm["X_staig_input"]`` of
    the same width and these are stacked unchanged. An explicit ``image`` is rejected: extract image features
    per slice first and pass the AnnData objects.
    """
    from .io import _slice_name, preprocess_defaults, read
    from .pp import concat_slices

    if image not in (None, "auto"):
        raise ValueError("image= applies to a single input; for several slices extract the image features per slice "
                         "(staig.img.extract_image_features) and pass the AnnData objects")
    if isinstance(inputs, dict):
        names, items = [str(k) for k in inputs], list(inputs.values())
    else:
        names, items = None, list(inputs)
    if not items:
        raise ValueError("no inputs given")
    bk = batch_key or "slice"
    parts, taken = [], set()
    for i, item in enumerate(items):
        part = item if isinstance(item, ad.AnnData) else read(item, **(read_kwargs or {}))
        if names is not None:
            part.obs[bk] = names[i]
        elif bk not in part.obs:
            src = item if not isinstance(item, ad.AnnData) else part.uns.get("staig", {}).get("source") or str(i)
            name = _slice_name(src, taken)
            taken.add(name)
            part.obs[bk] = name
        parts.append(part)
    platforms = [str(p.uns.get("staig", {}).get("platform", "h5ad")) for p in parts]
    categories = [str(p.obs[bk].iloc[0]) for p in parts]
    if len(set(categories)) != len(categories):
        raise ValueError(f"slice names must be unique, got {categories}; pass a dict {{name: input}}")
    if preprocess is False or preprocess is None:
        missing = [c for p, c in zip(parts, categories) if INPUT_KEY not in p.obsm]
        if missing:
            raise ValueError(f"preprocess=False with several inputs needs obsm['{INPUT_KEY}'] in every input; "
                             f"missing for {missing}")
        widths = {p.obsm[INPUT_KEY].shape[1] for p in parts}
        if len(widths) != 1:
            raise ValueError(f"obsm['{INPUT_KEY}'] has different widths across inputs: {sorted(widths)}")
        merged = ad.concat(parts, join="outer", index_unique="-", fill_value=0)
        merged.obs[bk] = pd.Categorical(merged.obs[bk].astype(str), categories=categories)
    else:
        defaults = [preprocess_defaults(p) for p in parts]
        pre = {"n_top_genes": None} if all(d.get("n_top_genes", 1) is None for d in defaults) else {}
        if any(d.get("n_top_genes", 1) is None for d in defaults) and not pre:
            logger.info("run: mixed platforms (%s); common highly variable genes are used", platforms)
        pre.update(preprocess if isinstance(preprocess, dict) else {})
        concat_keys = ("n_top_genes", "flavor", "normalize", "target_sum", "log1p", "scale", "zero_center",
                       "max_value", "strict_counts")
        merged = concat_slices(parts, batch_key=bk, batch_categories=categories,
                               **{k: v for k, v in pre.items() if k in concat_keys})
    info = merged.uns.setdefault("staig", {})
    info["platform"] = platforms[0] if len(set(platforms)) == 1 else "mixed"
    info["platforms"] = platforms
    info["sources"] = [str(p.uns.get("staig", {}).get("source", "")) for p in parts]
    info["slices"] = categories
    return merged, bk


def _clear_previous_results(adata: ad.AnnData, key_added: str, keep_smooth: bool, keep_domains: bool) -> None:
    """Drop result-level entries of an earlier ``run`` so that a reused AnnData never carries metadata
    (consensus, cluster, selection scores, metrics) or arrays that belong to another run."""
    info = adata.uns.get("staig", {})
    for key in ("consensus", "cluster", "n_clusters_selection", "n_clusters_selection_method", "metrics",
                "resolution"):
        info.pop(key, None)
    if not keep_smooth and SMOOTH_KEY in adata.obsm:
        del adata.obsm[SMOOTH_KEY]
    if not keep_domains and key_added in adata.obs:
        logger.info("run: removing obs[%r] of an earlier run (cluster_method=None)", key_added)
        del adata.obs[key_added]


def metrics_ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(a).astype(str), np.asarray(b).astype(str)))
