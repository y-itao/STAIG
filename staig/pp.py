"""Expression preprocessing, label loading and multi-slice concatenation.

The defaults are those of the paper (docs/dev/DESIGN.md):
HVG (seurat_v3 on raw counts) -> normalize_total(1e4) -> log1p -> scale(zero_center=False,
max_value=10) on the HVG subset, stored as ``adata.obsm["X_staig_input"]``.
``adata.X`` is left log-normalised (not scaled) so it stays useful for plotting/DE.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from ._utils import logger, to_dense_f32

INPUT_KEY = "X_staig_input"


def _is_integer_counts(X, n_check: int = 2000) -> bool:
    sub = X[:n_check]
    vals = sub.data if sp.issparse(sub) else np.asarray(sub)
    vals = vals[:200000]
    return bool(np.all(np.mod(vals, 1) == 0)) and bool(np.all(vals >= 0))


def _record(adata, section: str, params: dict) -> None:
    adata.uns.setdefault("staig", {})
    adata.uns["staig"][section] = params


def preprocess(
    adata: ad.AnnData,
    n_top_genes: int | None = 3000,
    flavor: str = "seurat_v3",
    normalize: bool = True,
    target_sum: float | None = 1e4,
    log1p: bool = True,
    scale: bool = True,
    zero_center: bool = False,
    max_value: float | None = 10.0,
    layer: str | None = None,
    strict_counts: bool = True,
    genes: Sequence[str] | None = None,
    key_added: str = INPUT_KEY,
    copy: bool = False,
) -> ad.AnnData | None:
    """Select HVGs, normalise and build the dense model input matrix.

    Parameters mirror the v1 behaviour by default. Set ``normalize=False, log1p=False`` for
    data that is already log-normalised (e.g. the Stereo-seq/Slide-seq h5ad files shipped with
    the paper). ``flavor="seurat_v3"`` needs raw counts; it is computed *before* normalisation,
    other flavors are computed after log-transformation. ``strict_counts=False`` applies
    ``seurat_v3`` even to non-integer matrices (v1 did this for the pre-normalised h5ad files).
    ``genes`` bypasses HVG selection and uses the given genes (e.g. spatially variable genes from
    SpatialDE/Squidpy, a co-expression module, or a curated panel). ``n_top_genes=None`` keeps every
    gene (targeted panels: Xenium, MERSCOPE, CosMx, seqFISH ...).
    """
    adata = adata.copy() if copy else adata
    adata.var_names_make_unique()
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    n_top_genes = adata.n_vars if n_top_genes is None else int(min(n_top_genes, adata.n_vars))

    if genes is not None:
        genes = [g for g in genes if g in adata.var_names]
        if len(genes) < 2:
            raise ValueError("fewer than two of the requested genes are present in adata.var_names")
        adata.var["highly_variable"] = adata.var_names.isin(genes)
        flavor = "custom"
    elif n_top_genes >= adata.n_vars:                 # targeted panels / small matrices: every gene is used
        adata.var["highly_variable"] = True
        flavor = "all"
    hvg_note = None
    if flavor == "seurat_v3":
        if strict_counts and not _is_integer_counts(adata.X):
            warnings.warn(
                "flavor='seurat_v3' expects raw integer counts but adata.X is not integer; "
                "falling back to flavor='seurat' computed after log1p.", stacklevel=2)
            flavor = "seurat"
        else:
            flavor, hvg_note = _seurat_v3_hvg(adata, n_top_genes)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)
    if flavor not in {"seurat_v3", "custom", "all"}:
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top_genes)

    sub = adata[:, adata.var["highly_variable"].values].copy()
    if scale:
        sc.pp.scale(sub, zero_center=zero_center, max_value=max_value)
    adata.obsm[key_added] = to_dense_f32(sub.X)
    _record(adata, "preprocess", dict(n_top_genes=n_top_genes, flavor=flavor, normalize=normalize,
                                      target_sum=target_sum, log1p=log1p, scale=scale,
                                      zero_center=zero_center, max_value=max_value,
                                      n_features=int(adata.obsm[key_added].shape[1]), hvg_note=hvg_note))
    logger.info("preprocess: %d spots x %d HVGs -> obsm['%s']", adata.n_obs,
                adata.obsm[key_added].shape[1], key_added)
    return adata if copy else None


def _seurat_v3_hvg(adata: ad.AnnData, n_top_genes: int, min_cells: int = 3) -> tuple[str, str | None]:
    """seurat_v3 HVGs with recovery for sparse matrices (Visium HD bins, Stereo-seq, Slide-seq): the loess
    fit of scanpy/scikit-misc fails with a singularity error when many genes are empty. Then the fit is
    repeated on genes detected in at least ``min_cells`` observations (the result is written back to
    ``adata.var``), and as a last resort the log-based ``seurat`` flavour is used. Returns the flavour
    actually used and a note for ``uns``."""
    x = adata.X
    if sp.issparse(x):
        mean = np.asarray(x.mean(0)).ravel()
        sq = np.asarray(x.multiply(x).mean(0)).ravel()
    else:
        xx = np.asarray(x, dtype=np.float64)
        mean, sq = xx.mean(0), (xx * xx).mean(0)
    var = sq - mean ** 2
    nonconst = var > 1e-12
    n_nc = int(nonconst.sum())
    if n_nc <= n_top_genes:                          # nothing to rank (and the loess fit would be degenerate)
        adata.var["highly_variable"] = nonconst
        note = f"only {n_nc} genes vary across observations; all of them are used (no loess fit)"
        logger.info("preprocess: %s", note)
        return "all", note
    if n_nc < 10:                                    # too few points for the loess fit: rank by variance instead
        top = np.argsort(-np.where(nonconst, var, -np.inf))[:n_top_genes]
        hv = np.zeros(adata.n_vars, dtype=bool)
        hv[top] = True
        adata.var["highly_variable"] = hv
        note = f"only {n_nc} genes vary; the {n_top_genes} most variable were taken without a loess fit"
        logger.info("preprocess: %s", note)
        return "all", note
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        return "seurat_v3", None
    except (ValueError, FloatingPointError) as first:
        n_cells = np.asarray((adata.X > 0).sum(0)).ravel()
        detected = n_cells >= min_cells
        if detected.sum() > n_top_genes:
            try:
                sub = adata[:, detected].copy()
                sc.pp.highly_variable_genes(sub, flavor="seurat_v3", n_top_genes=n_top_genes)
                hv = set(sub.var_names[sub.var["highly_variable"].to_numpy()])
                adata.var["highly_variable"] = adata.var_names.isin(hv)
                note = (f"seurat_v3 fit failed on all genes ({str(first)[:60]}); fitted on {int(detected.sum())} genes "
                        f"detected in >= {min_cells} observations")
                logger.info("preprocess: %s", note)
                return "seurat_v3", note
            except (ValueError, FloatingPointError):
                pass
        warnings.warn("flavor='seurat_v3' failed on this matrix (loess singularity); using flavor='seurat' after "
                      "log1p instead", stacklevel=3)
        return "seurat", f"seurat_v3 failed ({str(first)[:60]}); flavor='seurat' used"


def load_labels(
    adata: ad.AnnData,
    source,
    key: str = "ground_truth",
    drop_na: bool = True,
    copy: bool = False,
) -> ad.AnnData | None:
    """Attach ground-truth labels from a ``truth.txt`` (tab separated: barcode, label),
    a CSV/TSV with header, a pandas Series (indexed by barcode) or an array.

    Rows are aligned by barcode when possible; otherwise positional assignment with a warning
    (v1 always assigned by position). With ``drop_na=True`` unlabelled spots are removed, as in v1.
    """
    adata = adata.copy() if copy else adata
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        path = str(source)
        df = pd.read_csv(path, sep="\t" if path.endswith((".txt", ".tsv")) else ",", header=None)
        first_col = df.iloc[:, 0].astype(str)
        if df.shape[1] >= 2 and first_col.isin(adata.obs_names).mean() > 0.5:
            labels = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].astype(str).values)
        elif df.shape[1] >= 2 and str(df.iloc[0, 0]).lower() in {"barcode", "spot", "cell", ""}:
            df = pd.read_csv(path, sep="\t" if path.endswith((".txt", ".tsv")) else ",", index_col=0)
            labels = df.iloc[:, 0]
        else:
            labels = pd.Series(df.iloc[:, -1].values)
    elif isinstance(source, pd.Series):
        labels = source
    else:
        labels = pd.Series(np.asarray(source))

    # barcodes may be stored as integers on either side (AnnData allows integer obs_names): compare as strings
    obs_names = pd.Index(np.asarray(adata.obs_names).astype(str))
    index = pd.Index(np.asarray(labels.index).astype(str))
    frac = float(index.isin(obs_names).mean()) if len(index) else 0.0
    if frac > 0.5:                                   # barcodes recognised: align by barcode, never by position
        labels = pd.Series(np.asarray(labels.values), index=index)
        labels = labels[~labels.index.duplicated(keep="first")]
        aligned = labels.reindex(obs_names)
        aligned.index = adata.obs_names
        n_missing = int(aligned.isna().sum())
        if frac < 1.0:
            warnings.warn(f"load_labels: {len(labels) - int(labels.index.isin(obs_names).sum())} label "
                          f"barcodes are not in adata.obs_names (ignored); {n_missing} spots have no label.",
                          stacklevel=2)
        elif n_missing:
            logger.info("load_labels: %d spots have no label", n_missing)
    else:
        if len(labels) != adata.n_obs:
            raise ValueError(f"Got {len(labels)} labels for {adata.n_obs} spots and the barcodes do not match "
                             f"({frac:.0%} of the label index found in adata.obs_names)")
        warnings.warn(f"Labels assigned by position ({frac:.0%} of the label barcodes match obs_names).",
                      stacklevel=2)
        aligned = pd.Series(labels.values, index=adata.obs_names)
    aligned = aligned.replace({"NA": np.nan, "nan": np.nan, "": np.nan})
    adata.obs[key] = pd.Categorical(aligned.astype(object).where(aligned.notna(), None))
    if drop_na:
        keep = adata.obs[key].notna().values
        n_drop = int((~keep).sum())
        if n_drop:
            adata._inplace_subset_obs(keep)
            logger.info("load_labels: dropped %d unlabelled spots", n_drop)
    return adata if copy else None


def concat_slices(
    adatas: Sequence[ad.AnnData],
    batch_key: str = "slice",
    batch_categories: Sequence[str] | None = None,
    n_top_genes: int | None = 5000,
    flavor: str = "seurat_v3",
    normalize: bool = True,
    target_sum: float | None = 1e4,
    log1p: bool = True,
    scale: bool = False,
    zero_center: bool = False,
    max_value: float | None = 10.0,
    strict_counts: bool = True,
    key_added: str = INPUT_KEY,
    image_features_key: str = "image_features",
) -> ad.AnnData:
    """Concatenate several slices for alignment-free integration (v1 ``LoadBatch10xAdata``).

    Each slice: HVG(n_top_genes) -> normalize -> log1p (no scaling by default, as in v1).
    The model input uses the *intersection* of the per-slice HVG sets. Image features (if present
    in ``obsm[image_features_key]``) are carried over; per-slice PCs are computed later by
    ``staig.graph.image_pcs(..., batch_key=batch_key)``.
    """
    if len(adatas) < 2:
        raise ValueError("concat_slices needs at least two AnnData objects")
    if batch_categories is None:
        batch_categories = [str(a.obs.get(batch_key, pd.Series([i] * a.n_obs)).iloc[0])
                            if batch_key in a.obs else str(i) for i, a in enumerate(adatas)]
    batch_categories = [str(b) for b in batch_categories]
    if len(batch_categories) != len(adatas):
        raise ValueError(f"batch_categories has {len(batch_categories)} names for {len(adatas)} slices")
    if len(set(batch_categories)) != len(batch_categories):
        raise ValueError(f"batch_categories must be unique, got {batch_categories}")
    processed, hvg_sets = [], []
    for a, name in zip(adatas, batch_categories):
        a = a.copy()
        a.var_names_make_unique()
        n_top = a.n_vars if n_top_genes is None else min(n_top_genes, a.n_vars)
        if n_top >= a.n_vars:                                   # targeted panels: every gene is used
            a.var["highly_variable"] = True
            fl = "all"
        elif flavor == "seurat_v3" and (not strict_counts or _is_integer_counts(a.X)):
            fl, _ = _seurat_v3_hvg(a, n_top)                      # same recovery as preprocess()
        else:
            fl = "seurat" if flavor == "seurat_v3" else flavor
        if normalize:
            sc.pp.normalize_total(a, target_sum=target_sum)
        if log1p:
            sc.pp.log1p(a)
        if fl not in ("seurat_v3", "all"):
            sc.pp.highly_variable_genes(a, flavor=fl, n_top_genes=n_top)
        a.obs[batch_key] = name
        hvg_sets.append(set(a.var_names[a.var["highly_variable"].values]))
        processed.append(a)
    common = sorted(set.intersection(*hvg_sets))
    if len(common) == 0:
        raise ValueError("the slices share no highly variable gene (different gene panels or names?); "
                         "check var_names or pass n_top_genes=None to use every shared gene")
    if len(common) < 50:
        warnings.warn(f"Only {len(common)} genes are highly variable in every slice; consider a "
                      "larger n_top_genes.", stacklevel=2)
    merged = ad.concat(processed, join="outer", label=batch_key, keys=batch_categories,
                       index_unique="-", fill_value=0)
    merged.obs[batch_key] = pd.Categorical(merged.obs[batch_key].astype(str), categories=batch_categories)
    sub = merged[:, common].copy()
    if scale:
        sc.pp.scale(sub, zero_center=zero_center, max_value=max_value)
    merged.obsm[key_added] = to_dense_f32(sub.X)
    merged.var["highly_variable"] = merged.var_names.isin(common)
    if all(image_features_key in a.obsm for a in adatas):
        merged.obsm[image_features_key] = np.vstack([np.asarray(a.obsm[image_features_key], dtype=np.float32)
                                                     for a in adatas])
    merged.uns.setdefault("staig", {})
    merged.uns["staig"]["concat"] = dict(batch_key=batch_key, slices=list(batch_categories),
                                         n_common_hvg=len(common), n_top_genes=n_top_genes)
    logger.info("concat_slices: %d slices, %d spots, %d common HVGs", len(adatas), merged.n_obs, len(common))
    return merged
