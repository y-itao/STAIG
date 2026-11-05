"""Readers for the files that spatial platforms actually write (``staig.read``).

Users rarely have a ready-made ``.h5ad``; they have the output folder of Space Ranger, Xenium Onboard
Analysis, the MERSCOPE instrument, the CosMx/AtoMx export, SAW (Stereo-seq), the Slide-seq / Curio
pipeline or a pair of CSV files. ``staig.read(path)`` detects the layout and returns an AnnData with

* ``X``: raw counts (CSR, float32, non-negative), control features removed (``var`` keeps
  ``feature_types`` when the platform provides it),
* ``obsm["spatial"]``: finite x/y in the platform's own units (``uns["staig"]["coordinate_unit"]``),
* ``obs``: the per-cell/per-spot metadata of the platform, ``obs["control_counts"]`` when control
  features were removed,
* ``uns["staig"]``: ``platform``, ``source``, ``qc`` (what was removed and why), ``dropped_features``
  and platform details; Visium-like data also carry the scanpy ``uns["spatial"]`` entry (scale factors
  and images) so that plotting and histology feature extraction work.

Observation identifiers are kept as the strings written by the platform (leading zeros included).
Low-count cells are removed by a per-platform ``min_counts`` default (recorded in ``uns``), which is
all the preprocessing a user has to think about: ``staig.run(path)`` then applies the platform's
preprocessing preset (HVG selection for sequencing-based data, all genes for targeted panels).
``.h5ad`` files are the one exception: their ``X`` is kept as stored (it may already be normalised).

Validated on the vendor sample files listed in ``docs/input_formats.md``.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._utils import logger

PLATFORMS: dict[str, str] = {
    "visium": "10x Visium (Space Ranger outs/: filtered_feature_bc_matrix.h5 + spatial/)",
    "visium_hd": "10x Visium HD (Space Ranger binned_outputs/square_XXXum or segmented_outputs)",
    "xenium": "10x Xenium (Xenium Onboard Analysis / Xenium Ranger outs/)",
    "merscope": "Vizgen MERSCOPE (cell_by_gene.csv + cell_metadata.csv)",
    "cosmx": "Bruker/NanoString CosMx SMI (AtoMx flat files: *exprMat_file.csv + *metadata_file.csv)",
    "stereo_seq": "BGI Stereo-seq (SAW .gef / .gem / .gem.gz, square bins or cell bins)",
    "slide_seq": "Slide-seq / Slide-seqV2 / Curio Seeker (DGE or MTX + bead locations)",
    "seqfish": "Spatial Genomics seqFISH (*CellxGene.csv + *CellCoordinates.csv)",
    "table": "generic: counts table (CSV/TSV/MTX/10x h5) + coordinates table",
    "transcripts": "generic: transcript list with a cell column (Xenium transcripts, CosMx tx_file, STARmap ...)",
    "h5ad": "AnnData .h5ad with obsm['spatial'] (or x/y columns in obs)",
}

# default minimum number of counts per cell/spot/bin (0 = keep everything)
MIN_COUNTS: dict[str, int] = {"visium": 0, "visium_hd": 1, "xenium": 10, "merscope": 20, "cosmx": 20,
                              "stereo_seq": 20, "slide_seq": 10, "seqfish": 10, "table": 0, "transcripts": 10,
                              "h5ad": 0}
CHUNK_ROWS = 4_000_000          # rows per chunk when aggregating GEM / transcript tables
MISSING_TOKENS = {"", "nan", "none", "na", "null", "unassigned", "<na>"}

_CONTROL_RE = re.compile(r"^(NegPrb|Negative\d|Negative$|SystemControl|FalseCode|Blank[-_ ]|BLANK[-_ ]|Blank\d|"
                         r"UnassignedCodeword|DeprecatedCodeword|NegControl)", re.I)
_X_CANDIDATES = ["x", "X", "x_centroid", "center_x", "centroid_x", "xcoord", "x_coord", "coord_x", "pos_x",
                 "CenterX_global_px", "global_x", "x_global_px", "x_location", "spatial_x", "imagecol",
                 "pxl_col_in_fullres", "X_centroid", "x_position", "sdimx"]
_Y_CANDIDATES = ["y", "Y", "y_centroid", "center_y", "centroid_y", "ycoord", "y_coord", "coord_y", "pos_y",
                 "CenterY_global_px", "global_y", "y_global_px", "y_location", "spatial_y", "imagerow",
                 "pxl_row_in_fullres", "Y_centroid", "y_position", "sdimy"]
# transcript tables: whole-slide coordinates first, per-FOV ("local") pixels last
_TX_X_CANDIDATES = ["x_global_px", "global_x", "x_location", "x_centroid", "CenterX_global_px", "x", "X",
                    "x_local_px", "local_x"]
_TX_Y_CANDIDATES = ["y_global_px", "global_y", "y_location", "y_centroid", "CenterY_global_px", "y", "Y",
                    "y_local_px", "local_y"]
_LOCAL_COORDS = {"x_local_px", "y_local_px", "local_x", "local_y"}
_ID_CANDIDATES = ["cell_id", "cell_ID", "cellID", "cell", "EntityID", "barcode", "barcodes", "label", "name", "id",
                  "ID", "spot", "bead", "cell_name", "Cell", "index"]
_GENE_CANDIDATES = ["gene", "feature_name", "geneID", "gene_id", "target", "name", "Gene", "gene_name", "feature"]
_COUNT_CANDIDATES = ["MIDCount", "MIDCounts", "UMICount", "UMICounts", "count", "counts", "umi", "UMI", "n"]
_CELL_CANDIDATES = ["cell_id", "cell_ID", "cellID", "cell", "CellID", "label"]


# ----------------------------------------------------------------------------------------- helpers
def _dir(path) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _find(folder: Path, *patterns: str, depth: int = 0) -> list[Path]:
    """Files in ``folder`` (and ``depth`` levels below) matching the patterns, in pattern order then sorted."""
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        hits: list[Path] = []
        for d in range(depth + 1):
            hits += [p for p in folder.glob("*/" * d + pat) if not p.name.startswith("._")]
        for p in sorted(set(hits)):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _first(folder: Path, *patterns: str, depth: int = 0) -> Path | None:
    """First match of the first pattern that matches (patterns are tried in order)."""
    for pat in patterns:
        hits = _find(folder, pat, depth=depth)
        if hits:
            return hits[0]
    return None


def _only(folder: Path, *patterns: str, what: str, depth: int = 0) -> Path | None:
    """Like ``_first`` but refuses to guess when the first matching pattern has several hits."""
    for pat in patterns:
        hits = _find(folder, pat, depth=depth)
        if len(hits) > 1:
            raise FileNotFoundError(f"several {what} files under {folder}: {[p.name for p in hits[:8]]}; "
                                    f"pass the one you want explicitly")
        if hits:
            return hits[0]
    return None


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as e:
        raise ImportError(f"reading {path.name} needs pyarrow: pip install 'staig[io]' (or pyarrow)") from e


def _sep_for(path: Path, sep: str | None) -> str:
    if sep is not None:
        return sep
    name = path.name.lower().replace(".gz", "")
    if name.endswith((".tsv", ".txt", ".gem")):
        with _open_text(path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    return "\t" if line.count("\t") >= line.count(",") else ","
        return "\t"
    return ","


def _open_text(path: Path):
    import gzip

    return gzip.open(path, "rt") if path.name.lower().endswith(".gz") else open(path)


def _header(path: Path, sep: str) -> list[str]:
    """Column names of a (possibly '#'-commented, possibly gzipped) delimited text file."""
    with _open_text(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                return [c.strip() for c in line.rstrip("\r\n").split(sep)]
    raise ValueError(f"{path}: no header line")


def _gem_header_meta(path: Path) -> dict:
    """``#key=value`` lines at the top of a GEM file (FileFormat, BinSize, OffsetX, OffsetY, StereoChip ...)."""
    meta = {}
    with _open_text(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "=" in line:
                k, v = line[1:].strip().split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def _canonical_ids(values) -> np.ndarray:
    """Identifiers as the platform wrote them: strings; integral floats (from NaN promotion) -> ints."""
    s = pd.Series(np.asarray(values))
    if pd.api.types.is_float_dtype(s):
        out = s.map(lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
    elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        out = s.astype(str)
    else:
        out = s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out.to_numpy(dtype=object)


def _is_missing(ids: np.ndarray) -> np.ndarray:
    return np.asarray([str(v).strip().lower() in MISSING_TOKENS for v in ids])


def _read_10x_h5(path: Path) -> ad.AnnData:
    import scanpy as sc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_10x_h5(path, gex_only=False)
    adata.var_names_make_unique()
    return adata


def _read_10x_mtx(folder: Path) -> ad.AnnData:
    import scanpy as sc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_10x_mtx(folder, gex_only=False)
    adata.var_names_make_unique()
    return adata


def _keep_gene_expression(adata: ad.AnnData, drop_controls: bool) -> ad.AnnData:
    """Keep ``feature_types == "Gene Expression"`` (10x h5/mtx); other types are counted per cell and recorded."""
    if "feature_types" not in adata.var or not drop_controls:
        return adata
    ft = adata.var["feature_types"].astype(str)
    keep = (ft == "Gene Expression").to_numpy()
    if keep.all():
        return adata
    dropped = ft[~keep].value_counts().to_dict()
    control = adata[:, ~keep].X
    out = adata[:, keep].copy()
    out.obs["control_counts"] = np.asarray(control.sum(1)).ravel()
    out.uns.setdefault("staig", {})["dropped_features"] = {str(k): int(v) for k, v in dropped.items()}
    logger.info("dropped non-gene features: %s", dropped)
    return out


def _drop_control_names(adata: ad.AnnData, drop_controls: bool) -> ad.AnnData:
    """Drop features whose names mark them as controls (``NegPrb*``, ``Blank-*`` ...); counts kept per cell."""
    if not drop_controls:
        return adata
    ctrl = np.asarray([bool(_CONTROL_RE.match(str(v))) for v in adata.var_names])
    if not ctrl.any():
        return adata
    control = adata[:, ctrl].X
    out = adata[:, ~ctrl].copy()
    prev = out.obs["control_counts"].to_numpy() if "control_counts" in out.obs else 0
    out.obs["control_counts"] = prev + np.asarray(control.sum(1)).ravel()
    out.uns.setdefault("staig", {}).setdefault("dropped_features", {})["control"] = int(ctrl.sum())
    return out


def _drop_control_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    ctrl = [c for c in df.columns if _CONTROL_RE.match(str(c))]
    if not ctrl:
        return df, None, []
    return df.drop(columns=ctrl), df[ctrl].sum(1), ctrl


def _counts_frame_to_anndata(df: pd.DataFrame, chunk: int = 20000) -> ad.AnnData:
    """cells x genes DataFrame of counts -> AnnData (CSR float32), converted in row chunks."""
    bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"non-numeric columns in the counts table: {bad[:5]}; pass the coordinates / metadata "
                         "columns through `coordinates=` or remove them")
    blocks = [sp.csr_matrix(df.iloc[i:i + chunk].to_numpy(dtype=np.float32)) for i in range(0, len(df), chunk)]
    x = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, df.shape[1]), dtype=np.float32)
    adata = ad.AnnData(X=x, obs=pd.DataFrame(index=pd.Index(_canonical_ids(df.index.to_numpy()))),
                       var=pd.DataFrame(index=pd.Index(df.columns.astype(str))))
    adata.var_names_make_unique()
    return adata


def _pick(columns, candidates, what: str, required: bool = True) -> str | None:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"could not find a {what} column among {cols[:20]}; pass it explicitly")
    return None


def _read_csv_str_ids(path: Path, sep: str, index: bool = True) -> pd.DataFrame:
    """CSV/TSV with the first column read as strings (identifiers keep their leading zeros)."""
    names = pd.read_csv(path, sep=sep, nrows=0).columns
    first = names[0] if len(names) else None
    kw = {"dtype": {first: str}} if first is not None else {}
    df = pd.read_csv(path, sep=sep, index_col=0 if index else None, **kw)
    if index:
        df.index = pd.Index(_canonical_ids(df.index.to_numpy()))
    return df


def _parse_bool(values) -> np.ndarray:
    """Explicit boolean parsing (``True``/``False``, 1/0, yes/no, t/f); anything else is an error."""
    s = pd.Series(np.asarray(values))
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)
    truthy, falsy = {"true", "1", "t", "yes", "y"}, {"false", "0", "f", "no", "n"}
    low = s.astype(str).str.strip().str.lower()
    bad = ~(low.isin(truthy) | low.isin(falsy))
    if bad.any():
        raise ValueError(f"unrecognised boolean values {sorted(set(low[bad]))[:5]}")
    return low.isin(truthy).to_numpy()


def _filter_min_counts(adata: ad.AnnData, min_counts: int | None, platform: str) -> tuple[ad.AnnData, int]:
    if min_counts is None:
        min_counts = MIN_COUNTS.get(platform, 0)
    total = np.asarray(adata.X.sum(1)).ravel()
    adata.obs["total_counts"] = total
    keep = total >= min_counts
    n_drop = int((~keep).sum())
    if n_drop:
        adata = adata[keep].copy()
        logger.info("%s: removed %d observations with fewer than %d counts", platform, n_drop, min_counts)
    adata.uns.setdefault("staig", {}).setdefault("qc", {})["min_counts"] = int(min_counts)
    return adata, n_drop


def _finish(adata: ad.AnnData, platform: str, source, unit: str, min_counts: int | None, removed: dict | None = None,
            **extra) -> ad.AnnData:
    """Common post-processing: CSR float32 counts, finite coordinates, QC, provenance."""
    x = sp.csr_matrix(adata.X, dtype=np.float32) if sp.issparse(adata.X) else \
        sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    if x.nnz and x.data.min() < 0:
        raise ValueError(f"{source}: negative values in the counts matrix; STAIG expects raw counts")
    adata.X = x
    adata.obs_names = pd.Index(adata.obs_names.astype(str))
    removed = dict(removed or {})
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{source}: coordinates must be an (n, 2) array")
    coords = coords[:, :2]
    finite = np.isfinite(coords).all(1)
    if not finite.all():
        removed["missing_coordinates"] = int((~finite).sum())
        warnings.warn(f"{int((~finite).sum())} observations without coordinates were removed", stacklevel=3)
        adata = adata[finite].copy()
        coords = coords[finite]
    adata.obsm["spatial"] = coords
    adata, n_low = _filter_min_counts(adata, min_counts, platform)
    if n_low:
        removed["low_counts"] = n_low
    if adata.n_obs == 0:
        raise ValueError(f"{source}: no observations left after QC ({removed}); lower min_counts or check the input")
    if adata.n_vars == 0:
        raise ValueError(f"{source}: no genes left after removing control features")
    info = adata.uns.setdefault("staig", {})
    info.setdefault("qc", {}).update(n_removed=int(sum(removed.values())), removed=removed)
    info.update(platform=platform, source=str(source), coordinate_unit=unit, **extra)
    logger.info("read %s: %d observations x %d genes from %s", platform, adata.n_obs, adata.n_vars, source)
    return adata


def _attach_images(adata: ad.AnnData, spatial_dir: Path, library_id: str, metadata: dict | None = None,
                   load_images: bool = True) -> None:
    """Build the scanpy ``uns["spatial"]`` entry (scale factors + hires/lowres images) from a spatial/ folder."""
    entry: dict = {"images": {}, "scalefactors": {}, "metadata": dict(metadata or {})}
    sf = spatial_dir / "scalefactors_json.json"
    if sf.exists():
        entry["scalefactors"] = json.load(open(sf))
    if load_images:
        for key, name in (("hires", "tissue_hires_image.png"), ("lowres", "tissue_lowres_image.png")):
            img = spatial_dir / name
            if img.exists():
                try:
                    from matplotlib.image import imread

                    entry["images"][key] = imread(img)
                except Exception as e:  # pragma: no cover - image optional
                    logger.info("could not read %s: %s", img, e)
    adata.uns["spatial"] = {library_id: entry}


# ------------------------------------------------------------------------------------- 10x Visium
def read_visium(path, counts_file: str | None = None, library_id: str | None = None, labels: str | None = "truth.txt",
                image_features: str | None = "embeddings.npy", drop_na: bool = True, load_images: bool = True,
                source_image_path: str | None = None, drop_controls: bool = True,
                min_counts: int | None = None) -> ad.AnnData:
    """One Space Ranger output folder (``outs/`` or any folder with ``*filtered_feature_bc_matrix.h5`` and
    ``spatial/``; a ``filtered_feature_bc_matrix/`` MTX folder also works; a path to the ``.h5`` file means
    that file in that folder). Handles Space Ranger 1.x (``tissue_positions_list.csv``) and 2.x+
    (``tissue_positions.csv``). Non-gene features (e.g. Antibody Capture) are dropped and counted in
    ``obs["control_counts"]``. Optional ``truth.txt`` labels and ``embeddings.npy`` image features next
    to the folder are attached (paper data layout). ``load_images=False`` keeps coordinates and scale
    factors but skips the images."""
    from .pp import load_labels

    path = _dir(path)
    if path.is_file():
        counts_file, path = path.name, path.parent
    if (path / "outs").is_dir() and not _find(path, "*feature_bc_matrix.h5"):
        path = path / "outs"
    h5 = path / counts_file if counts_file else _first(
        path, "filtered_feature_bc_matrix.h5", "*filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5",
        "*raw_feature_bc_matrix.h5")
    if h5 is not None and h5.exists():
        adata = _keep_gene_expression(_read_10x_h5(h5), drop_controls)
    else:
        mtx = _first(path, "filtered_feature_bc_matrix", "raw_feature_bc_matrix")
        if mtx is None or not mtx.is_dir():
            raise FileNotFoundError(f"no *feature_bc_matrix.h5 or feature_bc_matrix/ folder in {path}")
        adata = _keep_gene_expression(_read_10x_mtx(mtx), drop_controls)
    pos = _read_visium_positions(path / "spatial")
    missing = int((~adata.obs_names.isin(pos.index)).sum())
    adata = adata[adata.obs_names.isin(pos.index)].copy()
    pos = pos.loc[adata.obs_names]
    adata.obs = adata.obs.join(pos)
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)
    lib = library_id or path.name
    _attach_images(adata, path / "spatial", lib, load_images=load_images)
    if source_image_path:
        adata.uns["spatial"][lib]["metadata"]["source_image_path"] = str(source_image_path)
    removed = {"no_position": missing} if missing else {}
    if labels and (path / labels).exists():
        before = adata.n_obs
        load_labels(adata, path / labels, drop_na=drop_na)
        if adata.n_obs < before:
            removed["unlabelled"] = before - adata.n_obs
    if image_features and (path / image_features).exists():
        feats = np.load(path / image_features)
        feats = feats.reshape(feats.shape[0], -1)
        if feats.shape[0] != adata.n_obs:
            raise ValueError(f"{image_features} has {feats.shape[0]} rows but adata has {adata.n_obs} spots "
                             "(features must be computed after dropping unlabelled spots)")
        adata.obsm["image_features"] = feats.astype(np.float32)
    return _finish(adata, "visium", path, "pixel", min_counts, removed)


def _read_visium_positions(spatial_dir: Path) -> pd.DataFrame:
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    f = spatial_dir / "tissue_positions.csv"
    if f.exists():
        pos = pd.read_csv(f, dtype={"barcode": str})
        if "barcode" not in pos.columns:                     # some exports have no header
            pos = pd.read_csv(f, header=None, names=cols, dtype={"barcode": str})
    elif (spatial_dir / "tissue_positions_list.csv").exists():
        pos = pd.read_csv(spatial_dir / "tissue_positions_list.csv", header=None, names=cols, dtype={"barcode": str})
    elif (spatial_dir / "tissue_positions.parquet").exists():
        pos = _read_parquet(spatial_dir / "tissue_positions.parquet")
    else:
        raise FileNotFoundError(f"no tissue_positions(.csv|_list.csv|.parquet) in {spatial_dir}")
    pos["barcode"] = pos["barcode"].astype(str)
    return pos.set_index("barcode")


# ---------------------------------------------------------------------------------- 10x Visium HD
def read_visium_hd(path, bin_size: int | str = 8, level: str = "bins", counts: str = "filtered",
                   in_tissue_only: bool = True, load_images: bool = True, drop_controls: bool = True,
                   min_counts: int | None = None) -> ad.AnnData:
    """Space Ranger Visium HD outputs.

    ``path``: the run folder or its ``outs/`` (containing ``binned_outputs/``), a ``square_XXXum`` bin
    folder (or its ``.h5`` file), or (``level="cells"``) a folder with ``segmented_outputs/``.
    ``bin_size`` in microns (2, 8, 16, ...) or a folder name such as ``"square_008um"``. With
    ``level="cells"`` the segmented cell matrix (``filtered_feature_cell_matrix.h5``) is read and cell
    positions are the mean position of the 2 µm bins assigned to each cell (``barcode_mappings.parquet``),
    or the area centroid of the segmentation polygon. Coordinates are full-resolution image pixels
    (``microns_per_pixel`` in ``uns["staig"]``)."""
    root = _dir(path)
    if root.is_file():
        root = root.parent
    if (root / "outs").is_dir() and not (root / "binned_outputs").is_dir() and not root.name.startswith("square_"):
        root = root / "outs"
    if level == "cells":
        return _read_visium_hd_cells(root, counts, load_images, drop_controls, min_counts)
    if root.name.startswith("square_"):
        bdir, root = root, root.parent.parent
    else:
        binned = root / "binned_outputs" if (root / "binned_outputs").is_dir() else root
        name = bin_size if isinstance(bin_size, str) else f"square_{int(bin_size):03d}um"
        bdir = binned / name
        if not bdir.is_dir():
            avail = sorted(p.name for p in binned.glob("square_*um"))
            raise FileNotFoundError(f"{bdir} not found; available bin folders: {avail}")
    h5 = bdir / f"{counts}_feature_bc_matrix.h5"
    adata = _read_10x_h5(h5) if h5.exists() else _read_10x_mtx(bdir / f"{counts}_feature_bc_matrix")
    adata = _keep_gene_expression(adata, drop_controls)
    pos = _read_visium_positions(bdir / "spatial")
    removed = {}
    missing = int((~adata.obs_names.isin(pos.index)).sum())
    if missing:
        removed["no_position"] = missing
    adata = adata[adata.obs_names.isin(pos.index)].copy()
    pos = pos.loc[adata.obs_names]
    if in_tissue_only and "in_tissue" in pos:
        off = (pos["in_tissue"].to_numpy() != 1)
        if off.any():
            removed["off_tissue"] = int(off.sum())
            adata = adata[~off].copy()
            pos = pos.loc[adata.obs_names]
    adata.obs = adata.obs.join(pos)
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)
    sf_path = bdir / "spatial" / "scalefactors_json.json"
    sf = json.load(open(sf_path)) if sf_path.exists() else {}
    _attach_images(adata, bdir / "spatial", f"{root.name}_{bdir.name}", {"bin_size_um": sf.get("bin_size_um")},
                   load_images=load_images)
    return _finish(adata, "visium_hd", bdir, "pixel", min_counts, removed, level="bins",
                   bin_size_um=sf.get("bin_size_um"), microns_per_pixel=sf.get("microns_per_pixel"))


def _read_visium_hd_cells(root: Path, counts: str, load_images: bool, drop_controls: bool,
                          min_counts: int | None) -> ad.AnnData:
    seg = root / "segmented_outputs" if (root / "segmented_outputs").is_dir() else root
    h5 = seg / f"{counts}_feature_cell_matrix.h5"
    adata = _read_10x_h5(h5) if h5.exists() else _read_10x_mtx(seg / f"{counts}_feature_cell_matrix")
    adata = _keep_gene_expression(adata, drop_controls)
    outs = seg.parent if seg.name == "segmented_outputs" else root
    mapping = _first(seg, "barcode_mappings.parquet") or _first(outs, "barcode_mappings.parquet")
    pos2 = outs / "binned_outputs" / "square_002um" / "spatial" / "tissue_positions.parquet"
    coords = None
    how = None
    if mapping is not None and pos2.exists():
        m = _read_parquet(mapping)
        m = m[m["in_cell"].astype(bool) & m["cell_id"].notna()][["square_002um", "cell_id"]]
        p = _read_parquet(pos2).set_index("barcode")[["pxl_col_in_fullres", "pxl_row_in_fullres"]]
        joined = m.join(p, on="square_002um").dropna()
        coords = joined.groupby("cell_id")[["pxl_col_in_fullres", "pxl_row_in_fullres"]].mean()
        coords.index = coords.index.astype(str)
        how = "mean of the 2 um bins of each cell"
    elif (seg / "cell_segmentations.geojson").exists():
        coords = _geojson_centroids(seg / "cell_segmentations.geojson")
        how = "area centroid of the segmentation polygon"
    if coords is None:
        raise FileNotFoundError("cell positions need barcode_mappings.parquet + binned_outputs/square_002um or "
                                "cell_segmentations.geojson next to the segmented outputs")
    missing = int((~adata.obs_names.isin(coords.index)).sum())
    adata = adata[adata.obs_names.isin(coords.index)].copy()
    adata.obsm["spatial"] = coords.loc[adata.obs_names].to_numpy(dtype=np.float64)
    sf_path = seg / "spatial" / "scalefactors_json.json"
    sf = json.load(open(sf_path)) if sf_path.exists() else {}
    if (seg / "spatial").is_dir():
        _attach_images(adata, seg / "spatial", f"{outs.name}_cells", load_images=load_images)
    return _finish(adata, "visium_hd", seg, "pixel", min_counts, {"no_position": missing} if missing else {},
                   level="cells", cell_position=how, microns_per_pixel=sf.get("microns_per_pixel"))


def _ring_area_centroid(ring: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed shoelace area and area centroid of a closed or open ring of vertices."""
    r = np.asarray(ring, dtype=np.float64)
    if len(r) and not np.allclose(r[0], r[-1]):
        r = np.vstack([r, r[:1]])
    x, y = r[:, 0], r[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * cross.sum()
    if abs(area) < 1e-12:
        return 0.0, r[:-1].mean(0) if len(r) > 1 else r[0]
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * area)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * area)
    return area, np.array([cx, cy])


def _polygon_centroid(polygon: list) -> np.ndarray:
    """Area centroid of a polygon with holes (exterior ring minus hole rings), area-weighted."""
    num, den = np.zeros(2), 0.0
    for i, ring in enumerate(polygon):
        area, c = _ring_area_centroid(np.asarray(ring, dtype=np.float64))
        w = abs(area) if i == 0 else -abs(area)
        num += w * c
        den += w
    return num / den if den > 0 else _ring_area_centroid(np.asarray(polygon[0], dtype=np.float64))[1]


def _geojson_centroids(path: Path) -> pd.DataFrame:
    """Area centroids of the cell polygons, keyed like the h5 cell ids (``cellid_000000001-1``)."""
    g = json.load(open(path))
    rows = {}
    for feat in g.get("features", []):
        cid = feat.get("properties", {}).get("cell_id")
        geom = feat.get("geometry", {})
        pts = geom.get("coordinates", [])
        if cid is None or not pts:
            continue
        if geom.get("type") == "Polygon":
            c = _polygon_centroid(pts)
        elif geom.get("type") == "MultiPolygon":
            parts = [(abs(_ring_area_centroid(np.asarray(p[0], dtype=np.float64))[0]), _polygon_centroid(p))
                     for p in pts]
            w = np.array([a for a, _ in parts])
            c = (np.vstack([pc for _, pc in parts]) * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else parts[0][1]
        else:
            continue
        key = f"cellid_{int(cid):09d}-1" if not isinstance(cid, str) or cid.isdigit() else str(cid)
        rows[key] = c
    return pd.DataFrame.from_dict(rows, orient="index", columns=["pxl_col_in_fullres", "pxl_row_in_fullres"])


# ---------------------------------------------------------------------------------------- Xenium
def read_xenium(path, drop_controls: bool = True, min_counts: int | None = None,
                cells_file: str | None = None) -> ad.AnnData:
    """Xenium Onboard Analysis / Xenium Ranger ``outs/``: ``cell_feature_matrix.h5`` (or the MTX folder)
    plus ``cells.parquet`` / ``cells.csv.gz``; control probes/codewords are dropped (counted in
    ``obs["control_counts"]``); coordinates are ``x_centroid``/``y_centroid`` in microns;
    ``experiment.xenium`` details go to ``uns["staig"]["xenium"]``."""
    path = _dir(path)
    if path.is_file():
        path = path.parent
    if (path / "outs").is_dir() and not (path / "cell_feature_matrix.h5").exists() and \
            not (path / "cell_feature_matrix").is_dir():
        path = path / "outs"
    h5 = path / "cell_feature_matrix.h5"
    if h5.exists():
        adata = _read_10x_h5(h5)
    elif (path / "cell_feature_matrix").is_dir():
        adata = _read_10x_mtx(path / "cell_feature_matrix")
    else:
        raise FileNotFoundError(f"no cell_feature_matrix.h5 (or cell_feature_matrix/) in {path}")
    adata = _keep_gene_expression(adata, drop_controls)
    if cells_file:
        cells = path / cells_file
        meta = _read_parquet(cells) if cells.suffix == ".parquet" else pd.read_csv(cells, dtype={"cell_id": str})
    else:
        meta = None
        if (path / "cells.parquet").exists():
            try:
                meta = _read_parquet(path / "cells.parquet")
            except ImportError:
                meta = None
        if meta is None:
            if not (path / "cells.csv.gz").exists():
                raise FileNotFoundError(f"no cells.parquet (pyarrow needed) or cells.csv.gz in {path}")
            meta = pd.read_csv(path / "cells.csv.gz", dtype={"cell_id": str})
    meta["cell_id"] = _canonical_ids(meta["cell_id"].to_numpy())
    meta = meta.set_index("cell_id")
    missing = int((~adata.obs_names.isin(meta.index)).sum())
    adata = adata[adata.obs_names.isin(meta.index)].copy()
    meta = meta.loc[adata.obs_names]
    adata.obs = adata.obs.join(meta)
    adata.obsm["spatial"] = meta[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    extra = {}
    if (path / "experiment.xenium").exists():
        spec = json.load(open(path / "experiment.xenium"))
        extra["xenium"] = {k: spec[k] for k in ("analysis_sw_version", "pixel_size", "panel_name", "region_name",
                                                "num_cells", "run_name") if k in spec}
    return _finish(adata, "xenium", path, "micron", min_counts, {"no_metadata": missing} if missing else {}, **extra)


# -------------------------------------------------------------------------------------- MERSCOPE
def read_merscope(path, counts=None, metadata=None, drop_blanks: bool = True,
                  min_counts: int | None = None) -> ad.AnnData:
    """Vizgen MERSCOPE cell-level output: ``cell_by_gene.csv`` (cells x genes, ``Blank-*`` columns are
    blank barcodes) and ``cell_metadata.csv`` (``EntityID``/``cell``, ``fov``, ``volume``, ``center_x``,
    ``center_y`` in microns). Prefixed names (``<sample>_cell_by_gene.csv``) are found automatically;
    several candidates in one folder must be passed explicitly. Cell ids are kept as strings."""
    path = Path(path).expanduser()
    if path.is_dir():
        counts = Path(counts) if counts else _only(path, "*cell_by_gene*.csv*", what="cell_by_gene", depth=1)
        metadata = Path(metadata) if metadata else _only(path, "*cell_metadata*.csv*", what="cell_metadata", depth=1)
    elif counts is None:
        counts = path
        metadata = Path(metadata) if metadata else _only(path.parent, "*cell_metadata*.csv*", what="cell_metadata")
    if counts is None or metadata is None:
        raise FileNotFoundError(f"need cell_by_gene.csv and cell_metadata.csv (looked in {path})")
    df = _read_csv_str_ids(Path(counts), ",")
    meta = _read_csv_str_ids(Path(metadata), ",")
    if "EntityID" in meta.columns:
        meta.index = pd.Index(_canonical_ids(meta["EntityID"].to_numpy()))
        meta = meta.drop(columns=["EntityID"])
    if df.index.duplicated().any():
        raise ValueError("duplicated cell ids in cell_by_gene.csv")
    meta = meta[~meta.index.duplicated(keep="first")]
    df, blanks, blank_names = (_drop_control_columns(df) if drop_blanks else (df, None, []))
    common = df.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError("cell ids of cell_by_gene.csv and cell_metadata.csv do not match")
    removed = {"no_metadata": int(len(df) - len(common))} if len(common) < len(df) else {}
    df, meta = df.loc[common], meta.loc[common]
    adata = _counts_frame_to_anndata(df)
    adata.obs = adata.obs.join(meta)
    if blanks is not None:
        adata.obs["control_counts"] = blanks.loc[common].to_numpy()
        adata.uns.setdefault("staig", {})["dropped_features"] = {"blank": len(blank_names)}
    adata.obsm["spatial"] = meta[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    return _finish(adata, "merscope", counts, "micron", min_counts, removed)


# ------------------------------------------------------------------------------------------ CosMx
def read_cosmx(path, counts=None, metadata=None, drop_controls: bool = True, min_counts: int | None = None,
               coordinates: str = "global") -> ad.AnnData:
    """CosMx SMI flat files (AtoMx export): ``*exprMat_file.csv[.gz]`` (``fov``, ``cell_ID``, genes,
    ``NegPrb*``/``SystemControl*`` controls) and ``*metadata_file.csv[.gz]`` (``CenterX/Y_global_px`` ...).
    Cells are keyed ``<fov>_<cell_ID>`` (or the ``cell`` column of newer exports); ``cell_ID == 0``
    (background) is dropped. Coordinates are whole-slide pixels (``coordinates="local"`` for per-FOV
    pixels). One slide per call: a folder with several ``exprMat`` files is rejected (use ``read([...])``)."""
    path = Path(path).expanduser()
    if path.is_dir():
        found = _find(path, "*exprMat_file.csv", "*exprMat_file.csv.gz", depth=2)
        if counts is None:
            if len(found) != 1:
                raise FileNotFoundError(f"expected exactly one *exprMat_file.csv under {path}, found {len(found)}: "
                                        f"{[str(f) for f in found[:6]]}")
            counts = found[0]
    elif counts is None:
        counts = path
    counts = Path(counts)
    if metadata is None:
        prefix = counts.name.replace("exprMat_file.csv.gz", "").replace("exprMat_file.csv", "")
        metadata = _only(counts.parent, f"{prefix}metadata_file.csv", f"{prefix}metadata_file.csv.gz",
                         what="metadata")
        if metadata is None:
            raise FileNotFoundError(f"no {prefix}metadata_file.csv next to {counts}")
    df = pd.read_csv(counts)
    meta = pd.read_csv(Path(metadata))
    if "cell" in df.columns and "cell" in meta.columns:
        df["cell"], meta["cell"] = _canonical_ids(df["cell"]), _canonical_ids(meta["cell"])
    else:
        for t in (df, meta):
            t["cell"] = (pd.Series(_canonical_ids(t["fov"])).astype(str) + "_"
                         + pd.Series(_canonical_ids(t["cell_ID"])).astype(str)).to_numpy()
    removed = {}
    if "cell_ID" in df.columns:
        bg = _canonical_ids(df["cell_ID"]) == "0"
        removed["background"] = int(bg.sum())
        df = df[~bg]
    df = df.set_index(pd.Index(df["cell"])).drop(columns=[c for c in ("cell", "fov", "cell_ID") if c in df.columns])
    meta = meta.set_index(pd.Index(meta["cell"])).drop(columns=["cell"])
    meta = meta[~meta.index.duplicated(keep="first")]
    if df.index.duplicated().any():
        raise ValueError("duplicated cell keys in the expression matrix")
    df, ctrl, ctrl_names = (_drop_control_columns(df) if drop_controls else (df, None, []))
    common = df.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError("cell keys of exprMat and metadata files do not match")
    if len(common) < len(df):
        removed["no_metadata"] = int(len(df) - len(common))
    df, meta = df.loc[common], meta.loc[common]
    adata = _counts_frame_to_anndata(df)
    adata.obs = adata.obs.join(meta)
    if ctrl is not None:
        adata.obs["control_counts"] = ctrl.loc[common].to_numpy()
        adata.uns.setdefault("staig", {})["dropped_features"] = {"control": len(ctrl_names)}
    xy = ("CenterX_global_px", "CenterY_global_px") if coordinates == "global" else \
        ("CenterX_local_px", "CenterY_local_px")
    adata.obsm["spatial"] = meta[list(xy)].to_numpy(dtype=np.float64)
    if "fov" in adata.obs:
        adata.obs["fov"] = adata.obs["fov"].astype(str).astype("category")
    return _finish(adata, "cosmx", counts, "pixel", min_counts, removed, coordinates=coordinates)


# ------------------------------------------------------------------------------------- Stereo-seq
def read_stereo_seq(path, bin_size: int = 50, cellbin: bool | str = "auto", min_counts: int | None = None,
                    sep: str | None = None) -> ad.AnnData:
    """SAW outputs: ``.gef`` (HDF5 square-bin or cell-bin GEF; ``geneExp/binN`` and ``cellBin`` groups with
    their ``offset`` tables honoured) or ``.gem`` / ``.gem.gz`` / ``.tsv`` text (``geneID x y MIDCount
    [ExonCount] [CellID]``; ``MIDCounts``/``UMICount`` and ``label``/``cell_id`` are accepted; ``#`` header
    lines are kept in ``uns["staig"]["gem_header"]``). Square bins of ``bin_size`` DNBs are aggregated on
    the absolute grid (bin = ``floor(coordinate / bin_size)``; stored coordinate = bin centre); a GEM with a
    cell column gives one observation per cell (position = mean of its DNBs) unless ``cellbin=False``.
    Coordinates are DNB indices; 1 DNB = 0.5 µm (500 nm pitch)."""
    path = Path(path).expanduser()
    if path.is_dir():
        f = _only(path, "*.gef", "*.gem", "*.gem.gz", "*.gem.txt", "*GEM*.tsv*", "*.gem.tsv*", what="GEM/GEF", depth=2)
        if f is None:
            raise FileNotFoundError(f"no .gef or .gem(.gz) file under {path}")
        path = f
    name = path.name.lower()
    if name.endswith((".gef", ".h5", ".hdf5")):
        return _read_gef(path, bin_size, cellbin, min_counts)
    return _read_gem(path, bin_size, cellbin, min_counts, sep)


def _aggregate_rows(keys: np.ndarray, gene_codes: np.ndarray, counts: np.ndarray, n_genes: int,
                    xs: np.ndarray | None = None, ys: np.ndarray | None = None):
    """Rows (key, gene, count[, x, y]) -> CSR matrix (keys x genes), ordered unique keys, mean x/y per key."""
    obs, inv = np.unique(keys, return_inverse=True)
    x = sp.coo_matrix((counts.astype(np.float32), (inv, gene_codes)), shape=(len(obs), n_genes)).tocsr()
    x.sum_duplicates()
    coords = None
    if xs is not None:
        n = np.bincount(inv, minlength=len(obs)).astype(np.float64)
        coords = np.stack([np.bincount(inv, weights=xs, minlength=len(obs)) / n,
                           np.bincount(inv, weights=ys, minlength=len(obs)) / n], 1)
    return obs, x, coords


def _read_gem(path: Path, bin_size: int, cellbin, min_counts, sep) -> ad.AnnData:
    sep = _sep_for(path, sep)
    header = _header(path, sep)
    meta = _gem_header_meta(path)
    gene = _pick(header, _GENE_CANDIDATES, "gene")
    xcol, ycol = _pick(header, ["x"], "x"), _pick(header, ["y"], "y")
    cnt = _pick(header, _COUNT_CANDIDATES, "count")
    cell = _pick(header, ["CellID", "cell_id", "cellID", "label", "cell"], "cell", required=False)
    use_cells = bool(cell) and cellbin is not False
    usecols = [gene, xcol, ycol, cnt] + ([cell] if use_cells else [])
    genes = pd.Index([], dtype=object)
    key_parts, code_parts, cnt_parts, x_parts, y_parts = [], [], [], [], []
    n_background = 0
    reader = pd.read_csv(path, sep=sep, comment="#", usecols=usecols, chunksize=CHUNK_ROWS,
                         dtype={xcol: np.int64, ycol: np.int64, cnt: np.float32}, low_memory=False)
    for chunk in reader:
        g = pd.Categorical(chunk[gene].astype(str))
        genes = genes.union(pd.Index(g.categories), sort=False) if len(genes) else pd.Index(g.categories)
        codes = genes.get_indexer(g.categories)[g.codes]
        if use_cells:
            ids = _canonical_ids(chunk[cell].to_numpy())
            bad = _is_missing(ids) | (ids == "0") | (ids == "-1")
            n_background += int(bad.sum())
            keep = ~bad
            key_parts.append(ids[keep])
            code_parts.append(codes[keep])
            cnt_parts.append(chunk[cnt].to_numpy()[keep])
            x_parts.append(chunk[xcol].to_numpy(np.float64)[keep])
            y_parts.append(chunk[ycol].to_numpy(np.float64)[keep])
        else:
            bx = np.floor_divide(chunk[xcol].to_numpy(), bin_size)
            by = np.floor_divide(chunk[ycol].to_numpy(), bin_size)
            key_parts.append(bx.astype(np.int64) * (1 << 31) + by.astype(np.int64))
            code_parts.append(codes)
            cnt_parts.append(chunk[cnt].to_numpy())
    if not key_parts:
        raise ValueError(f"{path}: no data rows")
    keys = np.concatenate(key_parts)
    codes = np.concatenate(code_parts)
    counts = np.concatenate(cnt_parts)
    if use_cells:
        obs, x, coords = _aggregate_rows(keys, codes, counts, len(genes), np.concatenate(x_parts),
                                         np.concatenate(y_parts))
        obs_names = pd.Index(obs.astype(str))
        level = "cells"
    else:
        obs, x, _ = _aggregate_rows(keys, codes, counts, len(genes))
        bx, by = obs // (1 << 31), obs % (1 << 31)
        coords = np.stack([bx * bin_size + bin_size / 2.0, by * bin_size + bin_size / 2.0], 1).astype(np.float64)
        obs_names = pd.Index([f"bin{bin_size}_{int(i)}_{int(j)}" for i, j in zip(bx, by)])
        level = "bins"
    adata = ad.AnnData(X=x, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=pd.Index(genes.astype(str))))
    adata.var_names_make_unique()
    adata.obsm["spatial"] = coords
    if level == "cells":
        adata.obs["cell_id"] = adata.obs_names
    removed = {"background": n_background} if n_background else {}
    return _finish(adata, "stereo_seq", path, "dnb", min_counts, removed, level=level,
                   bin_size=int(bin_size) if level == "bins" else None, microns_per_pixel=0.5,
                   gem_header=meta or None)


def _decode(a) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def _index_from_offsets(offsets: np.ndarray, counts: np.ndarray, n_rows: int, what: str) -> np.ndarray:
    """Row -> group index from a GEF offset/count table (validated: every row belongs to exactly one group)."""
    offsets = offsets.astype(np.int64)
    counts = counts.astype(np.int64)
    if np.array_equal(offsets, np.cumsum(counts) - counts) and counts.sum() == n_rows:
        return np.repeat(np.arange(len(counts)), counts)
    idx = np.full(n_rows, -1, dtype=np.int64)
    for i, (o, c) in enumerate(zip(offsets, counts)):
        if o < 0 or o + c > n_rows:
            raise ValueError(f"GEF {what} table: offset {o} + count {c} exceeds {n_rows} expression rows")
        if (idx[o:o + c] != -1).any():
            raise ValueError(f"GEF {what} table: overlapping offsets")
        idx[o:o + c] = i
    if (idx == -1).any():
        raise ValueError(f"GEF {what} table does not cover every expression row")
    return idx


def _read_gef(path: Path, bin_size: int, cellbin, min_counts) -> ad.AnnData:
    import h5py

    with h5py.File(path, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()
                 if k in ("offsetX", "offsetY", "resolution", "version", "omics")}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in attrs.items()}
        if "cellBin" in f and cellbin is not False:
            cell = f["cellBin/cell"][:]
            exp = f["cellBin/cellExp"][:]
            gene_ds = f["cellBin/gene"][:]
            names = _decode(gene_ds["gene"] if "gene" in gene_ds.dtype.names else gene_ds["geneName"])
            n_cells = cell.shape[0]
            rows = _index_from_offsets(cell["offset"], cell["geneCount"], exp.shape[0], "cell")
            cols = exp["geneID"].astype(np.int64)
            x = sp.coo_matrix((exp["count"].astype(np.float32), (rows, cols)), shape=(n_cells, len(names))).tocsr()
            ids = cell["id"].astype(np.int64) if "id" in cell.dtype.names else np.arange(n_cells)
            adata = ad.AnnData(X=x, obs=pd.DataFrame({"cell_id": ids}, index=ids.astype(str)),
                               var=pd.DataFrame(index=pd.Index(names)))
            adata.var_names_make_unique()
            adata.obsm["spatial"] = np.stack([cell["x"], cell["y"]], 1).astype(np.float64)
            for fld in ("area", "expCount", "geneCount", "dnbCount", "clusterID", "cellTypeID"):
                if fld in cell.dtype.names:
                    adata.obs[fld] = cell[fld]
            res = int(attrs.get("resolution", 500))
            return _finish(adata, "stereo_seq", path, "dnb", min_counts, level="cells", microns_per_pixel=res / 1000,
                           gef_attrs=attrs or None)
        if "geneExp" not in f:
            raise ValueError(f"{path}: neither a square-bin GEF (geneExp) nor a cell-bin GEF (cellBin)")
        bins = sorted(f["geneExp"].keys(), key=lambda s: int(re.sub(r"\D", "", s) or 0))
        wanted = f"bin{int(bin_size)}"
        src = wanted if wanted in f["geneExp"] else "bin1"
        if src not in f["geneExp"]:
            raise ValueError(f"{path}: bin{bin_size} not stored and no bin1 to aggregate from (available: {bins})")
        g = f["geneExp"][src]
        expr = g["expression"][:]
        gene_ds = g["gene"][:]
        names = _decode(gene_ds["gene"] if "gene" in gene_ds.dtype.names else gene_ds["geneName"])
        gene_idx = _index_from_offsets(gene_ds["offset"], gene_ds["count"], expr.shape[0], "gene")
        ds_attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in g["expression"].attrs.items()
                    if k in ("minX", "minY", "maxX", "maxY", "resolution")}
        res = int(ds_attrs.get("resolution", 500))
        stored = int(re.sub(r"\D", "", src))
    xs, ys, cnt = expr["x"].astype(np.int64), expr["y"].astype(np.int64), expr["count"].astype(np.float32)
    if stored == int(bin_size):
        bx, by = xs // stored, ys // stored                   # stored bin coordinates are multiples of the bin size
    else:
        bx, by = xs // bin_size, ys // bin_size               # aggregate a finer bin (bin1) on the absolute grid
    keys = bx * (1 << 31) + by
    obs, x, _ = _aggregate_rows(keys, gene_idx, cnt, len(names))
    bx, by = obs // (1 << 31), obs % (1 << 31)
    coords = np.stack([bx * bin_size + bin_size / 2.0, by * bin_size + bin_size / 2.0], 1).astype(np.float64)
    obs_names = pd.Index([f"bin{bin_size}_{int(i)}_{int(j)}" for i, j in zip(bx, by)])
    adata = ad.AnnData(X=x, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=pd.Index(names)))
    adata.var_names_make_unique()
    adata.obsm["spatial"] = coords
    return _finish(adata, "stereo_seq", path, "dnb", min_counts, level="bins", bin_size=int(bin_size),
                   microns_per_pixel=res / 1000, gef_attrs={**attrs, **ds_attrs} or None)


# -------------------------------------------------------------------------------------- Slide-seq
def read_slide_seq(path, counts=None, locations=None, min_counts: int | None = None) -> ad.AnnData:
    """Slide-seq / Slide-seqV2 pucks (``*.digital_expression.txt[.gz]`` genes x beads + ``*bead_locations.csv``
    / ``BeadLocationsForR.csv`` with ``barcodes, xcoord, ycoord``) and Curio Seeker outputs
    (``*MoleculesPerMatchedBead.mtx`` + ``barcodes.tsv`` + ``genes.tsv`` + ``*MatchedBeadLocation.csv``)."""
    path = Path(path).expanduser()
    folder = path if path.is_dir() else path.parent
    if counts:
        counts = Path(counts)
    elif path.is_file():
        counts = path
    else:
        counts = _only(folder, "*digital_expression*", "*.dge.txt*", "*MoleculesPerMatchedBead*.mtx*", "*matrix.mtx*",
                       "*counts*.csv*", "*counts*.tsv*", what="counts", depth=1)
    locations = Path(locations) if locations else _only(folder, "*bead_locations*.csv*", "BeadLocationsForR.csv",
                                                        "*MatchedBeadLocation*.csv*", "*bead*location*.csv*",
                                                        "*coords*.csv*", "*locations*.csv*", what="bead location",
                                                        depth=1)
    if counts is None or locations is None:
        raise FileNotFoundError(f"need a counts file and a bead-location file (looked in {folder})")
    loc = _read_coordinates(locations)
    if counts.name.lower().replace(".gz", "").endswith(".mtx"):
        adata = _read_mtx_triplet(counts)
    else:
        df = _read_csv_str_ids(counts, _sep_for(counts, None))
        in_rows = df.index.isin(loc.index).mean()
        in_cols = pd.Index(df.columns.astype(str)).isin(loc.index).mean()
        if in_cols > in_rows:                                    # genes x beads (DGE) -> beads x genes
            df = df.T
            df.index = pd.Index(df.index.astype(str))
        adata = _counts_frame_to_anndata(df)
    return _join_coordinates(adata, loc, "slide_seq", counts, "pixel", min_counts)


def _read_mtx_triplet(mtx: Path, drop_controls: bool = True) -> ad.AnnData:
    import scanpy as sc

    folder = mtx.parent
    stem = mtx.name.lower().replace(".gz", "")
    barcodes = _first(folder, "*barcodes*.tsv*", "*barcodes*.txt*")
    genes = _first(folder, "*genes*.tsv*", "*features*.tsv*", "*genes*.txt*")
    tenx_layout = (stem == "matrix.mtx" and barcodes is not None and barcodes.name.startswith("barcodes")
                   and genes is not None and genes.name.startswith(("genes", "features")))
    if tenx_layout:
        adata = _keep_gene_expression(_read_10x_mtx(folder), drop_controls)
    else:
        x = sc.read_mtx(mtx).X                                      # genes x cells in MatrixMarket
        if barcodes is None or genes is None:
            raise FileNotFoundError(f"{mtx.name} needs barcodes and genes files next to it")
        b = pd.read_csv(barcodes, sep="\t", header=None, dtype=str)[0].to_numpy()
        g = pd.read_csv(genes, sep="\t", header=None, dtype=str)
        gname = g.iloc[:, 1] if g.shape[1] > 1 else g.iloc[:, 0]
        x = x.T if x.shape[0] == len(g) and x.shape[1] == len(b) else x
        adata = ad.AnnData(X=sp.csr_matrix(x, dtype=np.float32), obs=pd.DataFrame(index=pd.Index(b)),
                           var=pd.DataFrame(index=pd.Index(gname.astype(str).to_numpy())))
    adata.var_names_make_unique()
    return _drop_control_names(adata, drop_controls)


# ------------------------------------------------------------------------- generic tables / seqFISH
def _read_coordinates(path, id_col: str | None = None, x_col: str | None = None, y_col: str | None = None,
                      sep: str | None = None) -> pd.DataFrame:
    """Coordinates table -> DataFrame indexed by observation id (strings) with columns x, y (+ other columns)."""
    path = Path(path)
    sep = _sep_for(path, sep)
    df = pd.read_csv(path, sep=sep, dtype=str)
    idc = id_col or _pick(df.columns, _ID_CANDIDATES, "id", required=False)
    if idc is None:
        first = df.columns[0]
        if str(first).startswith("Unnamed") or not pd.to_numeric(df[first], errors="coerce").notna().all():
            idc = first
        else:
            df.index = pd.Index([str(i) for i in range(len(df))])
    if idc is not None:
        df = df.set_index(pd.Index(_canonical_ids(df[idc].to_numpy()))).drop(columns=[idc])
    xc = x_col or _pick(df.columns, _X_CANDIDATES, "x")
    yc = y_col or _pick(df.columns, _Y_CANDIDATES, "y")
    df = df.rename(columns={xc: "x", yc: "y"})
    df["x"], df["y"] = pd.to_numeric(df["x"], errors="coerce"), pd.to_numeric(df["y"], errors="coerce")
    for c in df.columns:
        if c not in ("x", "y"):
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().sum() == df[c].notna().sum():
                df[c] = num
    dup = df.index.duplicated(keep="first")
    if dup.any():
        conflicting = df[df.index.isin(df.index[dup])].groupby(level=0)[["x", "y"]].nunique().max(1).gt(1).sum()
        warnings.warn(f"{int(dup.sum())} duplicated ids in {path.name} (first occurrence kept; "
                      f"{int(conflicting)} with conflicting coordinates)", stacklevel=3)
        df = df[~dup]
    return df


def _join_coordinates(adata: ad.AnnData, loc: pd.DataFrame, platform: str, source, unit: str, min_counts,
                      **extra) -> ad.AnnData:
    adata.obs_names = pd.Index(_canonical_ids(adata.obs_names.to_numpy()))
    if adata.obs_names.duplicated().any():
        raise ValueError(f"duplicated observation ids in the counts matrix of {source}")
    common = adata.obs_names.intersection(loc.index)
    if len(common) == 0:
        raise ValueError(f"no observation id of the counts matches the coordinates table "
                         f"(counts ids e.g. {list(adata.obs_names[:3])}, coordinates e.g. {list(loc.index[:3])})")
    removed = {}
    if len(common) < adata.n_obs:
        removed["no_coordinates"] = int(adata.n_obs - len(common))
        logger.info("%s: %d observations without coordinates dropped", platform, adata.n_obs - len(common))
    adata = adata[adata.obs_names.isin(common)].copy()
    loc = loc.loc[adata.obs_names]
    adata.obsm["spatial"] = loc[["x", "y"]].to_numpy(dtype=np.float64)
    extra_cols = [c for c in loc.columns if c not in ("x", "y") and c not in adata.obs.columns]
    if extra_cols:
        adata.obs = adata.obs.join(loc[extra_cols])
    return _finish(adata, platform, source, unit, min_counts, removed, **extra)


def read_table(counts, coordinates=None, transpose: bool | str = "auto", id_col: str | None = None,
               x_col: str | None = None, y_col: str | None = None, sep: str | None = None,
               drop_controls: bool = True, min_counts: int | None = None, platform: str = "table",
               unit: str = "pixel") -> ad.AnnData:
    """Generic reader: ``counts`` is a cells x genes (or genes x cells) CSV/TSV, a MatrixMarket file with
    barcodes/genes files next to it, a 10x ``.h5`` or an ``.h5ad``; ``coordinates`` a CSV/TSV with an id
    column (``cell``, ``cell_id``, ``barcode``, ``label`` ... or the first column) and x/y columns
    (``x/y``, ``center_x/center_y``, ``xcoord/ycoord``, ``X/Y`` ...). When ``coordinates`` is None the x/y
    columns are taken from the counts table itself. Identifiers are kept as strings. Control columns
    (``Blank-*``, ``NegPrb*`` ...) are dropped and counted in ``obs["control_counts"]``."""
    counts = Path(counts).expanduser()
    name = counts.name.lower()
    coords_df = _read_coordinates(coordinates, id_col, x_col, y_col, sep) if coordinates is not None else None
    if name.endswith(".h5ad"):
        adata = ad.read_h5ad(counts)
    elif name.endswith(".h5"):
        adata = _drop_control_names(_keep_gene_expression(_read_10x_h5(counts), drop_controls), drop_controls)
    elif name.replace(".gz", "").endswith(".mtx"):
        adata = _read_mtx_triplet(counts, drop_controls)
    else:
        df = _read_csv_str_ids(counts, _sep_for(counts, sep))
        if coords_df is None:
            xc = x_col or _pick(df.columns, _X_CANDIDATES, "x")
            yc = y_col or _pick(df.columns, _Y_CANDIDATES, "y")
            coords_df = df[[xc, yc]].rename(columns={xc: "x", yc: "y"}).apply(pd.to_numeric, errors="coerce")
            df = df.drop(columns=[c for c in df.columns if c in (xc, yc) or not pd.api.types.is_numeric_dtype(df[c])])
        if transpose == "auto":
            in_rows = df.index.isin(coords_df.index).mean()
            in_cols = pd.Index(df.columns.astype(str)).isin(coords_df.index).mean()
            transpose = in_cols > in_rows
        if transpose:
            df = df.T
            df.index = pd.Index(df.index.astype(str))
        if drop_controls:
            df, ctrl, ctrl_names = _drop_control_columns(df)
        else:
            ctrl, ctrl_names = None, []
        adata = _counts_frame_to_anndata(df)
        if ctrl is not None:
            adata.obs["control_counts"] = ctrl.to_numpy()
            adata.uns.setdefault("staig", {})["dropped_features"] = {"control": len(ctrl_names)}
    if coords_df is None:
        raise ValueError("coordinates are required for this counts format")
    return _join_coordinates(adata, coords_df, platform, counts, unit, min_counts)


def read_seqfish(path, min_counts: int | None = None, drop_controls: bool = True) -> ad.AnnData:
    """Spatial Genomics seqFISH exports: ``*CellxGene.csv`` (cells x genes, first column = cell label) and
    ``*CellCoordinates.csv`` (``label, area, center_x, center_y``). One region of interest per call."""
    path = _dir(path)
    counts = _only(path, "*CellxGene*.csv", what="CellxGene", depth=1)
    coords = _only(path, "*CellCoordinates*.csv", what="CellCoordinates", depth=1)
    if counts is None or coords is None:
        raise FileNotFoundError(f"need *CellxGene.csv and *CellCoordinates.csv under {path}")
    return read_table(counts, coords, id_col="label", min_counts=min_counts, platform="seqfish", unit="micron",
                      drop_controls=drop_controls)


def read_transcripts(path, gene_col: str | None = None, x_col: str | None = None, y_col: str | None = None,
                     cell_col: str | None = None, min_counts: int | None = None, sep: str | None = None,
                     drop_controls: bool = True, unassigned=("UNASSIGNED", "0", "-1")) -> ad.AnnData:
    """Aggregate a transcript/molecule table (CSV/TSV/parquet; Xenium ``transcripts.parquet``, CosMx
    ``tx_file.csv``, MERSCOPE ``detected_transcripts.csv`` with cell ids, STARmap lists ...) into a cells x
    genes matrix. The cell position is the count-weighted mean position of its transcripts; whole-slide
    coordinates are preferred over per-FOV pixels (which are rejected when several FOVs are present).
    CosMx cells are keyed ``<fov>_<cell_ID>``. Rows without a cell assignment (``unassigned`` tokens, empty
    values) are skipped; control genes are dropped and counted in ``obs["control_counts"]``."""
    path = Path(path).expanduser()
    if path.suffix == ".parquet":
        df = _read_parquet(path)
    else:
        df = pd.read_csv(path, sep=_sep_for(path, sep), low_memory=False)
    g = gene_col or _pick(df.columns, _GENE_CANDIDATES, "gene")
    xc = x_col or _pick(df.columns, _TX_X_CANDIDATES, "x")
    yc = y_col or _pick(df.columns, _TX_Y_CANDIDATES, "y")
    c = cell_col or _pick(df.columns, _CELL_CANDIDATES, "cell")
    if xc in _LOCAL_COORDS and "fov" in df.columns and df["fov"].nunique() > 1 and x_col is None:
        raise ValueError(f"{path.name}: only per-FOV pixel coordinates ({xc}) are available for several FOVs; "
                         "they cannot be combined into one map. Use the whole-slide columns or the cell-level export")
    ids = _canonical_ids(df[c].to_numpy())
    tokens = {str(t).lower() for t in unassigned}
    assigned = ~(_is_missing(ids) | np.isin(np.char.lower(ids.astype(str)), list(tokens)))
    if "fov" in df.columns and c != "cell":                     # CosMx: cell_ID is unique per FOV only
        ids = np.char.add(np.char.add(_canonical_ids(df["fov"].to_numpy()).astype(str), "_"), ids.astype(str))
    removed = {"unassigned": int((~assigned).sum())}
    df = df.assign(_cell=ids)[assigned]
    genes_all = df[g].astype(str).to_numpy()
    is_ctrl = np.asarray([bool(_CONTROL_RE.match(v)) for v in genes_all])
    if "is_gene" in df.columns:
        is_ctrl |= ~_parse_bool(df["is_gene"].to_numpy())
    cnt = _pick(df.columns, _COUNT_CANDIDATES, "count", required=False)
    weights = df[cnt].to_numpy(np.float32) if cnt else np.ones(len(df), np.float32)
    obs_key = pd.Categorical(df["_cell"])
    control_counts = np.bincount(obs_key.codes[is_ctrl], weights=weights[is_ctrl], minlength=len(obs_key.categories)) \
        if (drop_controls and is_ctrl.any()) else None
    if drop_controls and is_ctrl.any():
        keep = ~is_ctrl
        df, weights, obs_key = df[keep], weights[keep], pd.Categorical(df["_cell"][keep], categories=obs_key.categories)
    genes = pd.Categorical(df[g].astype(str))
    x = sp.coo_matrix((weights, (obs_key.codes, genes.codes)),
                      shape=(len(obs_key.categories), len(genes.categories))).tocsr()
    x.sum_duplicates()
    wsum = np.bincount(obs_key.codes, weights=weights, minlength=len(obs_key.categories))
    wx = np.bincount(obs_key.codes, weights=weights * df[xc].to_numpy(np.float64), minlength=len(wsum))
    wy = np.bincount(obs_key.codes, weights=weights * df[yc].to_numpy(np.float64), minlength=len(wsum))
    coords = np.stack([wx, wy], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        coords = coords / wsum[:, None]
    adata = ad.AnnData(X=x, obs=pd.DataFrame(index=pd.Index(obs_key.categories.astype(str))),
                       var=pd.DataFrame(index=pd.Index(genes.categories.astype(str))))
    adata.var_names_make_unique()
    adata.obsm["spatial"] = coords
    if control_counts is not None:
        adata.obs["control_counts"] = control_counts
        adata.uns.setdefault("staig", {})["dropped_features"] = {"control": int(len(set(genes_all[is_ctrl])))}
    return _finish(adata, "transcripts", path, "local" if xc in _LOCAL_COORDS else "platform", min_counts, removed)


def read_h5ad(path, min_counts: int | None = None) -> ad.AnnData:
    """AnnData file; ``obsm["spatial"]`` is required unless x/y columns exist in ``obs``. ``X`` is kept
    as stored (it may already be normalised; ``staig.run`` checks), observations without finite
    coordinates are removed, and ``uns["staig"]`` gets ``platform="h5ad"``, ``source`` and ``qc``."""
    path = Path(path).expanduser()
    if path.is_dir():
        files = sorted(path.glob("*.h5ad"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one .h5ad in {path}, found {[f.name for f in files]}")
        path = files[0]
    adata = ad.read_h5ad(path)
    if "spatial" not in adata.obsm:
        xc = _pick(adata.obs.columns, _X_CANDIDATES, "x", required=False)
        yc = _pick(adata.obs.columns, _Y_CANDIDATES, "y", required=False)
        if xc is None or yc is None:
            raise KeyError("adata.obsm['spatial'] is missing and no x/y columns were found in obs")
        adata.obsm["spatial"] = adata.obs[[xc, yc]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    finite = np.isfinite(coords).all(1)
    removed = {}
    if not finite.all():
        removed["missing_coordinates"] = int((~finite).sum())
        warnings.warn(f"{int((~finite).sum())} observations without coordinates were removed", stacklevel=2)
        adata = adata[finite].copy()
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    if min_counts:
        adata, n_low = _filter_min_counts(adata, min_counts, "h5ad")
        if n_low:
            removed["low_counts"] = n_low
    if adata.n_obs == 0:
        raise ValueError(f"{path}: no observations left ({removed})")
    info = adata.uns.setdefault("staig", {})
    info.update(platform="h5ad", source=str(path), coordinate_unit=info.get("coordinate_unit", "as stored"))
    info.setdefault("qc", {}).update(min_counts=int(min_counts or 0), n_removed=int(sum(removed.values())),
                                     removed=removed)
    return adata


# ------------------------------------------------------------------------------------- detection
def _looks_like_gem(path: Path) -> bool:
    """A text table whose header has geneID/x/y/count columns is a Stereo-seq GEM whatever its extension."""
    try:
        cols = set(_header(path, _sep_for(path, None)))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return bool({"x", "y"} <= cols and (cols & set(_GENE_CANDIDATES)) and (cols & set(_COUNT_CANDIDATES)))


def _layout_markers(path: Path) -> list[str]:
    """Platforms whose layout markers are present in ``path`` (several = ambiguous)."""
    found = []
    if (path / "experiment.xenium").exists() or (
            ((path / "cell_feature_matrix.h5").exists() or (path / "cell_feature_matrix").is_dir())
            and _first(path, "cells.parquet", "cells.csv.gz") is not None):
        found.append("xenium")
    if (path / "binned_outputs").is_dir() or path.name.startswith("square_") or (path / "segmented_outputs").is_dir() \
            or (path / "spatial" / "tissue_positions.parquet").exists():
        found.append("visium_hd")
    elif (path / "spatial").is_dir() and (_first(path, "*feature_bc_matrix.h5") or _first(path, "*feature_bc_matrix")):
        found.append("visium")
    if _first(path, "*cell_by_gene*.csv*", depth=1) and _first(path, "*cell_metadata*.csv*", depth=1):
        found.append("merscope")
    if _first(path, "*exprMat_file.csv*", depth=2) and _first(path, "*metadata_file.csv*", depth=2):
        found.append("cosmx")
    if _first(path, "*.gef", "*.gem", "*.gem.gz", "*.gem.txt", depth=1):
        found.append("stereo_seq")
    if _first(path, "*digital_expression*", "*MoleculesPerMatchedBead*", depth=1) or \
            _first(path, "*bead_locations*", "BeadLocationsForR.csv", "*MatchedBeadLocation*", depth=1):
        found.append("slide_seq")
    if _first(path, "*CellxGene*.csv", depth=1) and _first(path, "*CellCoordinates*.csv", depth=1):
        found.append("seqfish")
    return found


def detect_platform(path) -> str:
    """Guess the platform from the files found at ``path``; raises when nothing or several layouts match."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        name = path.name.lower()
        if name.endswith(".h5ad"):
            return "h5ad"
        if name.endswith((".gef", ".gem", ".gem.gz", ".gem.txt")):
            return "stereo_seq"
        if "exprmat_file" in name:
            return "cosmx"
        if "cell_by_gene" in name:
            return "merscope"
        if "digital_expression" in name or "moleculespermatchedbead" in name:
            return "slide_seq"
        if name.endswith(".parquet") or "tx_file" in name or "transcripts" in name or "detected_transcripts" in name:
            return "transcripts"
        if name.endswith("feature_bc_matrix.h5") and (path.parent / "spatial").is_dir():
            return "visium_hd" if (path.parent / "spatial" / "tissue_positions.parquet").exists() else "visium"
        if name == "cell_feature_matrix.h5":
            return "xenium"
        if name.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz", ".csv", ".csv.gz")) and _looks_like_gem(path):
            return "stereo_seq"
        return "table"
    if (path / "outs").is_dir() and not _layout_markers(path):
        path = path / "outs"
    found = _layout_markers(path)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"{path} matches several layouts {found}; pass platform=... explicitly")
    h5ads = sorted(path.glob("*.h5ad"))
    if len(h5ads) == 1:
        return "h5ad"
    listing = sorted(p.name for p in path.iterdir())[:30]
    raise ValueError(f"could not recognise the platform layout of {path}; files: {listing}. Supported layouts:\n"
                     + "\n".join(f"  {k}: {v}" for k, v in PLATFORMS.items())
                     + "\nFor a generic counts + coordinates pair use staig.io.read_table(counts, coordinates).")


_READERS = {"visium": read_visium, "visium_hd": read_visium_hd, "xenium": read_xenium, "merscope": read_merscope,
            "cosmx": read_cosmx, "stereo_seq": read_stereo_seq, "slide_seq": read_slide_seq, "seqfish": read_seqfish,
            "table": read_table, "transcripts": read_transcripts, "h5ad": read_h5ad}


def _slice_name(p, taken: set[str]) -> str:
    p = Path(str(p))
    name = p.stem if p.is_file() else p.name
    if name in ("outs", "") and p.parent.name:            # several ".../outs" folders: use the run folder's name
        name = p.parent.name
    base, i = name, 1
    while name in taken:
        i += 1
        name = f"{base}_{i}"
    return name


def read(path, platform: str = "auto", **kwargs) -> ad.AnnData | list[ad.AnnData]:
    """Read platform-native output (see ``staig.io.PLATFORMS``) into an AnnData ready for ``staig.run``.

    ``path`` may be a folder, a file or a list/dict of them (one AnnData per entry, with ``obs["slice"]``
    set from the dict keys or the folder names, for integration). ``platform="auto"`` detects the layout;
    keyword arguments are passed to the reader that accepts them (``bin_size``, ``level``, ``min_counts``,
    ``coordinates`` ...) and ignored with a warning otherwise.
    """
    if isinstance(path, dict):
        out = []
        for name, p in path.items():
            a = read(p, platform=platform, **kwargs)
            a.obs["slice"] = str(name)
            out.append(a)
        return out
    if isinstance(path, (list, tuple)):
        out, taken = [], set()
        for p in path:
            a = read(p, platform=platform, **kwargs)
            name = _slice_name(p, taken)
            taken.add(name)
            a.obs["slice"] = name
            out.append(a)
        return out
    platform = detect_platform(path) if platform == "auto" else platform
    if platform not in _READERS:
        raise ValueError(f"platform must be one of {sorted(_READERS)} or 'auto', got {platform!r}")
    reader = _READERS[platform]
    accepted = inspect.signature(reader).parameters
    use = {k: v for k, v in kwargs.items() if k in accepted}
    ignored = sorted(set(kwargs) - set(use))
    if ignored:
        warnings.warn(f"{reader.__name__} does not take {ignored}; ignored", stacklevel=2)
    logger.info("read: platform=%s path=%s", platform, path)
    return reader(path, **use)


# --------------------------------------------------------------- preprocessing defaults per platform
TARGETED = {"xenium", "merscope", "cosmx", "seqfish", "merfish", "starmap", "transcripts"}
SEQUENCING = {"visium", "visium_hd", "stereo_seq", "slide_seq"}


def preprocess_defaults(adata: ad.AnnData) -> dict:
    """Keyword arguments for :func:`staig.pp.preprocess` suited to the platform recorded in ``uns``:
    sequencing-based data get 3000 HVGs (seurat_v3 on counts); targeted panels use all genes."""
    platform = str(adata.uns.get("staig", {}).get("platform", ""))
    if platform in TARGETED or (platform in ("h5ad", "") and adata.n_vars <= 1500):
        return {"n_top_genes": None}
    return {}


def preset_for_platform(platform: str | None, has_image: bool = False) -> str | None:
    """Training preset for a platform name (None when unknown)."""
    if platform in ("visium",):
        return "visium" if has_image else "visium_no_image"
    if platform in ("visium_hd", "stereo_seq"):
        return "stereo_seq"
    if platform == "slide_seq":
        return "slide_seq"
    if platform in TARGETED:
        return "merfish"
    return None


def summary(adata: ad.AnnData) -> str:
    info = adata.uns.get("staig", {})
    parts = [f"{adata.n_obs} observations x {adata.n_vars} genes", f"platform={info.get('platform')}",
             f"coordinates in {info.get('coordinate_unit')}"]
    if info.get("qc"):
        parts.append(f"min_counts={info['qc'].get('min_counts')} (removed {info['qc'].get('removed') or 0})")
    if info.get("dropped_features"):
        parts.append(f"dropped features {info['dropped_features']}")
    return "; ".join(parts)


__all__ = ["read", "detect_platform", "read_visium", "read_visium_hd", "read_xenium", "read_merscope", "read_cosmx",
           "read_stereo_seq", "read_slide_seq", "read_seqfish", "read_table", "read_transcripts", "read_h5ad",
           "PLATFORMS", "MIN_COUNTS", "preprocess_defaults", "preset_for_platform", "summary"]
