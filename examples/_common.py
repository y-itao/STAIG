"""Shared I/O and JSON reporting for the four executable STAIG examples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc

import staig


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser(description: str, *, integration: bool = False) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--input", required=True, nargs="+" if integration else None,
                    help="Visium folder or .h5ad; integration accepts two or more paths")
    ap.add_argument("--n-clusters", type=positive_int, default=None,
                    help="default: silhouette selection (expensive for large data)")
    ap.add_argument("--output", required=True, type=Path, help="new annotated .h5ad file")
    ap.add_argument("--preset", default="auto", choices=["auto", *sorted(staig.PRESETS)])
    ap.add_argument("--n-seeds", type=positive_int, default=1)
    ap.add_argument("--epochs", type=positive_int, default=None, help="override the preset for a short smoke run")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="default: cpu")
    ap.add_argument("--log-normalized", action="store_true",
                    help="X is already log-normalized: disable normalize/log1p; use seurat HVGs")
    return ap


def load(path: str, *, image_features: bool = True):
    source = Path(path).expanduser()
    if source.is_dir():
        adata = staig.datasets.load_visium_dir(
            source, image_features="embeddings.npy" if image_features else None,
        )
    elif source.suffix.lower() == ".h5ad":
        adata = sc.read_h5ad(source)
    else:
        raise ValueError("input must be a .h5ad file or a Visium directory")
    coords = adata.obsm.get("spatial")
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2 or not np.isfinite(coords).all():
        raise ValueError('input needs finite (n_obs, >=2) coordinates in obsm["spatial"]')
    if not image_features:
        for key in ("image_features", "image_pcs"):
            adata.obsm.pop(key, None)
    # A recipe rerun must rebuild graph/input state rather than silently reuse stale caches.
    if "X_staig_input" in adata.obsm or "spatial_connectivities" in adata.obsp:
        raise ValueError("use a fresh input without X_staig_input or spatial_connectivities")
    return adata


def preprocessing(log_normalized: bool) -> dict:
    return {"flavor": "seurat", "normalize": False, "log1p": False} if log_normalized else {}


def overrides(args) -> dict:
    result = {"device": args.device, "verbose": False}
    if args.epochs is not None:
        result["epochs"] = args.epochs
    return result


def check_output(path: Path) -> None:
    if path.suffix.lower() != ".h5ad":
        raise ValueError("--output must end in .h5ad")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)


def finish(adata, output: Path, started: float, *, batch_key: str | None = None,
           metric_sample: int | None = None) -> None:
    """Evaluate, persist and print strict JSON; sample expensive metrics only when requested."""
    evaluation = adata
    if metric_sample is not None and adata.n_obs > metric_sample:
        indices = np.sort(np.random.default_rng(0).choice(adata.n_obs, metric_sample, replace=False))
        evaluation = adata[indices].copy()
    scores = staig.metrics.evaluate(evaluation, batch_key=batch_key)
    adata.uns.setdefault("staig", {})["metrics"] = scores
    adata.uns["staig"]["metric_n_obs"] = evaluation.n_obs
    # AnnData HDF5 mappings require string keys; silhouette selection uses integer k keys.
    selection = adata.uns["staig"].get("n_clusters_selection")
    if selection is not None:
        adata.uns["staig"]["n_clusters_selection"] = {str(k): v for k, v in selection.items()}
    adata.write_h5ad(output)
    report = {
        "output_path": str(output.resolve()), "n_obs": adata.n_obs, "n_vars": adata.n_vars,
        "n_domains": int(adata.obs["staig_domain"].nunique()), "metrics": scores,
        "metric_n_obs": evaluation.n_obs,
        "keys_written": {slot: sorted(getattr(adata, slot).keys())
                         for slot in ("obs", "var", "obsm", "obsp", "uns")},
        "time_seconds": time.perf_counter() - started,
        "train_seconds_last_seed": adata.uns["staig"]["timings"]["train"],
    }
    print(json.dumps(report, allow_nan=False, sort_keys=True))
