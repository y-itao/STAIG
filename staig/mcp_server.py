"""Optional local stdio MCP server. Install with ``pip install 'mcp>=1.2,<2'``."""

from __future__ import annotations

import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError("STAIG's FastMCP server needs `pip install 'mcp>=1.2,<2'` (MCP 1.x)") from exc

from . import PRESETS, img, metrics, preset_config, run

mcp = FastMCP("STAIG")


def _load(input_path: str, *, image_features: bool = True, platform: str = "auto", **read_kwargs):
    from .io import detect_platform, read

    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"input_path does not exist: {path}")
    platform = detect_platform(path) if platform == "auto" else platform
    if platform == "visium":
        read_kwargs.setdefault("image_features", "embeddings.npy" if image_features else None)
    adata = read(path, platform=platform, **read_kwargs)
    spatial = adata.obsm.get("spatial")
    if spatial is None or spatial.ndim != 2 or spatial.shape[1] < 2 or not np.isfinite(spatial).all():
        raise ValueError('input needs finite coordinates in obsm["spatial"] with shape (n_obs, >=2)')
    return adata


def _output(path: Path, suffix: str) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() != suffix:
        raise ValueError(f"output must end in {suffix}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@mcp.tool()
def staig_presets() -> dict:
    """List resolved STAIG preset configurations; 'auto' selects from data at run time."""
    return {name: preset_config(name).to_dict() for name in sorted(PRESETS)}


@mcp.tool()
def staig_run(input_path: str, n_clusters: int | None = None, preset: str = "auto",
              batch_key: str | None = None, n_seeds: int = 1, output_path: str | None = None) -> dict:
    """Load raw-count .h5ad or Visium data, run STAIG, write a NEW .h5ad and return metrics.

    Requires obsm['spatial']. For log-normalized input, preprocess explicitly with the Python API
    and save X_staig_input first. Several slices must already be concatenated; supply their batch_key.
    Defaults use the preset's full epoch count and may extract discoverable full-resolution histology.
    n_clusters=None performs silhouette selection; n_seeds>1 adds consensus cost. All paths are local
    to this server. Existing outputs (including the input) are never overwritten.
    """
    if n_seeds < 1 or (n_clusters is not None and n_clusters < 2):
        raise ValueError("n_seeds must be >=1; n_clusters must be >=2 or None")
    source = Path(input_path).expanduser().resolve()
    target = _output(Path(output_path) if output_path else source.with_name(f"{source.stem}_staig.h5ad"),
                     ".h5ad")
    started = time.perf_counter()
    # stdout belongs exclusively to the MCP transport, including third-party progress messages.
    with redirect_stdout(sys.stderr):
        adata = _load(str(source))
        if adata.n_obs < 3 or (n_clusters is not None and n_clusters >= adata.n_obs):
            raise ValueError("need at least 3 spots and n_clusters < n_obs")
        if batch_key is not None and (batch_key not in adata.obs or adata.obs[batch_key].isna().any()):
            raise ValueError("batch_key must name a non-missing slice column in obs")
        run(adata, n_clusters=n_clusters, preset=preset, batch_key=batch_key,
            n_seeds=n_seeds, verbose=False)
        scores = metrics.evaluate(adata, batch_key=batch_key)
        selection = adata.uns["staig"].get("n_clusters_selection")
        if selection is not None:
            adata.uns["staig"]["n_clusters_selection"] = {str(k): v for k, v in selection.items()}
        adata.write_h5ad(target)
    return {"output_path": str(target), "metrics": scores, "n_obs": adata.n_obs,
            "n_domains": int(adata.obs["staig_domain"].nunique()),
            "keys_written": ["obsm/X_staig", "obs/staig_domain", "uns/staig"],
            "time_seconds": time.perf_counter() - started}


@mcp.tool()
def staig_extract_image_features(input_path: str, image_path: str, output_npy: str) -> dict:
    """Extract BYOL H&E features to a NEW .npy file, in the loaded AnnData's obs_names order.

    input_path is .h5ad or a Visium directory; image_path must be the full-resolution image.
    Visium full-resolution pixel coordinates and fiducial_diameter_fullres metadata are required.
    A Visium loader drops unlabelled spots when truth.txt is present; use that same loader/order when
    attaching the saved array. ImageNet weights are downloaded on first use. Input is not modified.
    """
    image_path = str(Path(image_path).expanduser().resolve())
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    target = _output(Path(output_npy), ".npy")
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        adata = _load(input_path, image_features=False)
        img.extract_image_features(adata, image=image_path, num_workers=0, n_jobs=1)
        np.save(target, adata.obsm["image_features"], allow_pickle=False)
    return {"output_path": str(target), "shape": list(adata.obsm["image_features"].shape),
            "dtype": str(adata.obsm["image_features"].dtype), "row_order": "loaded adata.obs_names",
            "time_seconds": time.perf_counter() - started}


def main() -> None:
    """Serve MCP over stdio; no HTTP listener or remote upload is started."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
