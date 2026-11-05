"""Crop a square patch around every spot from the full-resolution histology image (v1 ``image_step1``).

Patch side = ``patch_scale`` x ``fiducial_diameter_fullres`` (3.5 d in the paper), resized to
``patch_size`` x ``patch_size`` with bilinear interpolation. Patches are returned as a uint8 RGB
array of shape (n_spots, patch_size, patch_size, 3) in ``adata.obs`` order.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .._utils import logger


def _read_image(image) -> np.ndarray:
    """Return an RGB uint8 HxWx3 array from a path or array (TIFF via tifffile, else PIL/cv2)."""
    if isinstance(image, np.ndarray):
        arr = image
    else:
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(path)
        arr = None
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                import tifffile

                arr = tifffile.imread(str(path))
            except ImportError:
                arr = None
        if arr is None:
            try:
                from PIL import Image

                Image.MAX_IMAGE_PIXELS = None
                arr = np.asarray(Image.open(path).convert("RGB"))
            except Exception:  # pragma: no cover - fallback
                import cv2

                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise OSError(f"could not read image {path}") from None
                arr = bgr[..., ::-1]
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = (arr * 255).round()
        elif arr.dtype == np.uint16:
            arr = arr / 257.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def resolve_image_path(adata, library_id: str | None = None) -> str | None:
    """Best-effort lookup of the full-resolution image next to the Visium folder the data came from."""
    cands = []
    src = adata.uns.get("staig", {}).get("source")
    if src:
        cands += [Path(src) / "spatial" / "tissue_full_image.tif", Path(src) / "spatial" / "tissue_full_image.png"]
    if "spatial" in adata.uns:
        for lib, d in adata.uns["spatial"].items():
            if library_id is not None and lib != library_id:
                continue
            meta = d.get("metadata", {}) if isinstance(d, dict) else {}
            p = meta.get("source_image_path") or meta.get("full_image_path")
            if p:
                cands.append(Path(p))
    for c in cands:
        if c.exists():
            return str(c)
    return None


def fiducial_diameter(adata, library_id: str | None = None) -> float:
    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        raise KeyError("adata.uns['spatial'] missing: cannot read fiducial_diameter_fullres; pass patch_px explicitly")
    lib = library_id or next(iter(adata.uns["spatial"]))
    sf = adata.uns["spatial"][lib]["scalefactors"]
    return float(sf["fiducial_diameter_fullres"])


def extract_patches(
    adata,
    image=None,
    library_id: str | None = None,
    patch_scale: float = 3.5,
    patch_px: int | None = None,
    patch_size: int = 512,
    spatial_key: str = "spatial",
    cache_dir: str | os.PathLike | None = None,
    dump_png_dir: str | os.PathLike | None = None,
) -> np.ndarray:
    """Crop and resize one patch per spot.

    ``image``: path or RGB array; when None the full-resolution image is looked up next to the
    data folder. ``patch_px`` overrides ``patch_scale * fiducial_diameter_fullres``.
    Out-of-bounds patches are zero padded.
    """
    import cv2

    if image is None:
        image = resolve_image_path(adata, library_id)
        if image is None:
            raise FileNotFoundError("full-resolution image not found; pass image=<path to tissue_full_image.tif>")
    img = _read_image(image)
    h, w = img.shape[:2]
    if patch_px is None:
        patch_px = int(patch_scale * fiducial_diameter(adata, library_id))
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    n = coords.shape[0]
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(cache_dir / "patches.npy", mode="w+", dtype=np.uint8,
                                        shape=(n, patch_size, patch_size, 3))
    else:
        out = np.zeros((n, patch_size, patch_size, 3), dtype=np.uint8)
    if dump_png_dir is not None:
        dump_png_dir = Path(dump_png_dir)
        dump_png_dir.mkdir(parents=True, exist_ok=True)
    n_pad = 0
    half = patch_px / 2
    for i, (x, y) in enumerate(coords):
        left, top = int(x - half), int(y - half)
        right, bottom = left + patch_px, top + patch_px
        l0, r0 = min(max(left, 0), w), max(min(right, w), 0)        # both bounds clipped into [0, w]
        t0, b0 = min(max(top, 0), h), max(min(bottom, h), 0)
        if r0 <= l0 or b0 <= t0:                                     # entirely outside the image
            n_pad += 1
            patch = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)
        else:
            patch = img[t0:b0, l0:r0]
            if patch.shape[0] != patch_px or patch.shape[1] != patch_px:
                n_pad += 1
                full = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)
                full[t0 - top:t0 - top + patch.shape[0], l0 - left:l0 - left + patch.shape[1]] = patch
                patch = full
        if patch_px != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        out[i] = patch
        if dump_png_dir is not None:
            cv2.imwrite(str(dump_png_dir / f"{i}.png"), patch[..., ::-1])
    if n_pad:
        warnings.warn(f"{n_pad} patches touched the image border and were zero padded", stacklevel=2)
    if cache_dir is not None:
        out.flush()
    scale = patch_px / max(fiducial_diameter(adata, library_id), 1e-9) if "spatial" in adata.uns else float("nan")
    logger.info("extract_patches: %d patches of %d px (%.1f x d) -> %dx%d", n, patch_px, scale, patch_size, patch_size)
    src = "array" if isinstance(image, np.ndarray) else str(image)
    adata.uns.setdefault("staig", {})["patches"] = dict(patch_px=int(patch_px), patch_size=int(patch_size), image=src)
    return out
