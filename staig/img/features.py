"""End-to-end image feature extraction: patches -> filter -> BYOL -> ``adata.obsm["image_features"]``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .._utils import get_device, logger
from .filters import filter_patches
from .patches import extract_patches

IMAGE_FEATURES_KEY = "image_features"


def extract_image_features(
    adata,
    image=None,
    library_id: str | None = None,
    method: str = "byol",
    patch_scale: float = 3.5,
    patch_px: int | None = None,
    patch_size: int = 512,
    band: tuple[int, int] = (245, 275),
    blur_pre: int = 5,
    blur_post: int = 15,
    filter: bool = True,
    epochs: int = 1,
    batch_size: int = 43,
    lr: float = 3e-4,
    image_size: int = 256,
    backbone: str = "resnet50",
    seed: int = 0,
    device: str = "auto",
    amp: bool = False,
    num_workers: int = 0,
    n_jobs: int | None = None,
    cache_dir: str | os.PathLike | None = None,
    key_added: str = IMAGE_FEATURES_KEY,
    copy: bool = False,
):
    """Compute self-supervised H&E features for every spot (paper: BYOL on band-pass filtered patches).

    ``method``: ``"byol"`` (default), ``"resnet50"`` (ImageNet features, no training) or an array /
    ``.npy`` path with pre-computed features (n_obs, d), e.g. from a pathology foundation model.
    With ``cache_dir`` the raw and filtered patches are memory-mapped on disk and reused.
    """
    adata = adata.copy() if copy else adata
    t0 = time.perf_counter()
    info: dict = dict(method=method)
    if isinstance(method, (np.ndarray, str)) and (isinstance(method, np.ndarray) or method.endswith(".npy")):
        feats = np.load(method) if isinstance(method, str) else np.asarray(method)
        feats = feats.reshape(feats.shape[0], -1).astype(np.float32)
        if feats.shape[0] != adata.n_obs:
            raise ValueError(f"features have {feats.shape[0]} rows, adata has {adata.n_obs} spots")
        adata.obsm[key_added] = feats
        info["method"] = "precomputed"
        adata.uns.setdefault("staig", {})["image"] = info
        return adata if copy else None

    dev = get_device(device)
    if dev.type != "cuda":                       # CPU / MPS: warn about run time
        logger.info("image: running on %s; BYOL training of %d patches may take several minutes "
                    "(use method='resnet50' or backbone='resnet18' for a quicker run)", dev, adata.n_obs)
    if n_jobs is None:
        n_jobs = 1 if os.name == "nt" else max(1, (os.cpu_count() or 2) // 2)
    cache = Path(cache_dir) if cache_dir is not None else None
    filt_path = cache / "patches_filtered.npy" if cache else None
    if filt_path is not None and filt_path.exists():
        patches_in = np.load(filt_path, mmap_mode="r")
        if patches_in.shape[0] != adata.n_obs:
            raise ValueError("cached patches do not match adata.n_obs; delete the cache")
        logger.info("image: using cached filtered patches %s", filt_path)
    else:
        raw = extract_patches(adata, image=image, library_id=library_id, patch_scale=patch_scale,
                              patch_px=patch_px, patch_size=patch_size, spatial_key="spatial",
                              cache_dir=cache)
        if filter:
            out = None
            if cache is not None:
                out = np.lib.format.open_memmap(filt_path, mode="w+", dtype=np.uint8,
                                                shape=(raw.shape[0], patch_size, patch_size, 3))
            patches_in = filter_patches(raw, band=band, blur_pre=blur_pre, blur_post=blur_post, size=patch_size,
                                        n_jobs=n_jobs, out=out)
            if out is not None:
                out.flush()
        else:
            patches_in = raw
    info.update(patch_scale=patch_scale, patch_px=adata.uns.get("staig", {}).get("patches", {}).get("patch_px"),
                patch_size=patch_size, band=list(band), blur_pre=blur_pre, blur_post=blur_post, filter=filter,
                t_patches=time.perf_counter() - t0)
    t1 = time.perf_counter()
    patches_src = filt_path if (filt_path is not None and filt_path.exists() and filter) else patches_in
    if method == "byol":
        from .byol import train_byol

        feats, binfo = train_byol(patches_src, epochs=epochs, batch_size=batch_size, lr=lr, image_size=image_size,
                                  backbone=backbone, seed=seed, device=dev, amp=amp, num_workers=num_workers)
        info.update(binfo)
    elif method == "resnet50":
        from .byol import pretrained_features

        feats = pretrained_features(patches_src, image_size=image_size, backbone="resnet50", device=dev,
                                    num_workers=num_workers)
    else:
        raise ValueError(f"unknown method {method!r}")
    info["t_features"] = time.perf_counter() - t1
    adata.obsm[key_added] = feats
    adata.uns.setdefault("staig", {})["image"] = info
    logger.info("image: %s features %s in %.1fs (patches %.1fs)", method, feats.shape, info["t_features"],
                info["t_patches"])
    return adata if copy else None
