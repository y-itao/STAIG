"""Frequency-domain filtering of patches (v1 ``image_step2`` / notebook ``Fig-3b`` step 2).

Pipeline per patch (numerically identical to v1, including the uint8 round trip through PNG):
gray -> GaussianBlur(5x5) -> FFT -> fftshift -> multiply by a square window that is 1 on
rows/cols ``[band[0], band[1])`` (for 512x512 patches the DC term is at (256, 256), so the
default 245-275 window keeps only the lowest spatial frequencies) -> inverse FFT -> magnitude ->
GaussianBlur(15x15) -> round/clip to uint8 -> replicated to 3 channels.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np


def filter_patch(patch: np.ndarray, band: tuple[int, int] = (245, 275), blur_pre: int = 5,
                 blur_post: int = 15, size: int = 512) -> np.ndarray:
    """Return the filtered patch as uint8 (size, size, 3)."""
    import cv2

    if patch.ndim == 3:
        gray = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2GRAY)
    else:
        gray = patch
    if gray.shape[0] != size or gray.shape[1] != size:
        gray = cv2.resize(gray, (size, size))
    if blur_pre and blur_pre > 1:
        gray = cv2.GaussianBlur(gray, (blur_pre, blur_pre), 0)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mask = np.zeros(gray.shape, dtype=np.uint8)
    lo, hi = band
    mask[lo:hi, lo:hi] = 1
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
    if blur_post and blur_post > 1:
        filtered = cv2.GaussianBlur(filtered, (blur_post, blur_post), 0)
    u8 = np.clip(np.rint(filtered.astype(np.float32)), 0, 255).astype(np.uint8)   # cv2.imwrite(float32) semantics
    return np.repeat(u8[..., None], 3, axis=-1)


def _filter_block(block: np.ndarray, **kw) -> np.ndarray:
    return np.stack([filter_patch(p, **kw) for p in block])


def filter_patches(patches: np.ndarray, band: tuple[int, int] = (245, 275), blur_pre: int = 5,
                   blur_post: int = 15, size: int = 512, n_jobs: int | None = None,
                   out: np.ndarray | None = None) -> np.ndarray:
    """Filter all patches (n, h, w, 3) -> (n, size, size, 3) uint8, optionally in parallel."""
    n = patches.shape[0]
    if out is None:
        out = np.zeros((n, size, size, 3), dtype=np.uint8)
    fn = partial(_filter_block, band=band, blur_pre=blur_pre, blur_post=blur_post, size=size)
    n_jobs = n_jobs or 1
    if n_jobs == 1 or n < 64:
        step = 64
        for s in range(0, n, step):
            out[s:s + step] = fn(np.asarray(patches[s:s + step]))
        return out
    step = max(16, n // (n_jobs * 4))
    starts = list(range(0, n, step))
    # threads, not processes: OpenCV and the FFT release the GIL, and forking a process that already
    # runs torch / BLAS thread pools can deadlock (spawn needs an importable __main__ on Windows/REPL)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for s, res in zip(starts, ex.map(lambda s: fn(np.asarray(patches[s:s + step])), starts)):
            out[s:s + res.shape[0]] = res
    return out
