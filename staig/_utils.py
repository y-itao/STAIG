"""Small shared helpers: seeding, device selection, logging, sparse conversions."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger("staig")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[staig] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def set_verbosity(verbose: bool) -> None:
    logger.setLevel(logging.INFO if verbose else logging.WARNING)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA).

    With ``deterministic=True`` cuDNN autotuning is disabled and PyTorch is asked to use
    deterministic kernels (``warn_only`` so that an op without a deterministic
    implementation emits a warning instead of raising).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def as_csr(m, dtype=np.float32) -> sp.csr_matrix:
    """Return a scipy CSR copy of ``m`` (dense, sparse or torch)."""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    if sp.issparse(m):
        return sp.csr_matrix(m, dtype=dtype)
    return sp.csr_matrix(np.asarray(m, dtype=dtype))


def to_dense_f32(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


@contextmanager
def timer(name: str, store: dict | None = None) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[name] = dt
    logger.info("%s: %.2fs", name, dt)


def fingerprint(*arrays) -> str:
    """Short content hash of arrays (shape, dtype and bytes; sparse: data/indices/indptr).

    Used to detect artefacts (graph, image PCs, edge probabilities, pseudo-labels) that were derived
    from inputs which have since changed, so that ``STAIG.prepare()`` recomputes them.
    """
    h = hashlib.blake2b(digest_size=8)
    for arr in arrays:
        if arr is None:
            h.update(b"None")
            continue
        parts = (arr.data, arr.indices, arr.indptr) if sp.issparse(arr) else (np.asarray(arr),)
        for p in parts:
            p = np.ascontiguousarray(p)
            if p.dtype == object:
                p = p.astype(str)
            h.update(str((p.shape, p.dtype.str)).encode())
            h.update(p.reshape(-1).view(np.uint8) if p.size else b"")
    return h.hexdigest()


def write_h5ad(adata, path: str | os.PathLike, overwrite: bool = False) -> Path:
    """Write ``adata`` atomically (temporary file in the same directory, then rename).

    Refuses to replace an existing file unless ``overwrite=True``; a failed write never leaves a
    partial file under the target name.
    """
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory")
    exists_msg = f"{path} exists; choose another name or pass overwrite=True (CLI: --overwrite)"
    if path.exists() and not overwrite:
        raise FileExistsError(exists_msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        adata.write_h5ad(tmp)
        if overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)                      # atomic "create only if absent"
            except FileExistsError:
                raise FileExistsError(exists_msg) from None
            except OSError:                             # file system without hard links (FAT/exFAT): best effort
                if path.exists():
                    raise FileExistsError(exists_msg) from None
                os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def unique_path(path: str | os.PathLike) -> Path:
    """``path`` if it does not exist, else ``<stem>_1<suffix>``, ``<stem>_2<suffix>``, ..."""
    path = Path(path)
    if not path.exists():
        return path
    for i in range(1, 100000):
        cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not cand.exists():
            return cand
    raise FileExistsError(path)


def check_adata_key(adata, where: str, key: str, hint: str) -> None:
    container = getattr(adata, where)
    if key not in container:
        raise KeyError(f"adata.{where}['{key}'] not found. {hint}")
