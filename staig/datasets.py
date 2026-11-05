"""Loaders for the datasets used in the paper.

The curated data archive is on Zenodo (record 14231773, ``Dataset.zip``, 8.8 GB). ``STAIG_DATA``
(environment variable) or the ``root`` argument points to the unpacked ``Dataset`` folder.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

import anndata as ad
import pandas as pd
import requests
import scanpy as sc

ZENODO_RECORD = "14231773"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/Dataset.zip/content"
DLPFC_SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
                "151671", "151672", "151673", "151674", "151675", "151676"]


class _HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded HTTP reads for zipfile, including ZIP64 archives.

    Only directory reads are cached. File payloads are streamed by ZipExtFile so
    even multi-GB entries do not accumulate in memory.
    """

    _TAIL_SIZE = 65535 + 22 + 20  # Maximum ZIP comment, end record, ZIP64 locator.

    def __init__(self, url, size=None, ranges=(), cache=False):
        super().__init__()
        self.url = url
        self.size = size
        self.ranges = list(ranges)
        self.cache = cache
        self.position = 0
        self.session = requests.Session()

    def _fetch(self, start=None, end=None):
        byte_range = f"bytes=-{self._TAIL_SIZE}" if start is None else f"bytes={start}-{end}"
        with self.session.get(
            self.url,
            headers={"Range": byte_range, "Accept-Encoding": "identity"},
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise OSError("Server must support HTTP Range requests (206); refusing a full download")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if match is None:
                raise OSError("Missing or invalid HTTP Content-Range")
            first, last, total = map(int, match.groups())
            expected_first = max(0, total - self._TAIL_SIZE) if start is None else start
            expected_last = total - 1 if start is None else end
            if (first, last) != (expected_first, expected_last) or not 0 <= first <= last < total:
                raise OSError("Server returned a different byte range than requested")
            if self.size is not None and self.size != total:
                raise OSError("Remote archive size changed; restart the download")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise OSError("Range response must not use HTTP content encoding")
            length = last - first + 1
            body = response.raw.read(length + 1)
            if len(body) != length:
                raise OSError("Truncated or oversized HTTP range response")
            self.size = total
        if self.cache or start is None:
            self.ranges.append((first, body))
        return body

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        self._checkClosed()
        if whence == os.SEEK_END:
            if self.size is None:
                self._fetch()
            position = self.size + offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_SET:
            position = offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        if self.size is None:
            self._fetch()
        end = self.size if size is None or size < 0 else min(self.position + size, self.size)
        start = self.position
        if end <= start:
            return b""
        for first, body in self.ranges:
            if first <= start and end <= first + len(body):
                result = body[start - first:end - first]
                break
        else:
            result = self._fetch(start, end - 1)
        self.position = end
        return result

    def close(self):
        self.session.close()
        super().close()


@lru_cache(maxsize=4)
def _central_directory(url):
    with _HTTPRangeReader(url, cache=True) as remote:
        # Validate HTTP support before zipfile can turn transport errors into BadZipFile.
        remote.seek(0, os.SEEK_END)
        with zipfile.ZipFile(remote) as archive:
            return remote.size, tuple(remote.ranges), tuple(archive.infolist())


def list_members() -> pd.DataFrame:
    """Return the ZIP member table, caching its central directory in memory.

    Columns: ``name`` (relative to Dataset/), ``filename`` (original ZIP name),
    ``size``, ``compressed_size``, ``compression``, ``crc32`` and ``is_dir``.
    Includes directory and metadata entries, so the published archive has 707
    rows. Only the ZIP tail and central directory are fetched, never the archive.
    """
    _, _, entries = _central_directory(ZENODO_URL)
    return pd.DataFrame.from_records(
        [(entry.filename.removeprefix("Dataset/"), entry.filename, entry.file_size,
          entry.compress_size, entry.compress_type, entry.CRC, entry.is_dir()) for entry in entries],
        columns=["name", "filename", "size", "compressed_size", "compression", "crc32", "is_dir"],
    )


def _root_path(root=None):
    return Path(root or os.environ.get("STAIG_DATA", "./Dataset")).expanduser()


def _safe_member(name):
    name = str(name).removeprefix("Dataset/").rstrip("/")
    if (not name or name.startswith("/") or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe dataset member: {name!r}")
    return name


def download_dataset(root=None, members=None, include_full_image=True, overwrite=False, verbose=True):
    """Extract selected Zenodo ZIP entries using HTTP Range requests.

    ``root`` is the Dataset directory itself, defaulting to ``STAIG_DATA`` or
    ``./Dataset``. ``members`` is a prefix or iterable of prefixes relative to
    Dataset/, e.g. ``["DLPFC/151673", "other/starmap"]``; None selects all data.
    Exact filenames are also accepted. Set ``include_full_image=False`` to omit
    tissue_full_image.tif/.tiff. Existing files of the right size are skipped
    unless ``overwrite=True``. Resumption is at file boundaries: incomplete
    files are downloaded again. Files are CRC-checked and atomically replaced.
    Returns only the paths written during this call (not skipped files).
    """
    if isinstance(members, str):
        members = [members]
    prefixes = None if members is None else [_safe_member(name) for name in members]
    if prefixes == []:
        return []
    size, ranges, entries = _central_directory(ZENODO_URL)
    selected = []
    matched = set()
    root = _root_path(root)
    resolved_root = root.resolve()
    for entry in entries:
        if entry.is_dir() or not entry.filename.startswith("Dataset/"):
            continue
        name = entry.filename.removeprefix("Dataset/")
        matching = [] if prefixes is None else [
            prefix for prefix in prefixes if name == prefix or name.startswith(prefix + "/")
        ]
        if prefixes is not None and not matching:
            continue
        matched.update(matching)
        name = _safe_member(name)
        if not include_full_image and PurePosixPath(name).name.lower() in {
            "tissue_full_image.tif", "tissue_full_image.tiff",
        }:
            continue
        path = root / name
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Dataset path escapes root: {path}")
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"ZIP symlink is not a dataset file: {name}")
        selected.append((entry, path))
    if prefixes is not None and (missing := set(prefixes) - matched):
        raise FileNotFoundError(f"No dataset members match: {sorted(missing)}")

    pending = [(entry, path) for entry, path in selected
               if overwrite or not path.is_file() or path.stat().st_size != entry.file_size]
    if not pending:
        return []
    written = []
    with _HTTPRangeReader(ZENODO_URL, size, ranges) as remote, zipfile.ZipFile(remote) as archive:
        for entry, path in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Downloading {entry.filename} ({entry.file_size:,} bytes)")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                                 suffix=".part", delete=False) as target:
                    temporary = Path(target.name)
                    with archive.open(entry) as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                if temporary.stat().st_size != entry.file_size:
                    raise zipfile.BadZipFile(f"Incorrect extracted size for {entry.filename}")
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            written.append(path)
    return written


def data_root(root: str | os.PathLike | None = None) -> Path:
    root = _root_path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root {root} not found. Use download_dataset(root=...) or a loader with "
            "download=True, or pass root=... / set STAIG_DATA to an unpacked Dataset folder.")
    return root


def load_visium_dir(path: str | os.PathLike, labels: str | None = "truth.txt",
                    image_features: str | None = "embeddings.npy",
                    drop_na: bool = True, load_images: bool = True) -> ad.AnnData:
    """Read one 10x Visium folder (``filtered_feature_bc_matrix.h5`` + ``spatial/``) with optional
    labels (``truth.txt``) and pre-computed image features (``embeddings.npy``, v1 BYOL output)."""
    from .io import read_visium

    return read_visium(path, labels=labels, image_features=image_features, drop_na=drop_na, load_images=load_images)


def load_dlpfc(slice_id: str = "151673", root=None, download=False, **kwargs) -> ad.AnnData:
    """One DLPFC slice with layer annotations; optionally download a missing folder."""
    slice_id = str(slice_id)
    if slice_id not in DLPFC_SLICES:
        raise ValueError(f"slice_id must be one of {DLPFC_SLICES}")
    return _load_visium_member(f"DLPFC/{slice_id}", root, download, **kwargs)


def load_visium(name: str, root=None, download=False, **kwargs) -> ad.AnnData:
    """``Human_Breast_Cancer``, ``Mouse_Brain_Anterior``, ``Mouse_Brain_Posterior``, ... (``v10x`` folder)."""
    return _load_visium_member(f"v10x/{_safe_member(name)}", root, download, **kwargs)


def _load_visium_member(member, root, download, **kwargs):
    root = _root_path(root)
    if download and not (root / member).exists():
        download_dataset(root=root, members=[member])
    return load_visium_dir(data_root(root) / member, **kwargs)


def load_h5ad(name: str, root=None, download=False) -> ad.AnnData:
    """Non-Visium datasets shipped as h5ad under ``other/`` (Stereo-seq, Slide-seqV2, STARmap, MERFISH)."""
    root = _root_path(root)
    name = _safe_member(name)
    pattern = name if name.endswith(".h5ad") else f"{name}*.h5ad"
    matches = sorted((root / "other").rglob(pattern))
    if not matches and download:
        table = list_members()
        candidates = sorted(row.name for row in table.itertuples()
                            if row.filename.startswith("Dataset/other/") and not row.is_dir
                            and PurePosixPath(row.name.removeprefix("other/")).match(pattern))
        if candidates:
            download_dataset(root=root, members=[candidates[0]])
            matches = sorted((root / "other").rglob(pattern))
    root = data_root(root) / "other"
    if not matches:
        raise FileNotFoundError(f"no h5ad matching {name!r} under {root}")
    return sc.read_h5ad(matches[0])


def n_clusters_for(adata: ad.AnnData, label_key: str = "ground_truth") -> int:
    return int(pd.Series(adata.obs[label_key]).dropna().nunique())


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Download selected files from the STAIG Zenodo dataset")
    parser.add_argument("--root", help="Dataset directory (default: STAIG_DATA or ./Dataset)")
    parser.add_argument("--members", nargs="+", help="Dataset prefixes, e.g. DLPFC/151673 other/starmap")
    parser.add_argument("--include-full-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    written = download_dataset(root=args.root, members=args.members,
                               include_full_image=args.include_full_image,
                               overwrite=args.overwrite, verbose=not args.quiet)
    if not args.quiet:
        print(f"Wrote {len(written)} file(s)")


if __name__ == "__main__":
    _main()
