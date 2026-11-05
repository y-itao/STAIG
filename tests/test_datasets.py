"""Dataset downloads are tested offline against an actual HTTP Range server."""

from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from staig import datasets


@contextmanager
def range_server(path):
    state = {"requests": [], "bytes": 0, "mode": "normal"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            size = path.stat().st_size
            header = self.headers.get("Range", "")
            state["requests"].append(header)
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if not match:
                self.send_error(400, "A Range header is required")
                return
            left, right = match.groups()
            start = int(left) if left else max(0, size - int(right))
            end = min(int(right), size - 1) if left and right else size - 1
            if start >= size or start > end:
                self.send_error(416)
                return
            if state["mode"] == "ignore":
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                return  # The client must refuse without reading the body.
            self.send_response(206)
            content_range = f"bytes {start}-{end}/{size}"
            if state["mode"] == "bad_range":
                content_range = f"bytes {start + 1}-{end}/{size}"
            body_size = end - start + 1
            if state["mode"] == "truncated":
                body_size -= 1
            self.send_header("Content-Range", content_range)
            self.send_header("Content-Length", str(body_size))
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                body = source.read(body_size)
            self.wfile.write(body)
            state["bytes"] += len(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/Dataset.zip", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def remote_zip(tmp_path, monkeypatch):
    path = tmp_path / "fixture.zip"
    contents = {
        "Dataset/DLPFC/151673/truth.txt": b"spot1\tLayer1\nspot2\tLayer2\n",
        "Dataset/DLPFC/151673/embeddings.npy": b"fake embeddings" * 300,
        "Dataset/DLPFC/151673/spatial/tissue_full_image.tif": b"full image" * 100,
        "Dataset/v10x/Human_Breast_Cancer/truth.txt": b"spot1\tTumor\n",
        "Dataset/other/starmap/starmap.h5ad": b"fake h5ad" * 100,
        "Dataset/DLPFC/1516730/truth.txt": b"prefix must end at a path boundary",
        "__MACOSX/._Dataset": b"metadata",
        # Keep selected entries outside the directory tail; this must not be downloaded.
        "Dataset/unselected.bin": np.random.default_rng(0).bytes(200_000),
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Dataset/", b"")
        archive.writestr("Dataset/DLPFC/151673/", b"")
        for index, (name, body) in enumerate(contents.items()):
            archive.writestr(name, body, compress_type=zipfile.ZIP_STORED if index % 2 == 0
                             else zipfile.ZIP_DEFLATED)
    with range_server(path) as (url, state):
        monkeypatch.setattr(datasets, "ZENODO_URL", url)
        datasets._central_directory.cache_clear()
        yield path, contents, state
        datasets._central_directory.cache_clear()


def test_list_members_caches_directory(remote_zip):
    path, contents, state = remote_zip
    table = datasets.list_members()
    assert len(table) == len(contents) + 2
    assert table.is_dir.sum() == 2
    assert set(table.compression) == {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
    for row in table[~table.is_dir].itertuples():
        assert row.size == len(contents[row.filename])
    assert "DLPFC/151673/truth.txt" in set(table.name)
    assert state["requests"] and all(item.startswith("bytes=") for item in state["requests"])
    assert state["bytes"] < path.stat().st_size // 2
    count = len(state["requests"])
    table.loc[0, "size"] = -1
    assert datasets.list_members().iloc[0]["size"] == 0
    assert len(state["requests"]) == count


def test_selective_extraction_resume_and_overwrite(remote_zip, tmp_path):
    archive_path, contents, state = remote_zip
    root = tmp_path / "output"
    kwargs = dict(root=root, members=["DLPFC/151673", "other/starmap"],
                  include_full_image=False, verbose=False)
    written = datasets.download_dataset(**kwargs)
    expected = [name for name in contents if name.startswith(("Dataset/DLPFC/151673/",
                                                             "Dataset/other/starmap/"))
                and not name.endswith(".tif")]
    assert set(written) == {root / name.removeprefix("Dataset/") for name in expected}
    for name in expected:
        assert (root / name.removeprefix("Dataset/")).read_bytes() == contents[name]
    assert not (root / "Dataset").exists()
    assert not (root / "DLPFC/1516730").exists()
    assert not (root / "DLPFC/151673/spatial/tissue_full_image.tif").exists()
    assert not (root / "__MACOSX").exists()
    assert state["bytes"] < archive_path.stat().st_size // 2
    count = len(state["requests"])
    assert datasets.download_dataset(**kwargs) == []
    assert len(state["requests"]) == count

    damaged = written[0]
    damaged.write_bytes(b"partial")
    assert datasets.download_dataset(**kwargs) == [damaged]
    original = contents[f"Dataset/{damaged.relative_to(root).as_posix()}"]
    assert damaged.read_bytes() == original
    damaged.write_bytes(b"x" * len(original))
    assert datasets.download_dataset(**kwargs) == []  # Resume is explicitly size-based.
    assert set(datasets.download_dataset(**kwargs, overwrite=True)) == set(written)
    assert damaged.read_bytes() == original
    assert not list(root.rglob("*.part"))


def test_full_image_and_environment_root(remote_zip, tmp_path, monkeypatch):
    _, contents, _ = remote_zip
    root = tmp_path / "environment-root"
    monkeypatch.setenv("STAIG_DATA", str(root))
    written = datasets.download_dataset(members="Dataset/DLPFC/151673/", verbose=False)
    image = root / "DLPFC/151673/spatial/tissue_full_image.tif"
    assert image in written
    assert image.read_bytes() == contents["Dataset/DLPFC/151673/spatial/tissue_full_image.tif"]
    assert datasets.data_root() == root


def test_all_members_and_empty_selection(remote_zip, tmp_path):
    _, contents, state = remote_zip
    assert datasets.download_dataset(tmp_path / "empty", members=[]) == []
    assert not state["requests"]
    root = tmp_path / "all"
    written = datasets.download_dataset(root, verbose=False)
    assert set(written) == {root / name.removeprefix("Dataset/") for name in contents
                            if name.startswith("Dataset/")}


def test_unknown_member_is_not_silently_ignored(remote_zip, tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="No dataset members match"):
        datasets.download_dataset(root, members=["DLPFC/151673", "DLPFC/unknown"])
    assert not root.exists()


@pytest.mark.parametrize("member", ["../escape", "/absolute", "DLPFC/../escape", "C:\\escape", ""])
def test_unsafe_selection_is_rejected(tmp_path, member):
    with pytest.raises(ValueError, match="Unsafe dataset member"):
        datasets.download_dataset(tmp_path / "output", members=[member])


@pytest.mark.parametrize("mode, message", [
    ("ignore", "must support HTTP Range"),
    ("bad_range", "different byte range"),
    ("truncated", "Truncated"),
])
def test_invalid_range_responses(remote_zip, mode, message):
    _, _, state = remote_zip
    state["mode"] = mode
    with pytest.raises(OSError, match=message):
        datasets.list_members()


def test_corrupt_member_keeps_existing_file(remote_zip, tmp_path):
    path, _, _ = remote_zip
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("Dataset/DLPFC/151673/truth.txt")
    # Damage stored content while retaining its original CRC in the directory.
    with path.open("r+b") as stream:
        stream.seek(info.header_offset + 26)
        name_length, extra_length = struct.unpack("<HH", stream.read(4))
        stream.seek(info.header_offset + 30 + name_length + extra_length)
        stream.write(b"!")
    root = tmp_path / "output"
    target = root / "DLPFC/151673/truth.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing file")
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        datasets.download_dataset(root, members=["DLPFC/151673/truth.txt"], verbose=False)
    assert target.read_bytes() == b"existing file"
    assert not list(root.rglob("*.part"))


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_unsafe_zip_entries(tmp_path, monkeypatch, kind):
    path = tmp_path / "unsafe.zip"
    entry = zipfile.ZipInfo("Dataset/../escape" if kind == "traversal" else "Dataset/link")
    if kind == "symlink":
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(entry, b"target")
    with range_server(path) as (url, _):
        monkeypatch.setattr(datasets, "ZENODO_URL", url)
        with pytest.raises(ValueError, match="Unsafe dataset member|ZIP symlink"):
            datasets.download_dataset(tmp_path / "output", verbose=False)
    assert not (tmp_path / "escape").exists()


def test_existing_symlink_cannot_escape_root(remote_zip, tmp_path):
    root = tmp_path / "output"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "DLPFC").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes root"):
        datasets.download_dataset(root, members="DLPFC/151673", verbose=False)
    assert not list(outside.iterdir())


def test_zip64_directory(tmp_path, monkeypatch):
    path = tmp_path / "zip64.zip"
    payload = b"stored ZIP64 data" * 100
    with monkeypatch.context() as patch:
        patch.setattr(zipfile, "ZIP64_LIMIT", 100)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("Dataset/other/example/file.h5ad", payload)
    assert b"PK\x06\x06" in path.read_bytes()
    with range_server(path) as (url, _):
        monkeypatch.setattr(datasets, "ZENODO_URL", url)
        assert len(datasets.list_members()) == 1
        written = datasets.download_dataset(tmp_path / "output", members="other/example", verbose=False)
    assert written[0].read_bytes() == payload


@pytest.mark.parametrize("loader, name, member", [
    (datasets.load_dlpfc, "151673", "DLPFC/151673"),
    (datasets.load_visium, "Human_Breast_Cancer", "v10x/Human_Breast_Cancer"),
])
def test_visium_loader_download_is_opt_in(remote_zip, tmp_path, monkeypatch, loader, name, member):
    _, _, state = remote_zip
    root = tmp_path / "output"
    calls = []
    marker = object()

    def load(path, **kwargs):
        assert (path / "truth.txt").is_file()
        calls.append((path, kwargs))
        return marker

    monkeypatch.setattr(datasets, "load_visium_dir", load)
    with pytest.raises(FileNotFoundError):
        loader(name, root=root)
    assert not state["requests"]
    assert loader(name, root=root, download=True, load_images=False) is marker
    assert calls == [(root / member, {"load_images": False})]
    count = len(state["requests"])
    assert loader(name, root=root, download=True) is marker
    assert len(state["requests"]) == count


def test_missing_slice_under_existing_root_downloads(remote_zip, tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "load_visium_dir", lambda path, **kwargs: path)
    path = datasets.load_dlpfc("151673", root=tmp_path, download=True)
    assert (path / "truth.txt").exists()


@pytest.mark.parametrize("name", ["starmap", "starmap/starmap.h5ad"])
def test_h5ad_loader_download_is_opt_in(remote_zip, tmp_path, monkeypatch, name):
    _, contents, state = remote_zip
    root = tmp_path / "output"
    monkeypatch.setattr(datasets.sc, "read_h5ad", lambda path: path.read_bytes())
    with pytest.raises(FileNotFoundError):
        datasets.load_h5ad(name, root=root)
    assert not state["requests"]
    expected = contents["Dataset/other/starmap/starmap.h5ad"]
    assert datasets.load_h5ad(name, root=root, download=True) == expected
    count = len(state["requests"])
    assert datasets.load_h5ad(name, root=root, download=True) == expected
    assert len(state["requests"]) == count


def test_module_cli(remote_zip, tmp_path, capsys):
    _, _, state = remote_zip
    root = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-m", "staig.datasets", "--help"],
                            capture_output=True, text=True, check=True, timeout=60)
    assert "--members" in result.stdout
    datasets._main(["--root", str(root), "--members", "DLPFC/151673", "other/starmap",
                    "--no-include-full-image"])
    assert "Wrote 3 file(s)" in capsys.readouterr().out
    assert state["requests"]
    assert (root / "other/starmap/starmap.h5ad").exists()
    assert not (root / "DLPFC/151673/spatial/tissue_full_image.tif").exists()


_LOCAL_SLICE = Path(os.environ.get("STAIG_DATA", "/nonexistent")) / "DLPFC" / "151673"


@pytest.mark.skipif(not _LOCAL_SLICE.exists(), reason="Local paper dataset is not available")
def test_load_visium_dir_local_paper_slice():
    adata = datasets.load_visium_dir(_LOCAL_SLICE)
    assert adata.n_obs > 0 and adata.n_vars > 0
    assert adata.var_names.is_unique
    assert adata.obs_names.is_unique
    assert adata.obs["ground_truth"].notna().all()
    assert datasets.n_clusters_for(adata) == 7
    assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
    assert adata.obsm["image_features"].shape == (adata.n_obs, 2048)
    assert adata.obsm["image_features"].dtype == np.float32
    assert adata.uns["spatial"]
    assert adata.uns["staig"]["source"] == str(_LOCAL_SLICE)


@pytest.mark.skipif(not _LOCAL_SLICE.exists(), reason="Local paper archive is not available")
def test_local_paper_archive_directory(tmp_path, monkeypatch):
    """Use HTTP Range against the local real ZIP64 archive; extract only truth.txt."""
    path = _LOCAL_SLICE.parents[3] / "Dataset.zip"
    if not path.exists():
        pytest.skip("Local Dataset.zip is not available")
    with range_server(path) as (url, state):
        monkeypatch.setattr(datasets, "ZENODO_URL", url)
        table = datasets.list_members()
        assert len(table) == 707
        info = table.loc[table.name == "DLPFC/151673/embeddings.npy"].iloc[0]
        assert info["size"] == (_LOCAL_SLICE / "embeddings.npy").stat().st_size
        written = datasets.download_dataset(tmp_path / "output", members="DLPFC/151673/truth.txt",
                                            verbose=False)
        assert written[0].read_bytes() == (_LOCAL_SLICE / "truth.txt").read_bytes()
        assert state["bytes"] < 300_000


def test_range_reader_seek_contract(remote_zip):
    path, _, _ = remote_zip
    with datasets._HTTPRangeReader(datasets.ZENODO_URL) as reader:
        assert reader.readable() and reader.seekable()
        assert reader.seek(-22, os.SEEK_END) == path.stat().st_size - 22
        assert reader.read(4) == b"PK\x05\x06"
        assert reader.seek(0, os.SEEK_END) == path.stat().st_size
        assert reader.read() == b""
        assert reader.seek(-4, os.SEEK_CUR) == path.stat().st_size - 4
        assert reader.read(0) == b""
        assert reader.seek(0) == 0
        assert reader.read(4) == b"PK\x03\x04"
        with pytest.raises(ValueError, match="negative"):
            reader.seek(-1)
        with pytest.raises(ValueError, match="whence"):
            reader.seek(0, 99)
    with pytest.raises(ValueError, match="closed"):
        reader.read(1)
