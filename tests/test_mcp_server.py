"""Optional MCP protocol and output-preservation checks (CPU, no downloads)."""

import json

import anndata as ad
import numpy as np
import pytest

pytest.importorskip("mcp.server.fastmcp")
memory = pytest.importorskip("mcp.shared.memory")
import anyio  # noqa: E402

from staig.mcp_server import mcp, staig_extract_image_features, staig_run  # noqa: E402


def test_protocol_run_and_readback(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rng = np.random.default_rng(7)
    adata = ad.AnnData(rng.poisson(3, (40, 12)).astype(np.float32))
    adata.obsm["spatial"] = rng.random((40, 2))
    # Explicitly prepared input: isolate the protocol/persistence contract from HVG selection.
    adata.obsm["X_staig_input"] = rng.random((40, 12), dtype=np.float32)
    source, target = tmp_path / "input.h5ad", tmp_path / "output.h5ad"
    adata.write_h5ad(source)
    original = source.read_bytes()

    async def exercise():
        async with memory.create_connected_server_and_client_session(mcp) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == {"staig_presets", "staig_run", "staig_extract_image_features"}
            presets = await client.call_tool("staig_presets", {})
            assert not presets.isError
            assert "paper_151673" in json.loads(presets.content[0].text)
            result = await client.call_tool("staig_run", {
                "input_path": str(source), "output_path": str(target),
                "preset": "merfish", "n_clusters": None,
            })
            assert not result.isError, result
            report = json.loads(result.content[0].text)
            assert report["output_path"] == str(target)
            assert set(report["metrics"]) == {"SC", "DB"}
            denied = await client.call_tool("staig_run", {
                "input_path": str(source), "output_path": str(source), "n_clusters": 3,
            })
            assert denied.isError

    anyio.run(exercise)
    assert source.read_bytes() == original
    saved = ad.read_h5ad(target)
    assert saved.obsm["X_staig"].shape == (40, 64)
    assert np.isfinite(saved.obsm["X_staig"]).all()
    assert saved.obs["staig_domain"].notna().all()
    assert all(isinstance(key, str) for key in saved.uns["staig"]["n_clusters_selection"])


def test_input_errors_do_not_create_outputs(tmp_path):
    source, target = tmp_path / "input.h5ad", tmp_path / "output.h5ad"
    adata = ad.AnnData(np.ones((5, 3), dtype=np.float32))
    adata.write_h5ad(source)
    with pytest.raises(ValueError, match="spatial"):
        staig_run(str(source), n_clusters=2, output_path=str(target))
    assert not target.exists()
    adata.obsm["spatial"] = np.ones((5, 2))
    adata.write_h5ad(source)
    with pytest.raises(ValueError, match="batch_key"):
        staig_run(str(source), n_clusters=2, batch_key="missing", output_path=str(target))
    with pytest.raises(FileNotFoundError):
        staig_extract_image_features(str(source), str(tmp_path / "missing.tif"), str(tmp_path / "f.npy"))
    assert not target.exists()
