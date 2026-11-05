"""Readers on real vendor sample files (run when STAIG_SAMPLES points to the samples folder, see
docs/input_formats.md for the download commands)."""

import os
from pathlib import Path

import numpy as np
import pytest

from staig import io

ROOT = os.environ.get("STAIG_SAMPLES")
pytestmark = pytest.mark.skipif(not ROOT, reason="STAIG_SAMPLES not set")
S = Path(ROOT or ".")


def _skip_unless(path: Path):
    if not path.exists():
        pytest.skip(f"{path} missing")


def test_xenium_tiny():
    _skip_unless(S / "xenium_tiny")
    a = io.read(S / "xenium_tiny")
    assert a.uns["staig"]["platform"] == "xenium" and a.n_vars == 5006
    assert set(a.uns["staig"]["dropped_features"]) >= {"Unassigned Codeword", "Negative Control Probe"}
    assert a.uns["staig"]["xenium"]["pixel_size"] == 0.2125 and np.isfinite(a.obsm["spatial"]).all()


def test_visium_hd_tiny():
    _skip_unless(S / "visium_hd_tiny")
    a = io.read(S / "visium_hd_tiny")
    assert a.uns["staig"]["platform"] == "visium_hd" and a.uns["staig"]["bin_size_um"] == 8.0 and a.n_obs > 5000
    c = io.read(S / "visium_hd_tiny", level="cells")
    assert c.n_obs == 612 and np.isfinite(c.obsm["spatial"]).all()


def test_visium_cytassist():
    _skip_unless(S / "visium_cytassist")
    a = io.read(S / "visium_cytassist")
    assert a.n_obs == 4169 and a.n_vars == 18085 and "spatial" in a.uns


def test_cosmx_exports():
    _skip_unless(S / "cosmx_hgsoc")
    a = io.read(S / "cosmx_hgsoc")
    assert a.uns["staig"]["platform"] == "cosmx" and a.n_vars == 960 and "NegPrb1" not in a.var_names
    b = io.read(S / "cosmx_raw" / "CosMx_raw_data" / "HS_FOV16")
    assert b.n_vars == 968 and "0" not in {n.split("_")[1] for n in b.obs_names}


def test_stereo_seq_samples():
    _skip_unless(S / "cartloader" / "stereoseq" / "raw" / "input.tsv.gz")
    a = io.read(S / "cartloader" / "stereoseq" / "raw" / "input.tsv.gz", bin_size=50)
    assert a.uns["staig"]["platform"] == "stereo_seq" and a.uns["staig"]["level"] == "bins" and a.n_obs > 5000


def test_seqfish_sample():
    _skip_unless(S / "seqfish2")
    a = io.read(S / "seqfish2" / "instrument 2 official")
    assert a.uns["staig"]["platform"] == "seqfish" and a.n_vars == 12
