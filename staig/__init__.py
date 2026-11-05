"""STAIG: spatial transcriptomics analysis via image-aided graph contrastive learning.

Quick start::

    import staig
    adata = staig.read("path/to/spaceranger_outs")   # Visium, Visium HD, Xenium, MERSCOPE, CosMx, Stereo-seq, ...
    model = staig.run(adata, n_clusters=7)           # preprocess -> graph -> train -> cluster
    staig.pl.spatial(adata, color="staig_domain")

``staig.run("path/to/outs")`` does both steps; the histology branch is added automatically when the
full-resolution H&E image is found next to a Visium folder.
"""

from __future__ import annotations

from . import augment, cluster, datasets, graph, img, io, loss, metrics, model, pl, pp
from ._config import PRESETS, STAIGConfig, auto_preset, preset_config
from .io import detect_platform, read
from .train import STAIG, run

__version__ = "2.0.0"

__all__ = [
    "STAIG", "run", "read", "detect_platform", "STAIGConfig", "PRESETS", "preset_config", "auto_preset",
    "pp", "graph", "augment", "model", "loss", "img", "io", "pl", "cluster", "metrics", "datasets",
    "__version__",
]
