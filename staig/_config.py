"""Configuration dataclass, presets and YAML helpers."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# n_negatives="auto": exact loss on GPU; on CPU exact up to CPU_EXACT_MAX_N spots, else AUTO_N_NEGATIVES sampled
CPU_EXACT_MAX_N = 8000
GPU_EXACT_MAX_N = 65536      # exact denominator on a GPU up to this many observations (Visium HD / Stereo-seq beyond)
GPU_AUTO_N_NEGATIVES = 8192
AUTO_N_NEGATIVES = 4096


@dataclass
class STAIGConfig:
    """All hyper-parameters of the graph model (see docs/dev/DESIGN.md)."""

    # ---- graph -------------------------------------------------------------------
    n_neighbors: int = 5
    edge_removal: str = "auto"      # "image" | "gene" | "fixed" | "auto"
    fixed_drop_prob: float = 0.2
    n_image_pcs: int = 16           # PCs of the image features used for edge weights / pseudo-labels
    n_gene_pcs: int = 64            # PCs of expression used for gene-similarity edge weights
    # ---- augmentation ------------------------------------------------------------
    drop_feature_rates: tuple[float, float] = (0.1, 0.2)
    n_graph_samples: int = 1        # >1: average n sampled graphs, keep edges present in >= 50 %
    # ---- negatives ----------------------------------------------------------------
    debias: str = "auto"            # "pseudo_label" | "none" | "auto"
    n_pseudo_labels: int = 40
    within_slice_negatives: bool = True
    # "auto": exact on GPU and for <= 8000 spots on CPU, else 4096 sampled; int = K sampled; None/0 = exact
    n_negatives: int | str | None = "auto"
    # ---- model -------------------------------------------------------------------
    hidden_dim: int = 64
    proj_dim: int = 64
    n_layers: int = 1
    activation: str = "prelu"       # "prelu" | "relu"
    embedding: str = "projection"   # "projection" (v1) | "encoder"
    init: str = "glorot"            # "glorot" (v1, random) | "pca" (first GCN layer from PCA of the propagated input)
    # ---- optimisation -------------------------------------------------------------
    tau: float = 10.0
    epochs: int = 400
    lr: float = 5e-4
    weight_decay: float = 1e-5
    lr_schedule: str = "none"       # "none" | "cosine"
    loss_variant: str = "full"      # "full" (every cross-view term in the denominator) | "v1" (STAIG 1.x convention)
    dtype: str = "float32"
    loss_chunk_size: int = 4096
    # ---- misc --------------------------------------------------------------------
    seed: int = 0
    device: str = "auto"
    deterministic: bool = True
    verbose: bool = True
    log_every: int = 50

    # ------------------------------------------------------------------------------
    def validate(self) -> STAIGConfig:
        if self.edge_removal not in {"image", "gene", "fixed", "auto"}:
            raise ValueError(f"edge_removal must be image|gene|fixed|auto, got {self.edge_removal!r}")
        if self.debias not in {"pseudo_label", "none", "auto"}:
            raise ValueError(f"debias must be pseudo_label|none|auto, got {self.debias!r}")
        if self.embedding not in {"projection", "encoder"}:
            raise ValueError("embedding must be 'projection' or 'encoder'")
        if self.loss_variant not in {"full", "v1"}:
            raise ValueError("loss_variant must be 'full' or 'v1'")
        if self.init not in {"glorot", "pca"}:
            raise ValueError("init must be 'glorot' or 'pca'")
        if self.activation not in {"prelu", "relu"}:
            raise ValueError("activation must be 'prelu' or 'relu'")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if not (0 < self.fixed_drop_prob < 1):
            raise ValueError("fixed_drop_prob must be in (0, 1)")
        r1, r2 = self.drop_feature_rates
        if not (0 <= r1 < 1 and 0 <= r2 < 1):
            raise ValueError("drop_feature_rates must be in [0, 1)")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not (self.n_negatives in (None, "auto") or (isinstance(self.n_negatives, int) and self.n_negatives >= 0)):
            raise ValueError("n_negatives must be None, 'auto' or a non-negative int")
        if self.n_graph_samples < 1:
            raise ValueError("n_graph_samples must be >= 1")
        return self

    def update(self, **overrides: Any) -> STAIGConfig:
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"Unknown config field(s): {sorted(unknown)}")
        cfg = dataclasses.replace(self, **overrides)
        if isinstance(cfg.drop_feature_rates, list):
            cfg.drop_feature_rates = tuple(cfg.drop_feature_rates)
        return cfg.validate()

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["drop_feature_rates"] = list(self.drop_feature_rates)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> STAIGConfig:
        return cls().update(**d)

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> STAIGConfig:
        if name not in PRESETS:
            raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
        return cls().update(**PRESETS[name]).update(**overrides)

    @classmethod
    def from_yaml(cls, path: str | Path, section: str | None = None) -> STAIGConfig:
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        if section is not None:
            d = d[section]
        preset = d.pop("preset", None)
        base = cls.from_preset(preset) if preset else cls()
        return base.update(**d)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


# ----------------------------------------------------------------------------------
# Presets. "default" = the paper's recommended values for 10x Visium with H&E.
# Keys named paper_* reproduce the exact per-dataset settings shipped with v1.
# ----------------------------------------------------------------------------------
# "default"/"visium": the recommended Visium configuration — tau 35, 300 epochs, 80 image pseudo-labels,
# 16 image PCs, PCA initialisation of the first GCN layer.
_VISIUM = {"tau": 35.0, "epochs": 300, "n_pseudo_labels": 80, "n_image_pcs": 16, "drop_feature_rates": (0.1, 0.2),
           "init": "pca"}
PRESETS: dict[str, dict[str, Any]] = {
    "default": dict(_VISIUM),
    "visium": dict(_VISIUM),
    "visium_no_image": {"edge_removal": "fixed", "debias": "none"},
    "stereo_seq": {"edge_removal": "fixed", "debias": "none", "n_layers": 2},
    "slide_seq": {"edge_removal": "fixed", "debias": "none", "n_layers": 2},
    "starmap": {"edge_removal": "fixed", "debias": "none", "n_layers": 2},
    "merfish": {"edge_removal": "fixed", "debias": "none", "n_layers": 1, "epochs": 200},
    "integration": {"debias": "none", "within_slice_negatives": True},
    # ---- platform aliases (same settings as the presets above) ------------------------------
    "visium_hd": {"edge_removal": "fixed", "debias": "none", "n_layers": 2},
    "xenium": {"edge_removal": "fixed", "debias": "none", "n_layers": 1, "epochs": 200},
    "merscope": {"edge_removal": "fixed", "debias": "none", "n_layers": 1, "epochs": 200},
    "cosmx": {"edge_removal": "fixed", "debias": "none", "n_layers": 1, "epochs": 200},
    "seqfish": {"edge_removal": "fixed", "debias": "none", "n_layers": 1, "epochs": 200},
    # ---- exact v1 settings -----------------------------------------------------------
    "paper_151673": {"tau": 35.0, "epochs": 300, "n_pseudo_labels": 80, "seed": 39788,
                     "drop_feature_rates": (0.1, 0.2), "n_image_pcs": 16, "n_layers": 1,
                     "debias": "pseudo_label", "edge_removal": "image", "dtype": "float64",
                     "loss_variant": "v1"},
    "paper_integration_vertical": {"tau": 30.0, "epochs": 180, "seed": 39788, "debias": "none",
                                   "drop_feature_rates": (0.1, 0.2), "n_image_pcs": 128,
                                   "edge_removal": "image", "dtype": "float64"},
    "paper_integration_horizontal": {"tau": 37.0, "epochs": 80, "seed": 36666, "debias": "none",
                                     "drop_feature_rates": (0.2, 0.2), "n_neighbors": 3,
                                     "edge_removal": "gene", "dtype": "float64"},
    "paper_mouse_olfactory_stereo": {"tau": 39.0, "epochs": 340, "seed": 39788, "debias": "none",
                                     "drop_feature_rates": (0.2, 0.2), "edge_removal": "gene",
                                     "n_layers": 1, "dtype": "float64"},
    "paper_starmap": {"tau": 2.0, "epochs": 500, "seed": 39788, "debias": "none", "n_layers": 2,
                      "drop_feature_rates": (0.01, 0.02), "edge_removal": "gene", "dtype": "float64"},
    "paper_merfish": {"tau": 1.0, "epochs": 400, "seed": 39788, "debias": "none", "n_layers": 1,
                      "drop_feature_rates": (0.01, 0.02), "edge_removal": "gene", "dtype": "float64"},
    "paper_slideseq_olfactory": {"tau": 22.0, "epochs": 310, "seed": 39788, "debias": "none", "n_layers": 2,
                                 "drop_feature_rates": (0.01, 0.02), "edge_removal": "gene", "dtype": "float64"},
}


def auto_preset(adata, batch_key: str | None = None) -> str:
    """Guess a preset from the data: Visium (image if features are present), targeted panels
    (<= 1500 genes: MERFISH / STARmap / CosMx / Xenium) or high-resolution sequencing platforms
    (Stereo-seq / Slide-seq / Visium HD bins). Multi-slice runs use the integration preset."""
    if batch_key is not None and batch_key in adata.obs and adata.obs[batch_key].nunique() > 1:
        return "integration"
    has_img = "image_features" in adata.obsm or "image_pcs" in adata.obsm
    platform = str(adata.uns.get("staig", {}).get("platform", "") or "")
    if platform and platform != "h5ad":
        from .io import preset_for_platform

        guess = preset_for_platform(platform, has_img)
        if guess is not None:
            return guess
    is_visium = "spatial" in adata.uns and any(isinstance(v, dict) and "scalefactors" in v
                                                for v in adata.uns["spatial"].values())
    if is_visium or has_img:
        return "visium" if has_img else "visium_no_image"
    if adata.n_vars <= 1500:
        return "merfish"
    return "stereo_seq"


def preset_config(name: str = "default", **overrides: Any) -> STAIGConfig:
    return STAIGConfig.from_preset(name, **overrides)


def export_presets(directory: str | Path | None = None) -> list[Path]:
    """Write every preset as a full YAML file (``staig/presets/<name>.yaml``).

    Run after editing ``PRESETS`` so the YAML mirrors stay identical to the source
    (``tests/test_robustness.py::test_preset_yaml_mirrors_match_source`` checks this).
    """
    directory = Path(directory) if directory is not None else Path(__file__).parent / "presets"
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, overrides in PRESETS.items():
        path = directory / f"{name}.yaml"
        with open(path, "w") as fh:
            fh.write(f"# STAIG preset '{name}' — full configuration (overrides vs. defaults: {overrides})\n")
            yaml.safe_dump(STAIGConfig.from_preset(name).to_dict(), fh, sort_keys=False)
        written.append(path)
    return written


def merge_configs(base: STAIGConfig, other: dict[str, Any] | STAIGConfig | None) -> STAIGConfig:
    if other is None:
        return copy.deepcopy(base)
    if isinstance(other, STAIGConfig):
        other = other.to_dict()
    return base.update(**other)
