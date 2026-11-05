# Running the settings of the paper

The experiments of [Yang *et al.* (2025)](https://doi.org/10.1038/s41467-025-56276-0) are available as
presets and scripts. The `paper_*` presets carry the per-dataset settings of the paper (float64, the
seeds, τ, epochs, image pseudo-labels, image PCs and the loss convention of version 1); the curated
datasets, including the released histology features, come from
[Zenodo record 14231773](https://zenodo.org/records/14231773).

## Presets

```python
import staig

for name in sorted(staig.PRESETS):
    if name.startswith("paper_"):
        print(name, staig.preset_config(name).to_dict())
```

| preset | experiment |
|---|---|
| `paper_151673` | DLPFC slice 151673 with histology (Figure 2) |
| `paper_integration_vertical` | adjacent DLPFC slices 151675 + 151676 (Figure 5a) |
| `paper_integration_horizontal` | mouse brain anterior + posterior (Figure 5b) |
| `paper_mouse_olfactory_stereo` | Stereo-seq mouse olfactory bulb (Figure 4) |
| `paper_slideseq_olfactory` | Slide-seqV2 mouse olfactory bulb (Figure 4) |
| `paper_starmap` | STARmap mouse visual cortex (Figure 4) |
| `paper_merfish` | MERFISH mouse visual cortex (Figure 4) |

## A DLPFC slice with histology

```python
import os
import staig
from staig import datasets, metrics, pl

adata = datasets.load_dlpfc("151673", root=os.getenv("STAIG_DATA"), download=True)   # counts, labels, features
model = staig.run(adata, n_clusters=7, preset="paper_151673", n_seeds=1, smooth=False, image=None)
print(metrics.evaluate(adata))            # ARI / NMI against the layer annotation
pl.spatial(adata, color=["ground_truth", "staig_domain"])
```

`image=None` keeps the released histology features of the slice (`obsm["image_features"]`) instead of
extracting new ones; drop it to recompute them from the full-resolution image. `n_seeds=1` and
`smooth=False` run the single training of the paper; the version 2 defaults (`n_seeds=3` on a GPU,
`smooth=True`) add the consensus and smoothing steps on top of it.

## The command-line scripts

`benchmarks/run_dlpfc.py` runs any configuration on one or all annotated DLPFC slices and writes a
per-seed CSV (ARI/NMI before and after refinement, training time):

```bash
python benchmarks/run_dlpfc.py --root "$STAIG_DATA" --slices 151673 --preset paper_151673 \
  --seeds 39788 --cluster gmm --tag paper
python benchmarks/run_dlpfc.py --root "$STAIG_DATA" --slices all \
  --paper-configs configs/paper/train_img_config.yaml --seeds paper --tag paper_all
```

`--paper-configs` replays the per-slice YAML settings shipped with version 1 (`configs/paper/`);
`--seeds 0,1,2` runs several seeds; `--features-dir` reads replacement image features named
`<slice>.npy`; `--cluster mclust` uses R mclust through the optional `[r]` extra.
`benchmarks/run_no_image.py` covers the Stereo-seq, Slide-seqV2, STARmap and MERFISH experiments and
`benchmarks/run_integration.py` the integration scenarios (`dlpfc_7576`, `dlpfc_4`, `mouse_brain`,
`merfish_vis`, `cross_platform`), with BatchKL and iLISI.

## Reporting

Run several seeds (`--seeds 0,1,2` or `run(n_seeds=...)`) and report the distribution; record the
preset, overrides, package versions, device and dtype with the result. The consensus labelling of
`run(n_seeds=...)` is a stable summary of the seeds; `uns["staig"]["consensus"]` records the seeds and
their agreement with the consensus.
