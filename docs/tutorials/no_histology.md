# Data without histology

Expression and `obsm["spatial"]` are sufficient. `run` defaults to
`preset="auto"`, which selects a configuration from the batch key, image/Visium
metadata and gene count. Choose the platform preset explicitly when you know
the assay; this heuristic does not identify every platform uniquely.

| Platform | Preset | Example archive member under `other/` | Expression state |
| --- | --- | --- | --- |
| Stereo-seq | `stereo_seq` | `Mouse_Olfactory_Stereo-seq/filtered_feature_bc_matrix_norm.h5ad` | Already normalised |
| Slide-seqV2 | `slide_seq` | `Mouse_Olfactory_slide-seqV2/v2.h5ad` | Already normalised |
| STARmap | `starmap` | `starmap/starmap_norm.h5ad` | Already normalised |
| MERFISH | `merfish` | `Merfish/mouse1.AUD_TEA_VIS.242.unexpand_cluster3.h5ad` | Counts in this release |

These presets use fixed edge removal without image pseudo-label debiasing.
Stereo-seq, Slide-seq and STARmap presets use two graph layers; MERFISH uses one.
Inspect your own h5ad's provenance rather than inferring normalisation from the
platform name. The archive also contains count-based Slide-seq files.

## Stereo-seq example

```python
import os
import staig
from staig import datasets, pp, pl, metrics

adata = datasets.load_h5ad(
    "Mouse_Olfactory_Stereo-seq/filtered_feature_bc_matrix_norm.h5ad",
    root=os.getenv("STAIG_DATA"), download=True,
)
adata.var_names_make_unique()
if "spatial" not in adata.obsm:
    columns = ["X", "Y"] if {"X", "Y"} <= set(adata.obs) else ["x", "y"]
    adata.obsm["spatial"] = adata.obs[columns].to_numpy()

pp.preprocess(
    adata, n_top_genes=3000, normalize=False, log1p=False,
    flavor="seurat", scale=False,
)
model = staig.run(
    adata, n_clusters=7, preset="stereo_seq", preprocess=False,
    epochs=5, device="cpu", loss_chunk_size=256, n_negatives=4096, image=None,
    cluster_method="kmeans", refine=False, verbose=False,
)
pl.spatial(adata, color="staig_domain", img_key=None)
print(metrics.evaluate(adata, label_key=None))
```

The small epoch count, sampled negatives and k-means make this an execution
example, not a paper reproduction. `image=None` disables automatic image
extraction. The [notebook](../examples/02_no_histology_stereo_seq.ipynb)
records its full-data run and finite-loss checks.

For log-normalised inputs, `normalize=False, log1p=False` prevents a second
transformation; `flavor="seurat"` selects HVGs in that space. The paper's own
processing applied `seurat_v3` to these files, which `strict_counts=False`
reproduces.

For raw-count inputs, use `pp.preprocess(adata)` before `run(...,
preprocess=False)`, or let `run` preprocess automatically. For the example
MERFISH files, annotations may be named `cluster_L2`; pass
`label_key="cluster_L2"` to `metrics.evaluate` when using those labels. Without
annotations, SC and DB describe embedding structure, not anatomical truth.
Choose a domain count for your scientific question; automatic cluster-number
selection is available by omitting `n_clusters` but can be expensive.
