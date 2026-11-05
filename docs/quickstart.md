# A single Visium slice with H&E

!!! tip "Starting from the platform's own files"
    `staig.run("path/to/outs")` reads Space Ranger, Visium HD, Xenium, MERSCOPE, CosMx, Stereo-seq,
    Slide-seq and seqFISH outputs directly; see [Input formats](input_formats.md). This page uses the
    paper's DLPFC slice.

Install STAIG with `[image]` for fresh image extraction. `STAIG_DATA` should point
to the unpacked `Dataset` directory itself, containing `DLPFC/`, `other/` and
`v10x/`. Passing `root=` overrides the environment variable. With neither set,
the default is `./Dataset`.

```python
import os
import staig
from staig import datasets, img, metrics, pl

adata = datasets.load_dlpfc(
    "151673", root=os.getenv("STAIG_DATA"), download=True,
)
print(adata.shape, adata.obsm["spatial"].shape)
```

`download=True` downloads a missing slice folder from
[Zenodo record 14231773](https://zenodo.org/records/14231773). An existing folder
is loaded directly; if it is incomplete, call `datasets.download_dataset` with
`members=["DLPFC/151673"]` explicitly. Labels are aligned to barcodes and
unlabelled spots are dropped by default, matching the released embeddings.

## Use the released histology features

The loader attaches `embeddings.npy` to `obsm["image_features"]`. Passing the
array through the image API demonstrates the same interface used for externally
computed features. It does not retrain BYOL.

```python
img.extract_image_features(adata, method=adata.obsm["image_features"].copy())
model = staig.run(adata, n_clusters=7)
pl.spatial(adata, color=["ground_truth", "staig_domain"])
scores = metrics.evaluate(adata)
print(scores)
```

`run` preprocesses expression, constructs the graph, trains, embeds and clusters.
It returns a fitted `STAIG` model and writes results into `adata`. The default
configuration uses 400 epochs; for a short CPU check, replace the training line
with `staig.run(adata, n_clusters=7, epochs=30, device="cpu", verbose=False)`.
`preset="auto"` selects a configuration from the data. `image="auto"` can
extract BYOL features when a full-resolution image is discoverable and features
are absent; use `image=None` to skip automatic extraction. The loader-supplied
features in this example are reused. Default single-slice clustering uses Python
GMM and spatial refinement. Evaluation
stores ARI/NMI (where labels exist), silhouette coefficient (SC) and
Davies–Bouldin index (DB).

## Extract features from the full-resolution image

This is an alternative starting from a fresh AnnData object. It needs the full
H&E image, image extras and ImageNet weights (downloaded by torchvision on first
use). It is substantially more expensive than loading the released embeddings.

```python
fresh = datasets.load_dlpfc(
    "151673", root=os.getenv("STAIG_DATA"), download=True, image_features=None,
)
img.extract_image_features(
    fresh, method="byol", epochs=1, batch_size=16,
    num_workers=0, n_jobs=1, cache_dir=".cache/151673-byol-default",
)
fresh_model = staig.run(fresh, n_clusters=7)
```

The loader records the source folder, so the image extractor can find
`spatial/tissue_full_image.tif`. For your own folder, pass `image=` explicitly
if lookup fails. See the [histology tutorial](tutorials/visium_with_histology.md)
for coordinate conventions and cache rules.

## AnnData keys

These are the keys the pipeline writes (see `docs/dev/DESIGN.md`). Entries are
created only when the corresponding step runs.

| Key | Written by | Contents |
| --- | --- | --- |
| `obsm["spatial"]` | User / dataset loader | Full-resolution `(x, y)` pixel coordinates for Visium; platform coordinates otherwise |
| `obs[batch_key]` | `pp.concat_slices` | Categorical slice identifier for integration |
| `obs["ground_truth"]` | `pp.load_labels` / loader | Optional labels; unlabelled spots dropped by default |
| `obsm["X_staig_input"]` | `pp.preprocess` / `pp.concat_slices` | Dense float32 matrix of model input features |
| `obsm["image_features"]` | `img.extract_image_features` / loader | Raw features, typically 2048 dimensions for ResNet-50/BYOL; custom dimensions allowed |
| `obsm["image_pcs"]` | `graph.image_pcs`, called by model preparation | Standardised/PCA image features; default 16 PCs |
| `obsp["spatial_connectivities"]` | `graph.spatial_neighbors` | Symmetric binary CSR graph, no self-loops; block-diagonal across slices |
| `obsp["staig_edge_prob"]` | `graph.edge_removal_probabilities` | Directed CSR edge-removal probabilities; row-normalised in image/gene modes when nonzero weights exist; constant in fixed mode |
| `obs["staig_pseudo_label"]` | `graph.pseudo_labels` | Image-PC k-means labels when pseudo-label debiasing is active |
| `obsm["X_staig"]` | `STAIG.embed`, also called by `fit` | Float32 embedding, 64 dimensions by default |
| `obs["staig_domain"]` | `cluster.cluster`, also called by `run` | Categorical domain labels, refined when requested |
| `uns["staig"]` | Pipeline stages / `metrics.evaluate` | Source, preprocessing, graph, image metadata, config, resolved negative count, training history/timings, optional consensus metadata and evaluated metrics |

The image extractor stores raw features; PCA occurs during model preparation
unless you call `graph.image_pcs` yourself. This revision does not automatically
record dependency versions in `uns["staig"]`. Most preprocessing/graph functions
mutate in place and return `None` unless `copy=True`; `run` returns a model, and
`embed` returns an array. There is no `staig.tl` namespace in this revision.
