# Visium with histology

The [quick start](../quickstart.md) uses released H&E features. The
[executed notebook](../examples/01_visium_dlpfc_with_histology.ipynb) uses the full
labelled slice, short CPU training and these same released features. Fresh BYOL
extraction below is a separate, more expensive workflow.

## Load expression and spatial metadata

```python
import os
import staig
from staig import datasets, img, metrics, pl

adata = datasets.load_dlpfc(
    "151673", root=os.getenv("STAIG_DATA"), download=True, image_features=None,
)
assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
assert adata.obs_names.is_unique
```

For your own Space Ranger output, use `datasets.load_visium_dir` with the folder
containing `filtered_feature_bc_matrix.h5` and `spatial/`. Label and feature files
are optional. The extractor expects full-resolution `(x, y)` coordinates, where
`x` is column and `y` is row, in exactly `adata.obs_names` order. A display image
such as `tissue_hires_image.png` has a different scale and should not replace
the full-resolution image without transforming coordinates and patch dimensions.

## Crop, filter and learn image features

```python
img.extract_image_features(
    adata,
    method="byol",
    epochs=1,
    batch_size=16,
    num_workers=0,
    n_jobs=1,
    cache_dir=".cache/151673-byol-default",
)
assert adata.obsm["image_features"].shape == (adata.n_obs, 2048)
```

The default patch width is `3.5 * fiducial_diameter_fullres`, then resized to
512 pixels per side. The filter uses the frequency window `(245, 275)` on those
patches. These are the settings of the paper (see
[DESIGN.md](https://github.com/y-itao/STAIG/blob/main/docs/dev/DESIGN.md)). Supply
`image=` and `patch_px=` when automatic image lookup or fiducial metadata is
unavailable. Edge patches are zero-padded with a warning.

Use a new cache directory whenever the image, barcode order, patch size, crop
scale or filtering settings change. Existing filtered patches are checked only
for row count, so an equal-length but differently ordered cache is not safe.
Keep BYOL batch sizes at least two and avoid a final singleton batch because
its projector uses batch normalisation. The published slice used here has no
singleton batch with the shown batch size.

`method="resnet50"` skips BYOL training and uses ImageNet features; it is a
different method. In this revision its inference batch is fixed internally at
128; the public `batch_size` argument controls BYOL training. Reusing
`embeddings.npy` through `method=<array>` avoids both extraction and weight
downloads and is the appropriate starting point for paper comparisons.

## Learn spatial domains and inspect them

```python
model = staig.run(
    adata, n_clusters=7, preset="visium", epochs=30,
    device="cpu", seed=0, verbose=False,
)
pl.spatial(adata, color=["ground_truth", "staig_domain"])
print(metrics.evaluate(adata))
pl.loss_curve(adata)
```

Thirty epochs are a workflow demonstration. Compare the loss curve, histology
and labels, and record the configuration before interpreting domain boundaries.
Newly extracted features can change clusters even with the same graph-model
seed. The [reproduction tutorial](reproducing_the_paper.md) separates these
choices from the published experiment.
