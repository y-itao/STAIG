# STAIG v1 reference code

These files preserve the implementation used by the original paper. They are
kept for reference only, are excluded from the v2 package, and are not importable
as supported STAIG v2 modules. Their historical imports and scripts require the
old environment, including `torch_geometric`, R/`rpy2`, `byol-pytorch` and other
v1 dependencies. Several scripts also assume the original data layout and local
paths; copying them into a v2 environment is not a migration procedure.

| File | Historical purpose |
| --- | --- |
| `staig.py` | v1 training wrapper, pseudo-labels, graph construction and embedding workflow. |
| `net.py` | PyG GCN encoders, single/multi-view models, contrastive losses and graph/feature dropout. |
| `adata_processing.py` | Single-slice and multi-slice Visium/h5ad loaders, expression preprocessing, labels and image feature preparation. |
| `utils.py` | R/mclust and graph clustering, resolution search and spatial label refinement. |
| `metrics.py` | v1 evaluation helpers, including ARI/NMI, silhouette, kBET and R-backed batch mixing metrics. |
| `batchKL.R` | R implementation of the BatchKL mixing metric. |
| `calLISI.R` | R helper for local inverse Simpson index (LISI) evaluation. |
| `example_train_with_image.py` | Single-slice training example using histology-derived image features. |
| `example_train_without_image.py` | Single-slice training example without image features. |
| `example_integration_horizontal.py` | Multi-slice horizontal integration example. |
| `example_integration_vertical.py` | Multi-slice vertical integration example. |
| `image_step1_clip.py` | Crops full-resolution histology into spot-centred image patches. |
| `image_step2_filter.py` | Applies the original grayscale, blur and Fourier-window patch filter. |
| `image_step3_byol.py` | Trains the external BYOL model and exports image embeddings. |
| `image_step4_show.py` | Clusters and visualises the exported image embeddings. |

The original tracked tree is available through the `v1-archive` tag:

```bash
git ls-tree -r --name-only v1-archive
git show v1-archive:staig/staig.py
git show v1-archive:staig/net.py
git show v1-archive:example/Fig-3b.ipynb
```

Use `git show v1-archive:<original-path>` for a file that existed at that tag.
The historical standalone image scripts were deleted earlier and can instead
be inspected with `git show e60d926:image_step1_clip.py` (and the corresponding
step 2–4 filenames). Reference notebooks are also retained in
`examples_v1/notebooks/` in the working v2 tree. The design of version 2 is
documented in [DESIGN.md](../docs/dev/DESIGN.md).
