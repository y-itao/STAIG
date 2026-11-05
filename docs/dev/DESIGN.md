# STAIG 2.0 — design notes for contributors

STAIG 2.0 implements the method of Yang *et al.*, *Nat Commun* 16, 1067 (2025) as an installable,
cross-platform package. These notes describe how the package is organised and the contracts that the
modules keep, so that contributions stay consistent.

## Goals

| goal | how |
|---|---|
| usable without data preparation | `staig.read` / `staig.run` on the platform's own output files; presets per platform; automatic histology for Visium |
| efficient | float32 training, sparse graphs, chunked and checkpointed loss (memory O(chunk × N)), optional sampled negatives, pure PyTorch (no R, no PyTorch Geometric) |
| robust | deterministic seeding, validated inputs, explicit errors, content fingerprints for cached artefacts, multi-seed consensus |
| portable | Linux, macOS, Windows; CPU or GPU; CI on all three |
| automatable | CLI with JSON output, GUI, MCP server, machine-readable documentation |

## Package layout

```
staig/
  __init__.py        public API: read, run, STAIG, STAIGConfig, PRESETS, modules
  _config.py         STAIGConfig dataclass, presets, YAML export (export_presets)
  _utils.py          seeding, device selection, logging, fingerprints, atomic h5ad writes
  io.py              platform readers (Visium, Visium HD, Xenium, MERSCOPE, CosMx, Stereo-seq GEF/GEM,
                     Slide-seq / Curio, seqFISH, tables, transcript lists, h5ad), detection, QC defaults
  pp.py              preprocess() (HVG -> normalise -> log1p -> scale), load_labels(), concat_slices()
  graph.py           spatial kNN graph (CSR), image PCs, pseudo-labels, edge-removal probabilities
  augment.py         edge removal (with optional resampling average) and feature masking
  model.py           sparse GCN layers, encoder, projection head, PCA initialisation
  loss.py            neighbour contrastive loss, chunked + checkpointed, exact or sampled negatives
  train.py           STAIG (prepare / fit / embed / save / load) and the one-call run() pipeline
  cluster.py         GMM (mclust-EEE-like), optional R mclust, leiden/louvain, refinement, smoothing,
                     silhouette-based choice of k, multi-seed consensus
  metrics.py         ARI/NMI, silhouette, Davies-Bouldin, BatchKL and LISI in pure Python
  pl.py              spatial and UMAP plots
  img/               patches.py (spot patches), filters.py (band-pass filter), byol.py (own BYOL),
                     features.py (extract_image_features)
  datasets.py        selective Zenodo downloads and loaders for the paper's data
  cli.py, gui.py, mcp_server.py   front-ends
```

## Data contract (AnnData keys)

| key | written by | content |
|---|---|---|
| `X` | reader | raw counts (CSR float32); `.h5ad` inputs keep what they store |
| `obsm["spatial"]` | reader | (x, y) in the platform's units |
| `obsm["X_staig_input"]` | `pp.preprocess` / `pp.concat_slices` | dense float32 model input (HVGs, scaled) |
| `obsm["image_features"]`, `obsm["image_pcs"]` | `img.extract_image_features`, `graph.image_pcs` | histology features and their PCs (per batch) |
| `obs["staig_pseudo_label"]` | `graph.pseudo_labels` | k-means labels on image PCs (debiased negatives) |
| `obsp["spatial_connectivities"]` | `graph.spatial_neighbors` | symmetric binary kNN graph, block-diagonal across batches |
| `obsp["staig_edge_prob"]` | `graph.edge_removal_probabilities` | per-edge removal probability (same sparsity as the graph) |
| `obsm["X_staig"]`, `obsm["X_staig_smooth"]` | `STAIG.embed`, `cluster.smooth_embedding` | embedding (projection head output) and its smoothed version |
| `obs["staig_domain"]` | `cluster.cluster` | domains |
| `uns["staig"]` | all | platform, source, QC, preprocessing, graph/edge/pseudo-label parameters with fingerprints, config, history, timings, cluster/consensus metadata, metrics |

Every artefact under `uns["staig"]` records the parameters and a content fingerprint of the data it
was derived from; `STAIG.prepare()` recomputes an artefact whenever these differ, so a reused AnnData
never carries stale state. `run()` clears the result-level entries of an earlier run before it starts.

## Method as implemented

1. **Preprocessing**: `seurat_v3` HVGs on counts (3000; every gene for targeted panels), total-count
   normalisation to 10⁴, `log1p`, scaling without centring (`max_value=10`). Sparse matrices in which
   the loess fit of `seurat_v3` cannot be computed fall back to detected genes and then to the
   log-based `seurat` flavour, recorded in `uns["staig"]["preprocess"]["hvg_note"]`.
2. **Histology** (optional): patches of 3.5 × fiducial diameter around every spot, resized to 512 px,
   Gaussian blur, band-pass filter in the frequency domain, blur; BYOL on ResNet-50 (ImageNet V1
   weights), one epoch, 2048-d features; standardised PCA (16 PCs single slice, per slice for
   integration); k-means pseudo-labels (80) for the debiased negatives.
3. **Graph**: kNN (k = 5) on the coordinates, ties broken by spot index, block-diagonal across slices.
   Edge-removal probabilities: distance in image-PC space normalised per row (`image`), cosine distance
   on gene PCs with a row softmax (`gene`), or a fixed rate (`fixed`).
4. **Training**: two augmented views per epoch (edge removal with those probabilities, random masking
   of gene columns at two rates), shared GCN (`Â = D^{-1/2}(A+I)D^{-1/2}`), PReLU, projection head.
   Neighbour contrastive loss with the positives (the spot itself across views and its spatial
   neighbours in both views) over the masked negatives; image pseudo-labels remove image-similar spots
   from the negatives, integration keeps negatives within a slice. The loss is evaluated in log space
   per chunk of anchors with gradient checkpointing; above a size threshold (65 536 observations on a
   GPU, 8 000 on a CPU) a shared random subset of negatives is used with an N/K rescaling.
   Adam, 300–400 epochs depending on the preset, PCA initialisation of the first GCN layer.
5. **Domains**: embedding on the un-augmented graph, one smoothing pass over the spatial graph,
   Gaussian-mixture clustering with tied covariance initialised from Ward clustering (k-means++ above
   15 000 points), majority-vote refinement over the 15 nearest spots, silhouette-based choice of k
   when it is not given, and a co-association consensus over seeds (`run(n_seeds=...)`).

## Numerical contracts (checked by the tests)

* `loss.neighbor_contrastive_loss` equals the dense reference implementation to 1e-10 for all mask
  modes with `variant="v1"`; the default `variant="full"` is exact in log space for any τ (tested at
  τ = 0.001 against the analytic value) and the sampled estimator equals the exact loss at K = N.
* `model.GCNLayer` matches `torch_geometric.nn.GCNConv` with identical weights (when PyG is installed).
* `graph.spatial_neighbors` equals the dense argsort construction on tie-free coordinates and is
  deterministic by index on ties.
* `img.filters.filter_patch` reproduces the PNG round-trip numerics of the original pipeline bit for bit.
* `metrics.lisi` equals `harmonypy.compute_lisi` to 1e-15 on test data; `metrics.batch_kl` follows the
  R implementation (k = min(5 B, 100), log2).
* CPU runs are deterministic for a given seed; GPU runs request deterministic kernels.

## Presets

`PRESETS` holds the defaults per platform (`default`/`visium`, `visium_no_image`, `visium_hd`,
`stereo_seq`, `slide_seq`, `starmap`, `merfish`, `xenium`, `merscope`, `cosmx`, `seqfish`,
`integration`) and the per-dataset settings of the paper as `paper_*` presets (float64, the seeds, τ,
epochs, pseudo-labels, image PCs and the STAIG 1.x loss convention). `staig/presets/*.yaml` are
generated from `PRESETS` by `export_presets()`; a test keeps them equal.

## Testing

`tests/` runs on CPU in a few minutes: loss and graph numerics against reference implementations,
model equivalence, readers on synthetic layouts of every platform (`tests/test_io.py`) and on the
vendor sample files when `STAIG_SAMPLES` is set, pipeline end to end on synthetic data, clustering and
metrics, CLI and GUI backends, dataset downloads against a local HTTP server. CI runs the suite on
Linux (Python 3.10–3.12), Windows and macOS.
