# Using STAIG from an AI agent

STAIG learns spatial domains from expression, spatial neighbours and optional
histology, and integrates several slices without coordinate alignment. Install
from a checkout with `python -m pip install -e ".[image,cluster]"`; see
[installation](installation.md), the [API](api.md) and the
[AnnData keys contract](quickstart.md#anndata-keys). The public interfaces are
`staig.run`, `staig.pp`, `staig.img`, `staig.cluster` and `staig.metrics`.

## Decide whether STAIG fits

Input is an AnnData object with expression in `X` and finite coordinates in
`obsm["spatial"]`, or a Space Ranger Visium folder. Default preprocessing expects
raw counts. A full-resolution H&E image is optional; its pixel coordinates must
match the spots. Aligned precomputed image features can replace extraction.
For integration, each spot needs a slice identity in `obs[batch_key]`.

Outputs are categorical `obs["staig_domain"]`, embedding `obsm["X_staig"]`, sparse
graphs in `obsp`, and metadata in `uns["staig"]`. `staig.run` changes the input
AnnData **in place** and returns a fitted model. Save the AnnData to preserve
domains and embeddings; `model.save()` alone does not save those annotations.

Use STAIG for spatial domain exploration and comparing shared structure across
slices. It does not assign validated cell-type names, perform geometric image
registration, infer causal mechanisms, or establish clinical diagnoses. Without
spatial coordinates it is not the appropriate general scRNA-seq clustering tool.
Platform support means suitable expression/coordinate inputs can be used, not
that every bin size, panel, imaging modality or tissue has been validated.

## Platform outputs without an h5ad

`staig.read(path)` / `staig.run(path)` and `staig run path` accept Space Ranger (Visium, Visium HD),
Xenium, MERSCOPE, CosMx, SAW (Stereo-seq), Slide-seq / Curio, seqFISH, generic counts + coordinates
tables, transcript lists and `.h5ad`; `staig inspect path` prints the detected platform and QC
summary. Reader options: `read_kwargs={"bin_size": 16}` (Visium HD µm / Stereo-seq DNBs),
`{"level": "cells"}` (Visium HD segmented cells), `{"min_counts": 0}`, `{"coordinates": "pos.csv"}`
(generic tables). Several paths = integration with `batch_key="slice"`. Details and the validated
sample files: [Input formats](input_formats.md).

## Four runnable recipes

Save each block as its own script after installing STAIG. Paths passed on the
command line are placeholders for your files. These examples use raw counts and
a known `n_clusters=7`; replace that value using biological context. Use fresh
AnnData: `run` reuses existing `X_staig_input`, graph and image-PC keys.

All four scripts print a JSON summary and write a new `.h5ad`. The longer
argparse versions in [`examples/`](https://github.com/y-itao/STAIG/tree/v2/examples)
accept `--input --n-clusters --output --preset --n-seeds --epochs`. They default
to CPU, reject overwriting an output, and offer `--log-normalized` for expression
already normalised and log-transformed. Their final stdout line is JSON; warnings
and progress can appear on stderr. Small `--epochs` values are smoke checks,
not paper-quality training.

### Visium plus H&E

Save as `visium_recipe.py`; run
`python visium_recipe.py spaceranger_out full_resolution_image.tif result.h5ad`.
This explicitly extracts BYOL features and may download ImageNet weights once.
The alternative `examples/run_visium.py` also accepts aligned released
`embeddings.npy`, avoiding fresh extraction.

```python
import json
import sys
import time
from pathlib import Path

import staig

started = time.perf_counter()
folder, image, output = sys.argv[1:]
if Path(output).exists():
    raise FileExistsError(output)
adata = staig.datasets.load_visium_dir(folder, image_features=None)
staig.img.extract_image_features(
    adata, image=image, device="cpu", num_workers=0, n_jobs=1,
)
staig.run(adata, n_clusters=7, preset="visium", image=None, device="cpu")
scores = staig.metrics.evaluate(adata)
adata.write_h5ad(output)
print(json.dumps({"output_path": output, "metrics": scores,
                  "keys_written": ["obs/staig_domain", "obsm/X_staig", "uns/staig"],
                  "time_seconds": time.perf_counter() - started}, allow_nan=False))
```

The Visium loader drops unlabelled spots when `truth.txt` exists, matching the
released embeddings. For your own data without that filtering, use `labels=None`
or `drop_na=False` and generate features for that exact barcode order.

### No image

Save as `no_image_recipe.py`; run `python no_image_recipe.py counts.h5ad result.h5ad`.
For already log-normalised data, replace `preprocess=True` with
`preprocess={"flavor": "seurat", "normalize": False, "log1p": False}`.

```python
import json
import sys
import time
from pathlib import Path

import scanpy as sc
import staig

started = time.perf_counter()
source, output = sys.argv[1:]
if Path(output).exists():
    raise FileExistsError(output)
adata = sc.read_h5ad(source)
for key in ("image_features", "image_pcs"):
    adata.obsm.pop(key, None)
staig.run(adata, n_clusters=7, preset="auto", image=None, device="cpu", preprocess=True)
scores = staig.metrics.evaluate(adata)
adata.write_h5ad(output)
print(json.dumps({"output_path": output, "metrics": scores,
                  "keys_written": ["obs/staig_domain", "obsm/X_staig", "uns/staig"],
                  "time_seconds": time.perf_counter() - started}, allow_nan=False))
```

`image=None` skips extraction; existing image features can still affect training.
That is why the no-image recipes remove both image feature keys on fresh input.

### Several slices

Save as `integration_recipe.py`; run
`python integration_recipe.py slice1.h5ad slice2.h5ad integrated.h5ad`.
All slices below contain raw counts. For log-normalised slices, pass
`flavor="seurat", normalize=False, log1p=False` to `concat_slices`.

```python
import json
import sys
import time
from pathlib import Path

import scanpy as sc
import staig

started = time.perf_counter()
*inputs, output = sys.argv[1:]
if Path(output).exists():
    raise FileExistsError(output)
slices = [sc.read_h5ad(path) for path in inputs]
adata = staig.pp.concat_slices(
    slices, batch_key="slice", batch_categories=[f"slice_{i}" for i in range(len(slices))],
)
staig.run(adata, n_clusters=7, preset="integration", batch_key="slice",
          image=None, preprocess=False, refine=False, device="cpu")
scores = staig.metrics.evaluate(adata, batch_key="slice")
adata.write_h5ad(output)
print(json.dumps({"output_path": output, "metrics": scores,
                  "keys_written": ["obs/slice", "obs/staig_domain", "obsm/X_staig", "uns/staig"],
                  "time_seconds": time.perf_counter() - started}, allow_nan=False))
```

`concat_slices` preprocesses each slice and uses the intersection of its HVGs.
The spatial graph stays block-diagonal; coordinates need not be aligned.
Images are not automatically extracted in batch mode. Extract per slice first
if wanted, with aligned `image_features` present on every input.

### Large data on CPU

Save as `large_recipe.py`; run `python large_recipe.py counts.h5ad result.h5ad`.
This uses approximate negative sampling, k-means and at most 2,000 spots for
evaluation. It avoids full-data silhouette selection by specifying the cluster
count. The same normalisation caveat as the no-image recipe applies.

```python
import json
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import staig

started = time.perf_counter()
source, output = sys.argv[1:]
if Path(output).exists():
    raise FileExistsError(output)
adata = sc.read_h5ad(source)
for key in ("image_features", "image_pcs"):
    adata.obsm.pop(key, None)
staig.run(adata, n_clusters=7, preset="auto", image=None, device="cpu",
          n_seeds=1, loss_chunk_size=256, n_negatives=1024,
          cluster_method="kmeans", refine=False)
idx = np.sort(np.random.default_rng(0).choice(adata.n_obs, min(2000, adata.n_obs), replace=False))
sample = adata[idx].copy()
scores = staig.metrics.evaluate(sample)
adata.uns["staig"]["metrics"] = scores
adata.uns["staig"]["metric_n_obs"] = sample.n_obs
adata.write_h5ad(output)
print(json.dumps({"output_path": output, "metrics": scores, "metric_n_obs": sample.n_obs,
                  "keys_written": ["obs/staig_domain", "obsm/X_staig", "uns/staig"],
                  "time_seconds": time.perf_counter() - started}, allow_nan=False))
```

The model is still full-graph, with dense N-by-HVG input. Sampling negatives
changes the objective; it does not make every preprocessing/clustering step
constant-memory. See [large datasets](tutorials/large_datasets.md).

## Choosing n_clusters, preset and n_seeds

- **`n_clusters`:** use anatomical knowledge or an independent reference when
  available; seven is a DLPFC example, not a universal default. `None` searches
  3–15 clusters by silhouette in this revision. Inspect marker expression and
  spatial maps, and report the candidates tried. Large data should specify a
  count and compare candidates on a representative sample to limit cost.
- **`preset`:** `auto` chooses `integration` for multiple batches; otherwise
  Visium metadata/image features select `visium` or `visium_no_image`, <=1,500
  genes select `merfish`, and other inputs select `stereo_seq`. This is a
  heuristic. Explicit options include `slide_seq`, `starmap`, `merfish`,
  `stereo_seq`, `visium`, `visium_no_image` and `integration`. List all settings
  with `staig presets --show-default` or `staig.PRESETS`; resolve one with
  `staig.preset_config("visium").to_dict()`. `paper_*` presets replicate the per-dataset settings of
  the paper (float64, seeds, τ, epochs).
- **`n_seeds`:** the default `"auto"` means 3 (consensus) on a GPU and 1 on CPU; use 1 for a
  smoke run and 3–5 for consensus when resources permit. Training cost grows roughly with seed count,
  plus clustering/consensus. Up to 20,000 spots consensus materialises a dense
  N-by-N matrix temporarily; larger inputs use one-hot cluster memberships and
  k-means. Start with 1 for memory-constrained jobs. `X_staig` is the representative
  seed's embedding; the returned model/config/history/timings describe the last
  fitted seed, not all seeds or the total elapsed time. Record wall time yourself.

## Run times

Training time grows with the number of spots. A Visium section (3–5 k spots, 300 epochs) trains in
seconds on a GPU and in minutes on a laptop CPU; a 20 k-spot section takes a few minutes on a GPU;
above 65 536 observations on a GPU (8 000 on a CPU) the loss samples its negatives automatically and
the time grows linearly with the data. Histology feature extraction adds about a minute per Visium
section on a GPU (patch extraction and one BYOL epoch). Multi-seed consensus multiplies the training
time by `n_seeds`. Measure on the target machine before scheduling large jobs and record spot/gene
counts, preset, overrides, seeds, device, thread count and versions with the result.

## Validate the result

Run `staig.metrics.evaluate(adata)` after training: ARI/NMI are included only when
`obs["ground_truth"]` exists; `SC` (silhouette) and `DB` measure embedding separation
when more than one domain exists. For integration, supply `batch_key="slice"` to
add BatchKL and iLISI. These require the corresponding batch column and multiple
batches. Metric computation can be costly; the large-data recipe reports the
evaluation sample size explicitly.

Higher silhouette is better separation, not proof of anatomically correct
domains. Batch mixing can improve by erasing real biological differences, so
inspect tissue structure and marker expression per slice as well. Ensure
embeddings and losses are finite, labels cover all spots, and graphs have no
cross-slice edges. Compare independent seeds, unsmoothed and refined maps, and
any available held-out anatomical labels. Do not tune on the labels used to
claim final performance without disclosing that choice.

For marker review on a saved result whose `X` contains log-normalised expression:

```python
import scanpy as sc

adata = sc.read_h5ad("result.h5ad")
sc.tl.rank_genes_groups(adata, groupby="staig_domain", method="wilcoxon", use_raw=False)
sc.pl.rank_genes_groups(adata, n_genes=10)
```

Inspect known tissue markers and spatial maps too. Markers selected from the same
expression data that produced the domains are exploratory, not independent
validation. Keep raw counts separately when downstream analyses require them.

## Common errors and fixes

| Symptom | Action |
| --- | --- |
| Missing `obsm["spatial"]` | Load the spatial coordinate table and align rows by barcode. For Visium use full-resolution `(x, y)` pixels; do not substitute UMAP/PCA coordinates. |
| Non-integer counts warning / `strict_counts` | Check preprocessing history. `strict_counts=True` warns and switches from `seurat_v3` to `seurat`; it does not reject the input or prevent double normalisation. For already log-normalised expression use `preprocess={"flavor": "seurat", "normalize": False, "log1p": False}`. `strict_counts=False` forces the v1-style HVG calculation on non-integers; use only for deliberate historical compatibility. If raw counts are in a layer, pass `preprocess={"layer": "counts"}`. |
| No full-resolution image | Provide `image=".../full_resolution.tif"`, use aligned released/custom features, or run the no-image path. The low/hires images in `uns["spatial"]` are not automatic replacements for the full-resolution image. `image="auto"` skips extraction when none is discoverable. |
| CUDA out of memory | First lower `loss_chunk_size` (e.g. 256); the exact loss still uses chunk-by-N similarity work. Set `n_negatives=1024` for an approximate sampled objective, or use `device="cpu"`. Image extraction has its own `batch_size` and device controls via `image_kwargs`. `n_negatives="auto"` uses 4,096 sampled negatives only on CPU with >8,000 spots; otherwise it is exact. None, zero, or K>=N means exact. |
| Feature rows do not match spots | Generate/reindex features after barcode filtering; a matching row count alone does not prove alignment. |
| Stale results after parameter changes | `STAIG.prepare()` / `run()` recompute the graph, image PCs, edge probabilities and pseudo-labels whenever their parameters or the underlying coordinates, features or batch membership change (content fingerprints in `uns["staig"]`). `obsm["X_staig_input"]` is the one cached input: re-run `pp.preprocess` (or reload) when you change preprocessing, and avoid double preprocessing. |
| `.h5ad` save after automatic cluster selection | Works as is: `uns["staig"]["n_clusters_selection"]` uses string keys. The CLI refuses to overwrite an existing `--output` unless `--overwrite` is given. |

For tool-driven execution and the three tool schemas, see [MCP integration](mcp.md).
For the complete text bundle and discovery index, see [llms-full.txt](llms-full.txt)
and [llms.txt](llms.txt). The index follows the [llms.txt convention](https://llmstxt.org/).
