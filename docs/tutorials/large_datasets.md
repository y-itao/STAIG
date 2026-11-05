# Large datasets and memory

STAIG trains on the full graph. Sparse spatial edges and chunked contrastive
loss reduce peak memory. Exact loss compares every anchor with all spots and
retains quadratic total work; sampled negatives approximate its denominator.
Neither mode is mini-batch GNN training.

## Estimate the working set

The following standalone calculation estimates individual arrays, not peak
process memory:

```python
n_spots, n_genes, chunk = 20_000, 3_000, 256
gib = 1024 ** 3
print("float32 expression input, GiB:", n_spots * n_genes * 4 / gib)
print("one loss block, GiB:", chunk * n_spots * 4 / gib)
print("one uint8 patch cache, GiB:", n_spots * 512 * 512 * 3 / gib)
```

Several loss blocks, masks, gradients and model states coexist. Preprocessing
materialises a dense `N × F` feature matrix in host RAM even when `adata.X` is
sparse. Float64 roughly doubles array storage. Lower `loss_chunk_size` reduces
temporary device memory at the cost of more chunk/recomputation overhead;
reduce `n_top_genes` before training if expression storage is the bottleneck.

For an executable full-data example with conservative CPU settings, see
[Stereo-seq](no_histology.md). A GPU run can use the same API with `device="cuda"`;
measure free GPU memory before choosing a chunk size. CPU execution avoids GPU
pressure but still needs host RAM and can take much longer.

## Sampled negatives with `n_negatives`

`n_negatives="auto"` is the current default: training uses 4,096 sampled nodes
on CPU when there are more than 8,000 spots, and exact negatives otherwise
(including CUDA and MPS). These thresholds are implementation settings, not
measured hardware limits. Use `n_negatives=None` or `0` for exact loss, or an
integer `K` smaller than the number of spots to request sampled negatives.
A value at least as large as the dataset also selects exact loss.

The sampled path keeps graph positives exact and estimates the denominator from
a random subset shared across anchors, rescaled by `N / K`. Its main similarity
blocks have shape `loss_chunk_size × K` rather than `loss_chunk_size × N`.
This is an approximation that can change the fitted result; do not label it a
numerically identical paper reproduction. Record the resolved value in
`adata.uns["staig"]["n_negatives"]` (`"all"` for exact) with the configuration.
The [Stereo-seq example](no_histology.md) explicitly sets `n_negatives=4096`.

## Image caches and other memory costs

`img.extract_image_features(..., cache_dir=...)` writes `patches.npy` and, when
filtering, `patches_filtered.npy` as uint8 memory maps. Each holds
`N × patch_size × patch_size × 3` bytes plus a small header; budget for both.
At 20,000 spots and 512 pixels, that is about 14.65 GiB per cache, a calculated
storage estimate. The full-resolution image itself is loaded into RAM in this
revision. Use `n_jobs=1` to limit filter-worker copies and `num_workers=0` to
limit image-loader workers. BYOL `batch_size` controls training; internal
embedding inference still uses batches of 128 in this revision.

Filtered caches are reused based only on row count. Use a separate cache for
every image, ordered barcode set and crop/filter configuration. Changing
`patch_size` also requires a matching frequency `band` centred on the new FFT
grid; leaving `(245, 275)` on a smaller patch is incorrect. Reusing released
features skips patch storage entirely.

Clustering and scoring can be more expensive than graph storage. Ward
initialisation in default GMM clustering can require quadratic memory; use
`cluster_method="kmeans"` when appropriate. Full silhouette evaluation also
performs pairwise-distance work. Consensus builds a dense co-association matrix
up to its `max_dense` threshold; lower that threshold to use the membership
representation path on large inputs. None of these choices is an automatic
guarantee of fitting a given CPU or GPU.
