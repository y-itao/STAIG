# Alignment-free integration of tissue slices

Use a separate AnnData object per slice. Coordinates remain in each slice's
native system; `batch_key` makes the spatial graph block-diagonal. Integration
learns a shared latent representation rather than registering tissue images.

```python
import os
import matplotlib.pyplot as plt
import staig
from staig import datasets, pp, metrics, pl

slice_ids = ["151675", "151676"]
slices = [datasets.load_dlpfc(s, root=os.getenv("STAIG_DATA"), download=True)
          for s in slice_ids]
merged = pp.concat_slices(
    slices, batch_key="slice", batch_categories=slice_ids, n_top_genes=5000,
)
model = staig.run(
    merged, n_clusters=7, preset="integration", batch_key="slice",
    preprocess=False, refine=False, epochs=10,
    device="cpu", loss_chunk_size=256, seed=0, verbose=False,
)
pl.spatial(merged, color="staig_domain", batch_key="slice")
plt.show()
print(metrics.evaluate(merged, batch_key="slice"))
```

`concat_slices` copies the inputs, selects HVGs per slice and uses their
intersection as `X_staig_input`. Its default normalisation is appropriate for
raw counts and its default `scale=False` follows the integration workflow. For
pre-normalised slices, pass `normalize=False, log1p=False, flavor="seurat"`.
Do not concatenate already preprocessed count slices and normalise again.

Raw image features are carried over when present in every input. Model
preparation computes image PCs per slice and uses image-based edge removal;
without images, the integration preset uses gene-based edge removal. For
image-free integration, load slices with `image_features=None`. The preset
disables pseudo-label debiasing and restricts negatives to the same slice.

Integration defaults to `refine=False`, matching the example. If refinement is
required, `run(..., refine=True, batch_key="slice")` now forwards the batch key
to its final refinement step. You can also call
`staig.cluster.refine_labels(merged, batch_key="slice")` explicitly. Neighbours
are selected separately within each slice. The plotting wrapper similarly draws
separate coordinate panels; concatenation does not preserve per-slice H&E
display metadata.

## Check mixing and biological structure together

`metrics.evaluate(..., batch_key="slice")` writes the returned metrics under
`merged.uns["staig"]["metrics"]`. BatchKL is a Python implementation: lower
values indicate neighbourhood batch proportions closer to the overall mixture.
iLISI (pure-Python port of the LISI algorithm) reports the median effective batch diversity: higher
values indicate more mixing. Neither metric proves that distinct biological
domains were preserved. Inspect ARI/NMI where annotations exist and spatial
maps of each slice alongside mixing scores.

The [executed notebook](../examples/03_integration_dlpfc_two_slices.ipynb)
also checks that no spatial edge crosses slice boundaries. Its ten-epoch CPU
run demonstrates the workflow of [paper Figure 5a](https://www.nature.com/articles/s41467-025-56276-0/figures/5);
use `preset="paper_integration_vertical"` with its full epoch count for the experiment itself.
