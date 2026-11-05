# Bring your own image features

Any model can supply one feature vector per retained spot. The image API
accepts a NumPy array or a **string** ending in `.npy` through `method=`. It
does not invoke the external encoder or infer barcode alignment.

## Runnable example with released features

```python
import os
import numpy as np
import staig
from staig import datasets, img, metrics

adata = datasets.load_dlpfc("151673", root=os.getenv("STAIG_DATA"), download=True)
features = np.asarray(adata.obsm.pop("image_features"), dtype=np.float32)
assert features.ndim == 2 and features.shape[0] == adata.n_obs
assert np.isfinite(features).all()
img.extract_image_features(adata, method=features)
model = staig.run(adata, n_clusters=7, epochs=10, device="cpu", verbose=False)
print(metrics.evaluate(adata))
```

Replace `features` with the output of your chosen pathology model. Use its
documented preprocessing, patch resolution and licensing conditions. Store
barcodes alongside feature rows and reindex them to `adata.obs_names` after
filtering; matching the row count alone cannot detect shuffled rows. Feature
width is flexible and is not restricted to ResNet-50's 2048 dimensions.

Assigning `adata.obsm["image_features"] = features` directly is also supported.
The API route additionally records `uns["staig"]["image"]["method"]` as
`"precomputed"`. It stores float32 raw features; model preparation standardises
and projects them through PCA. The `[image]` extra is not required merely to
attach an existing array and train the graph model.

## Replacing features after a previous run

Prefer a freshly loaded AnnData. If you replace features on a trained object,
clear the derived image PCs, pseudo-labels and edge probabilities before
retraining; otherwise preparation may reuse them. The following continues the
example above, retaining the same input array only to demonstrate invalidation:

```python
adata.obsm["image_features"] = features.copy()
adata.obsm.pop("image_pcs", None)
adata.obs.drop(columns=["staig_pseudo_label"], errors="ignore", inplace=True)
adata.obsp.pop("staig_edge_prob", None)
model = staig.run(adata, n_clusters=7, epochs=10, device="cpu", verbose=False)
```

Changing coordinates or the spot set also requires rebuilding the spatial graph.
The [executed custom-feature notebook](../examples/04_custom_image_features.ipynb)
extracts simple RGB patch statistics from real H&E data and passes them as an
array. Those descriptors demonstrate the interface; they are not a pretrained
foundation model or the published BYOL representation.
