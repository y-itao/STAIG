# STAIG v2 examples

Run the notebooks in a Python environment with STAIG and its `image,dev` extras.
Set `STAIG_DATA` to the unpacked `Dataset` root (the directory containing
`DLPFC/`, `other/` and `v10x/`). Without it, `download=True` fetches missing data
under `./Dataset` relative to the notebook's working directory.

| Notebook | Real input | Training in the saved execution |
| --- | --- | --- |
| `01_visium_dlpfc_with_histology.ipynb` | All labelled DLPFC 151673 spots; released BYOL features | CPU, 30 epochs, GMM |
| `02_no_histology_stereo_seq.ipynb` | Full normalised Stereo-seq olfactory bulb | CPU, 5 epochs, 4096 sampled negatives, k-means |
| `03_integration_dlpfc_two_slices.ipynb` | All labelled DLPFC 151675 + 151676 spots; released BYOL features | CPU, 10 epochs, GMM |
| `04_custom_image_features.ipynb` | All labelled DLPFC 151673 spots; RGB statistics from real H&E patches | CPU, 10 epochs, GMM |

Every code cell runs in sequence on the full data (no subsampling, no GPU needed). The epoch
counts are kept small so that the notebooks execute in minutes; use the presets' full epoch counts
for an analysis. Notebook 01 reuses the released H&E embeddings; notebook 04 exercises image
cropping and the custom array interface. Neither retrains BYOL.

From the repository root, after activating the environment and setting
`STAIG_DATA`, execute notebooks sequentially:

```bash
for notebook in examples/0*.ipynb; do
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 --inplace "$notebook" || break
done
```

If STAIG is not installed, first run `python -m pip install -e ".[image,dev]"`.
The Python 3 kernel must use that environment. Re-execution replaces the saved
outputs. Keep outputs below 1 MB each; do not commit cached data or patch arrays.
The website renders these same notebooks and offers their source for download;
it does not retrain them during a build.
