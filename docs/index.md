# STAIG

**Spatial domains and alignment-free slice integration for spatial transcriptomics, straight from the
files your platform writes.**

STAIG learns an embedding of every spot, bin or cell from its gene expression, its spatial
neighbourhood and, when the section has an H&E image, the histology around it, and clusters that
embedding into tissue domains. The same model embeds several sections into one space without aligning
their coordinates. Method: [Yang *et al.*, *Nature Communications* 16, 1067 (2025)](https://doi.org/10.1038/s41467-025-56276-0).

![STAIG overview (Figure 1 of the paper): histology patches are filtered and encoded by a self-supervised network; their similarities steer the augmentation of the spatial graph; a graph contrastive model embeds the spots for domain identification and alignment-free integration.](assets/staig_overview.png)

## Start here

```bash
pip install "staig[image,io] @ git+https://github.com/y-itao/STAIG"
staig inspect path/to/outs                                   # which platform, how many cells, QC
staig run path/to/outs --n-clusters 7 --output domains.h5ad  # the whole analysis
```

```python
import staig
model = staig.run("path/to/spaceranger_outs", n_clusters=7)
adata = model.adata                     # obs["staig_domain"], obsm["X_staig"], uns["staig"]
staig.pl.spatial(adata)
```

`path/to/outs` can be a Space Ranger folder (Visium, Visium HD), a Xenium, MERSCOPE, CosMx, SAW
(Stereo-seq), Slide-seq / Curio or seqFISH output, a counts + coordinates table pair or an `.h5ad`:
see [Input formats](input_formats.md). Several inputs are integrated in one call.

* [Installation](installation.md)
* [Input formats](input_formats.md): every supported layout, defaults and validated vendor samples
* [Quick start](quickstart.md) with the paper's DLPFC slice
* Tutorials: [Visium with histology](tutorials/visium_with_histology.md),
  [without histology](tutorials/no_histology.md), [integrating slices](tutorials/integration.md),
  [large datasets](tutorials/large_datasets.md), [custom image features](tutorials/custom_image_features.md),
  [reproducing the paper](tutorials/reproducing_the_paper.md)
* [API reference](api.md), [FAQ](faq.md)
* For AI agents: [agent guide](agent_guide.md), [MCP server](mcp.md), `llms.txt` / `llms-full.txt`
