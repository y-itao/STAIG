# Working on STAIG

STAIG identifies spatial transcriptomic domains and integrates slices without
coordinate alignment. Read `docs/dev/DESIGN.md` for the design, but
verify signatures against the current source: the specification also contains
design targets and historical interfaces. User/task scope overrides this guide.

## Environment and checks

Use Python >=3.10 with the appropriate PyTorch build installed first. For CPU CI:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,image,cluster]"
pytest -q
ruff check staig tests
```

The task environment uses `/home/yang/anaconda3/envs/staig2/bin/python`. When work
must stay inside the repository, put temporary files/caches and any additional
virtual environment inside a task-specific local directory. Run tests on CPU
unless a task explicitly requests a bounded GPU experiment. Dataset tests use a
local HTTP server; the Windows/macOS CI jobs exclude `tests/test_datasets.py`.

## Module map

| Module | Responsibility |
| --- | --- |
| `staig/__init__.py` | Public exports, version; there is no `staig.tl` namespace |
| `staig/_config.py`, `_utils.py` | Config/presets, seeding, device and sparse utilities |
| `staig/io.py` | Readers for platform outputs (`staig.read`): Visium/HD, Xenium, MERSCOPE, CosMx, Stereo-seq GEF/GEM, Slide-seq, seqFISH, tables, transcripts; platform preprocessing defaults |
| `staig/pp.py` | Counts/HVG preprocessing, labels, concatenating slices |
| `staig/graph.py`, `augment.py` | Spatial CSR graph, edge probabilities, graph/feature augmentation |
| `staig/model.py`, `loss.py` | Torch GCN, projection head, chunked contrastive loss |
| `staig/train.py` | `STAIG.fit/embed/save/load`, one-call `run` pipeline |
| `staig/img/` | Spot-centred patches, filtering, BYOL/ResNet features |
| `staig/cluster.py`, `metrics.py`, `pl.py` | Clustering/refinement/consensus, evaluation, plotting |
| `staig/datasets.py`, `cli.py` | Selective Zenodo downloads/loaders and command line |
| `staig/mcp_server.py` | Optional FastMCP stdio tools; not imported by the base package |
| `staig/gui.py` | Optional Gradio interface |
| `tests/`, `benchmarks/`, `examples/` | CPU checks, scripts for the experiments of the paper, runnable user workflows |
| `legacy/`, `examples_v1/`, `configs/paper/` | Historical paper implementation and settings |

## AnnData contract

| Key | Meaning |
| --- | --- |
| `X` or an explicitly selected counts layer | Expression matrix; defaults expect raw counts; do not normalise twice |
| `obsm["spatial"]` | Required spot coordinates, rows aligned to `obs_names`; Visium uses full-resolution `(x, y)` pixels |
| `obs[batch_key]` | Slice identities, required for batch-aware integration |
| `obs["ground_truth"]` | Optional reference labels; loader may drop unlabelled spots |
| `obsm["X_staig_input"]` | Dense float32 N-by-HVG input, created by preprocessing/concatenation |
| `obsm["image_features"]` | Optional aligned raw histology features; extractor/loader writes these |
| `obsm["image_pcs"]` | Image PCs, computed during model preparation |
| `obsp["spatial_connectivities"]` | Symmetric sparse CSR graph; block-diagonal across slices |
| `obsp["staig_edge_prob"]` | Sparse directed edge-removal probabilities |
| `obs["staig_pseudo_label"]` | Image-derived pseudo-labels when debiasing is used |
| `obsm["X_staig"]` | Final embedding (64 columns by default) |
| `obs["staig_domain"]` | Categorical domains |
| `uns["staig"]` | Config, history/timings, optional consensus and evaluated metrics |

`staig.run` mutates AnnData and returns the fitted **model**, not AnnData. With
multiple seeds it returns the last model, while the stored embedding comes from
the representative seed. `metrics.evaluate` adds metrics explicitly. Preserve
barcode order when supplying image features. Start from fresh input when changing
preprocessing, graphs or image features; cached keys are reused by preparation.

## Generated files

Preset YAML mirrors in `staig/presets/` are generated from `staig._config.PRESETS` by
`python -c "from staig._config import export_presets; export_presets()"`. Update the
source presets and regenerate the mirrors together; `tests/test_robustness.py::
test_preset_yaml_mirrors_match_source` fails when they diverge.
The architectural `configs/presets/` directory is currently empty.
Keep all `paper_*` presets unchanged unless explicitly tasked with a correction.

`llms-full.txt` and `docs/llms-full.txt` are generated, verbatim source bundles:

```bash
python docs/build_llms_full.py
```

That command also copies root `llms.txt` into `docs/llms.txt`. Edit the root index
and source Markdown, then regenerate. Do not edit generated copies by hand.

## Experiment scripts and invariants

`benchmarks/run_dlpfc.py`, `run_integration.py`, `run_no_image.py` and `sweep_dlpfc.py` run the
experiments of the paper (`--preset paper_*`, `--paper-configs configs/paper/*.yaml`) and write
per-seed CSV tables; see their `--help`. For example, with an unpacked Dataset root:

```bash
python benchmarks/run_dlpfc.py --root "$STAIG_DATA" --slices 151673 \
  --preset paper_151673 --seeds 39788 --cluster gmm --tag paper_check
```

When comparing configurations, keep slices, preprocessing, images, cluster method, refinement,
hardware and seeds identical, and report several seeds.

- Never store dense N-by-N matrices in AnnData. Preserve sparse graph storage.
- Run the tests **before** changing `loss.py` to establish a baseline; after any
  change rerun loss/reference-gradient tests and the relevant pipeline tests.
- Keep exact and sampled-negative objectives distinguishable. Do not infer
  biological correctness from low loss, silhouettes or successful execution.
- Do not replace/remove datasets, caches, outputs or another worker's edits
  without task authorization. Re-read shared files immediately before editing.
- Keep `paper_*` presets and historical paper artifacts unchanged.
