# Changelog

## 2.0.0 (2026-11)

A complete rewrite of the code of Yang *et al.*, *Nat Commun* 16, 1067 (2025) as an installable,
cross-platform tool. The method is unchanged; the settings of the paper are available as `paper_*`
presets.

### Use it on your own data
- `staig.read(path)` / `staig.run(path)` / `staig run path` read the output folders of the platforms
  directly: Space Ranger (Visium, Visium HD bins or segmented cells), Xenium Onboard Analysis / Xenium
  Ranger, MERSCOPE, CosMx (AtoMx flat files), SAW (Stereo-seq `.gef`/`.gem`, square bins and cell bins),
  Slide-seq / Curio Seeker, seqFISH, generic counts + coordinates tables, transcript lists and `.h5ad`.
  Control probes, background and low-count cells are removed with recorded defaults, identifiers are
  kept as written, and each platform gets its preprocessing (3000 highly variable genes for
  sequencing-based data, every gene for targeted panels). Validated on the official sample bundles
  listed in `docs/input_formats.md`; `staig inspect` shows what would be read.
- One call: `staig.run(adata_or_path)` with automatic preset selection, automatic histology feature
  extraction for Visium folders, silhouette-based choice of the number of domains, multi-seed
  consensus, spatial smoothing and refinement. Several inputs are integrated without alignment.
  Scanpy-style modules `pp`, `graph`, `img`, `cluster`, `metrics`, `pl`, `datasets`; `STAIG` class
  with `fit/embed/save/load`.
- Front-ends: command line (`staig inspect`, `staig run`, `staig image`, `staig presets`, JSON
  output), local web interface (`staig gui`), MCP server for AI agents (`staig-mcp`), runnable example
  scripts and notebooks, selective Zenodo downloads.
- Runs on Linux, macOS and Windows, CPU or GPU, with pure-Python dependencies: pure-PyTorch GCN, own
  BYOL implementation, Python Gaussian-mixture clustering (R mclust available through the optional
  `[r]` extra), pure-Python BatchKL and LISI. No R, PyTorch Geometric, byol-pytorch or harmonypy
  required.
- Presets per platform (`visium`, `visium_no_image`, `visium_hd`, `stereo_seq`, `slide_seq`,
  `starmap`, `merfish`, `xenium`, `merscope`, `cosmx`, `seqfish`, `integration`) plus the settings of
  the paper as `paper_*`; YAML export in `staig/presets/`.

### Efficiency
- float32 training, sparse kNN graphs, chunked and checkpointed contrastive loss (memory
  O(chunk × N)); a Visium section trains in seconds on a GPU. Whole Visium HD capture areas and large
  Stereo-seq chips train in bounded memory through automatic negative sampling.
- Histology features in about a minute per section on a GPU: in-memory patches, thread-parallel
  filtering, batched BYOL.
- Large tables (Stereo-seq GEM, transcript lists) are aggregated in chunks.

### Robustness
- Deterministic seeding and graph construction, validated inputs with explicit errors, content
  fingerprints for every cached artefact (graph, image PCs, edge probabilities, pseudo-labels), atomic
  output writes, exact float64 checkpoints.
- `run(n_seeds=...)` returns a co-association consensus over seeds for stable domain labels.
- Preprocessing recovers automatically when the `seurat_v3` loess fit cannot be computed on very
  sparse matrices.

### Documentation and agent-friendliness
- MkDocs site with tutorials, API reference and FAQ; `llms.txt` / `llms-full.txt`, `AGENTS.md`,
  `codemeta.json`, `CITATION.cff`, issue templates, CI on Linux/Windows/macOS, an agent guide with
  complete recipes and a JSON-output MCP server.

### Compatibility notes
- `obsm`/`obs`/`obsp` key names changed (see the README table). The version 1 files are kept under
  `legacy/` and tag `v1-archive`.

## 1.x (2023–2025)

Code used for the paper: `git checkout v1-archive`.
