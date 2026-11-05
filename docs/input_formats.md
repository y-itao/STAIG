# Input formats: start from what your platform writes

Most users do not have an `.h5ad`; they have the output folder of the instrument software. `staig.read`
recognises those folders and files, builds the AnnData STAIG needs (raw counts, coordinates, metadata,
control features removed, low-count cells removed) and records what it did in `uns["staig"]`.
`staig.run` accepts the same paths directly, so the whole analysis is

```python
import staig
model = staig.run("path/to/spaceranger_outs")        # any supported layout, see the table below
adata = model.adata                                   # obs["staig_domain"], obsm["X_staig"], uns["staig"]
```

or, on the command line,

```bash
staig inspect path/to/outs                            # what would be read: platform, cells, genes, QC
staig run path/to/outs --output domains.h5ad          # read -> preprocess -> train -> cluster
staig run slide_a/ slide_b/ --output integrated.h5ad  # several inputs = alignment-free integration
```

The GUI (`staig gui`) has the same platform selector. Nothing has to be converted by hand.

## Supported layouts

| platform | what `staig.read` looks for | coordinates | removed by default | preset |
|---|---|---|---|---|
| 10x Visium (Space Ranger 1.x–4.x) | the run folder or its `outs/`, or any folder with `*filtered_feature_bc_matrix.h5` (or the MTX folder) and `spatial/` (`tissue_positions.csv` or `tissue_positions_list.csv`, `scalefactors_json.json`, images); optional `truth.txt`, `embeddings.npy`; `load_images=False` keeps coordinates and scale factors | full-resolution pixels | non-gene features (e.g. Antibody Capture; counted in `obs["control_counts"]`), spots without a position | `visium` / `visium_no_image` |
| 10x Visium HD (Space Ranger 3.x/4.x) | the run folder or its `outs/` with `binned_outputs/square_XXXum/` (`bin_size=8` µm by default), one bin folder or its `.h5`; `level="cells"` reads `segmented_outputs/filtered_feature_cell_matrix.h5` with cell positions from `barcode_mappings.parquet` + 2 µm bins (mean bin position) or, failing that, the area centroid of the segmentation polygon | full-resolution pixels (`microns_per_pixel` recorded) | bins with `in_tissue == 0`, empty bins | `visium_hd` (= `stereo_seq`) |
| 10x Xenium (XOA 1.x–4.x, Xenium Ranger) | `outs/` with `cell_feature_matrix.h5` (or the MTX folder), `cells.parquet` / `cells.csv.gz`, `experiment.xenium` | microns (`x_centroid`, `y_centroid`) | Negative Control Probe/Codeword, Unassigned/Deprecated Codeword, Genomic Control features; cells < 10 transcripts | `xenium` (= `merfish`) |
| Vizgen MERSCOPE | `*cell_by_gene.csv` + `*cell_metadata.csv` (`EntityID`/`cell` ids; `center_x`, `center_y`, `fov`, `volume`) | microns | `Blank-*` barcodes; cells < 20 counts | `merscope` (= `merfish`) |
| Bruker/NanoString CosMx SMI (AtoMx flat files) | `*exprMat_file.csv[.gz]` + `*metadata_file.csv[.gz]` (one slide per folder; cells keyed `fov_cell_ID` or the `cell` column) | whole-slide pixels (`CenterX/Y_global_px`; `coordinates="local"` for per-FOV pixels) | `NegPrb*`, `Negative*`, `SystemControl*`, `FalseCode*` probes; `cell_ID == 0` (background); cells < 20 counts | `cosmx` (= `merfish`) |
| BGI Stereo-seq (SAW) | `.gef` (square-bin or cell-bin HDF5 after the SAW file-format manual, `offset` tables honoured), `.gem` / `.gem.gz` / `.tsv` with `geneID x y MIDCount [ExonCount] [CellID]` (`MIDCounts`, `UMICount`, `label` accepted; `#` header lines kept in `uns["staig"]["gem_header"]`). Bins are formed on the absolute grid `floor(coordinate / bin_size)` (stored coordinate = bin centre), read in 4-million-row chunks | DNB indices (1 DNB = 0.5 µm) | background labels `0`/`-1`, bins/cells < 20 counts | `stereo_seq` |
| Slide-seq / Slide-seqV2 / Curio Seeker | `*.digital_expression.txt[.gz]` (genes × beads) + `*bead_locations.csv` / `BeadLocationsForR.csv` (`barcodes, xcoord, ycoord`), or `*MoleculesPerMatchedBead.mtx` + `barcodes.tsv` + `genes.tsv` + `*MatchedBeadLocation.csv` | bead coordinates as written | beads < 10 counts | `slide_seq` |
| Spatial Genomics seqFISH | `*CellxGene.csv` + `*CellCoordinates.csv` (`label, area, center_x, center_y`) | microns | cells < 10 counts | `seqfish` (= `merfish`) |
| generic counts + coordinates | `staig.io.read_table(counts, coordinates)`: CSV/TSV (cells × genes or genes × cells, orientation detected), MTX + barcodes/genes, 10x `.h5`; coordinates CSV/TSV with an id column and x/y columns (`x/y`, `X/Y`, `center_x/center_y`, `xcoord/ycoord`, `CenterX_global_px/...`); x/y columns inside the counts table also work | as written | `Blank*`, `NegPrb*` ... columns | by size (`auto`) |
| transcript lists | `staig.io.read_transcripts(path)`: Xenium `transcripts.parquet`, CosMx `tx_file.csv` (cells keyed `fov_cell_ID`), MERSCOPE `detected_transcripts.csv` with cell ids, STARmap/seqFISH molecule tables — aggregated to cells (position = count-weighted mean of the cell's transcripts; whole-slide coordinates preferred, per-FOV pixels rejected for several FOVs) | as written | unassigned transcripts, control genes (counted per cell); cells < 10 transcripts | by size |
| AnnData `.h5ad` | `obsm["spatial"]`, or x/y columns in `obs`; `X` is kept as stored (it may already be normalised; `staig run` detects that) | as stored | observations without finite coordinates | `auto` |

Every platform reader returns raw counts in `X` (CSR float32, negative values are rejected),
`obsm["spatial"]` as finite `(x, y)`, the platform's per-cell metadata in `obs`, `var["feature_types"]`
when the platform provides it, observation ids as the platform wrote them (strings, leading zeros kept),
and in `uns["staig"]`: `platform`, `source`, `coordinate_unit`, `qc` (`min_counts`, `n_removed` and a
`removed` breakdown: low counts, missing coordinates/metadata, off-tissue bins, background, unassigned
transcripts, unlabelled spots), `dropped_features` with `obs["control_counts"]` whenever control
features were removed, and platform details (`bin_size_um`, `microns_per_pixel`, the `experiment.xenium`
summary, GEM/GEF headers ...). `min_counts=0` disables only the count filter; `drop_controls=False`
(`drop_blanks=False` for MERSCOPE) keeps control features where a reader has them (Visium, Visium HD,
Xenium, CosMx, seqFISH, tables, transcripts). An input that ends up with no observation raises an error
naming the reasons instead of returning an empty object.

Coordinates are left in the platform's units: STAIG only uses nearest-neighbour relations, so the unit
never changes the result. Visium and Visium HD data also get the scanpy `uns["spatial"]` entry (scale
factors, hires/lowres images), so `staig.pl.spatial` and histology feature extraction work as for the
paper data.

## What `staig.run` does with each platform

* **Preprocessing** (`preprocess=True`): sequencing-based data (Visium, Visium HD, Stereo-seq,
  Slide-seq) get 3000 highly variable genes (`seurat_v3` on counts), normalisation to 10⁴, `log1p` and
  scaling — the pipeline of the paper. Targeted panels (Xenium, MERSCOPE, CosMx, seqFISH, MERFISH,
  STARmap) use every gene. Matrices in which fewer genes vary than are requested skip the loess fit
  (all varying genes are used); when the `seurat_v3` loess fit fails on a very sparse matrix (common
  for Visium HD subsets), the fit is repeated on detected genes and, failing that, the log-based
  `seurat` flavour is used; `uns["staig"]["preprocess"]["hvg_note"]` records this. The same recovery is
  used when slices are concatenated. `preprocess=False` keeps an existing `obsm["X_staig_input"]`
  untouched, also for lists of inputs.
* **Preset** (`preset="auto"`): chosen from `uns["staig"]["platform"]` (table above) — image-guided
  edge removal and debiased negatives for Visium with histology features, fixed edge removal and no
  pseudo-labels otherwise; two GCN layers for high-resolution sequencing platforms.
* **Scale**: the exact contrastive denominator is used up to 65 536 observations on a GPU (8 000 on a
  CPU); above that `n_negatives="auto"` samples 8 192 (4 096 on CPU) negatives per step, so a whole
  Visium HD capture area at 8 µm (hundreds of thousands of bins) trains in bounded memory. Use 16 µm
  bins for a first look.
* **Histology**: for Visium folders (and `.h5ad` files) the full-resolution image found next to the
  data is used automatically (`image="auto"`); for every other platform `image="auto"` does nothing
  (the patch size of the paper is tied to the 55 µm spot). Pass `image=` and
  `image_kwargs={"patch_px": ...}` explicitly if you want to try histology on Visium HD. With several
  inputs an explicit `image=` is rejected: extract features per slice first and pass the AnnData objects.
* **Several inputs**: `staig.run([path_a, path_b])`, `staig run a/ b/` or `a;b` in the GUI read every
  input, name the slices after the folders (dict keys, or the run folder's name for `.../outs`), and
  concatenate them: every gene when all inputs are targeted panels, otherwise the genes among the
  5 000 most variable of every slice (the paper's integration setting). The integration preset is used
  with `batch_key="slice"` (or the key you pass). `uns["staig"]` records `platforms`, `sources` and
  `slices`; the merged `platform` is `"mixed"` when they differ.

## Validation on vendor sample files

The readers were developed against the official sample bundles below and the raw files shipped with the
paper's Zenodo dataset; `tests/test_io.py` replays the same layouts synthetically on every CI run and
`tests/test_io_samples.py` checks the real files when `STAIG_SAMPLES` points to a folder holding them.

| sample | software / source | read as | `staig run` (1 seed, RTX 5090) |
|---|---|---|---|
| `Xenium_Prime_Mouse_Ileum_tiny` (10x) | Xenium Onboard Analysis 3.0, 23 cells, 5 006 genes | `xenium`: 11 cells after QC, 8 774 control features dropped | too few cells to cluster |
| `Visium_HD_Tiny_3prime_Dataset` (10x) | Space Ranger 4.0.1 | `visium_hd` 8 µm: 10 605 bins; 16 µm: 3 321; `level="cells"`: 612 segmented cells (mapping-derived positions; the GeoJSON fallback gives area centroids within 7 px of them) | 62 s, 3 domains; cells: 16 s, 9 domains |
| `CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer` (10x) | Space Ranger 2.1, `tissue_positions.csv`, 35 Antibody Capture features | `visium`: 4 169 spots × 18 085 genes | 18 s, 3 domains |
| seqFISH v2 test data set (Spatial Genomics, via scverse) | `Roi1_CellxGene.csv`, `Roi1_CellCoordinates.csv` | `seqfish`: 58 cells × 12 genes | 5 s |
| CosMx HGSOC slide (Zenodo 8287971) | AtoMx flat files, 980 columns incl. 19 `NegPrb` | `cosmx`: 38 161 cells × 960 genes | 232 s, 3 domains |
| CosMx per-FOV exports (Zenodo 18602259) | two FOVs of different slides, `cell_ID == 0` background rows | `cosmx`: 2 054 and 3 352 cells × 968 genes | 12 s; both slides integrated: 24 s |
| Stereo-seq bin1 GEM (cartloader starter, Zenodo 17953582) | 18 M rows `geneID x y MIDCounts` | `stereo_seq` bin 50: 9 082 bins × 23 463 genes | 56 s, 4 domains |
| Stereo-seq cell-bin table (paper data, `Cell_GetExp_gene.txt`) | `geneID x y UMICount label` | `stereo_seq` cells: 19 247 cells × 27 106 genes | 104 s, 3 domains |
| STARmap transcript list (cartloader starter) | `X Y gene count cell_id Z` | `read_transcripts`: 14 816 cells × 1 022 genes | — |

The square-bin and cell-bin GEF readers follow the SAW file-format manual (software V7.1, document
A4.2: `geneExp/binN/expression` with `x, y, count`, `geneExp/binN/gene` with `gene, offset, count`;
`cellBin/cell`, `cellBin/cellExp`, `cellBin/gene`; the `offset` tables are honoured and validated) and
are tested on files written to that specification, including reversed offsets; a GEF from a SAW run
that behaves differently is a bug report we want. Slide-seq and Curio layouts follow the Broad / Curio
documentation and the STAGATE tutorial that loaded the paper's Slide-seqV2 data and are tested on
synthetic files only.

Download commands for the public samples:

```bash
curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip
curl -O https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Tiny_3prime_Dataset/Visium_HD_Tiny_3prime_Dataset_outs.zip
curl -O https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
curl -O https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_spatial.tar.gz
curl -O https://s3.embl.de/spatialdata/raw_data/seqfish-2-test-dataset.zip
curl -O "https://zenodo.org/records/8287971/files/Run5452_S2_exprMat_file.csv?download=1"   # + metadata_file, fov_positions_file
curl -O "https://zenodo.org/records/17953582/files/stereoseq_starter.raw.tar.gz?download=1"
```

## Notes and limits

* `.parquet` files (Visium HD positions, Xenium tables) need `pyarrow`: `pip install "staig[io]"`.
  Xenium falls back to `cells.csv.gz` when pyarrow is missing.
* `staig inspect` shows the detection and the QC summary without training; in Python
  `staig.io.detect_platform` returns the platform name (it reads nothing) and `staig.io.summary(adata)`
  prints the QC summary. An unrecognised folder raises an error listing its files and the supported
  layouts, and a folder that matches several layouts asks for `platform=`; use `read_table` for anything
  else.
* The silhouette-based choice of the number of domains is a heuristic (it tends to pick few domains on
  large single-cell data sets); pass `n_clusters` when you know the number of domains.
* Negative-control counts are kept per cell in `obs["control_counts"]` for QC.
* MERSCOPE `detected_transcripts.csv` can only be aggregated when it carries a cell-id column (newer
  Vizgen post-processing); otherwise use the `cell_by_gene.csv` export.
* Memory: GEM and transcript tables are aggregated in 4-million-row chunks and dense tables are
  converted to sparse in row blocks; a whole 18-million-row Stereo-seq GEM still needs a few GB of RAM
  while it is read, and training keeps a dense `N × genes` input matrix (3 000 columns after HVG
  selection, every gene for panels).
* Stereo-seq bins are anchored at multiples of the bin size in chip coordinates (stereopy anchors at
  the smallest observed coordinate instead), so the same bin grid is obtained whatever the tissue
  region; header offsets are recorded, not applied.
* CosMx per-FOV pixel coordinates (`coordinates="local"`, `x_local_px`) overlap between FOVs and are
  only meaningful for one FOV.
