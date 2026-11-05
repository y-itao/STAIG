# Experiment scripts

Scripts that run the experiments of the paper with version 2. All of them take `--root` (or the
`STAIG_DATA` environment variable) pointing to the unpacked Zenodo `Dataset` folder and write per-seed
CSV tables (`--tag` names the output file).

| script | what it runs |
|---|---|
| `run_dlpfc.py` | one configuration on one or all annotated DLPFC slices (ARI/NMI before and after refinement, training time); `--paper-configs configs/paper/train_img_config.yaml` replays the per-slice settings of version 1 |
| `sweep_dlpfc.py` | many configurations × slices × seeds with several clustering post-processings on the same embedding and the consensus over seeds; embeddings are saved for `analyze_embeddings.py` |
| `analyze_embeddings.py` | ensembles (consensus, concatenated PCA) from saved embeddings |
| `run_no_image.py` | Stereo-seq / Slide-seqV2 / STARmap / MERFISH without histology (SC, DB, ARI/NMI where labels exist, time, peak GPU memory) |
| `run_integration.py` | the integration scenarios of the paper (ARI/NMI, BatchKL, iLISI) |

Conventions: `ARI_ref`/`NMI_ref` are after the majority-vote refinement used for DLPFC in the paper;
`seed = -1` rows are consensus results; times are wall-clock seconds for `STAIG.fit` on the machine the
script ran on.
