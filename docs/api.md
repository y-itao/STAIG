# API reference

These signatures are extracted from the checked-out source. The public modules
are `pp`, `graph`, `img`, `train`, `cluster`, `metrics`, `pl`, `datasets` and
`cli`; use `staig.cluster.cluster`, not `staig.tl.cluster`.
See the [AnnData key table](quickstart.md#anndata-keys) and the
[negative-sampling settings](tutorials/large_datasets.md#sampled-negatives-with-n_negatives).

## Package and configuration

::: staig
    options:
      members: [STAIGConfig, PRESETS, preset_config, STAIG, run]

## Reading platform outputs

::: staig.io

## Preprocessing

::: staig.pp

## Spatial graphs

::: staig.graph

## Image features

::: staig.img

## Training

::: staig.train
    options:
      members: [STAIG, run]
      show_root_full_path: true

## Clustering

::: staig.cluster

## Evaluation

::: staig.metrics

## Plotting

::: staig.pl

## Datasets

::: staig.datasets
    options:
      members: [list_members, download_dataset, data_root, load_visium_dir, load_dlpfc, load_visium, load_h5ad, n_clusters_for]

## Command line

`staig run`, `staig image` and `staig presets` are the installed CLI commands.
`python -m staig.datasets` is the selective downloader; benchmark scripts live
under `benchmarks/` rather than a `staig benchmark` command.

```bash
staig --help
staig presets --show-default
python -m staig.datasets --help
```

For CLI `run`, add `--counts` when the input contains raw counts; otherwise
normalisation and log1p are disabled. Model configuration YAML accepts
`STAIGConfig` fields and an optional `preset` key.

::: staig.cli
    options:
      members: [main]
