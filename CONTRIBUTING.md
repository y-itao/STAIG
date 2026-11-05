# Contributing to STAIG

Read [DESIGN.md](docs/dev/DESIGN.md) before changing the method or its
AnnData contracts. The v1 reference implementations live in `legacy/`; production
changes belong in `staig/`.

## Development setup

Use Python 3.10, 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,image,cluster]"
```

The core package does not require R or PyG. The optional `r` extra is only needed
for comparisons with R/mclust. Do not add optional image or clustering libraries
as mandatory imports for unrelated functionality.

## Checks

Run CPU tests and Ruff before submitting a pull request:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m ruff check staig tests
```

CI runs these checks on Ubuntu with Python 3.10, 3.11 and 3.12. Tests must use small,
deterministic CPU fixtures. The downloader tests serve a tiny ZIP locally and do
not download the paper archive. The optional real-data loader check is skipped
when the local DLPFC slice is absent.

For dataset work alone:

```bash
python -m ruff check staig/datasets.py tests/test_datasets.py
CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_datasets.py -q
```

When working on the shared development machine, use the designated Python at
`/home/yang/anaconda3/envs/staig2/bin/python` and set `TMPDIR=/home/yang/tmp` for
commands that create temporary files. Keep downloaded data outside version
control. Set `STAIG_DATA` to the unpacked `Dataset` directory, or supply `root=`
to a loader. To download a single slice without its full-resolution image:

```bash
python -m staig.datasets --root ./Dataset --members DLPFC/151673 --no-include-full-image
```

Existing files with the correct size are skipped; incomplete files restart at the
file boundary. `--overwrite` replaces existing files. Omitting `--members`
selects the entire dataset, so use explicit prefixes for small downloads.

## Adding a preset

1. Add a named override dictionary to `PRESETS` in `staig/_config.py`, using only
   fields in `STAIGConfig`. Leave shared defaults unchanged unless measurements
   on several datasets and seeds support the change. Reserve `paper_*` names for
   the settings of the paper.
2. Regenerate the YAML mirrors: `python -c "from staig._config import export_presets;
   export_presets()"` (a test fails when `staig/presets/*.yaml` and `PRESETS` differ).
   Check that `STAIGConfig.from_preset("your_name").validate()` succeeds and add a
   focused CPU test for the relevant behavior. Check the configuration's AnnData
   requirements and the intended image/no-image or integration mode.
3. If a runnable configuration example is useful, add a YAML file under
   `configs/presets/` containing `preset: your_name` and any documented overrides.
   Load it with `STAIGConfig.from_yaml(path)`; placing a YAML file there alone does
   not register a new built-in preset.
4. Document the intended platform and rationale, with the data, seeds and settings
   used to choose the values.

Keep pull requests focused and describe the change and the checks performed.
Small `.npy` fixtures under `tests/` are allowed;
large datasets, generated arrays elsewhere and model outputs should stay local.
