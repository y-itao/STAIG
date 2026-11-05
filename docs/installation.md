# Installation

STAIG v2 requires Python 3.10 or newer. Install from GitHub:

```bash
pip install git+https://github.com/y-itao/STAIG
```

The intended PyPI package name is `staig` once v2 is released. During development,
use the branch containing v2 explicitly if the default branch still contains v1:

```bash
python -m pip install "staig @ git+https://github.com/y-itao/STAIG@v2"
```

## Optional dependencies

| Extra | Adds | Needed for |
| --- | --- | --- |
| `[image]` | torchvision, OpenCV, Pillow, TIFF support | Cropping/filtering H&E patches and BYOL/ResNet features |
| `[cluster]` | igraph, leidenalg | Leiden clustering (`cluster_method="leiden"`) |
| `[r]` | rpy2 | Optional R/mclust comparison; install R and R's `mclust` separately |
| `[docs]` | Material, mkdocstrings, mkdocs-jupyter | Building this site |
| `[dev]` | Jupyter, nbconvert, test tools | Running the example notebooks and development |

For image extraction and notebooks on the v2 branch:

```bash
python -m pip install "staig[image,dev] @ git+https://github.com/y-itao/STAIG@v2"
```

Replace `image,dev` with `cluster` or `r` when that is the only extra you need.
Python GMM clustering, BatchKL and iLISI need no R. The core workflow does not
require scGeneClust, PyWGCNA, SpatialDE or PyTorch Geometric.

## Conda and PyTorch

Create an isolated environment, then choose PyTorch for your hardware before
installing STAIG. For a Linux or Windows CPU environment:

```bash
conda create -n staig2 python=3.11 pip
conda activate staig2
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install "staig[image,dev] @ git+https://github.com/y-itao/STAIG@v2"
```

For NVIDIA GPUs, replace the CPU PyTorch installation line with the command from
the [official PyTorch selector](https://pytorch.org/get-started/locally/) for your
driver/platform. Keep `torch` and `torchvision` compatible. CUDA accelerates model
training and image extraction; CPU execution is supported with `device="cpu"`.
`device="auto"` selects CUDA, then MPS, then CPU according to availability.

```python
import torch
import staig

print("STAIG", staig.__version__, "PyTorch", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
```

### Apple Silicon

Use a native arm64 Python environment and the macOS PyTorch wheel from the
[PyTorch instructions](https://pytorch.org/get-started/locally/), without the Linux
CPU wheel index. MPS is a selectable training device (`device="mps"`), but this
revision uses sparse CSR graph operations: actual MPS training depends on support
in your installed PyTorch version. This documentation run did not validate MPS.
If a sparse operation is unsupported, select `device="cpu"` explicitly;
automatic selection does not catch unsupported-operator errors. BYOL is slow on
CPU, so the notebooks reuse released image features where possible.

### Windows

The Python workflow can use CPU or an appropriate CUDA wheel. Use WSL2 for the
optional R/mclust extra, with R, rpy2 and the `mclust` R package installed in the
same WSL environment. For image extraction in notebooks, `num_workers=0` avoids
worker-spawning complications.

## Build the documentation

From a v2 checkout:

```bash
python -m pip install -e ".[docs]"
python -m pip install -r docs/dev/requirements-site.txt
mkdocs build --strict
```

The generated site goes to `examples/.cache/site/`. Notebook outputs are rendered
without rerunning training. See [quick start](quickstart.md) for data setup and
the notebook execution commands in [examples](https://github.com/y-itao/STAIG/tree/main/examples).
