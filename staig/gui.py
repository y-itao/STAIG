"""Local web interface (``staig gui``): pick a dataset, press *Run*, look at the domains, download the result.

Requires the optional ``gui`` extra (``pip install "staig[gui]"``, i.e. gradio). The interface runs
``staig.run`` in the same process; the heavy lifting happens in a worker thread so the page stays
responsive. Nothing is uploaded anywhere — the app serves on localhost only.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from pathlib import Path

from .io import PLATFORMS


def preset_choices() -> list[str]:
    from ._config import PRESETS

    names = list(PRESETS)
    return ["auto"] + [p for p in names if not p.startswith("paper_")] + [p for p in names if p.startswith("paper_")]


def run_pipeline(input_path: str, image_path: str, n_clusters: int, preset: str, batch_key: str,
                 n_seeds: int, smooth: bool, epochs: int, counts: str | bool, output_dir: str,
                 platform: str = "auto", bin_size: int | None = None, min_counts: int | None = None,
                 coordinates: str = "", hd_level: str = "bins"):
    """Backend of the GUI (also usable without gradio). Returns (metrics_json, figure_path, h5ad_path, log).

    ``input_path`` may be any platform output recognised by :func:`staig.read` (several paths separated by
    ``;`` are integrated as slices)."""
    import matplotlib

    matplotlib.use("Agg")

    import staig
    from staig import metrics, pl
    from staig.io import read, summary

    log = []
    t0 = time.time()
    inputs = [p.strip() for p in (input_path or "").split(";") if p.strip()]
    if not inputs:
        raise FileNotFoundError("no input given")
    for p in inputs:
        if not Path(p).exists():
            raise FileNotFoundError(f"input not found: {p!r}")
    kw = {}
    if bin_size:
        kw["bin_size"] = int(bin_size)
    if min_counts is not None and int(min_counts) >= 0:
        kw["min_counts"] = int(min_counts)
    if coordinates and coordinates.strip():
        kw["coordinates"] = coordinates.strip()
    if hd_level and hd_level != "bins":
        kw["level"] = hd_level
    adatas = read(inputs, platform=platform or "auto", **kw) if len(inputs) > 1 else \
        [read(inputs[0], platform=platform or "auto", **kw)]
    for p, a in zip(inputs, adatas):
        log.append(f"read {p}: {summary(a)}")
    bk = batch_key.strip() if batch_key and batch_key.strip() else None
    is_h5ad = [p.lower().endswith(".h5ad") for p in inputs]
    if len(adatas) > 1:
        bk = bk or "slice"
        adata: object = adatas
    else:
        adata = adatas[0]
    image = image_path.strip() if image_path and image_path.strip() else "auto"
    from ._utils import unique_path, write_h5ad

    if isinstance(counts, str):                       # dropdown: "auto" | "raw counts" | "normalised"
        counts = None if counts == "auto" else counts.lower().startswith("raw")
    if counts is None:
        from .pp import _is_integer_counts

        raw = [True if not h5 else _is_integer_counts(a.X) for a, h5 in zip(adatas, is_h5ad)]
        if any(is_h5ad):
            log.append("h5ad input detected as " + ("raw counts" if all(raw) else "normalised values"))
        if not all(r == raw[0] for r in raw):
            raise ValueError("some inputs hold raw counts and others normalised values; choose the expression "
                             "values explicitly")
        counts = all(raw)
    pre = {} if counts else {"normalize": False, "log1p": False}
    kwargs = dict(n_clusters=int(n_clusters) if n_clusters and int(n_clusters) > 0 else None, preset=preset,
                  batch_key=bk, image=image, n_seeds=int(n_seeds), smooth=bool(smooth), preprocess=pre)
    if epochs and int(epochs) > 0:
        kwargs["epochs"] = int(epochs)
    model = staig.run(adata, **kwargs)
    adata = model.adata
    res = metrics.evaluate(adata, batch_key=bk)
    info = adata.uns["staig"]
    log.append(f"edge_removal={info['config'].get('edge_removal')} negatives={info.get('negatives')}")
    log.append(f"metrics: {json.dumps({k: round(v, 4) for k, v in res.items()})}")
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="staig_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(inputs[0]).stem or "staig"
    for _ in range(1000):                                              # never overwrite an earlier result
        h5ad_path = unique_path(out_dir / f"{stem}_staig.h5ad")
        try:
            write_h5ad(adata, h5ad_path)
            break
        except FileExistsError:                                        # created meanwhile: take the next name
            continue
    fig_path = unique_path(out_dir / f"{stem}_staig_domains.png")
    import matplotlib.pyplot as plt

    colors = ["staig_domain"] + (["ground_truth"] if "ground_truth" in adata.obs else [])
    try:
        pl.spatial(adata, color=colors, show=False, batch_key=bk)
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    except Exception as e:  # plotting must not hide a successful run
        log.append(f"plot failed: {e}")
        fig_path = None
    plt.close("all")
    log.append(f"written {h5ad_path} in {time.time() - t0:.1f}s")
    return json.dumps(res, indent=2), (str(fig_path) if fig_path else None), str(h5ad_path), "\n".join(log)


def build_app():
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover
        raise ImportError("the GUI needs gradio: pip install 'staig[gui]'") from e

    def _run(*args):
        try:
            args = list(args)
            args[11] = int(args[11]) if args[11] else None          # bin size 0 -> default
            args[12] = int(args[12]) if args[12] is not None and int(args[12]) >= 0 else None
            return run_pipeline(*args)
        except Exception as e:
            return "", None, "", f"ERROR: {e}\n\n{traceback.format_exc()}"

    with gr.Blocks(title="STAIG") as app:
        gr.Markdown("# STAIG — spatial domains from expression, coordinates and histology\n"
                    "Point to the output folder of your platform (Space Ranger / Visium HD / Xenium / MERSCOPE / "
                    "CosMx / Stereo-seq / Slide-seq / seqFISH) or an `.h5ad`, press **Run**. "
                    "Several paths separated by `;` are integrated as slices.")
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Input: platform output folder / file (or .h5ad); use ';' for several slices")
                with gr.Row():
                    platform = gr.Dropdown(["auto"] + list(PLATFORMS), value="auto", label="Platform")
                    bin_size = gr.Number(label="Bin size (Visium HD um / Stereo-seq DNBs; 0 = default)", value=0,
                                         precision=0)
                    min_counts = gr.Number(label="Min counts per cell (-1 = platform default)", value=-1, precision=0)
                with gr.Row():
                    coords = gr.Textbox(label="Coordinates table (only for a generic counts table)", value="")
                    hd_level = gr.Dropdown(["bins", "cells"], value="bins", label="Visium HD: bins or segmented cells")
                img = gr.Textbox(label="Full-resolution H&E image (optional; 'auto' looks next to the Visium folder)",
                                 value="auto")
                with gr.Row():
                    k = gr.Number(label="Number of domains (0 = choose by silhouette)", value=0, precision=0)
                    preset = gr.Dropdown(preset_choices(), value="auto", label="Preset")
                with gr.Row():
                    seeds = gr.Slider(1, 10, value=3, step=1, label="Seeds (consensus when > 1)")
                    epochs = gr.Number(label="Epochs (0 = preset default)", value=0, precision=0)
                with gr.Row():
                    smooth = gr.Checkbox(label="Smooth embedding before clustering", value=True)
                    counts = gr.Dropdown(["auto", "raw counts", "normalised"], value="auto",
                                         label="Expression values")
                batch = gr.Textbox(label="Batch key for multi-slice integration (optional)")
                outdir = gr.Textbox(label="Output folder (optional)")
                btn = gr.Button("Run STAIG", variant="primary")
            with gr.Column():
                fig = gr.Image(label="Spatial domains", type="filepath")
                met = gr.Code(label="Metrics", language="json")
                out = gr.Textbox(label="Result file (.h5ad)")
                logbox = gr.Textbox(label="Log", lines=8)
        btn.click(_run, [inp, img, k, preset, batch, seeds, smooth, epochs, counts, outdir, platform, bin_size,
                         min_counts, coords, hd_level], [met, fig, out, logbox])
    return app


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="staig-gui", description="Local web interface for STAIG")
    p.add_argument("--port", type=int, default=int(os.environ.get("STAIG_GUI_PORT", 7860)))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true", help="create a temporary public link (gradio)")
    args = p.parse_args(argv)
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
