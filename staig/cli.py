"""Command line interface: ``staig run``, ``staig image``, ``staig presets``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLATFORM_NAMES = ("visium", "visium_hd", "xenium", "merscope", "cosmx", "stereo_seq", "slide_seq", "seqfish", "table",
                  "transcripts", "h5ad")


def _cmd_presets(args):
    from ._config import PRESETS, STAIGConfig

    for name in sorted(PRESETS):
        print(f"{name}: {json.dumps(PRESETS[name])}")
    if args.show_default:
        print(json.dumps(STAIGConfig().to_dict(), indent=2))


def _read_kwargs(args) -> dict:
    kw = {}
    if getattr(args, "bin_size", None) is not None:
        kw["bin_size"] = args.bin_size
    if getattr(args, "hd_level", None):
        kw["level"] = args.hd_level
    if getattr(args, "min_counts", None) is not None:
        kw["min_counts"] = args.min_counts
    if getattr(args, "coordinates", None):
        kw["coordinates"] = args.coordinates
    return kw


def _load_inputs(args):
    """Read one or several platform outputs; returns (adata or list of adata, list of 'is .h5ad file' flags)."""
    from .io import read, summary

    kw = _read_kwargs(args)
    is_h5ad = [str(p).lower().endswith(".h5ad") for p in args.input]
    if len(args.input) == 1:
        adata = read(args.input[0], platform=args.platform, **kw)
        print(f"read {args.input[0]}: {summary(adata)}")
        return adata, is_h5ad
    adatas = read(list(args.input), platform=args.platform, **kw)
    for p, a in zip(args.input, adatas):
        print(f"read {p} (slice {a.obs['slice'].iloc[0]!r}): {summary(a)}")
    return adatas, is_h5ad


def _expression_mode(args, adatas, is_h5ad) -> dict:
    """Preprocessing overrides from --counts / --log-normalized / auto-detection on .h5ad inputs."""
    from .pp import _is_integer_counts

    if args.log_normalized:
        print("input treated as log-normalised values (--log-normalized)")
        return {"normalize": False, "log1p": False}
    if args.counts:
        print("input treated as raw counts (--counts)")
        return {}
    raw = []
    for a, h5 in zip(adatas, is_h5ad):
        raw.append(True if not h5 else _is_integer_counts(a.X))      # platform readers always give raw counts
    if all(raw):
        if any(is_h5ad):
            print("input treated as raw counts (auto-detected; use --counts/--log-normalized to force)")
        return {}
    if not all(r == raw[0] for r in raw):
        raise ValueError("some inputs hold raw counts and others normalised values; pass --counts or "
                         "--log-normalized, or preprocess the inputs consistently")
    print("input treated as normalised values (auto-detected; use --counts/--log-normalized to force)")
    return {"normalize": False, "log1p": False}


def _cmd_inspect(args):
    from .io import detect_platform, read, summary

    for p in args.input:
        platform = detect_platform(p) if args.platform == "auto" else args.platform
        print(f"{p}: platform={platform}")
        if not args.detect_only:
            adata = read(p, platform=platform, **_read_kwargs(args))
            print(f"  {summary(adata)}")
            print(f"  obs columns: {list(adata.obs.columns)[:12]}")


def _cmd_run(args):
    from . import metrics, run
    from ._config import STAIGConfig

    for target in (args.output, args.save_model):            # fail before training, not after
        if target and Path(target).exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite or choose another name")
    cfg = STAIGConfig.from_yaml(args.config) if args.config else None
    adata, is_h5ad = _load_inputs(args)
    if args.image is not None:
        if isinstance(adata, list):
            raise ValueError("--image applies to a single input")
        from .img import extract_image_features

        extract_image_features(adata, image=args.image, epochs=args.image_epochs)
    pre = _expression_mode(args, adata if isinstance(adata, list) else [adata], is_h5ad)
    overrides = {} if args.seed is None else {"seed": args.seed}  # keep the preset/YAML seed unless --seed
    batch_key = args.batch_key or ("slice" if isinstance(adata, list) else None)
    model = run(adata, n_clusters=args.n_clusters, preset=args.preset, config=cfg, batch_key=batch_key,
                image=None, n_seeds=args.n_seeds if args.n_seeds == "auto" else int(args.n_seeds),
                cluster_method=args.cluster, smooth=not args.no_smooth,
                refine=True if args.refine else None, preprocess=pre, **overrides)
    adata = model.adata
    res = metrics.evaluate(adata, batch_key=batch_key)
    print(json.dumps({k: round(v, 4) for k, v in res.items()}))
    if args.output:
        from ._utils import write_h5ad

        write_h5ad(adata, args.output, overwrite=args.overwrite)
        print(f"written {args.output}")
    if args.save_model:
        model.save(args.save_model)


def _cmd_image(args):
    from .img import extract_image_features
    from .io import detect_platform, read

    platform = detect_platform(args.input)
    adata = read(args.input, platform=platform, **({"image_features": None} if platform == "visium" else {}))
    extract_image_features(adata, image=args.image, epochs=args.epochs, batch_size=args.batch_size)
    import numpy as np

    np.save(args.output, adata.obsm["image_features"])
    print(f"written {args.output} {adata.obsm['image_features'].shape}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="staig", description="STAIG: image-aided graph contrastive learning for ST")
    sub = p.add_subparsers(dest="cmd", required=True)
    def add_input_options(s):
        s.add_argument("input", nargs="+", help="platform output folder/file(s): Space Ranger outs, Visium HD, "
                                                "Xenium, MERSCOPE, CosMx, Stereo-seq .gef/.gem, Slide-seq, seqFISH, "
                                                ".h5ad; several inputs = alignment-free integration")
        s.add_argument("--platform", default="auto", help="auto | " + " | ".join(PLATFORM_NAMES))
        s.add_argument("--bin-size", type=int, default=None, help="Visium HD bin (um) / Stereo-seq bin (DNBs)")
        s.add_argument("--hd-level", default=None, choices=["bins", "cells"], help="Visium HD: bins or segmented cells")
        s.add_argument("--min-counts", type=int, default=None, help="drop observations below this count total")
        s.add_argument("--coordinates", default=None, help="coordinates table for a generic counts table")

    s = sub.add_parser("inspect", help="detect the platform and summarise what would be read")
    add_input_options(s)
    s.add_argument("--detect-only", action="store_true")
    s.set_defaults(func=_cmd_inspect)
    s = sub.add_parser("run", help="read, preprocess, train, embed and cluster one or several datasets")
    add_input_options(s)
    s.add_argument("--n-clusters", type=int, default=None)
    s.add_argument("--preset", default="auto", help="auto | visium | visium_no_image | stereo_seq | merfish | ...")
    s.add_argument("--n-seeds", default="auto", help="'auto' (3 on GPU, 1 on CPU) or an int; >1 = consensus")
    s.add_argument("--no-smooth", action="store_true", help="do not smooth the embedding before clustering")
    s.add_argument("--config", default=None, help="YAML config (optionally with a 'preset:' key)")
    s.add_argument("--batch-key", default=None, help="obs column with the slice (default 'slice' for several inputs)")
    s.add_argument("--cluster", default="gmm", choices=["gmm", "mclust", "kmeans", "leiden", "louvain"])
    s.add_argument("--refine", action="store_true")
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--counts", action="store_true", help="force: the input holds raw counts (normalise + log1p)")
    mode.add_argument("--log-normalized", action="store_true",
                      help="force: the input is already log-normalised (auto-detected for .h5ad inputs)")
    s.add_argument("--image", default=None, help="full-resolution H&E image to extract features from")
    s.add_argument("--image-epochs", type=int, default=1)
    s.add_argument("--seed", type=int, default=None, help="override the seed of the preset / YAML config")
    s.add_argument("--output", default=None, help="write the annotated .h5ad here (refuses to overwrite)")
    s.add_argument("--save-model", default=None)
    s.add_argument("--overwrite", action="store_true", help="allow --output / --save-model to replace files")
    s.set_defaults(func=_cmd_run)
    s = sub.add_parser("image", help="extract BYOL image features for a Visium folder")
    s.add_argument("input")
    s.add_argument("--image", default=None)
    s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--output", default="embeddings.npy")
    s.set_defaults(func=_cmd_image)
    s = sub.add_parser("gui", help="open the local web interface (needs 'pip install staig[gui]')")
    s.add_argument("--port", type=int, default=7860)
    s.set_defaults(func=lambda a: __import__("staig.gui", fromlist=["main"]).main(["--port", str(a.port)]))
    s = sub.add_parser("presets", help="list presets")
    s.add_argument("--show-default", action="store_true")
    s.set_defaults(func=_cmd_presets)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
