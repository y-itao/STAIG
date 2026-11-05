"""Benchmarks without histology: Stereo-seq / Slide-seqV2 olfactory bulb, STARmap, MERFISH.

    python benchmarks/run_no_image.py --root ../data/zenodo/Dataset --dataset stereo --preset stereo_seq --seeds 0,1,2

Datasets (Zenodo layout):
  stereo      other/Mouse_Olfactory_Stereo-seq/filtered_feature_bc_matrix_norm.h5ad   (19 109 bins, log-normalised)
  slideseq    other/Mouse_Olfactory_slide-seqV2/v2.h5ad                               (20 139 beads, log-normalised)
  hippocampus other/Mouse_Hippocampus_Tissue_slide-seqV2/filtered_feature_bc_matrix.h5ad (52 869 beads, counts)
  starmap     other/starmap/starmap_norm.h5ad                                  (1 207 cells, obs['ground_truth'])
  merfish_mtg other/Merfish/H18.06.006.MTG.4000.expand.rep3.h5ad                        (3 970 cells, counts)
  merfish_vis other/Merfish/mouse1.AUD_TEA_VIS.242.unexpand_cluster3.h5ad               (5 995 cells, counts)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import staig  # noqa: E402
from staig import cluster as C  # noqa: E402
from staig import graph, metrics, pp  # noqa: E402

DATASETS = {
    "stereo": dict(path="other/Mouse_Olfactory_Stereo-seq/filtered_feature_bc_matrix_norm.h5ad", normalized=True,
                   n_clusters=7, n_top_genes=3000, label=None, preset="stereo_seq"),
    "slideseq": dict(path="other/Mouse_Olfactory_slide-seqV2/v2.h5ad", normalized=True, n_clusters=11,
                     n_top_genes=3000, label=None, preset="slide_seq"),
    "hippocampus": dict(path="other/Mouse_Hippocampus_Tissue_slide-seqV2/filtered_feature_bc_matrix.h5ad",
                        normalized=False, n_clusters=10, n_top_genes=3000, label=None, preset="slide_seq"),
    "starmap": dict(path="other/starmap/starmap_norm.h5ad", normalized=True, n_clusters=7, n_top_genes=1020,
                    label="ground_truth", preset="starmap"),
    "merfish_mtg": dict(path="other/Merfish/H18.06.006.MTG.4000.expand.rep3.h5ad", normalized=False, n_clusters=None,
                        n_top_genes=4000, label="cluster_L2", preset="merfish"),
    "merfish_vis": dict(path="other/Merfish/mouse1.AUD_TEA_VIS.242.unexpand_cluster3.h5ad", normalized=False,
                        n_clusters=None, n_top_genes=234, label="cluster_L2", preset="merfish"),
}


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = tuple(v) if isinstance(v, list) else v
    return out


def load(name, root, n_top_genes=None):
    spec = DATASETS[name]
    adata = sc.read_h5ad(Path(root) / spec["path"])
    adata.var_names_make_unique()
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["X", "Y"]].values if "X" in adata.obs else adata.obs[["x", "y"]].values
    ntg = n_top_genes or spec["n_top_genes"]
    if spec["normalized"]:
        pp.preprocess(adata, n_top_genes=ntg, normalize=False, log1p=False, scale=False, strict_counts=False)
    else:
        pp.preprocess(adata, n_top_genes=ntg, scale=True)
    return adata, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("STAIG_DATA", "./Dataset"))
    ap.add_argument("--dataset", default="stereo", choices=sorted(DATASETS))
    ap.add_argument("--preset", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--n-clusters", type=int, default=None)
    ap.add_argument("--n-top-genes", type=int, default=None)
    ap.add_argument("--cluster", default="gmm")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--save-h5ad", default=None)
    args = ap.parse_args()
    adata0, spec = load(args.dataset, args.root, args.n_top_genes)
    preset = args.preset or spec["preset"]
    overrides = parse_overrides(args.override)
    n_clusters = args.n_clusters or spec["n_clusters"]
    if n_clusters is None and spec["label"]:
        n_clusters = int(adata0.obs[spec["label"]].nunique())
    out = Path(__file__).parent / "results" / f"{args.dataset}_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        adata = adata0.copy()
        cfg = staig.STAIGConfig.from_preset(preset).update(seed=seed, verbose=False, **overrides)
        graph.spatial_neighbors(adata, n_neighbors=cfg.n_neighbors)
        model = staig.STAIG(cfg)
        t0 = time.perf_counter()
        model.fit(adata)
        t_train = time.perf_counter() - t0
        mem = torch_peak_mem()
        C.cluster(adata, n_clusters=n_clusters, method=args.cluster, refine=args.refine, random_state=seed)
        res = metrics.evaluate(adata, label_key=spec["label"])
        row = dict(dataset=args.dataset, seed=seed, preset=preset, overrides=json.dumps(overrides, sort_keys=True),
                   n_obs=adata.n_obs, n_features=adata.obsm["X_staig_input"].shape[1], n_clusters=n_clusters,
                   train_s=t_train, peak_gpu_gb=mem, **{k: round(v, 4) for k, v in res.items()})
        rows.append(row)
        print(row, flush=True)
        if args.save_h5ad and seed == rows[0]["seed"]:
            adata.write_h5ad(args.save_h5ad)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written {out}")


def torch_peak_mem():
    import torch

    if torch.cuda.is_available():
        m = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return round(m, 2)
    return float("nan")


if __name__ == "__main__":
    main()
