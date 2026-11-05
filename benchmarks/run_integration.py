"""Alignment-free integration benchmarks (paper Fig. 5).

    python benchmarks/run_integration.py --root ../data/zenodo/Dataset --scenario dlpfc_7576 --seeds 0,1,2

Scenarios: dlpfc_7576 (vertical, same donor), dlpfc_4 (two donors), mouse_brain (anterior+posterior,
horizontal), merfish_vis (two mice), cross_platform (Stereo-seq + Slide-seqV2 olfactory bulb).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import staig  # noqa: E402
from staig import cluster as C  # noqa: E402
from staig import datasets, graph, metrics, pp  # noqa: E402


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


def build(scenario, root, subsample=None, seed=0):
    root = Path(root)
    if scenario in {"dlpfc_7576", "dlpfc_4"}:
        ids = ["151675", "151676"] if scenario == "dlpfc_7576" else ["151507", "151508", "151675", "151676"]
        ads = [datasets.load_dlpfc(s, root=root) for s in ids]
        merged = pp.concat_slices(ads, batch_key="slice", batch_categories=ids, n_top_genes=5000)
        graph.image_pcs(merged, n_comps=128, batch_key="slice")
        return merged, dict(label="ground_truth", n_clusters=7, preset="paper_integration_vertical", image=True)
    if scenario == "mouse_brain":
        ids = ["Mouse_Brain_Anterior", "Mouse_Brain_Posterior"]
        ads = [datasets.load_visium(s, root=root, labels=None, image_features=None) for s in ids]
        merged = pp.concat_slices(ads, batch_key="slice", batch_categories=["anterior", "posterior"], n_top_genes=5000)
        return merged, dict(label=None, n_clusters=25, preset="paper_integration_horizontal", image=False)
    if scenario == "merfish_vis":
        ads = [sc.read_h5ad(root / "other" / "Merfish" / f"mouse{i}.AUD_TEA_VIS.242.unexpand_cluster3.h5ad")
               for i in (1, 2)]
        merged = pp.concat_slices(ads, batch_key="slice", batch_categories=["mouse1", "mouse2"], n_top_genes=234)
        return merged, dict(label="cluster_L2", n_clusters=8, preset="integration", image=False,
                            overrides=dict(n_layers=1, tau=1.0, epochs=400, drop_feature_rates=(0.1, 0.2)))
    if scenario == "cross_platform":
        a = sc.read_h5ad(root / "other" / "Mouse_Olfactory_Stereo-seq" / "filtered_feature_bc_matrix_norm.h5ad")
        b = sc.read_h5ad(root / "other" / "Mouse_Olfactory_slide-seqV2" / "v2.h5ad")
        if subsample:
            rng = np.random.RandomState(seed)
            a = a[rng.choice(a.n_obs, subsample, replace=False)].copy()
            b = b[rng.choice(b.n_obs, subsample, replace=False)].copy()
        for x in (a, b):
            x.var_names_make_unique()
        merged = pp.concat_slices([a, b], batch_key="slice", batch_categories=["stereo_seq", "slide_seqv2"],
                                  n_top_genes=5000, normalize=False, log1p=False, flavor="seurat_v3",
                                  strict_counts=False)       # v1 applied seurat_v3 to the normalised matrices
        return merged, dict(label=None, n_clusters=7, preset="integration", image=False,
                            overrides=dict(n_layers=2, tau=10.0, epochs=400))
    raise ValueError(scenario)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("STAIG_DATA", "./Dataset"))
    ap.add_argument("--scenario", default="dlpfc_7576")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--n-clusters", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--cluster", default="gmm")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--save-h5ad", default=None)
    args = ap.parse_args()
    merged0, spec = build(args.scenario, args.root, args.subsample)
    preset = args.preset or spec["preset"]
    overrides = dict(spec.get("overrides", {}))
    overrides.update(parse_overrides(args.override))
    n_clusters = args.n_clusters or spec["n_clusters"]
    out = Path(__file__).parent / "results" / f"integration_{args.scenario}_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        adata = merged0.copy()
        cfg = staig.STAIGConfig.from_preset(preset).update(seed=seed, verbose=False, **overrides)
        graph.spatial_neighbors(adata, n_neighbors=cfg.n_neighbors, batch_key="slice")
        model = staig.STAIG(cfg)
        t0 = time.perf_counter()
        model.fit(adata, batch_key="slice")
        t_train = time.perf_counter() - t0
        C.cluster(adata, n_clusters=n_clusters, method=args.cluster, refine=False, random_state=seed)
        res = metrics.evaluate(adata, label_key=spec["label"], batch_key="slice")
        import torch

        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
        row = dict(scenario=args.scenario, seed=seed, preset=preset, overrides=json.dumps(overrides, sort_keys=True),
                   n_obs=adata.n_obs, n_features=adata.obsm["X_staig_input"].shape[1], n_clusters=n_clusters,
                   negatives=adata.uns["staig"]["negatives"], edge_mode=adata.uns["staig"]["edge_removal"]["mode"],
                   train_s=round(t_train, 1), peak_gpu_gb=round(mem, 2), **{k: round(v, 4) for k, v in res.items()})
        rows.append(row)
        print(row, flush=True)
        if args.save_h5ad and len(rows) == 1:
            adata.write_h5ad(args.save_h5ad)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written {out}")


if __name__ == "__main__":
    main()
