"""DLPFC benchmark: 12 slices x seeds x configs -> CSV.

Example::

    python benchmarks/run_dlpfc.py --root ../data/zenodo/Dataset --slices all \
        --preset paper_151673 --seeds 39788 --tag v1compat --cluster gmm,mclust

Each row: slice, seed, preset, overrides, cluster method, ARI/NMI before and after refinement,
training time. Uses the image features shipped on Zenodo (``embeddings.npy``) unless
``--features-dir`` points to a folder with ``<slice>.npy`` files.
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


V1_KEYS = {"tau": "tau", "num_epochs": "epochs", "num_layers": "n_layers", "seed": "seed",
           "num_neigh": "n_neighbors", "learning_rate": "lr", "weight_decay": "weight_decay",
           "num_hidden": "hidden_dim", "num_proj_hidden": "proj_dim", "k": "n_pseudo_labels"}


def v1_section_to_overrides(section: dict) -> dict:
    """Map a v1 YAML section (train_img_config.yaml / mv_config.yaml) onto STAIGConfig fields."""
    out = {V1_KEYS[k]: v for k, v in section.items() if k in V1_KEYS}
    if "drop_feature_rate_1" in section:
        r1 = float(section["drop_feature_rate_1"])
        out["drop_feature_rates"] = (r1, float(section.get("drop_feature_rate_2", r1)))
    return out


def run_one(slice_id, root, preset, overrides, seed, cluster_methods, features_dir=None, refine_neighbors=15,
            n_image_pcs=None, pseudo_k=None, verbose=False, zero_center=False, smooth_steps=0):
    adata = datasets.load_dlpfc(slice_id, root=root)
    if features_dir is not None:
        f = np.load(Path(features_dir) / f"{slice_id}.npy")
        adata.obsm["image_features"] = f.reshape(f.shape[0], -1).astype(np.float32)
    overrides = {k: v for k, v in overrides.items() if k != "seed"}
    cfg = staig.STAIGConfig.from_preset(preset).update(seed=seed, verbose=verbose, **overrides)
    pp.preprocess(adata, n_top_genes=3000, zero_center=zero_center)
    graph.image_pcs(adata, n_comps=n_image_pcs or cfg.n_image_pcs)
    graph.spatial_neighbors(adata, n_neighbors=cfg.n_neighbors)
    model = staig.STAIG(cfg)
    if pseudo_k is not None:
        model.config = model.config.update(n_pseudo_labels=pseudo_k)
    t0 = time.perf_counter()
    model.fit(adata)
    t_train = time.perf_counter() - t0
    rows = []
    n_clusters = datasets.n_clusters_for(adata)
    rep = "X_staig"
    if smooth_steps:
        C.smooth_embedding(adata, n_steps=smooth_steps, key_added="X_staig_smooth")
        rep = "X_staig_smooth"
    for method in cluster_methods:
        try:
            C.cluster(adata, n_clusters=n_clusters, method=method, refine=False, random_state=seed,
                      key_added="dom", use_rep=rep)
        except ImportError as e:
            print(f"[skip] {method}: {e}")
            continue
        ari0 = metrics.ari(adata.obs["ground_truth"], adata.obs["dom"])
        nmi0 = metrics.nmi(adata.obs["ground_truth"], adata.obs["dom"])
        C.refine_labels(adata, key="dom", n_neighbors=refine_neighbors, key_added="dom_ref")
        ari1 = metrics.ari(adata.obs["ground_truth"], adata.obs["dom_ref"])
        nmi1 = metrics.nmi(adata.obs["ground_truth"], adata.obs["dom_ref"])
        rows.append(dict(slice=slice_id, seed=seed, preset=preset, overrides=json.dumps(overrides, sort_keys=True),
                         cluster=method, n_clusters=n_clusters, ARI=ari0, NMI=nmi0, ARI_refined=ari1,
                         NMI_refined=nmi1, train_s=t_train, negatives=adata.uns["staig"]["negatives"],
                         edge_mode=adata.uns["staig"]["edge_removal"]["mode"], n_obs=adata.n_obs,
                         final_loss=float(model.history["loss"][-1])))
        print(f"{slice_id} seed={seed} {method}: ARI {ari0:.3f} -> {ari1:.3f} (refined), NMI {nmi0:.3f} -> {nmi1:.3f}, "
              f"train {t_train:.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("STAIG_DATA", "./Dataset"))
    ap.add_argument("--slices", default="151673")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--cluster", default="gmm")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--n-image-pcs", type=int, default=None)
    ap.add_argument("--pseudo-k", type=int, default=None)
    ap.add_argument("--refine-neighbors", type=int, default=15)
    ap.add_argument("--paper-configs", default=None, help="v1 YAML with one section per slice (overrides per slice)")
    ap.add_argument("--zero-center", action="store_true", help="mean-centre the scaled input (paper text; v1 did not)")
    ap.add_argument("--smooth-steps", type=int, default=0, help="smooth the embedding before clustering")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    slices = datasets.DLPFC_SLICES if args.slices == "all" else args.slices.split(",")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds != "paper" else [0]
    overrides = parse_overrides(args.override)
    methods = args.cluster.split(",")
    out = Path(args.out or Path(__file__).parent / "results" / f"dlpfc_{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    paper = None
    if args.paper_configs:
        import yaml

        with open(args.paper_configs) as fh:
            paper = yaml.safe_load(fh)
    rows = []
    for s in slices:
        ov = dict(overrides)
        if paper is not None:
            sec = paper.get(int(s), paper.get(s))
            if sec is None:
                print(f"[skip] no paper config for {s}")
                continue
            ov = {**v1_section_to_overrides(sec), **overrides}
        run_seeds = seeds if not (paper is not None and args.seeds == "paper") else [int(ov.get("seed", 0))]
        for seed in run_seeds:
            rows += run_one(s, args.root, args.preset, ov, seed, methods, args.features_dir,
                            args.refine_neighbors, args.n_image_pcs, args.pseudo_k, args.verbose,
                            args.zero_center, args.smooth_steps)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    import pandas as pd

    df = pd.DataFrame(rows)
    print("\nsummary (mean over slices and seeds):")
    cols = ["ARI", "NMI", "ARI_refined", "NMI_refined", "train_s"]
    print(df.groupby("cluster")[cols].agg(["mean", "median"]).round(3))
    print(f"written {out}")


if __name__ == "__main__":
    main()
