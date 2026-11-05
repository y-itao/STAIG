"""Variant sweep on DLPFC: train once per (variant, slice, seed), evaluate several clustering
post-processings on the same embedding, and a multi-seed consensus per (variant, slice).

    python benchmarks/sweep_dlpfc.py --root ../data/zenodo/Dataset --slices all --seeds 0,1,2 \
        --variants base,paper_default --tag sweep1

Outputs benchmarks/results/sweep_<tag>.csv (one row per variant/slice/seed/post-processing) and the
embeddings under benchmarks/results/emb/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import staig  # noqa: E402
from staig import cluster as C  # noqa: E402
from staig import datasets, graph, metrics, pp  # noqa: E402

VARIANTS = {
    # name: (preset, overrides)
    "base": ("paper_151673", dict(dtype="float32")),
    "v2_default": ("default", {}),            # the shipped default (paper-form loss, PCA init, float32)
    "paper_default": ("default", dict(tau=10.0, epochs=400, n_pseudo_labels=40, n_image_pcs=16)),
    "paper_default_pcs64": ("default", dict(tau=10.0, epochs=400, n_pseudo_labels=40, n_image_pcs=64)),
    "base_resample5": ("paper_151673", dict(dtype="float32", n_graph_samples=5)),
    "base_encoder": ("paper_151673", dict(dtype="float32", embedding="encoder")),
    "base_cosine": ("paper_151673", dict(dtype="float32", lr_schedule="cosine")),
    "base_mask01": ("paper_151673", dict(dtype="float32", drop_feature_rates=(0.1, 0.1))),
    "base_tau10": ("paper_151673", dict(dtype="float32", tau=10.0)),
    "base_ep600": ("paper_151673", dict(dtype="float32", epochs=600)),
    "base_nodebias": ("paper_151673", dict(dtype="float32", debias="none")),
    "base_gene": ("paper_151673", dict(dtype="float32", edge_removal="gene", debias="none")),
    "base_fixed": ("paper_151673", dict(dtype="float32", edge_removal="fixed", debias="none")),
    "base_2layer": ("paper_151673", dict(dtype="float32", n_layers=2)),
    "base_k3": ("paper_151673", dict(dtype="float32", n_neighbors=3)),
    "base_k7": ("paper_151673", dict(dtype="float32", n_neighbors=7)),
    "base_q40": ("paper_151673", dict(dtype="float32", n_pseudo_labels=40)),
    "base_tau1": ("paper_151673", dict(dtype="float32", tau=1.0)),
    "base_tau3": ("paper_151673", dict(dtype="float32", tau=3.0)),
    "base_lr1e3": ("paper_151673", dict(dtype="float32", lr=1e-3)),
    "base_hidden128": ("paper_151673", dict(dtype="float32", hidden_dim=128, proj_dim=128)),
    "base_ep100": ("paper_151673", dict(dtype="float32", epochs=100)),
    "base_pcainit": ("paper_151673", dict(dtype="float32", init="pca")),
    "base_pcainit_tau10": ("paper_151673", dict(dtype="float32", init="pca", tau=10.0)),
    "base_pcainit_2layer": ("paper_151673", dict(dtype="float32", init="pca", n_layers=2)),
}


def post_variants(adata, emb, seed, n_clusters, methods, posts=("smooth1", "smooth2", "l2")):
    """Yield (name, labels) for several clustering post-processings of one embedding."""
    out = {}
    adata.obsm["_e"] = emb
    for m in methods:
        C.cluster(adata, n_clusters=n_clusters, method=m, use_rep="_e", key_added="_d", random_state=seed)
        out[f"{m}"] = adata.obs["_d"].values.copy()
    if "smooth1" in posts:
        sm = C.smooth_embedding(adata, use_rep="_e", n_steps=1, alpha=0.5)
        adata.obsm["_s"] = sm.astype(np.float32)
        C.cluster(adata, n_clusters=n_clusters, method="gmm", use_rep="_s", key_added="_d", random_state=seed)
        out["gmm_smooth1"] = adata.obs["_d"].values.copy()
    if "smooth2" in posts:
        sm2 = C.smooth_embedding(adata, use_rep="_e", n_steps=2, alpha=0.5)
        adata.obsm["_s2"] = sm2.astype(np.float32)
        C.cluster(adata, n_clusters=n_clusters, method="gmm", use_rep="_s2", key_added="_d", random_state=seed)
        out["gmm_smooth2"] = adata.obs["_d"].values.copy()
    if "l2" in posts:
        adata.obsm["_n"] = (emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        C.cluster(adata, n_clusters=n_clusters, method="gmm", use_rep="_n", key_added="_d", random_state=seed)
        out["gmm_l2"] = adata.obs["_d"].values.copy()
    return out


def score(adata, labels, refine_neighbors=15):
    gt = adata.obs["ground_truth"].values
    adata.obs["_lab"] = labels
    r0 = (metrics.ari(gt, labels), metrics.nmi(gt, labels))
    ref = C.refine_labels(adata, key="_lab", n_neighbors=refine_neighbors, key_added="_ref")
    r1 = (metrics.ari(gt, ref), metrics.nmi(gt, ref))
    return r0, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("STAIG_DATA", "./Dataset"))
    ap.add_argument("--slices", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--methods", default="gmm", help="comma list of base clusterers: gmm,mclust,kmeans")
    ap.add_argument("--posts", default="smooth1,smooth2,l2",
                    help="extra post-processings (subset of smooth1,smooth2,l2)")
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()
    slices = datasets.DLPFC_SLICES if args.slices == "all" else args.slices.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    methods = args.methods.split(",")
    out = Path(__file__).parent / "results" / f"sweep_{args.tag}.csv"
    emb_dir = Path(__file__).parent / "results" / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def flush():
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for s in slices:
        base_adata = datasets.load_dlpfc(s, root=args.root)
        pp.preprocess(base_adata, n_top_genes=3000)
        n_clusters = datasets.n_clusters_for(base_adata)
        for v in variants:
            preset, ov = VARIANTS[v]
            label_sets = {}
            for seed in seeds:
                adata = base_adata.copy()
                cfg = staig.STAIGConfig.from_preset(preset).update(seed=seed, verbose=False, **ov)
                graph.image_pcs(adata, n_comps=cfg.n_image_pcs)
                graph.spatial_neighbors(adata, n_neighbors=cfg.n_neighbors)
                t0 = time.perf_counter()
                staig.STAIG(cfg).fit(adata)
                t_train = time.perf_counter() - t0
                emb = adata.obsm["X_staig"]
                np.save(emb_dir / f"{v}_{s}_{seed}.npy", emb)
                post_names = tuple(p for p in args.posts.split(",") if p)
                posts = post_variants(adata, emb, seed, n_clusters, methods, post_names)
                for pname, lab in posts.items():
                    (a0, n0), (a1, n1) = score(adata, lab)
                    label_sets.setdefault(pname, []).append(lab)
                    rows.append(dict(variant=v, slice=s, seed=seed, post=pname, ARI=a0, NMI=n0, ARI_ref=a1,
                                     NMI_ref=n1, train_s=round(t_train, 2), n_obs=adata.n_obs))
                    print(f"{v:22s} {s} seed={seed} {pname:12s} ARI {a0:.3f}/{a1:.3f} "
                          f"NMI {n0:.3f}/{n1:.3f} ({t_train:.1f}s)", flush=True)
            # consensus over seeds for each post-processing
            if len(seeds) > 1:
                for pname, labs in label_sets.items():
                    cons = C.consensus_labels([np.asarray(lab) for lab in labs], n_clusters)
                    (a0, n0), (a1, n1) = score(adata, cons.astype(str))
                    rows.append(dict(variant=v, slice=s, seed=-1, post=f"{pname}+consensus", ARI=a0, NMI=n0,
                                     ARI_ref=a1, NMI_ref=n1, train_s=float("nan"), n_obs=adata.n_obs))
                    print(f"{v:22s} {s} consensus {pname:12s} ARI {a0:.3f}/{a1:.3f} NMI {n0:.3f}/{n1:.3f}", flush=True)
            flush()
    import pandas as pd

    df = pd.DataFrame(rows)
    summ = df.groupby(["variant", "post"])[["ARI", "ARI_ref", "NMI", "NMI_ref"]].agg(["mean", "median", "std"]).round(3)
    print(summ.to_string())
    print(f"written {out}")


if __name__ == "__main__":
    main()
