"""Offline analysis of saved sweep embeddings: ensembles and post-processing variants.

    python benchmarks/analyze_embeddings.py --root ../data/zenodo/Dataset --variant base --seeds 0,1,2

For every slice it loads benchmarks/results/emb/<variant>_<slice>_<seed>.npy and evaluates:
  single        : GMM on each seed's embedding (mean over seeds)
  consensus     : co-association consensus of the per-seed GMM labels
  concat_pca    : GMM on PCA(64) of the concatenated L2-normalised per-seed embeddings
  smooth1       : GMM on the spatially smoothed embedding (per seed, mean)
  smooth1+cons  : consensus of the smoothed-embedding labels
all with and without the v1 refinement.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staig import cluster as C  # noqa: E402
from staig import datasets, graph, metrics  # noqa: E402


def evaluate(adata, labels, refine_neighbors=15):
    gt = adata.obs["ground_truth"].values
    adata.obs["_l"] = pd.Categorical(np.asarray(labels).astype(str))
    r0 = metrics.ari(gt, adata.obs["_l"].values), metrics.nmi(gt, adata.obs["_l"].values)
    ref = C.refine_labels(adata, key="_l", n_neighbors=refine_neighbors, key_added="_r")
    return r0, (metrics.ari(gt, ref), metrics.nmi(gt, ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("STAIG_DATA", "./Dataset"))
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--slices", default="all")
    ap.add_argument("--emb-dir", default=str(Path(__file__).parent / "results" / "emb"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    slices = datasets.DLPFC_SLICES if args.slices == "all" else args.slices.split(",")
    rows = []
    for s in slices:
        adata = datasets.load_dlpfc(s, root=args.root, image_features=None)
        graph.spatial_neighbors(adata, n_neighbors=5)
        k = datasets.n_clusters_for(adata)
        embs = []
        for seed in seeds:
            f = Path(args.emb_dir) / f"{args.variant}_{s}_{seed}.npy"
            if not f.exists():
                print(f"[missing] {f}")
                continue
            embs.append(np.load(f))
        if not embs:
            continue
        singles, smooth_labels, single_labels = [], [], []
        for i, e in enumerate(embs):
            lab = C.gmm_labels(e, k, random_state=seeds[i])
            single_labels.append(lab)
            singles.append(evaluate(adata, lab))
            adata.obsm["_e"] = e
            sm = C.smooth_embedding(adata, use_rep="_e")
            labs = C.gmm_labels(sm, k, random_state=seeds[i])
            smooth_labels.append(labs)
            singles.append(None)
            r0 = singles[-2]
            rows.append(dict(slice=s, variant=args.variant, post="single", seed=seeds[i],
                             ARI=r0[0][0], NMI=r0[0][1], ARI_ref=r0[1][0], NMI_ref=r0[1][1]))
            r = evaluate(adata, labs)
            rows.append(dict(slice=s, variant=args.variant, post="smooth1", seed=seeds[i],
                             ARI=r[0][0], NMI=r[0][1], ARI_ref=r[1][0], NMI_ref=r[1][1]))
        if len(embs) > 1:
            for name, labsets in [("consensus", single_labels), ("smooth1+consensus", smooth_labels)]:
                cons = C.consensus_labels(labsets, k)
                r = evaluate(adata, cons)
                rows.append(dict(slice=s, variant=args.variant, post=name, seed=-1,
                                 ARI=r[0][0], NMI=r[0][1], ARI_ref=r[1][0], NMI_ref=r[1][1]))
            cat = np.hstack([e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9) for e in embs])
            z = PCA(n_components=min(64, cat.shape[1]), random_state=0).fit_transform(cat)
            lab = C.gmm_labels(z, k, random_state=0)
            r = evaluate(adata, lab)
            rows.append(dict(slice=s, variant=args.variant, post="concat_pca", seed=-1,
                             ARI=r[0][0], NMI=r[0][1], ARI_ref=r[1][0], NMI_ref=r[1][1]))
            adata.obsm["_z"] = z
            zs = C.smooth_embedding(adata, use_rep="_z")
            lab = C.gmm_labels(zs, k, random_state=0)
            r = evaluate(adata, lab)
            rows.append(dict(slice=s, variant=args.variant, post="concat_pca+smooth1", seed=-1,
                             ARI=r[0][0], NMI=r[0][1], ARI_ref=r[1][0], NMI_ref=r[1][1]))
        df = pd.DataFrame(rows)
        last = df[df.slice == s].groupby("post")[["ARI_ref"]].mean().round(3)
        print(s, last.to_dict()["ARI_ref"], flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / f"analysis_{args.tag or args.variant}.csv"
    df.to_csv(out, index=False)
    print(df.groupby("post")[["ARI", "ARI_ref", "NMI", "NMI_ref"]].agg(["mean", "median", "std"]).round(3).to_string())
    print(f"written {out}")


if __name__ == "__main__":
    main()
