"""Use sampled negatives, k-means and sampled evaluation for a large CPU-only section."""

import time

from _common import check_output, finish, load, overrides, parser, positive_int, preprocessing

import staig


def main():
    ap = parser(__doc__)
    ap.add_argument("--loss-chunk-size", type=positive_int, default=256)
    ap.add_argument("--n-negatives", type=positive_int, default=1024)
    ap.add_argument("--metric-sample", type=positive_int, default=2000)
    args = ap.parse_args()
    if args.device != "cpu":
        ap.error("this example requires --device cpu")
    if args.n_clusters is None:
        ap.error("set --n-clusters to avoid an expensive silhouette search on the full dataset")
    if args.metric_sample < 2:
        ap.error("--metric-sample must be at least 2")
    check_output(args.output)
    started = time.perf_counter()
    adata = load(args.input, image_features=False)
    staig.run(adata, n_clusters=args.n_clusters, preset=args.preset, n_seeds=args.n_seeds,
              image=None, cluster_method="kmeans", refine=False,
              loss_chunk_size=args.loss_chunk_size, n_negatives=args.n_negatives,
              preprocess=preprocessing(args.log_normalized) or True, **overrides(args))
    finish(adata, args.output, started, metric_sample=args.metric_sample)


if __name__ == "__main__":
    main()
