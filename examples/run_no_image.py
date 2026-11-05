"""Run one spatial section without histology (raw counts unless --log-normalized)."""

import time

from _common import check_output, finish, load, overrides, parser, preprocessing

import staig


def main():
    args = parser(__doc__).parse_args()
    check_output(args.output)
    started = time.perf_counter()
    adata = load(args.input, image_features=False)
    staig.run(adata, n_clusters=args.n_clusters, preset=args.preset, n_seeds=args.n_seeds,
              image=None, preprocess=preprocessing(args.log_normalized) or True, **overrides(args))
    finish(adata, args.output, started)


if __name__ == "__main__":
    main()
