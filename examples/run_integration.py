"""Integrate two or more slices without coordinate alignment."""

import time

from _common import check_output, finish, load, overrides, parser, preprocessing

import staig


def main():
    ap = parser(__doc__, integration=True)
    args = ap.parse_args()
    if len(args.input) < 2:
        ap.error("--input needs at least two slices")
    check_output(args.output)
    started = time.perf_counter()
    slices = [load(path) for path in args.input]
    merged = staig.pp.concat_slices(
        slices, batch_key="slice", batch_categories=[f"slice_{i}" for i in range(len(slices))],
        **preprocessing(args.log_normalized),
    )
    staig.run(merged, n_clusters=args.n_clusters, preset=args.preset, n_seeds=args.n_seeds,
              batch_key="slice", image=None, preprocess=False, refine=False, **overrides(args))
    finish(merged, args.output, started, batch_key="slice")


if __name__ == "__main__":
    main()
