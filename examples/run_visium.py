"""Run a Visium section using supplied H&E features or a full-resolution image."""

import time

from _common import check_output, finish, load, overrides, parser, positive_int, preprocessing

import staig
from staig.img.patches import resolve_image_path


def main():
    ap = parser(__doc__)
    ap.add_argument("--image", help="full-resolution H&E file; overrides loaded image features")
    ap.add_argument("--image-epochs", type=positive_int, default=1)
    args = ap.parse_args()
    check_output(args.output)
    started = time.perf_counter()
    adata = load(args.input, image_features=args.image is None)
    if args.image is not None or "image_features" not in adata.obsm:
        image_path = args.image or resolve_image_path(adata)
        if image_path is None:
            ap.error("provide --image with full-resolution H&E, or aligned image_features / embeddings.npy")
        staig.img.extract_image_features(
            adata, image=image_path, epochs=args.image_epochs, device=args.device,
            num_workers=0, n_jobs=1,
        )
    staig.run(adata, n_clusters=args.n_clusters, preset=args.preset, n_seeds=args.n_seeds,
              image=None, preprocess=preprocessing(args.log_normalized) or True, **overrides(args))
    finish(adata, args.output, started)


if __name__ == "__main__":
    main()
