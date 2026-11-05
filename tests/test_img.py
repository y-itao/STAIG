import anndata as ad
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from staig.img import filters, patches  # noqa: E402


def _v1_filter(gray_png_rgb: np.ndarray, lower=245, upper=275, size=512):
    """Port of the v1 notebook ``process_image`` (reads PNG as grayscale, writes float32 PNG)."""
    image = cv2.cvtColor(gray_png_rgb, cv2.COLOR_RGB2GRAY)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    mask = np.zeros(image.shape, np.uint8)
    mask[lower:upper, lower:upper] = 1
    f_ishift = np.fft.ifftshift(fshift * mask)
    image_filtered = np.abs(np.fft.ifft2(f_ishift))
    image_filtered = cv2.GaussianBlur(image_filtered, (15, 15), 0)
    rgb = cv2.cvtColor(np.float32(image_filtered), cv2.COLOR_GRAY2RGB)
    ok, buf = cv2.imencode(".png", rgb)           # cv2.imwrite(float32) round trip
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def test_filter_matches_v1_png_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(3):
        patch = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        ours = filters.filter_patch(patch)
        ref = _v1_filter(patch)
        assert ours.shape == ref.shape == (512, 512, 3)
        assert np.array_equal(ours, ref)


def test_extract_patches_centres_and_padding():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[100:110, 200:210] = 255                      # a bright square at (y=100..110, x=200..210)
    adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    adata.obsm["spatial"] = np.array([[205.0, 105.0], [2.0, 2.0]])   # (x, y)
    out = patches.extract_patches(adata, image=img, patch_px=20, patch_size=20)
    assert out.shape == (2, 20, 20, 3)
    assert out[0, 5:15, 5:15].min() == 255 and out[0, 0, 0, 0] == 0
    assert out[1].sum() == 0                          # fully padded corner patch


def test_resize_and_cache(tmp_path):
    img = np.random.default_rng(1).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    adata.obsm["spatial"] = np.array([[100.0, 100.0], [50.0, 60.0], [150.0, 120.0]])
    out = patches.extract_patches(adata, image=img, patch_px=40, patch_size=64, cache_dir=tmp_path)
    assert out.shape == (3, 64, 64, 3) and (tmp_path / "patches.npy").exists()
    filt = filters.filter_patches(out, size=64, band=(28, 36))
    assert filt.shape == (3, 64, 64, 3) and filt.dtype == np.uint8
