"""End-to-end CPU test on synthetic data: a 30x30 grid with four expression blocks."""

import anndata as ad
import numpy as np

import staig
from staig import metrics


def make_synthetic(seed=0, n_genes=200, noise=1.0, means=None, batch_factor=None):
    """30x30 grid, four spatial blocks with distinct expression programs (shared ``means`` across
    batches); ``batch_factor`` multiplies every gene by a per-gene factor (a batch effect)."""
    rng = np.random.default_rng(seed)
    g = 30
    xs, ys = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.stack([xs.ravel(), ys.ravel()], 1).astype(float) * 100
    block = (coords[:, 0] >= 1500).astype(int) * 2 + (coords[:, 1] >= 1500).astype(int)
    if means is None:
        means = np.random.default_rng(12345).random((4, n_genes)) * 3
    lam = np.exp(np.log1p(means[block]) + noise * rng.standard_normal((g * g, n_genes)) * 0.3)
    if batch_factor is not None:
        lam = lam * batch_factor[None, :]
    counts = rng.poisson(lam)
    adata = ad.AnnData(X=counts.astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["ground_truth"] = block.astype(str)
    adata.obs["ground_truth"] = adata.obs["ground_truth"].astype("category")
    return adata


def test_run_recovers_blocks_without_image():
    adata = make_synthetic()
    model = staig.run(adata, n_clusters=4, preset="visium_no_image", epochs=60, device="cpu",
                      preprocess={"n_top_genes": 150}, refine=True, verbose=False)
    assert adata.obsm["X_staig"].shape == (900, 64)
    assert len(model.history["loss"]) == 60
    res = metrics.evaluate(adata)
    assert res["ARI"] > 0.6, res


def test_determinism_cpu():
    a1, a2 = make_synthetic(), make_synthetic()
    for a in (a1, a2):
        staig.run(a, n_clusters=4, preset="visium_no_image", epochs=10, device="cpu", seed=3,
                  preprocess={"n_top_genes": 100}, verbose=False)
    assert np.allclose(a1.obsm["X_staig"], a2.obsm["X_staig"])


def test_integration_mode_mixes_batches():
    a = make_synthetic(seed=1)
    factor = np.exp(np.random.default_rng(7).standard_normal(200) * 0.5)   # per-gene batch effect
    b = make_synthetic(seed=2, batch_factor=factor)
    merged = staig.pp.concat_slices([a, b], batch_key="slice", n_top_genes=150)
    staig.graph.spatial_neighbors(merged, n_neighbors=5, batch_key="slice")
    from sklearn.decomposition import PCA

    raw_ilisi = metrics.ilisi(PCA(20).fit_transform(merged.obsm["X_staig_input"]), merged.obs["slice"].values)
    model = staig.STAIG(preset="integration", epochs=40, device="cpu", verbose=False, edge_removal="fixed")
    model.fit(merged, batch_key="slice")
    assert model._mask_mode == "batch"
    staig.cluster.cluster(merged, n_clusters=4, method="kmeans")
    res = metrics.evaluate(merged, batch_key="slice")
    assert res["ARI"] > 0.8 and res["iLISI"] > raw_ilisi, (res, raw_ilisi)


def test_run_options_consensus_smooth_auto_preset():
    adata = make_synthetic(seed=5)
    model = staig.run(adata, n_clusters=4, preset="auto", epochs=15, device="cpu", seed=1, n_seeds=3, smooth=True,
                      preprocess={"n_top_genes": 100}, verbose=False, image=None)
    assert adata.uns["staig"]["config"]["edge_removal"] == "fixed"          # auto -> merfish-like preset (<=1500 genes)
    assert adata.uns["staig"]["consensus"]["seeds"] == [1, 2, 3]
    assert "X_staig_smooth" in adata.obsm and adata.obs["staig_domain"].nunique() == 4
    assert model.config.seed == adata.uns["staig"]["consensus"]["representative_seed"]
    assert metrics.evaluate(adata)["ARI"] > 0.6


def test_sampled_negatives_path_runs_on_cpu():
    adata = make_synthetic(seed=6)
    staig.run(adata, n_clusters=4, preset="visium_no_image", epochs=15, device="cpu", n_negatives=200,
              preprocess={"n_top_genes": 100}, verbose=False)
    assert adata.uns["staig"]["n_negatives"] == 200
    assert metrics.evaluate(adata)["ARI"] > 0.6


def test_prepare_recomputes_when_parameters_change():
    adata = make_synthetic(seed=9)
    staig.pp.preprocess(adata, n_top_genes=100)
    m1 = staig.STAIG(preset="visium_no_image", epochs=2, device="cpu", verbose=False, n_neighbors=4)
    m1.prepare(adata)
    deg4 = adata.obsp["spatial_connectivities"].sum(1).mean()
    m2 = staig.STAIG(preset="visium_no_image", epochs=2, device="cpu", verbose=False, n_neighbors=8,
                     fixed_drop_prob=0.3)
    m2.prepare(adata)
    assert adata.obsp["spatial_connectivities"].sum(1).mean() > deg4
    assert np.allclose(adata.obsp["staig_edge_prob"].data, 0.3)
    adata.obsm["image_features"] = np.random.default_rng(0).standard_normal((adata.n_obs, 32)).astype(np.float32)
    m3 = staig.STAIG(preset="visium", epochs=2, device="cpu", verbose=False, n_image_pcs=8, n_pseudo_labels=5)
    m3.prepare(adata)
    assert adata.obsm["image_pcs"].shape[1] == 8 and adata.obs["staig_pseudo_label"].nunique() == 5
    m4 = staig.STAIG(preset="visium", epochs=2, device="cpu", verbose=False, n_image_pcs=4, n_pseudo_labels=3)
    m4.prepare(adata)
    assert adata.obsm["image_pcs"].shape[1] == 4 and adata.obs["staig_pseudo_label"].nunique() == 3


def test_zero_edge_graph_does_not_crash():
    adata = make_synthetic(seed=11)
    staig.pp.preprocess(adata, n_top_genes=60)
    adata.obs["b"] = [str(i) for i in range(adata.n_obs)]        # every spot alone -> no edges
    staig.graph.spatial_neighbors(adata, n_neighbors=5, batch_key="b")
    assert adata.obsp["spatial_connectivities"].nnz == 0
    model = staig.STAIG(preset="visium_no_image", epochs=2, device="cpu", verbose=False)
    model.fit(adata, batch_key="b")
    assert np.isfinite(adata.obsm["X_staig"]).all()
