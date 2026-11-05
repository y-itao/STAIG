"""Robustness and edge-case tests: serialisation, input validation, numerics, metrics and I/O."""

import io
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

import staig
from staig import cluster, graph, metrics, pp
from staig._utils import unique_path, write_h5ad
from staig.loss import GraphStructure, neighbor_contrastive_loss


def small(n=32, genes=20, seed=0):
    from test_pipeline import make_synthetic

    a = make_synthetic(seed=seed, n_genes=genes)[:n].copy()
    a.obsm["X_staig_input"] = np.log1p(a.X).astype(np.float32)
    return a


# ----- selection scores
def test_selection_scores_are_string_keyed_and_h5ad_writable(tmp_path):
    a = small(20)
    a.obsm["X_staig"] = a.obsm["X_staig_input"]
    cluster.cluster(a, n_clusters=None, n_clusters_range=[2, 3], method="kmeans")
    scores = a.uns["staig"]["n_clusters_selection"]
    assert set(scores) == {"2", "3"} and a.uns["staig"]["n_clusters_selection_method"] == "kmeans"
    a.write_h5ad(tmp_path / "x.h5ad")
    back = ad.read_h5ad(tmp_path / "x.h5ad")
    assert back.uns["staig"]["n_clusters_selection"]["2"] == pytest.approx(scores["2"])


def test_write_h5ad_is_atomic_and_refuses_overwrite(tmp_path):
    a = small(5)
    target = tmp_path / "out" / "a.h5ad"
    write_h5ad(a, target)
    assert target.exists() and not list(target.parent.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        write_h5ad(a, target)
    write_h5ad(a, target, overwrite=True)
    assert unique_path(target) == target.with_name("a_1.h5ad")


# ----- GUI counts choice
def test_gui_normalised_choice_skips_normalisation(monkeypatch, tmp_path):
    from staig import gui

    p = tmp_path / "syn.h5ad"
    small().write_h5ad(p)
    seen = {}

    def fake_run(adata, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr("staig.run", fake_run)
    for choice, expected in [("normalised", {"normalize": False, "log1p": False}), ("raw counts", {}),
                             ("auto", {})]:                      # the synthetic data are integer counts
        with pytest.raises(RuntimeError, match="stop"):
            gui.run_pipeline(str(p), "", 4, "auto", "", 1, False, 1, choice, str(tmp_path))
        assert seen["preprocess"] == expected, choice


# ----- label alignment
def test_load_labels_aligns_partially_overlapping_barcodes():
    a = small(10)
    s = pd.Series(np.arange(10), index=list(a.obs_names)[::-1][:-1] + ["foreign"])
    with pytest.warns(UserWarning, match="not in adata.obs_names"):
        pp.load_labels(a, s, drop_na=False)
    got = a.obs["ground_truth"]
    expected = s.reindex(a.obs_names)
    assert got.isna().values[0] and got.notna().sum() == 9
    assert np.array_equal(got.dropna().astype(int).values, expected.dropna().astype(int).values)
    b = small(10)                                     # obs_names "0".."9": an array aligns by identity
    pp.load_labels(b, np.arange(10), drop_na=False)
    assert b.obs["ground_truth"].astype(int).tolist() == list(range(10))
    c = small(10)
    c.obs_names = [f"bc{i}" for i in range(10)]
    with pytest.warns(UserWarning, match="by position"):
        pp.load_labels(c, np.arange(10)[::-1], drop_na=False)
    assert c.obs["ground_truth"].astype(int).tolist() == list(range(9, -1, -1))


# ----- concat validation
def test_concat_slices_validates_batch_names():
    from test_pipeline import make_synthetic

    a = make_synthetic(n_genes=100)
    with pytest.raises(ValueError, match="batch_categories"):
        pp.concat_slices([a, a.copy()], batch_categories=["only"], n_top_genes=80)
    with pytest.raises(ValueError, match="unique"):
        pp.concat_slices([a, a.copy()], batch_categories=["x", "x"], n_top_genes=80)
    m = pp.concat_slices([a, a.copy()], batch_categories=["x", "y"], n_top_genes=80)
    assert m.n_obs == 2 * a.n_obs and list(m.obs["slice"].cat.categories) == ["x", "y"]


# ----- unknown batch key
def test_unknown_batch_key_raises_everywhere():
    a = small(10)
    a.obs["slice"] = ["a", "b"] * 5
    m = staig.STAIG(preset="integration", device="cpu", edge_removal="fixed", verbose=False)
    with pytest.raises(KeyError, match="typo"):
        m.prepare(a, batch_key="typo")
    with pytest.raises(KeyError, match="typo"):
        graph.spatial_neighbors(a, n_neighbors=2, batch_key="typo")
    with pytest.raises(KeyError, match="typo"):
        staig.run(a, n_clusters=2, batch_key="typo", epochs=1, device="cpu", image=None, preprocess=False,
                  verbose=False)
    m.prepare(a, batch_key="slice")
    A = a.obsp["spatial_connectivities"].tocoo()
    assert m._mask_mode == "batch" and (a.obs["slice"].values[A.row] == a.obs["slice"].values[A.col]).all()


# ----- preset mirrors
def test_preset_yaml_mirrors_match_source():
    from staig._config import PRESETS, STAIGConfig

    d = Path(staig.__file__).parent / "presets"
    for name in PRESETS:
        assert STAIGConfig.from_yaml(d / f"{name}.yaml") == STAIGConfig.from_preset(name), name
    assert STAIGConfig.from_yaml(d / "paper_151673.yaml").loss_variant == "v1"


# ----- exact small-tau loss
def test_paper_loss_is_exact_for_tiny_tau():
    A = sp.block_diag([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], format="csr")
    g = GraphStructure(A.indptr, A.indices, 4, torch.device("cpu"))
    u = torch.eye(4, requires_grad=True)
    v = u.roll(2, 0)
    loss = neighbor_contrastive_loss(u, v, g, 0.001)
    loss.backward()
    analytic = torch.logaddexp(torch.tensor(1000.0), torch.log(torch.tensor(6.0)))   # log(exp(1/tau) + 6)
    assert abs(float(loss.detach()) - float(analytic)) < 1e-3
    assert torch.isfinite(u.grad).all() and float(u.grad.abs().max()) > 1.0


# ----- sampled self terms
@pytest.mark.parametrize("variant", ["full", "v1"])
def test_sampled_loss_handles_zero_vectors_and_singleton_batches(variant):
    a = small(8)
    graph.spatial_neighbors(a, n_neighbors=2)
    A = a.obsp["spatial_connectivities"]
    g = GraphStructure(A.indptr, A.indices, 8, torch.device("cpu"))
    z = torch.zeros((8, 20), dtype=torch.float64)
    exact = float(neighbor_contrastive_loss(z, z, g, 1.0, variant=variant))
    sampled = float(neighbor_contrastive_loss(z, z, g, 1.0, n_negatives=4, variant=variant,
                                              generator=torch.Generator().manual_seed(0)))
    assert exact == pytest.approx(np.log(15.0)) and abs(exact - sampled) < 0.02
    A0 = sp.csr_matrix((8, 8))
    g0 = GraphStructure(A0.indptr, A0.indices, 8, torch.device("cpu"))
    ones, labels = torch.ones((8, 4)), torch.arange(8)
    exact = float(neighbor_contrastive_loss(ones, ones, g0, 10.0, "batch", labels, variant=variant))
    sampled = float(neighbor_contrastive_loss(ones, ones, g0, 10.0, "batch", labels, n_negatives=3, variant=variant,
                                              generator=torch.Generator().manual_seed(0)))
    assert exact == pytest.approx(0.0, abs=1e-6) and sampled == pytest.approx(0.0, abs=1e-6)


# ----- stale image PCs
def test_prepare_recomputes_image_pcs_on_batch_or_feature_change():
    a = small(32)
    a.obsm["image_features"] = np.asarray(a.X[:, :6], dtype=np.float32)
    m = staig.STAIG(device="cpu", n_image_pcs=4, n_pseudo_labels=3, verbose=False)
    m.prepare(a)
    assert m._mask_mode == "pseudo_label"
    assert a.uns["staig"]["pseudo_labels"]["source"] == graph.obsm_fingerprint(a, "image_pcs")
    a.obsm["image_features"] = a.obsm["image_features"][::-1].copy()     # same shape, other content
    before = a.obsm["image_pcs"].copy()
    m.prepare(a)
    assert not np.array_equal(before, a.obsm["image_pcs"])
    assert a.uns["staig"]["pseudo_labels"]["source"] == graph.obsm_fingerprint(a, "image_pcs")
    single = a.obsm["image_pcs"].copy()
    a.obs["slice"] = ["a"] * 16 + ["b"] * 16
    m.prepare(a, batch_key="slice")                                      # per-slice PCA for integration
    assert not np.array_equal(single, a.obsm["image_pcs"]) and m._mask_mode == "batch"
    ref = a.copy()
    graph.image_pcs(ref, n_comps=4, batch_key="slice")
    assert np.allclose(ref.obsm["image_pcs"], a.obsm["image_pcs"])
    a.obs["slice"] = ["a"] * 8 + ["b"] * 24                              # same key, other membership
    before = a.obsm["image_pcs"].copy()
    m.prepare(a, batch_key="slice")
    assert not np.array_equal(before, a.obsm["image_pcs"])
    a.obsm["spatial"] = a.obsm["spatial"][::-1].copy()                   # other coordinates -> new graph
    before = a.obsp["spatial_connectivities"].copy()
    m.prepare(a, batch_key="slice")
    assert (before != a.obsp["spatial_connectivities"]).nnz > 0


# ----- consensus state
def test_consensus_keeps_embedding_smoothing_and_metadata_consistent():
    a = small(80)
    m = staig.run(a, n_clusters=3, n_seeds=3, epochs=2, device="cpu", image=None, preprocess=False, refine=True,
                  verbose=False, hidden_dim=8, proj_dim=8)
    rep = a.uns["staig"]["consensus"]["representative_seed"]
    assert m.config.seed == rep and a.uns["staig"]["config"]["seed"] == rep
    assert np.allclose(cluster.smooth_embedding(a), a.obsm["X_staig_smooth"], atol=1e-5)
    info = a.uns["staig"]["cluster"]
    assert info["refine"] is True and info["consensus"] is True and info["n_clusters_found"] <= 3


# ----- float64 checkpoints
def test_float64_checkpoint_roundtrip_is_exact():
    m = staig.STAIG(preset="visium_no_image", epochs=2, device="cpu", dtype="float64", verbose=False, hidden_dim=8,
                    proj_dim=8).fit(small())
    f = io.BytesIO()
    m.save(f)
    f.seek(0)
    r = staig.STAIG.load(f, device="cpu")
    for p, q in zip(m.net.parameters(), r.net.parameters()):
        assert q.dtype == torch.float64 and torch.equal(p, q)


# ----- CLI seed / overwrite
def test_cli_keeps_config_seed_unless_overridden(monkeypatch, tmp_path):
    from staig import cli

    seen = {}

    def fake_run(adata, **kw):
        seen.clear()
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr("staig.run", fake_run)
    p = tmp_path / "s.h5ad"
    small().write_h5ad(p)
    y = tmp_path / "c.yaml"
    staig.STAIGConfig(seed=39788).to_yaml(y)
    with pytest.raises(RuntimeError, match="stop"):
        cli.main(["run", str(p), "--config", str(y)])
    assert "seed" not in seen and seen["config"].seed == 39788
    with pytest.raises(RuntimeError, match="stop"):
        cli.main(["run", str(p), "--config", str(y), "--seed", "5"])
    assert seen["seed"] == 5
    out = tmp_path / "exists.h5ad"
    out.write_bytes(b"")
    seen.clear()
    with pytest.raises(FileExistsError):
        cli.main(["run", str(p), "--output", str(out)])
    assert not seen                                                       # refused before training


# ----- LISI / BatchKL
def test_lisi_excludes_self_by_index_and_metrics_ignore_missing_labels():
    x = np.zeros((6, 2))
    x[3:, 0] = [1, 2, 3]
    lab = np.array(["a", "b", "b", "a", "a", "a"])
    assert metrics.lisi(x, lab, perplexity=2, n_neighbors=2)[0] == pytest.approx(1.0)
    a = small(12)
    b = np.array(["a"] * 6 + [np.nan] * 6, dtype=object)
    full = metrics.batch_kl(a.obsm["X_staig_input"][:6], b[:6], replicates=1, n_cells=6)
    assert metrics.batch_kl(a.obsm["X_staig_input"], b, replicates=1, n_cells=12) == pytest.approx(full)
    li = metrics.lisi(a.obsm["X_staig_input"], b, perplexity=2)
    assert np.isnan(li[6:]).all() and np.allclose(li[:6], 1.0)


# ----- n_clusters validation
def test_cluster_validates_n_clusters_and_evaluate_guards_silhouette():
    a = small(8)
    a.obsm["X_staig"] = a.obsm["X_staig_input"]
    with pytest.raises(ValueError, match="n_clusters"):
        cluster.cluster(a, n_clusters=99)
    with pytest.raises(ValueError, match="n_clusters"):
        cluster.cluster(a, n_clusters=1)
    cluster.cluster(a, n_clusters=8, method="kmeans")
    assert a.uns["staig"]["cluster"]["n_clusters_found"] == 8
    assert "SC" not in metrics.evaluate(a)                               # silhouette undefined for k = n


# ----- patches outside image
def test_extract_patches_wholly_outside_image_is_zero_padded():
    from staig.img import patches

    a = small(2)
    a.obsm["spatial"] = np.array([[-20.0, 10.0], [16.0, 16.0]])
    with pytest.warns(UserWarning, match="zero padded"):
        out = patches.extract_patches(a, image=np.full((32, 32, 3), 7, np.uint8), patch_px=8, patch_size=8)
    assert out.shape == (2, 8, 8, 3) and out[0].max() == 0 and out[1].min() == 7


# ----- selection arguments
def test_select_n_clusters_rejects_unsupported_method_or_metric():
    x = small(16).obsm["X_staig_input"]
    with pytest.raises(ValueError, match="method"):
        cluster.select_n_clusters(x, [2, 3], method="leiden")
    with pytest.raises(ValueError, match="metric"):
        cluster.select_n_clusters(x, [2, 3], metric="nonsense")
    best, scores = cluster.select_n_clusters(x, [1, 2, 99], method="kmeans")
    assert best == 2 and list(scores) == [2]


# ----- gene-mode PCA as in v1
def test_gene_edge_probabilities_use_a_joint_pca():
    a = small(40)
    a.obs["slice"] = ["a"] * 20 + ["b"] * 20
    graph.spatial_neighbors(a, n_neighbors=3, batch_key="slice")
    graph.edge_removal_probabilities(a, mode="gene", n_gene_pcs=4, batch_key="slice")
    joint = a.obsp["staig_edge_prob"].copy()
    graph.edge_removal_probabilities(a, mode="gene", n_gene_pcs=4)        # no batch key: identical
    assert np.allclose(joint.toarray(), a.obsp["staig_edge_prob"].toarray())


# ----- deterministic ties
def test_knn_ties_are_broken_by_index():
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)      # square grid: 4 equidistant neighbours
    adj = graph._knn_symmetric(coords, 3).tocsr()
    knn = graph._knn_symmetric(coords[::-1], 3).tocsr()                # reversed order -> mirrored choice
    assert adj.nnz == knn.nnz and (adj != adj.T).nnz == 0 and adj.diagonal().sum() == 0
    i = 7                                                              # interior spot (1, 1): neighbours 1, 6, 8, 13
    nb = np.flatnonzero(graph._knn_symmetric(coords, 3).toarray()[i])
    assert {1, 6, 8}.issubset(set(nb))                                 # the three lowest indices among the ties
    shuffled = coords[np.random.default_rng(0).permutation(36)]
    a1 = graph._knn_symmetric(shuffled, 3).toarray()
    a2 = graph._knn_symmetric(shuffled.copy(), 3).toarray()
    assert np.array_equal(a1, a2)


# ----- identifiers, caches, state, metrics, files
def test_load_labels_aligns_integer_barcodes():
    a = small(10)                                                           # obs_names "0" ... "9"
    s = pd.Series(np.arange(10), index=pd.Index(range(9, -1, -1)))
    pp.load_labels(a, s, drop_na=False)
    assert a.obs["ground_truth"].astype(int).tolist() == list(range(9, -1, -1))
    b = small(10)
    with pytest.warns(UserWarning, match="by position"):
        pp.load_labels(b, pd.Series(np.arange(10), index=[f"x{i}" for i in range(10)]), drop_na=False)


def test_fingerprint_is_dtype_exact_and_sparse_safe():
    a = small(10)
    a.obsm["image_pcs"] = np.ones((10, 2), np.float64)
    a.obsm["image_pcs"][:, 0] += np.arange(10) * 1e-9
    m = staig.STAIG(device="cpu", n_neighbors=2, n_pseudo_labels=3, verbose=False, debias="none")
    m.prepare(a)
    fp = graph.obsm_fingerprint(a, "image_pcs")
    a.obsm["image_pcs"][0, 0] += 2e-8                                       # below float32 resolution
    assert graph.obsm_fingerprint(a, "image_pcs") != fp
    m.prepare(a)
    fresh = a.copy()
    graph.edge_removal_probabilities(fresh, mode="image")
    assert np.allclose(a.obsp["staig_edge_prob"].toarray(), fresh.obsp["staig_edge_prob"].toarray())
    a.obsm["image_features"] = sp.csr_matrix(np.eye(10, 6, dtype=np.float32))
    fp1 = graph.obsm_fingerprint(a, "image_features")
    a.obsm["image_features"] = sp.csr_matrix(2 * np.eye(10, 6, dtype=np.float32))
    assert graph.obsm_fingerprint(a, "image_features") != fp1


def test_run_clears_results_of_an_earlier_run():
    a = small(80)
    kw = dict(n_clusters=3, epochs=2, device="cpu", image=None, preprocess=False, verbose=False, hidden_dim=8,
              proj_dim=8)
    staig.run(a, n_seeds=3, seed=0, refine=True, **kw)
    m = staig.run(a, n_seeds=1, seed=20, refine=False, smooth=False, **kw)
    assert "consensus" not in a.uns["staig"] and "X_staig_smooth" not in a.obsm
    assert a.uns["staig"]["cluster"]["consensus"] is False and a.uns["staig"]["cluster"]["refine"] is False
    assert m.config.seed == 20 and a.uns["staig"]["config"]["seed"] == 20
    staig.run(a, n_seeds=1, seed=30, cluster_method=None, **kw)
    assert "staig_domain" not in a.obs and "cluster" not in a.uns["staig"]


def test_knn_lowest_index_rule_holds_beyond_the_tie_window():
    coords = np.zeros((81, 2))
    coords[:80] = np.repeat([[100, 0], [-100, 0], [0, 100], [0, -100]], 20, axis=0)
    for k in (1, 3, 5):
        A = graph._knn_symmetric(coords, k)
        assert A[80].indices.tolist() == list(range(k)), k           # 80 equidistant spots: lowest indices win
        assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0
    dup = np.zeros((20, 2))                                             # 20 identical coordinates, k = 5
    A = graph._knn_symmetric(dup, 5)
    assert A[19].indices.tolist()[:5] == [0, 1, 2, 3, 4] and A[0].indices.tolist()[:5] == [1, 2, 3, 4, 5]


def test_lisi_single_neighbour_and_batch_kl_all_missing():
    a = small(6)
    x = a.obsm["X_staig_input"].copy()
    x[1] = x[0]
    labels = np.array(["a", "a", None, None, None, None], dtype=object)
    li = metrics.lisi(x, labels, perplexity=2)
    assert li[0] == pytest.approx(1.0) and li[1] == pytest.approx(1.0) and np.isnan(li[2:]).all()
    with pytest.raises(ValueError, match="valid batch"):
        metrics.batch_kl(x, [None] * 6, replicates=1)


def test_write_h5ad_never_clobbers_a_file_created_meanwhile(tmp_path, monkeypatch):
    import anndata

    target = tmp_path / "raced.h5ad"
    original = anndata.AnnData.write_h5ad

    def delayed(self, path, *args, **kw):
        original(self, path, *args, **kw)
        target.write_bytes(b"other writer")

    monkeypatch.setattr(anndata.AnnData, "write_h5ad", delayed)
    with pytest.raises(FileExistsError):
        write_h5ad(small(5), target)
    assert target.read_bytes() == b"other writer" and not list(tmp_path.glob(".*.tmp"))
    monkeypatch.setattr(anndata.AnnData, "write_h5ad", original)
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(IsADirectoryError):
        write_h5ad(small(5), d, overwrite=True)


# ===================================================================== robustness of HVG selection
def test_preprocess_recovers_from_seurat_v3_singularity(monkeypatch):
    import scanpy as sc
    from test_pipeline import make_synthetic

    a = make_synthetic(seed=3, n_genes=120)
    a.X[:, 60:] = 0                                     # half of the genes never detected
    calls = {"n": 0, "fail": "first"}
    original = sc.pp.highly_variable_genes

    def flaky(adata, **kw):
        calls["n"] += 1
        v3 = kw.get("flavor") == "seurat_v3"
        if v3 and (calls["fail"] == "always" or (calls["fail"] == "first" and calls["n"] == 1)):
            raise ValueError(b"There are other near singularities as well. 0.03")
        return original(adata, **kw)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", flaky)
    pp.preprocess(a, n_top_genes=20)
    info = a.uns["staig"]["preprocess"]
    assert info["flavor"] == "seurat_v3" and "detected in >= 3" in info["hvg_note"]
    assert a.var["highly_variable"].sum() == 20 and not a.var["highly_variable"][60:].any()
    assert a.obsm["X_staig_input"].shape[1] == 20 and a.n_vars == 120     # var not subset
    b = make_synthetic(seed=4, n_genes=120)
    calls["fail"] = "always"                             # every seurat_v3 call fails -> seurat flavour
    with pytest.warns(UserWarning, match="flavor='seurat'"):
        pp.preprocess(b, n_top_genes=20)
    assert b.uns["staig"]["preprocess"]["flavor"] == "seurat"
    assert 20 <= b.obsm["X_staig_input"].shape[1] <= 25             # scanpy's seurat flavour keeps ties
