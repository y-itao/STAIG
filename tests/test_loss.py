import numpy as np
import pytest
import torch

from staig.loss import GraphStructure, neighbor_contrastive_loss, reference_dense_loss


def _random_graph(n=60, k=4, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))
    from staig.graph import _knn_symmetric

    adj = _knn_symmetric(coords, k)
    return adj


@pytest.mark.parametrize("mask_mode", ["none", "pseudo_label", "batch"])
@pytest.mark.parametrize("chunk", [7, 64])
def test_chunked_matches_reference(mask_mode, chunk):
    torch.manual_seed(0)
    n, d = 60, 8
    adj = _random_graph(n)
    adj.sort_indices()
    graph = GraphStructure(adj.indptr, adj.indices, n, torch.device("cpu"))
    z1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    z2 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 5, (n,)) if mask_mode == "pseudo_label" else torch.randint(0, 2, (n,))
    loss = neighbor_contrastive_loss(z1, z2, graph, tau=10.0, mask_mode=mask_mode,
                                     labels=None if mask_mode == "none" else labels, chunk_size=chunk,
                                     variant="v1")
    ref = reference_dense_loss(z1, z2, torch.tensor(adj.toarray(), dtype=torch.float64), 10.0, mask_mode,
                               None if mask_mode == "none" else labels)
    assert torch.allclose(loss, ref, atol=1e-10)
    paper = neighbor_contrastive_loss(z1, z2, graph, tau=10.0, mask_mode=mask_mode,
                                      labels=None if mask_mode == "none" else labels, chunk_size=chunk)
    if mask_mode == "pseudo_label":
        assert not torch.allclose(paper, ref, atol=1e-10)      # the full denominator keeps the anchor's cross-view term
        assert abs(float(paper.detach() - ref.detach())) < 0.05
    else:
        assert torch.allclose(paper, ref, atol=1e-10)
    g1 = torch.autograd.grad(loss, z1, retain_graph=True)[0]
    g2 = torch.autograd.grad(ref, z1, retain_graph=True)[0]
    assert torch.allclose(g1, g2, atol=1e-10)
    g3 = torch.autograd.grad(paper, z1)[0]                      # log-space form: same gradients where the values agree
    if mask_mode != "pseudo_label":
        assert torch.allclose(g3, g2, atol=1e-10)
    else:
        assert torch.isfinite(g3).all() and g3.abs().max() > 0


def test_loss_is_finite_and_positive_float32():
    n, d = 100, 16
    adj = _random_graph(n, k=5, seed=1)
    adj.sort_indices()
    graph = GraphStructure(adj.indptr, adj.indices, n, torch.device("cpu"))
    z1 = torch.randn(n, d)
    z2 = z1 + 0.1 * torch.randn(n, d)
    loss = neighbor_contrastive_loss(z1, z2, graph, tau=1.0)
    assert torch.isfinite(loss) and loss > 0


@pytest.mark.parametrize("mask_mode", ["none", "pseudo_label", "batch"])
def test_sampled_negatives_exact_when_k_equals_n_and_unbiased(mask_mode):
    torch.manual_seed(1)
    n, d = 80, 8
    adj = _random_graph(n, k=4, seed=3)
    adj.sort_indices()
    graph = GraphStructure(adj.indptr, adj.indices, n, torch.device("cpu"))
    z1 = torch.randn(n, d, dtype=torch.float64)
    z2 = torch.randn(n, d, dtype=torch.float64)
    labels = torch.randint(0, 4, (n,)) if mask_mode == "pseudo_label" else torch.randint(0, 2, (n,))
    lab = None if mask_mode == "none" else labels
    exact = neighbor_contrastive_loss(z1, z2, graph, 2.0, mask_mode, lab, chunk_size=32, reduce=False)
    tiny = neighbor_contrastive_loss(z1.float(), z2.float(), graph, 0.05, mask_mode, lab, chunk_size=32)
    assert torch.isfinite(tiny)                                     # no overflow for small tau (paper range 0.1-50)
    g = torch.Generator().manual_seed(0)
    # K = N-1 is nearly exact; K = N falls back to the exact path
    full = neighbor_contrastive_loss(z1, z2, graph, 2.0, mask_mode, lab, chunk_size=32, reduce=False,
                                     n_negatives=n, generator=g)
    assert torch.allclose(full, exact, atol=1e-10)
    ests = torch.stack([neighbor_contrastive_loss(z1, z2, graph, 2.0, mask_mode, lab, chunk_size=32,
                                                  reduce=False, n_negatives=40, generator=g) for _ in range(200)])
    rel = ((ests.mean(0) - exact).abs() / exact.abs()).max()
    assert rel < 0.05, rel
