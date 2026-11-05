import numpy as np
import pytest
import torch

from staig.model import GCNLayer, STAIGNet, gcn_norm_csr


def _edges(n=40, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.integers(0, n, 120)
    c = rng.integers(0, n, 120)
    keep = r != c
    r, c = r[keep], c[keep]
    return torch.tensor(np.stack([np.concatenate([r, c]), np.concatenate([c, r])]), dtype=torch.long)


def test_gcn_matches_pyg_if_available():
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GCNConv

    torch.manual_seed(0)
    n, din, dout = 40, 12, 6
    x = torch.randn(n, din)
    ei = _edges(n)
    ref = GCNConv(din, dout)
    ours = GCNLayer(din, dout)
    with torch.no_grad():
        ours.weight.copy_(ref.lin.weight.t())
        ours.bias.copy_(ref.bias)
    out_ref = ref(x, ei)
    out = ours(x, gcn_norm_csr(ei, n, x.dtype))
    assert torch.allclose(out, out_ref, atol=1e-5)


def test_gcn_norm_handles_duplicates_and_isolated_nodes():
    n = 10
    ei = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])   # duplicated edge
    a = gcn_norm_csr(ei, n, torch.float32).to_dense()
    assert torch.allclose(a, a.t())
    assert a[5, 5] == 1.0                              # isolated node keeps its self loop
    assert torch.isfinite(a).all()


def test_net_shapes_and_grad():
    torch.manual_seed(0)
    n, din = 50, 30
    x = torch.randn(n, din)
    ei = _edges(n, 1)
    net = STAIGNet(din, hidden_dim=8, proj_dim=8, n_layers=2)
    z, h = net(x, ei)
    assert z.shape == (n, 8) and h.shape == (n, 8)
    h.sum().backward()
    assert net.encoder.layers[0].weight.grad is not None
