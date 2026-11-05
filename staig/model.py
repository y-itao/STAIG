"""Graph encoder and projection head in pure PyTorch (no torch_geometric).

``GCNLayer`` reproduces ``torch_geometric.nn.GCNConv`` (self loops, symmetric normalisation,
bias) using a sparse CSR adjacency built from the sampled ``edge_index``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_norm_csr(edge_index: torch.Tensor, n_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 as a sparse CSR tensor (rows sorted, duplicates summed)."""
    dev = edge_index.device
    loops = torch.arange(n_nodes, device=dev)
    row = torch.cat([edge_index[0], loops])
    col = torch.cat([edge_index[1], loops])
    val = torch.ones(row.numel(), device=dev, dtype=dtype)
    a = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n_nodes, n_nodes)).coalesce()
    row, col = a.indices()
    val = a.values()
    deg = torch.zeros(n_nodes, device=dev, dtype=dtype).index_add_(0, row, val)
    dinv = deg.pow(-0.5)
    dinv = torch.where(torch.isinf(dinv), torch.zeros_like(dinv), dinv)
    val = dinv[row] * val * dinv[col]
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (n_nodes, n_nodes)).coalesce().to_sparse_csr()


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:          # glorot, as PyG GCNConv
        bound = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        out = torch.sparse.mm(adj_norm, x @ self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class Encoder(nn.Module):
    """k GCN layers; k=1: in->out, k>=2: in->2*out->...->out (v1 ``Encoder``)."""

    def __init__(self, in_dim: int, out_dim: int, n_layers: int = 1, activation: str = "prelu"):
        super().__init__()
        assert n_layers >= 1
        dims = [in_dim] + ([2 * out_dim] * (n_layers - 1)) + [out_dim]
        self.layers = nn.ModuleList(GCNLayer(dims[i], dims[i + 1]) for i in range(n_layers))
        self.activation = nn.PReLU() if activation == "prelu" else nn.ReLU()   # shared, as in v1

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x, adj_norm))
        return x


@torch.no_grad()
def pca_init_first_layer(net: STAIGNet, x: torch.Tensor, edge_index: torch.Tensor) -> None:
    """Initialise the first GCN layer with the principal axes of the propagated input Â X, scaled so
    that every output unit has unit variance. Makes the starting point of training deterministic
    and data-driven (a smoothed-expression PCA) instead of a random projection."""
    layer = net.encoder.layers[0]
    out_dim = layer.weight.shape[1]
    adj = gcn_norm_csr(edge_index, x.size(0), x.dtype)
    xa = torch.sparse.mm(adj, x)
    xa = xa - xa.mean(0, keepdim=True)
    q = min(out_dim, min(xa.shape) - 1)
    _, _, v = torch.pca_lowrank(xa, q=q, center=False, niter=4)
    w = torch.zeros_like(layer.weight)
    w[:, :q] = v[:, :q]
    scores = xa @ w
    scale = scores.std(0, keepdim=True).clamp_min(1e-6)
    w = w / scale
    if q < out_dim:                      # fill remaining units with the glorot values already there
        w[:, q:] = layer.weight[:, q:]
    layer.weight.copy_(w)
    if layer.bias is not None:
        layer.bias.zero_()


class STAIGNet(nn.Module):
    """Encoder + two-layer projection head. ``forward`` returns (z, h)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, proj_dim: int = 64,
                 n_layers: int = 1, activation: str = "prelu"):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, n_layers, activation)
        self.fc1 = nn.Linear(hidden_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adj_norm = gcn_norm_csr(edge_index, x.size(0), x.dtype)
        z = self.encoder(x, adj_norm)
        h = self.fc2(F.elu(self.fc1(z)))
        return z, h
