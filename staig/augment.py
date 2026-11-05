"""Graph and feature augmentation (docs/dev/DESIGN.md).

Edge removal is defined on undirected edges {i, j}, i < j, with the removal probability taken
from entry P[i, j] (the row of the smaller index), exactly as v1 ``dropout_adj(force_undirected=True)``.
Random numbers are drawn on the CPU from a ``torch.Generator`` so that results do not depend on
the device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class EdgeSampler:
    """Holds the upper-triangular edge list of the spatial graph and its removal probabilities."""

    row: torch.Tensor      # (E,) int64, i < j, on device
    col: torch.Tensor      # (E,) int64
    prob: torch.Tensor     # (E,) float32 on CPU
    n_nodes: int
    generator: torch.Generator
    n_samples: int = 1

    @classmethod
    def from_csr(cls, adj: sp.csr_matrix, prob: sp.csr_matrix | None, n_nodes: int,
                 device: torch.device, seed: int, n_samples: int = 1, fixed_prob: float = 0.2) -> EdgeSampler:
        adj = sp.triu(sp.csr_matrix(adj), k=1).tocoo()
        row, col = adj.row.astype(np.int64), adj.col.astype(np.int64)
        if prob is None or row.size == 0:
            p = np.full(row.size, fixed_prob, dtype=np.float32)
        else:
            prob = sp.csr_matrix(prob)
            p = np.asarray(prob[row, col]).reshape(-1).astype(np.float32)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        return cls(row=torch.from_numpy(row).to(device), col=torch.from_numpy(col).to(device),
                   prob=torch.from_numpy(p), n_nodes=n_nodes, generator=g, n_samples=n_samples)

    @property
    def full_edge_index(self) -> torch.Tensor:
        return torch.stack([torch.cat([self.row, self.col]), torch.cat([self.col, self.row])])

    def sample(self) -> torch.Tensor:
        """Return a directed ``edge_index`` (2, 2E') of one augmented graph."""
        e = self.prob.numel()
        if self.n_samples == 1:
            keep = torch.rand(e, generator=self.generator) >= self.prob
        else:
            keeps = torch.rand((self.n_samples, e), generator=self.generator) >= self.prob
            keep = keeps.float().mean(0) >= 0.5
        keep = keep.to(self.row.device)
        r, c = self.row[keep], self.col[keep]
        return torch.stack([torch.cat([r, c]), torch.cat([c, r])])


def drop_features(x: torch.Tensor, drop_prob: float, generator: torch.Generator) -> torch.Tensor:
    """Zero a random subset of feature columns (same mask for all nodes), as v1 ``drop_feature``."""
    if drop_prob <= 0:
        return x
    mask = torch.rand(x.size(1), generator=generator) < drop_prob
    x = x.clone()
    x[:, mask.to(x.device)] = 0
    return x
