"""Neighbour contrastive loss (docs/dev/DESIGN.md), chunked and checkpointed.

For anchors i (rows of view *a*) against view *b*, with ``S_aa = exp(cos(z_a, z_a) / tau)`` and
``S_ab = exp(cos(z_a, z_b) / tau)``::

    pos_i = S_ab[i,i] + sum_j A_ij S_aa[i,j] + sum_j A_ij S_ab[i,j]
    den_i = S_ab[i,i] + sum_{j != i} M_ij (S_aa[i,j] + S_ab[i,j])
    l_i   = -log( pos_i / den_i / (2 deg_i + 1) )

``M`` is the negative mask: all ones (``"none"``), ``1[y_i != y_j]`` for image pseudo-labels
(``"pseudo_label"``) or ``1[batch_i == batch_j]`` (``"batch"``). Positives are the spatial neighbours
of the original graph. The total loss is the mean over both directions.

``variant="full"`` (default) evaluates ``l_i`` in log space (``logsumexp`` over the positive and the
masked negative sets), so it is exact for any ``tau``. ``variant="v1"`` uses the denominator
convention of STAIG 1.x (the anchor's own cross-view term is not counted, and a pseudo-label mask
also removes the intra-view self term) in linear space, for runs that must match version 1 exactly.

``F.normalize`` keeps the loss finite for an all-zero embedding row.

Memory is O(chunk_size x N): the N x N similarity matrices are never materialised in full and
each chunk is recomputed in the backward pass (``torch.utils.checkpoint``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GraphStructure:
    """Device-resident CSR representation of the (binary, symmetric) spatial graph."""

    def __init__(self, indptr, indices, n_nodes: int, device: torch.device):
        self.indptr = torch.as_tensor(indptr, dtype=torch.int64, device=device)
        self.indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
        self.n_nodes = int(n_nodes)
        self.deg = (self.indptr[1:] - self.indptr[:-1]).to(torch.float32)

    def dense_rows(self, start: int, end: int, dtype: torch.dtype) -> torch.Tensor:
        """Dense (end-start, N) block of adjacency rows."""
        s, e = int(self.indptr[start]), int(self.indptr[end])
        cols = self.indices[s:e]
        counts = self.indptr[start + 1:end + 1] - self.indptr[start:end]
        rows = torch.repeat_interleave(torch.arange(end - start, device=cols.device), counts)
        block = torch.zeros((end - start, self.n_nodes), dtype=dtype, device=cols.device)
        block[rows, cols] = 1
        return block


def _mask(labels: torch.Tensor, start: int, end: int, mask_mode: str, cols: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean negative mask M for the anchors start..end against all nodes (or against ``cols``)."""
    lab_j = labels if cols is None else labels[cols]
    if mask_mode == "pseudo_label":
        return labels[start:end, None] != lab_j[None, :]
    if mask_mode == "batch":
        return labels[start:end, None] == lab_j[None, :]
    raise ValueError(mask_mode)


def _segment_sum(vals: torch.Tensor, counts: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Row sums of ``vals`` grouped by consecutive segments of lengths ``counts`` (deterministic)."""
    if vals.numel() == 0:
        return torch.zeros(n_rows, dtype=vals.dtype, device=vals.device)
    return torch.segment_reduce(vals, "sum", lengths=counts, axis=0, unsafe=True)


def _positive_logits(a: torch.Tensor, za: torch.Tensor, zb: torch.Tensor, graph: GraphStructure, start: int,
                     end: int, tau: float):
    """Logits of the positive set of every anchor, gathered from the CSR neighbour lists (no dense N columns).

    Returns ``(self_ab, nb_aa, nb_ab, counts)``: the cross-view self logit (c,), the intra-/cross-view
    neighbour logits (E_c,) in CSR order and the number of neighbours per anchor (c,).
    """
    s, e = int(graph.indptr[start]), int(graph.indptr[end])
    cols = graph.indices[s:e]
    counts = graph.indptr[start + 1:end + 1] - graph.indptr[start:end]
    rows = torch.repeat_interleave(torch.arange(end - start, device=za.device), counts)
    self_ab = (a * zb[start:end]).sum(1) / tau
    nb_aa = (a[rows] * za[cols]).sum(1) / tau
    nb_ab = (a[rows] * zb[cols]).sum(1) / tau
    return self_ab, nb_aa, nb_ab, counts


def _log_positive(self_ab: torch.Tensor, nb_aa: torch.Tensor, nb_ab: torch.Tensor,
                  counts: torch.Tensor) -> torch.Tensor:
    """log( S_ab_ii + sum_{j in N(i)} S_aa_ij + sum_{j in N(i)} S_ab_ij ) per anchor, via a padded logsumexp."""
    c = counts.numel()
    max_deg = int(counts.max().item()) if c else 0
    parts = torch.full((c, 1 + 2 * max_deg), float("-inf"), dtype=self_ab.dtype, device=self_ab.device)
    parts[:, 0] = self_ab
    if nb_aa.numel():
        rows = torch.repeat_interleave(torch.arange(c, device=self_ab.device), counts)
        offs = torch.arange(nb_aa.numel(), device=self_ab.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts)
        parts[rows, 1 + offs] = nb_aa
        parts[rows, 1 + max_deg + offs] = nb_ab
    return torch.logsumexp(parts, 1)


def _chunk_loss(za: torch.Tensor, zb: torch.Tensor, start: int, end: int, graph: GraphStructure,
                tau: float, mask_mode: str, labels: torch.Tensor | None, variant: str = "full") -> torch.Tensor:
    """l_i for anchors start..end (za, zb row-normalised).

    ``variant="full"``: computed in log space (logsumexp over the positive and the masked negative
    sets), exact for any tau; ``variant="v1"``: the STAIG 1.x convention in linear space.
    """
    a = za[start:end]
    c = end - start
    local = torch.arange(c, device=za.device)
    ar = torch.arange(start, end, device=za.device)
    l_aa = a @ za.t() / tau                                  # (c, N) logits
    l_ab = a @ zb.t() / tau
    n_pos = 2.0 * graph.deg[start:end].to(za.dtype) + 1.0
    neg_inf = float("-inf")
    if variant == "v1":
        adj = graph.dense_rows(start, end, za.dtype)
        s_aa = torch.exp(l_aa - 1.0 / tau)                  # shifted by exp(-1/tau); ratio unchanged
        s_ab = torch.exp(l_ab - 1.0 / tau)
        self_aa = s_aa[local, ar]
        self_ab = s_ab[local, ar]
        s_aa_off = s_aa.clone()
        s_aa_off[local, ar] = 0
        pos = self_ab + (adj * s_aa_off).sum(1) + (adj * s_ab).sum(1)
        if mask_mode == "none":
            den = s_aa_off.sum(1) + s_ab.sum(1)
        else:
            m = _mask(labels, start, end, mask_mode).to(za.dtype)
            den = (s_aa_off * m).sum(1) + (s_ab * m).sum(1)
            if mask_mode == "pseudo_label":
                den = den - self_aa                          # 1.x convention
        return -torch.log(pos.clamp_min(1e-30) / den.clamp_min(1e-30) / n_pos)
    # ---- full form, log space: positives from the CSR lists, negatives masked in place ----
    log_pos = _log_positive(*_positive_logits(a, za, zb, graph, start, end, tau))
    if mask_mode != "none":
        m = _mask(labels, start, end, mask_mode)
        l_aa.masked_fill_(~m, neg_inf)
        m[local, ar] = True                                  # S_ab_ii always counts (Eq. 7 / Eq. 10)
        l_ab.masked_fill_(~m, neg_inf)
    l_aa[local, ar] = neg_inf                                # intra-view self term never counts
    log_den = torch.logaddexp(torch.logsumexp(l_aa, 1), torch.logsumexp(l_ab, 1))
    return log_den - log_pos + torch.log(n_pos)


def _chunk_loss_sampled(za: torch.Tensor, zb: torch.Tensor, start: int, end: int, graph: GraphStructure,
                        tau: float, mask_mode: str, labels: torch.Tensor | None,
                        neg_idx: torch.Tensor, variant: str = "full") -> torch.Tensor:
    """Same loss with the denominator estimated from a shared random subset ``J`` of ``K`` nodes
    (positives exact). The sampled sum is rescaled by N/K; self terms are handled exactly (never
    subtracted as constants), so the estimate is always a valid positive sum. Cost O(chunk x K)."""
    a = za[start:end]
    c = end - start
    n = za.size(0)
    k = neg_idx.numel()
    self_ab, nb_aa, nb_ab, counts = _positive_logits(a, za, zb, graph, start, end, tau)
    l_aa = a @ za[neg_idx].t() / tau                                              # (c, K)
    l_ab = a @ zb[neg_idx].t() / tau
    n_pos = 2.0 * graph.deg[start:end].to(za.dtype) + 1.0
    is_self = neg_idx[None, :] == torch.arange(start, end, device=za.device)[:, None]   # (c, K)
    m = ~is_self if mask_mode == "none" else (_mask(labels, start, end, mask_mode, neg_idx) & ~is_self)
    scale = n / k
    if variant == "v1":
        shift = 1.0 / tau
        self_aa = torch.exp((a * a).sum(1) / tau - shift)                        # exact (0 for a zero vector)
        pos = torch.exp(self_ab - shift) + _segment_sum(torch.exp(nb_aa - shift), counts, c) \
            + _segment_sum(torch.exp(nb_ab - shift), counts, c)
        den = ((torch.exp(l_aa - shift) * m).sum(1) + (torch.exp(l_ab - shift) * m).sum(1)) * scale
        if mask_mode == "pseudo_label":
            den = den - self_aa                                                   # 1.x convention
        else:
            den = den + torch.exp(self_ab - shift)                                # exact S_ab_ii
        return -torch.log(pos.clamp_min(1e-30) / den.clamp_min(1e-30) / n_pos)
    # full form: exact self term + rescaled sampled rest, all in log space
    log_pos = _log_positive(self_ab, nb_aa, nb_ab, counts)
    neg_inf = float("-inf")
    l_aa.masked_fill_(~m, neg_inf)
    l_ab.masked_fill_(~m, neg_inf)
    log_scale = torch.log(torch.tensor(scale, dtype=za.dtype, device=za.device))
    sampled = torch.logaddexp(torch.logsumexp(l_aa, 1), torch.logsumexp(l_ab, 1)) + log_scale
    log_den = torch.logaddexp(sampled, self_ab)                                   # + exact S_ab_ii
    return log_den - log_pos + torch.log(n_pos)


def _direction(za: torch.Tensor, zb: torch.Tensor, graph: GraphStructure, tau: float, mask_mode: str,
               labels: torch.Tensor | None, chunk_size: int, use_checkpoint: bool,
               neg_idx: torch.Tensor | None = None, variant: str = "full") -> torch.Tensor:
    n = za.size(0)
    out = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if neg_idx is not None:
            fn, extra = _chunk_loss_sampled, (neg_idx, variant)
        else:
            fn, extra = _chunk_loss, (variant,)
        if use_checkpoint and za.requires_grad:
            out.append(checkpoint(fn, za, zb, start, end, graph, tau, mask_mode, labels, *extra,
                                  use_reentrant=False))
        else:
            out.append(fn(za, zb, start, end, graph, tau, mask_mode, labels, *extra))
    return torch.cat(out)


def neighbor_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    graph: GraphStructure,
    tau: float,
    mask_mode: str = "none",
    labels: torch.Tensor | None = None,
    chunk_size: int = 4096,
    use_checkpoint: bool = True,
    reduce: bool = True,
    n_negatives: int | None = None,
    generator: torch.Generator | None = None,
    variant: str = "full",
) -> torch.Tensor:
    """``n_negatives``: None/0 = exact denominator over all nodes; an int K < N draws K nodes (shared by
    all anchors of this call) and rescales their contribution by N/K (approximate, O(N x K)).
    ``variant``: ``"full"`` (default) or ``"v1"`` (STAIG 1.x convention, see module docstring)."""
    if mask_mode != "none" and labels is None:
        raise ValueError("labels are required for mask_mode != 'none'")
    za = F.normalize(z1, dim=1)
    zb = F.normalize(z2, dim=1)
    neg_idx = None
    n = za.size(0)
    if n_negatives is not None and 0 < n_negatives < n:
        perm = torch.randperm(n, generator=generator)[:n_negatives]
        neg_idx = perm.to(za.device)
    if variant not in {"full", "v1"}:
        raise ValueError("variant must be 'full' or 'v1'")
    l1 = _direction(za, zb, graph, tau, mask_mode, labels, chunk_size, use_checkpoint, neg_idx, variant)
    l2 = _direction(zb, za, graph, tau, mask_mode, labels, chunk_size, use_checkpoint, neg_idx, variant)
    loss = 0.5 * (l1 + l2)
    return loss.mean() if reduce else loss


# ----------------------------------------------------------------------------------------
# Dense reference implementation: line-by-line port of v1 ``nei_con_loss`` / ``nei_con_loss_bias``
# (used by the unit tests only).
# ----------------------------------------------------------------------------------------
def _v1_sim(z1, z2):
    return F.normalize(z1) @ F.normalize(z2).t()


def _v1_nei_con_loss(z1, z2, adj, tau, mask=None):
    adj = adj - torch.diag_embed(adj.diag())
    adj = (adj > 0).to(z1.dtype)
    nei_count = torch.sum(adj, 1) * 2 + 1
    f = lambda x: torch.exp(x / tau)  # noqa: E731
    intra = f(_v1_sim(z1, z1))
    inter = f(_v1_sim(z1, z2))
    if mask is None:
        den = intra.sum(1) + inter.sum(1) - intra.diag()
    else:
        den = (intra * mask).sum(1) + (inter * mask).sum(1) - intra.diag()
    loss = (inter.diag() + (intra * adj).sum(1) + (inter * adj).sum(1)) / den
    loss = loss / nei_count
    return -torch.log(loss)


def reference_dense_loss(z1, z2, adj_dense, tau, mask_mode="none", labels=None):
    if mask_mode == "none":
        mask = None
    elif mask_mode == "pseudo_label":
        mask = (labels.view(-1, 1) != labels.view(1, -1)).to(z1.dtype)
    elif mask_mode == "batch":
        mask = (labels.view(-1, 1) == labels.view(1, -1)).to(z1.dtype)
    else:
        raise ValueError(mask_mode)
    l1 = _v1_nei_con_loss(z1, z2, adj_dense, tau, mask)
    l2 = _v1_nei_con_loss(z2, z1, adj_dense, tau, mask)
    return (0.5 * (l1 + l2)).mean()
