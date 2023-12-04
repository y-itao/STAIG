import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor




class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 1
        self.k = k
        self.conv = [base_model(in_channels, out_channels if k == 1 else 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        if k > 1:
            self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class MVmodel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(MVmodel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        h = self.projection(z)
        return h

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, h1: torch.Tensor, h2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss(self,z1: torch.Tensor, z2: torch.Tensor, adj):
        '''neighbor contrastive loss'''
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj[adj > 0] = 1
        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / self.tau)
        intra_view_sim = f(self.sim(z1, z1))
        inter_view_sim = f(self.sim(z1, z2))

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)

    def contrastive_loss(self,z1: torch.Tensor, z2: torch.Tensor, adj,
                         mean: bool = True):
        l1 = self.nei_con_loss(z1, z2, adj)
        l2 = self.nei_con_loss(z2, z1, adj)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss_bias(self, z1: torch.Tensor, z2: torch.Tensor, adj, pseudo_labels):
        '''neighbor contrastive loss'''
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj[adj > 0] = 1
        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / self.tau)
        intra_view_sim = f(self.sim(z1, z1))
        inter_view_sim = f(self.sim(z1, z2))

        # Create a mask for negative samples with different pseudo labels
        negative_mask = (pseudo_labels.view(-1, 1) != pseudo_labels.view(1, -1)).float()

        # Apply the mask to intra_view_sim and inter_view_sim
        masked_intra_view_sim = intra_view_sim * negative_mask
        masked_inter_view_sim = inter_view_sim * negative_mask

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                masked_intra_view_sim.sum(1) + masked_inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)
    def contrastive_loss_bias(self,z1: torch.Tensor, z2: torch.Tensor, adj, pseudo_labels,
                         mean: bool = True):
        l1 = self.nei_con_loss_bias(z1, z2, adj, pseudo_labels)
        l2 = self.nei_con_loss_bias(z2, z1, adj, pseudo_labels)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

class SVmodel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(SVmodel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        h = self.projection(z)
        return h


    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, h1: torch.Tensor, h2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss(self,z1: torch.Tensor, z2: torch.Tensor, adj):
        '''neighbor contrastive loss'''
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj[adj > 0] = 1
        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / self.tau)
        intra_view_sim = f(self.sim(z1, z1))
        inter_view_sim = f(self.sim(z1, z2))

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)

    def contrastive_loss(self,z1: torch.Tensor, z2: torch.Tensor, adj,
                         mean: bool = True):
        l1 = self.nei_con_loss(z1, z2, adj)
        l2 = self.nei_con_loss(z2, z1, adj)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=torch.device('cpu')).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def filter_adj(row: Tensor, col: Tensor, edge_attr: OptTensor,
               mask: Tensor) -> Tuple[Tensor, Tensor, OptTensor]:
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(
        edge_index: Tensor,
        edge_attr: Tensor,
        min_p: float = 0.1,
        max_p: float = 0.9,
        force_undirected: bool = False,
        num_nodes: Optional[int] = None,
        training: bool = True,
) -> Tuple[Tensor, Tensor]:
    if not training:
        return edge_index, edge_attr

    row, col = edge_index

    if force_undirected:
        mask = row <= col
        row, col, edge_attr = row[mask], col[mask], edge_attr[mask]

    # 获取删除概率的最大值和最小值
    min_p_tensor = torch.tensor(min_p, device=torch.device('cpu'))
    max_p_tensor = torch.tensor(max_p, device=torch.device('cpu'))

    # 缩放 edge_attr 以使其位于给定的删除概率区间内
    edge_attr_scaled = min_p_tensor + (max_p_tensor - min_p_tensor) * edge_attr
    edge_attr_scaled_cpu = edge_attr_scaled.to('cpu')

    # 根据缩放后的 edge_attr（概率）值决定是否删除邻边
    mask = torch.rand(edge_attr_scaled.size(0), device=torch.device('cpu')) >= edge_attr_scaled_cpu

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def random_dropout_adj(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, OptTensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    .. warning::

        :class:`~torch_geometric.utils.dropout_adj` is deprecated and will
        be removed in a future release.
        Use :class:`torch_geometric.utils.dropout_edge` instead.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> dropout_adj(edge_index, edge_attr)
        (tensor([[0, 1, 2, 3],
                [1, 2, 3, 2]]),
        tensor([1, 3, 5, 6]))

        >>> # The returned graph is kept undirected
        >>> dropout_adj(edge_index, edge_attr, force_undirected=True)
        (tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]]),
        tensor([1, 3, 5, 1, 3, 5]))
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index

    mask = torch.rand(row.size(0), device=torch.device('cpu')) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr