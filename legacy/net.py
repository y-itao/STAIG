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

    def nei_con_loss(self,z1: torch.Tensor, z2: torch.Tensor, adj, mask=None):
        '''neighbor contrastive loss'''
        adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
        adj[adj > 0] = 1
        nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
        nei_count = torch.squeeze(torch.tensor(nei_count))

        f = lambda x: torch.exp(x / self.tau)

        if mask is None:
            intra_view_sim = f(self.sim(z1, z1))
            inter_view_sim = f(self.sim(z1, z2))
        else:
            intra_view_sim = f(self.sim(z1, z1)) * mask
            inter_view_sim = f(self.sim(z1, z2)) * mask

        loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
                intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
        loss = loss / nei_count  # divided by the number of positive pairs for each node

        return -torch.log(loss)

    def contrastive_loss(self,z1: torch.Tensor, z2: torch.Tensor, adj,mask=None,
                         mean: bool = True):
        l1 = self.nei_con_loss(z1, z2, adj, mask)
        l2 = self.nei_con_loss(z2, z1, adj, mask)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
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

    edge_attr_scaled = edge_attr
    edge_attr_scaled_cpu = edge_attr_scaled.to('cpu')


    mask = torch.rand(edge_attr_scaled.size(0), device=torch.device('cpu')) >= edge_attr_scaled_cpu

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr

def multiple_dropout_average(edge_index: Tensor,
                             edge_attr: Tensor,
                             num_trials: int = 10,
                             force_undirected: bool = False,
                             num_nodes: Optional[int] = None,
                             threshold_ratio: float = 0.5,
                             training: bool = True,
                             device: str = 'cuda') -> Tuple[Tensor, Tensor]:
    if not training:
        return edge_index, edge_attr

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
    simulation_flag = torch.tensor([False]) 

    if simulation_flag.item():
        edge_count = torch.zeros((num_nodes, num_nodes), dtype=torch.int32, device=device)
        for _ in range(num_trials):
            dropped_edge_index, _ = dropout_adj(
                edge_index, edge_attr, force_undirected)
            dropped_edge_index = dropped_edge_index.to(device)
            src, dest = dropped_edge_index
            edge_count[src, dest] += 1
            if force_undirected:
                edge_count[dest, src] += 1
        threshold = int(num_trials * threshold_ratio)
        mask = edge_count >= threshold
        final_edge_index = mask.nonzero().t().contiguous()
    else:
        final_edge_index, _ = dropout_adj(
            edge_index, edge_attr, force_undirected)
    
    return final_edge_index, edge_attr 


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

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.dis_layers = 1  
        self.dis_hid_dim = 64  
        self.dis_dropout = 0.2  
        self.dis_input_dropout = 0.1 

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = input_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dis_dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).view(-1)