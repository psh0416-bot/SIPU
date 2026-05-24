import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


def build_lsdan_edges(edge_index: torch.Tensor, num_nodes: int, num_distance: int, device: torch.device):
    """Build hop-specific edge lists A^1..A^K for LSDAN."""
    if num_distance < 1:
        raise ValueError("LSDAN requires num_distance >= 1.")

    base_edges = add_self_loops(edge_index)[0].cpu()
    values = torch.ones(base_edges.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(base_edges, values, (num_nodes, num_nodes)).coalesce()
    dense_adj = adj.to_dense()

    hop_edges = []
    reachability = dense_adj.clone()
    for hop_idx in range(num_distance):
        if hop_idx == 0:
            current = reachability
        else:
            reachability = torch.sparse.mm(reachability.to_sparse(), dense_adj).clamp_max_(1.0)
            current = reachability
        hop_edges.append(torch.nonzero(current > 0, as_tuple=False).t().contiguous().to(device))
    return hop_edges


class LSDAN(nn.Module):
    """LSDAN baseline with distance-specific GCN branches and attention fusion."""

    requires_signed_weights: bool = False

    def __init__(self, in_dim: int, hidden_dim: int = 16, layer_num: int = 2, dropout: float = 0.0):
        super().__init__()
        assert layer_num >= 1
        self.dropout = float(dropout)

        self.shorts = nn.ModuleList([
            GCNConv(in_dim, hidden_dim, cached=True)
            for _ in range(layer_num)
        ])
        self.match_dim = nn.Linear(in_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        edge_weight=None,
        return_embedding: bool = False,
    ):
        if not isinstance(edge_index, (list, tuple)):
            raise TypeError("LSDAN expects a list of hop-specific edge_index tensors.")
        if len(edge_index) < len(self.shorts):
            raise ValueError("Insufficient hop-specific edge lists for LSDAN.")

        outs = []
        coefs = []
        matched_x = self.match_dim(x)

        for hop_idx, conv in enumerate(self.shorts):
            h = conv(x, edge_index[hop_idx])
            h = F.leaky_relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            outs.append(h)
            coefs.append(torch.sum(h * matched_x, dim=-1))

        coef_tensor = torch.stack(coefs, dim=0)
        coef_tensor = F.softmax(coef_tensor, dim=0)
        stacked_outs = torch.stack(outs, dim=0)
        fused = torch.sum(stacked_outs * coef_tensor.unsqueeze(-1), dim=0)
        logits = self.linear(fused).view(-1)

        if return_embedding:
            return logits, fused
        return logits
