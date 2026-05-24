import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCN(nn.Module):
    """Standard GCN backbone for binary node classification under PU learning.

    - No signed-edge assumption.
    - Returns a single logit per node (shape: [N]).
    """

    requires_signed_weights: bool = False

    def __init__(self, in_dim: int, hidden_dim: int = 16, layer_num: int = 2, dropout: float = 0.0):
        super().__init__()
        assert layer_num >= 1
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(GCNConv(in_dim, 1))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(layer_num - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, return_embedding: bool = False):
        h = x
        last_hidden = None
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight=edge_weight)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
                last_hidden = h
        logits = h.view(-1)
        if return_embedding:
            emb = last_hidden if last_hidden is not None else logits.unsqueeze(-1)
            return logits, emb
        return logits


class GAT(nn.Module):
    """Standard GAT backbone for binary node classification under PU learning."""

    requires_signed_weights: bool = False

    def __init__(self, in_dim: int, hidden_dim: int = 16, layer_num: int = 2, dropout: float = 0.0):
        super().__init__()
        assert layer_num >= 1
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(GATConv(in_dim, 1, heads=1, concat=False, dropout=self.dropout))
        else:
            self.convs.append(GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=self.dropout))
            for _ in range(layer_num - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=self.dropout))
            self.convs.append(GATConv(hidden_dim, 1, heads=1, concat=False, dropout=self.dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None, return_embedding: bool = False):
        del edge_weight
        h = x
        last_hidden = None
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = F.elu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
                last_hidden = h
        logits = h.view(-1)
        if return_embedding:
            emb = last_hidden if last_hidden is not None else logits.unsqueeze(-1)
            return logits, emb
        return logits


class MLP(nn.Module):
    """Simple MLP backbone for binary node classification (edge-agnostic)."""

    requires_signed_weights: bool = False

    def __init__(self, in_dim: int, hidden_dim: int = 16, layer_num: int = 2, dropout: float = 0.0):
        super().__init__()
        assert layer_num >= 1
        self.dropout = float(dropout)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            num_inputs = in_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(num_inputs, hidden_dim))
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None, return_embedding: bool = False):
        h = x
        last_hidden = None
        for layer in self.layers:
            h = layer(h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            last_hidden = h
        logits = self.linear(h).view(-1)
        if return_embedding:
            emb = last_hidden if last_hidden is not None else logits.unsqueeze(-1)
            return logits, emb
        return logits
