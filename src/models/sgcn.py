from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from utils import SignedGraphConstruction


def build_sgcn_kwargs(signed_edges, device, use_conf):
    if signed_edges is None:
        raise ValueError("signed_edges required for signedpu.")
    if not isinstance(signed_edges, SignedGraphConstruction):
        signed_edges = SignedGraphConstruction.from_signed_edges(signed_edges)
    pos_edge_index, neg_edge_index, pos_weight, neg_weight = signed_edges.to_signedgcn_inputs(
        device, soft_sign=True, use_conf=use_conf
    )
    return {
        "pos_edge_index": pos_edge_index,
        "neg_edge_index": neg_edge_index,
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
    }


class LinkSignClassifier(nn.Module):
    """MLP classifier for link sign prediction from node embeddings."""

    def __init__(self, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, emb_u: Tensor, emb_v: Tensor) -> Tensor:
        return self.net(torch.cat([emb_u, emb_v], dim=-1)).view(-1)


class SignedGCNConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        norm_emb: bool = True,
        add_self_loops=True,
        use_attention: bool = True,
        weighted_mean: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.add_self_loops = add_self_loops
        self.norm_emb = norm_emb
        self.use_attention = use_attention
        self.weighted_mean = weighted_mean
        self.eps = 1e-12

        self.lin_b = torch.nn.Linear(in_dim, out_dim, bias)
        self.lin_u = torch.nn.Linear(in_dim, out_dim, bias)

        self.alpha_u = torch.nn.Linear(self.out_dim * 2, 1)
        self.alpha_b = torch.nn.Linear(self.out_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()
        torch.nn.init.xavier_normal_(self.alpha_b.weight)
        torch.nn.init.xavier_normal_(self.alpha_u.weight)

        self.alpha_b.reset_parameters()
        self.alpha_u.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos_edge_index: LongTensor,
        neg_edge_index: LongTensor,
        pos_weight,
        neg_weight,
    ):
        orig_pos_weight = pos_weight
        orig_neg_weight = neg_weight

        if self.first_aggr:
            emb_balanced = self.lin_b(x)
            emb_unbalanced = self.lin_u(x)

            pos_edges, pos_weights_no_self = remove_self_loops(pos_edge_index, edge_attr=orig_pos_weight)
            pos_edges, pos_weights = add_self_loops(pos_edges, edge_attr=pos_weights_no_self, fill_value=1.0)

            edge_types = torch.zeros(pos_edges.size(-1), dtype=torch.long)

            out_balanced = self.propagate(
                pos_edges, x1=emb_balanced, x2=emb_balanced,
                edge_p=edge_types, alpha_func=self.alpha_b, edge_attr=pos_weights
            )
            if self.weighted_mean:
                out_balanced = self._weighted_mean(out_balanced, pos_edges, pos_weights, emb_balanced.size(0))

            neg_edges, neg_weights_no_self = remove_self_loops(neg_edge_index, edge_attr=orig_neg_weight)
            neg_edges, neg_weights = add_self_loops(neg_edges, edge_attr=neg_weights_no_self, fill_value=1.0)

            edge_types = torch.zeros(neg_edges.size(-1), dtype=torch.long)

            out_unbalanced = self.propagate(
                neg_edges, x1=emb_unbalanced, x2=emb_unbalanced,
                edge_p=edge_types, alpha_func=self.alpha_u, edge_attr=neg_weights
            )
            if self.weighted_mean:
                out_unbalanced = self._weighted_mean(out_unbalanced, neg_edges, neg_weights, emb_unbalanced.size(0))

            combined_out = torch.cat([out_balanced, out_unbalanced], dim=-1)

        else:
            feature_dim = self.in_dim
            emb_balanced = x[..., :feature_dim]
            emb_unbalanced = x[..., feature_dim:]

            pos_edges_no_self, pos_weights_no_self = remove_self_loops(pos_edge_index, edge_attr=orig_pos_weight)
            pos_edges_with_self, pos_weights_with_self = add_self_loops(
                pos_edges_no_self, edge_attr=pos_weights_no_self, fill_value=1.0
            )
            neg_edges_no_self, neg_weights_no_self = remove_self_loops(neg_edge_index, edge_attr=neg_weight)
            merged_edges = torch.cat([pos_edges_with_self, neg_edges_no_self], dim=-1)
            merged_weights = torch.cat([pos_weights_with_self, neg_weights_no_self], dim=-1)

            edge_labels_pos = torch.zeros(pos_edges_with_self.size(-1), dtype=torch.long)
            edge_labels_neg = torch.ones(neg_edges_no_self.size(-1), dtype=torch.long)
            merged_edge_labels = torch.cat([edge_labels_pos, edge_labels_neg], dim=-1)

            transformed_balanced_x1 = self.lin_b(emb_balanced)
            transformed_balanced_x2 = self.lin_b(emb_unbalanced)

            out_balanced = self.propagate(
                merged_edges, x1=transformed_balanced_x1, x2=transformed_balanced_x2,
                edge_p=merged_edge_labels, alpha_func=self.alpha_b, edge_attr=merged_weights
            )
            if self.weighted_mean:
                out_balanced = self._weighted_mean(
                    out_balanced, merged_edges, merged_weights, transformed_balanced_x1.size(0)
                )

            pos_edges_no_self, pos_weights_no_self = remove_self_loops(pos_edge_index, edge_attr=orig_pos_weight)
            pos_edges_with_self, pos_weights_with_self = add_self_loops(
                pos_edges_no_self, edge_attr=pos_weights_no_self, fill_value=1.0
            )
            neg_edges_no_self, neg_weights_no_self = remove_self_loops(neg_edge_index, edge_attr=orig_neg_weight)

            merged_edges = torch.cat([pos_edges_with_self, neg_edges_no_self], dim=-1)
            merged_weights = torch.cat([pos_weights_with_self, neg_weights_no_self], dim=-1)

            edge_labels_pos = torch.zeros(pos_edges_with_self.size(-1), dtype=torch.long)
            edge_labels_neg = torch.ones(neg_edges_no_self.size(-1), dtype=torch.long)
            merged_edge_labels = torch.cat([edge_labels_pos, edge_labels_neg], dim=-1)

            transformed_unbalanced_x1 = self.lin_u(emb_unbalanced)
            transformed_unbalanced_x2 = self.lin_u(emb_balanced)

            out_unbalanced = self.propagate(
                merged_edges, x1=transformed_unbalanced_x1, x2=transformed_unbalanced_x2,
                edge_p=merged_edge_labels, alpha_func=self.alpha_u, edge_attr=merged_weights
            )
            if self.weighted_mean:
                out_unbalanced = self._weighted_mean(
                    out_unbalanced, merged_edges, merged_weights, transformed_unbalanced_x1.size(0)
                )

            combined_out = torch.cat([out_balanced, out_unbalanced], dim=-1)

        return combined_out

    def _weighted_mean(self, out: Tensor, edge_index: LongTensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        if edge_weight is None or edge_weight.numel() == 0:
            return out
        index = edge_index[1]
        denom = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        denom.index_add_(0, index, edge_weight)
        denom = denom.clamp_min(self.eps).view(-1, 1)
        return out / denom

    def message(
        self,
        x1_j: Tensor,
        x2_j: Tensor,
        x1_i: Tensor,
        x2_i: Tensor,
        edge_p: Tensor,
        alpha_func,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        edge_attr: Tensor,
    ) -> Tensor:
        x1 = torch.cat([x1_j, x1_i], dim=-1)
        x2 = torch.cat([x2_j, x2_i], dim=-1)
        edge_h = torch.stack([x1, x2], dim=-1)
        edge_h = edge_h[torch.arange(edge_h.size(0)), :, edge_p]

        alpha = None
        if self.use_attention:
            alpha = alpha_func(edge_h)
            alpha = torch.tanh(alpha)
            alpha = softmax(alpha, index, ptr, size_i)

        x_j = torch.stack([x1_j, x2_j], dim=-1)
        row = torch.arange(edge_h.size(0), device=edge_h.device)
        x_j = x_j[row, :, edge_p]
        if alpha is None:
            msg = x_j * edge_attr.view(-1, 1)
        else:
            msg = x_j * alpha * edge_attr.view(-1, 1)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


class SignedGCN(nn.Module):
    """SignedGCN-style signed propagation backbone + node PU classifier head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        layer_num: int = 2,
        dropout: float = 0.0,
        link_sign_dropout: float = 0.0,
        use_attention: bool = True,
        weighted_mean: bool = False,
        ego_channel: bool = False,
        link_sign_classifier: bool = False,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            hidden_dim += 1
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.link_sign_dropout = link_sign_dropout
        self.ego_channel = ego_channel

        self.conv1 = SignedGCNConv(
            in_dim, hidden_dim // 2, first_aggr=True,
            use_attention=use_attention, weighted_mean=weighted_mean
        )
        self.convs = nn.ModuleList([
            SignedGCNConv(
                hidden_dim // 2, hidden_dim // 2, first_aggr=False,
                use_attention=use_attention, weighted_mean=weighted_mean
            )
            for _ in range(max(0, layer_num - 1))
        ])
        self.ego_lin = None
        if self.ego_channel:
            self.ego_lin = nn.Linear(in_dim, hidden_dim // 2)
            self.proj = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        else:
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.link_sign_classifier = None
        if link_sign_classifier:
            self.link_sign_classifier = LinkSignClassifier(
                hidden_dim,
                dropout=self.link_sign_dropout,
            )

    def forward(
        self,
        x: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        pos_weight: Tensor,
        neg_weight: Tensor,
        return_embedding: bool = False,
    ) -> Tensor:
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index, pos_weight, neg_weight))
        z = nn.functional.dropout(z, p=self.dropout, training=self.training)
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index, pos_weight, neg_weight))
            z = nn.functional.dropout(z, p=self.dropout, training=self.training)
        if self.ego_channel:
            z_ego = self.ego_lin(x)
            z = torch.cat([z, z_ego], dim=-1)
        z = self.proj(z)
        logits = self.head(z).view(-1)

        if return_embedding:
            return logits, z
        return logits

    def link_sign_logits(self, embeddings: Tensor, u: Tensor, v: Tensor) -> Tensor:
        if self.link_sign_classifier is None:
            return (embeddings[u] * embeddings[v]).sum(dim=1)
        return self.link_sign_classifier(embeddings[u], embeddings[v])
