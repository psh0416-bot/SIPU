import torch
import torch.nn.functional as F
from torch.optim import Adam


def _to_prob_pos(predictions: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to positive-class probabilities (binary)."""
    if predictions.dim() == 1:
        return torch.sigmoid(predictions)
    if predictions.dim() == 2 and predictions.size(1) == 1:
        return torch.sigmoid(predictions.view(-1))
    if predictions.dim() == 2 and predictions.size(1) == 2:
        return torch.softmax(predictions, dim=1)[:, 1]
    raise ValueError("predictions must be [N], [N,1], or [N,2]")


def _group_balanced_bce(prob_pos: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Group-balanced BCE on probabilities (binary)."""
    eps = 1e-6
    prob_pos = prob_pos.clamp(min=eps, max=1 - eps)
    labels = (labels > 0).float()
    pos_mask = labels == 1
    neg_mask = labels == 0
    loss_pos = -torch.log(prob_pos[pos_mask]) if pos_mask.any() else None
    loss_neg = -torch.log(1.0 - prob_pos[neg_mask]) if neg_mask.any() else None

    if loss_pos is not None and loss_neg is not None:
        return loss_pos.mean() + loss_neg.mean()
    if loss_pos is not None:
        return loss_pos.mean()
    if loss_neg is not None:
        return loss_neg.mean()
    return torch.tensor(0.0, device=prob_pos.device)


def _train_edge_weights_from_prob_pos(
    prob_pos: torch.Tensor,
    relabeled_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_iters=20,
    epochs=100,
    lr=0.01,
):
    prob_pos = prob_pos.detach().reshape(-1)
    prob_matrix = torch.stack([1.0 - prob_pos, prob_pos], dim=1)
    num_nodes = prob_matrix.size(0)

    raw_edge_weight = torch.nn.Parameter(torch.zeros(edge_index.size(1), device=prob_matrix.device))
    optimizer = Adam([raw_edge_weight], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        pos_edge_weight = F.softplus(raw_edge_weight)
        propagated = prob_matrix
        for _ in range(num_iters):
            weighted_messages = pos_edge_weight.view(-1, 1) * propagated[edge_index[0]]

            base_prop = torch.zeros_like(prob_matrix) + 0 * weighted_messages.sum()
            propagated = base_prop.scatter_add(
                0,
                edge_index[1].unsqueeze(1).expand(-1, 2),
                weighted_messages,
            )

            base_degree = torch.zeros(num_nodes, device=prob_matrix.device) + 0 * pos_edge_weight.sum()
            degree = base_degree.scatter_add(0, edge_index[1], pos_edge_weight)
            degree = degree.clamp(min=1e-6)

            propagated = propagated / degree.view(-1, 1)

        prob_pos_prop = propagated[:, 1]
        loss = _group_balanced_bce(prob_pos_prop, relabeled_labels)

        loss.backward()
        optimizer.step()

    return F.softplus(raw_edge_weight).detach()


def train_edge_weights(
    predictions,
    relabeled_labels,
    edge_index,
    num_iters=20,
    epochs=100,
    lr=0.01,
):
    """Learn positive edge weights by matching propagated probs to relabeled labels."""
    prob_pos = _to_prob_pos(predictions)
    return _train_edge_weights_from_prob_pos(
        prob_pos,
        relabeled_labels,
        edge_index,
        num_iters=num_iters,
        epochs=epochs,
        lr=lr,
    )
