from collections import deque

import torch
from torch import nn


class DistPULoss(nn.Module):
    """Distance-aware PU loss adapted to the binary PU setting."""

    def __init__(self, edge_index, prior=None, delta=3):
        super().__init__()
        self.edge_index = edge_index
        self.prior = prior
        self.delta = int(delta)
        self.prior_near = 0.6
        self.prior_far = 0.3

    def _split_unlabeled_nodes(self, labels):
        edge_index = self.edge_index.detach().cpu()
        labels_cpu = labels.detach().cpu()
        num_nodes = int(labels_cpu.numel())

        pos_nodes = torch.where(labels_cpu > 0)[0].tolist()
        unl_nodes = torch.where(labels_cpu == 0)[0].tolist()
        if not unl_nodes:
            return set(), set()
        if not pos_nodes:
            return set(), set(unl_nodes)

        neighbors = [[] for _ in range(num_nodes)]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            neighbors[u].append(v)

        distances = [None] * num_nodes
        q = deque()
        for node in pos_nodes:
            distances[node] = 0
            q.append(node)

        while q:
            u = q.popleft()
            du = distances[u]
            for v in neighbors[u]:
                if distances[v] is None:
                    distances[v] = du + 1
                    q.append(v)

        near_set = set()
        far_set = set()
        for node in unl_nodes:
            d = distances[node]
            if d is not None and d <= self.delta:
                near_set.add(node)
            else:
                far_set.add(node)
        return near_set, far_set

    def forward(self, predictions, labels, embeddings=None, alpha=0.01):
        probs = torch.sigmoid(predictions.reshape(-1))
        labels = (labels > 0).float().reshape(-1)

        pos_nodes = torch.where(labels == 1)[0]
        pos_loss = torch.tensor(0.0, device=predictions.device)
        if pos_nodes.numel() > 0:
            pos_loss = (self.prior_far + self.prior_near) * 2.0 * torch.abs(
                probs[pos_nodes].mean() - 1.0
            )

        near_set, far_set = self._split_unlabeled_nodes(labels)

        near_loss = torch.tensor(0.0, device=predictions.device)
        if near_set:
            near_idx = torch.tensor(sorted(near_set), dtype=torch.long, device=predictions.device)
            near_probs = probs[near_idx]
            near_loss = torch.abs(near_probs.mean() - (self.prior_near / len(near_probs)))

        far_loss = torch.tensor(0.0, device=predictions.device)
        if far_set:
            far_idx = torch.tensor(sorted(far_set), dtype=torch.long, device=predictions.device)
            far_probs = probs[far_idx]
            far_loss = torch.abs(far_probs.mean() - (self.prior_far / len(far_probs)))

        return pos_loss + near_loss + far_loss
