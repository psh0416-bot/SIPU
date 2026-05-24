from torch import nn, sparse
import torch
import numpy as np
from .base import SigmoidLoss, LogitsBCE


class InferenceModel(nn.Module):
    def __init__(self, edges, potential=0.95, threshold=1e-6, max_iters=100):
        super().__init__()
        self.eps = 1e-12

        if isinstance(edges, np.ndarray):
            values = torch.ones(edges.shape[0])
            edge_index = torch.from_numpy(edges).t()
            edges = torch.sparse_coo_tensor(edge_index, values, dtype=torch.float32)
        self.threshold = threshold
        self.max_iters = max_iters
        self.softmax = nn.Softmax(dim=1)

        indices = edges.coalesce().indices()
        self.register_buffer("src_nodes", indices[0, :].long())
        self.register_buffer("dst_nodes", indices[1, :].long())
        self.num_nodes = edges.size(0)
        # noinspection PyProtectedMember
        self.num_edges = edges._nnz() // 2
        self.register_buffer("rev_edges", self.set_rev_edges(edges).long())
        self.register_buffer("potential", torch.full([2, 2], fill_value=(1 - potential) / 2))
        self.potential[0, 0] = potential / 2
        self.potential[1, 1] = potential / 2

    def set_rev_edges(self, edges):
        degrees = sparse.mm(edges, torch.ones([self.num_nodes, 1])).view(-1).int()
        zero = torch.zeros(1, dtype=torch.int64)
        indices = torch.cat([zero, degrees.cumsum(dim=0)[:-1]])
        counts = torch.zeros(self.num_nodes, dtype=torch.int64)
        rev_edges = torch.zeros(2 * self.num_edges, dtype=torch.int64)
        edge_idx = 0
        for dst, degree in enumerate(degrees):
            for _ in range(degree):
                src = self.dst_nodes[edge_idx]
                rev_edges[indices[src] + counts[src]] = edge_idx
                edge_idx += 1
                counts[src] += 1
        return rev_edges

    def update_messages(self, messages, beliefs):
        new_beliefs = beliefs[self.src_nodes].clamp_min(self.eps)
        rev_messages = messages[self.rev_edges].clamp_min(self.eps)
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential.to(messages.device))
        new_msgs = new_msgs.clamp_min(self.eps)
        return new_msgs / new_msgs.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def compute_beliefs(self, priors, messages):
        beliefs = priors.clamp_min(self.eps).log()
        beliefs.index_add_(0, self.dst_nodes, messages.clamp_min(self.eps).log())
        return self.softmax(beliefs)

    def forward(self, priors):
        beliefs = priors
        messages = torch.full([self.num_edges * 2, 2], fill_value=0.5, device=priors.device)
        for _ in range(self.max_iters):
            old_beliefs = beliefs
            messages = self.update_messages(messages, beliefs)
            beliefs = self.compute_beliefs(priors, messages)
            diff = (beliefs - old_beliefs).abs().max()
            if diff < self.threshold:
                break
        return beliefs


class BeliefRiskEstimator(nn.Module):
    def __init__(self, edges, priors, potential=0.9, bre_loss="sigmoid", recompute=False, labels=None):
        super().__init__()
        if isinstance(priors, float):
            self.pi = priors
            assert labels is not None
        if recompute:
            priors = self.to_priors(labels, priors)
            model = InferenceModel(edges, potential).to(priors.device)
            self.marginals = nn.Parameter(model(priors), requires_grad=False)
        else:
            priors = self.to_initial_priors(labels)
            self.marginals = priors
        if bre_loss == "sigmoid":
            self.loss = SigmoidLoss(reduction=False)
        elif bre_loss == "bce-logits":
            self.loss = LogitsBCE(reduction=False)
        else:
            raise ValueError(f"Unsupported bre_loss: {bre_loss}")

    @staticmethod
    def to_priors(labels, prior):
        num_nodes = labels.size(0)
        priors = torch.zeros(num_nodes, 2, device=labels.device)
        priors[labels == 1, 0] = 0
        priors[labels == 1, 1] = 1
        priors[labels == 2, 0] = prior
        priors[labels == 2, 1] = 1 - prior
        priors[labels == 0, 0] = 1 - prior
        priors[labels == 0, 1] = prior
        return priors

    @staticmethod
    def to_initial_priors(labels):
        num_nodes = labels.size(0)
        priors = torch.zeros(num_nodes, 2, device=labels.device)
        priors[labels == 1, 0] = 0
        priors[labels == 1, 1] = 1
        priors[labels == 0, 0] = 1
        priors[labels == 0, 1] = 0
        return priors

    def forward(self, predictions, labels):
        # Assume labels are already on the same device as predictions
        all_nodes = torch.arange(predictions.size(0), device=predictions.device)
        pos_nodes = all_nodes[labels == 1]
        unl_nodes = all_nodes[labels == 0]

        zero = predictions.new_zeros(())
        r_hat_plus_p = self.loss(predictions[pos_nodes], 1).mean() if pos_nodes.numel() > 0 else zero

        # ---- Key fix: indices used for CPU marginals must be on CPU ----
        unl_nodes_cpu = unl_nodes.detach().cpu()

        m_unl_pos = self.marginals[unl_nodes_cpu, 1].to(predictions.device)
        m_unl_neg = self.marginals[unl_nodes_cpu, 0].to(predictions.device)

        pred_unl = predictions[unl_nodes]
        if pred_unl.numel() > 0:
            r_hat_plus_u = (self.loss(pred_unl, 1) * m_unl_pos).mean()
            r_hat_minus_u = (self.loss(pred_unl, 0) * m_unl_neg).mean()
        else:
            r_hat_plus_u = zero
            r_hat_minus_u = zero

        return r_hat_plus_p + r_hat_plus_u + r_hat_minus_u

