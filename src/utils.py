from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def empty_edge_index(device, dtype):
    return torch.empty((2, 0), dtype=dtype, device=device)


def compute_prior(labels, trn_labels):
    """
    Compute the class prior, which indicates the number of positive nodes among the unlabeled.
    """
    if labels.device != trn_labels.device:
        labels = labels.to(trn_labels.device)
    unlabeled_mask = trn_labels == 0
    valid_mask = unlabeled_mask & (labels >= 0)
    if not bool(valid_mask.any().item()):
        return float("nan")
    return (labels[valid_mask].sum().float() / valid_mask.sum()).item()


def compute_observed_prior(trn_labels):
    """Observed positive ratio from the currently assigned PU labels."""
    return (trn_labels.sum().float() / len(trn_labels)).item()


def build_expected_labels(predictions, trn_nodes, pi_hat):
    """Build expected labels: 1 for P, top-pi U as pseudo-pos (2), rest U as 0."""
    num_nodes = int(predictions.size(0))
    trn_mask = torch.zeros(num_nodes, dtype=torch.bool, device=predictions.device)
    trn_mask[torch.as_tensor(trn_nodes, device=predictions.device)] = True
    unl_mask = ~trn_mask

    expected_trn_labels = torch.zeros(num_nodes, dtype=torch.float, device=predictions.device)
    expected_trn_labels[trn_mask] = 1.0

    unl_idx_all = unl_mask.nonzero(as_tuple=False).view(-1)
    num_unl_all = int(unl_idx_all.numel())
    k_all = int(round(float(pi_hat) * float(num_unl_all)))
    k_all = max(0, min(k_all, int(num_unl_all)))
    s_idx = torch.empty((0,), dtype=torch.long, device=predictions.device)
    if k_all > 0 and unl_idx_all.numel() > 0:
        scores_all = predictions[unl_idx_all]
        topk_all = torch.topk(scores_all, k=k_all, largest=True).indices
        s_idx = unl_idx_all[topk_all]
    k = int(s_idx.numel())
    if k > 0:
        expected_trn_labels[s_idx] = 2.0

    return expected_trn_labels


def estimate_prior_argmax(predictions, trn_labels, trn_nodes):
    """Hard (threshold) prior + expected labels."""
    trn_nodes_t = torch.as_tensor(trn_nodes, device=predictions.device)
    # Prior from hard 0/1 labels.
    prior_labels = trn_labels.clone()
    prior_labels[predictions >= 0.50] = 1
    prior_labels[predictions < 0.50] = 0
    trn_mask = torch.zeros_like(prior_labels, dtype=torch.bool)
    trn_mask[trn_nodes_t] = True
    unl_vals = prior_labels[~trn_mask]
    prior = (unl_vals.sum().float() / len(unl_vals)).item() if len(unl_vals) > 0 else float("nan")

    # Expected labels: pseudo positives as 2, labeled positives as 1.
    expected_trn_labels = trn_labels.clone()
    expected_trn_labels[predictions >= 0.50] = 2
    expected_trn_labels[predictions < 0.50] = 0
    expected_trn_labels[trn_nodes_t] = 1
    return prior, expected_trn_labels


def estimate_prior_cdf(
    probs,
    trn_nodes,
    *,
    grid=1001,
    eps=1e-12,
    min_qp=1e-6,
    min_qu=0.0,
):
    """CDF prior: pi_hat = min_{c in [0,1]} Q_u(c) / Q_p(c)."""
    if probs is None:
        return float("nan"), None
    z = probs.detach().float().view(-1).cpu()
    n = int(z.numel())
    if n == 0:
        return float("nan"), None
    trn_arr = np.asarray(trn_nodes, dtype=np.int64)
    trn_mask = torch.zeros(n, dtype=torch.bool)
    if trn_arr.size > 0:
        trn_mask[torch.from_numpy(trn_arr)] = True
    z_p = z[trn_mask]
    z_u = z[~trn_mask]
    if z_p.numel() == 0 or z_u.numel() == 0:
        return float("nan"), None

    z_p_sorted = torch.sort(z_p).values
    z_u_sorted = torch.sort(z_u).values
    grid_n = int(grid)
    grid_n = max(2, grid_n)
    c = torch.linspace(0.0, 1.0, steps=grid_n)

    idx_p = torch.searchsorted(z_p_sorted, c, right=False)
    cnt_p = z_p_sorted.numel() - idx_p
    idx_u = torch.searchsorted(z_u_sorted, c, right=False)
    cnt_u = z_u_sorted.numel() - idx_u

    q_p = cnt_p.float() / float(z_p_sorted.numel())
    q_u = cnt_u.float() / float(z_u_sorted.numel())

    valid = (q_p >= min_qp) & (q_u >= min_qu)
    if not bool(valid.any().item()):
        return float("nan"), None

    ratio = q_u[valid] / (q_p[valid] + eps)
    pi_hat = float(ratio.min().clamp(0.0, 1.0).item())

    expected_trn_labels = build_expected_labels(
        probs, trn_nodes, pi_hat
    )
    return pi_hat, expected_trn_labels


@dataclass
class SignedGraphConstruction:
    """Container for undirected signed edges with p_plus and optional confidence."""

    u: np.ndarray
    v: np.ndarray
    p_plus: np.ndarray
    conf: Optional[np.ndarray] = None

    def as_signed_edges(self) -> np.ndarray:
        return np.stack([self.u, self.v, self.p_plus], axis=1)

    def to_signedgcn_inputs(self, device, soft_sign: bool, use_conf: bool = False):
        if self.u.size == 0:
            empty = empty_edge_index(device, torch.long)
            w_empty = torch.empty((0,), dtype=torch.float32, device=device)
            return empty, empty, w_empty, w_empty

        u = self.u.astype(np.int64, copy=False)
        v = self.v.astype(np.int64, copy=False)
        src2 = np.concatenate([u, v], axis=0)
        dst2 = np.concatenate([v, u], axis=0)

        if soft_sign:
            pplus = self.p_plus.astype(np.float32, copy=False)
            pplus2 = np.concatenate([pplus, pplus], axis=0)
            pminus2 = (1.0 - pplus2).astype(np.float32, copy=False)
            if use_conf and self.conf is not None:
                conf = self.conf.astype(np.float32, copy=False)
                conf2 = np.concatenate([conf, conf], axis=0)
                pplus2 = pplus2 * conf2
                pminus2 = pminus2 * conf2
            edge_index = torch.from_numpy(np.stack([src2, dst2], axis=0)).long().to(device)
            pos_edge_index = edge_index
            neg_edge_index = edge_index.clone()
            pos_weight = torch.from_numpy(pplus2).to(device)
            neg_weight = torch.from_numpy(pminus2).to(device)
            return pos_edge_index, neg_edge_index, pos_weight, neg_weight

        sign = np.where(self.p_plus >= 0.5, 1, -1).astype(np.int64, copy=False)
        sign2 = np.concatenate([sign, sign], axis=0)
        pos_mask = sign2 > 0
        neg_mask = sign2 < 0

        if pos_mask.any():
            pos_edge_index = torch.from_numpy(
                np.stack([src2[pos_mask], dst2[pos_mask]], axis=0)
            ).long().to(device)
            pos_weight = torch.ones((pos_edge_index.size(1),), dtype=torch.float32, device=device)
        else:
            pos_edge_index = empty_edge_index(device, torch.long)
            pos_weight = torch.empty((0,), dtype=torch.float32, device=device)

        if neg_mask.any():
            neg_edge_index = torch.from_numpy(
                np.stack([src2[neg_mask], dst2[neg_mask]], axis=0)
            ).long().to(device)
            neg_weight = torch.ones((neg_edge_index.size(1),), dtype=torch.float32, device=device)
        else:
            neg_edge_index = empty_edge_index(device, torch.long)
            neg_weight = torch.empty((0,), dtype=torch.float32, device=device)

        return pos_edge_index, neg_edge_index, pos_weight, neg_weight

    @classmethod
    def from_signed_edges(cls, signed_edges):
        if signed_edges is None or getattr(signed_edges, "size", 0) == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return cls(
                u=empty,
                v=empty,
                p_plus=np.zeros((0,), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
            )
        u = signed_edges[:, 0].astype(np.int64, copy=False)
        v = signed_edges[:, 1].astype(np.int64, copy=False)
        s = signed_edges[:, 2].astype(np.float32, copy=False)
        if s.min() < 0 or s.max() > 1:
            p_plus = (s + 1.0) * 0.5
        else:
            p_plus = s
        conf = cls.compute_sign_confidence(p_plus)
        return cls(u=u, v=v, p_plus=p_plus, conf=conf)

    @staticmethod
    def compute_sign_confidence(p_plus):
        """Compute confidence for p_plus in [0,1] using 0.5 as center."""
        if p_plus is None or getattr(p_plus, "size", 0) == 0:
            return p_plus
        p_plus = np.asarray(p_plus, dtype=np.float32)
        conf = 2.0 * np.abs(p_plus - 0.5)
        return conf.astype(np.float32, copy=False)

    @classmethod
    def link_sign_estimation(
        cls,
        edges,
        pseudo_labels,
        node_probs=None,
    ):
        """Construct signed edges with p_plus from node posteriors (default)."""
        if edges is None:
            empty = np.zeros((0,), dtype=np.int64)
            return cls(
                u=empty,
                v=empty,
                p_plus=np.zeros((0,), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
            )
        src = edges[0].detach().cpu().numpy()
        dst = edges[1].detach().cpu().numpy()
        undirected_mask = src < dst
        undirected_edges = np.stack([src[undirected_mask], dst[undirected_mask]], axis=1)
        if undirected_edges.shape[0] == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return cls(
                u=empty,
                v=empty,
                p_plus=np.zeros((0,), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
            )
        u = undirected_edges[:, 0]
        v = undirected_edges[:, 1]
        if torch.is_tensor(node_probs):
            device = node_probs.device
        else:
            device = pseudo_labels.device

        u_t = torch.from_numpy(u).to(device)
        v_t = torch.from_numpy(v).to(device)

        if node_probs is not None:
            probs = node_probs
            if not torch.is_tensor(probs):
                probs = torch.as_tensor(probs, device=device)
            if probs.ndim > 1:
                probs = probs.view(-1)
            p_u = probs[u_t]
            p_v = probs[v_t]
            p_plus = p_u * p_v + (1.0 - p_u) * (1.0 - p_v)
        else:
            y = pseudo_labels.to(device)
            same = (y[u_t] > 0) == (y[v_t] > 0)
            p_plus = same.float()

        p_plus_np = p_plus.detach().cpu().numpy().astype(np.float32, copy=False)
        conf_np = cls.compute_sign_confidence(p_plus_np)
        return cls(u=u, v=v, p_plus=p_plus_np, conf=conf_np)
