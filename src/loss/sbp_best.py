import torch
import torch.nn as nn
import numpy as np
from torch import sparse
from .base import SigmoidLoss, LogitsBCE

class InferenceModel(nn.Module):
    def __init__(self, edges, potential=0.95, threshold=1e-6, max_iters=100, device=None,
                 soft_sign=False, conf_weight=False):
        super().__init__()

        self.threshold = threshold
        self.max_iters = max_iters
        self.softmax = nn.Softmax(dim=1)
        self.soft_sign = soft_sign
        self.conf_weight = conf_weight

        # --- Build sparse adjacency and signed values ---
        if isinstance(edges, np.ndarray):
            if edges.shape[1] == 2:
                edge_index = torch.from_numpy(edges).long()
                edge_sign = torch.ones(edge_index.size(0), dtype=torch.float32)
            elif edges.shape[1] == 3:
                edge_index = torch.from_numpy(edges[:, :2]).long()
                edge_sign = torch.from_numpy(edges[:, 2]).float()
            else:
                raise ValueError("edges np.ndarray must be (E,2) or (E,3) with sign")

            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            src2 = torch.cat([src, dst], dim=0)
            dst2 = torch.cat([dst, src], dim=0)
            n = int(torch.max(torch.stack([src2, dst2])) + 1)
            idx = torch.stack([src2, dst2], dim=0)
            if soft_sign:
                sign2 = torch.cat([edge_sign, edge_sign], dim=0)
                sign2 = sign2.float()
                if sign2.min().item() < 0 or sign2.max().item() > 1:
                    pplus2 = (sign2 + 1.0) * 0.5
                else:
                    pplus2 = sign2
                # sort to match coalesce ordering
                order = torch.argsort(src2 * n + dst2)
                src2 = src2[order]
                dst2 = dst2[order]
                pplus2 = pplus2[order]
                idx = torch.stack([src2, dst2], dim=0)
                adj = torch.sparse_coo_tensor(idx, torch.ones_like(pplus2), size=(n, n))
                adj = adj.coalesce()
                if adj._nnz() == pplus2.numel():
                    self.edge_pplus_dir = pplus2.to(adj.device)
                else:
                    # fallback: map (src,dst) -> pplus
                    key = (src2.cpu().numpy() * n + dst2.cpu().numpy()).astype(np.int64)
                    val = pplus2.cpu().numpy().astype(np.float32)
                    acc = {}
                    cnt = {}
                    for k, v in zip(key, val):
                        acc[k] = acc.get(k, 0.0) + float(v)
                        cnt[k] = cnt.get(k, 0) + 1
                    idx_np = adj.indices().cpu().numpy()
                    key2 = (idx_np[0] * n + idx_np[1]).astype(np.int64)
                    pvals = np.array([acc[k] / cnt[k] for k in key2], dtype=np.float32)
                    self.edge_pplus_dir = torch.from_numpy(pvals).to(adj.device)
                if self.conf_weight:
                    conf = 2.0 * (self.edge_pplus_dir - 0.5).abs()
                    self.edge_conf_dir = conf.clamp(0.0, 1.0)
            else:
                if edge_sign.min().item() >= 0 and edge_sign.max().item() <= 1:
                    edge_sign = torch.where(edge_sign >= 0.5, torch.ones_like(edge_sign), -torch.ones_like(edge_sign))
                sign2 = torch.cat([edge_sign, edge_sign], dim=0)
                # sort to match coalesce ordering
                order = torch.argsort(src2 * n + dst2)
                src2 = src2[order]
                dst2 = dst2[order]
                sign2 = sign2[order]
                idx = torch.stack([src2, dst2], dim=0)
                adj = torch.sparse_coo_tensor(idx, sign2, size=(n, n))
                adj = adj.coalesce()
        else:
            adj = edges.coalesce()


        if device is not None:
            adj = adj.to(device)
            if self.soft_sign and hasattr(self, "edge_pplus_dir"):
                self.edge_pplus_dir = self.edge_pplus_dir.to(adj.device)
            if self.soft_sign and hasattr(self, "edge_conf_dir"):
                self.edge_conf_dir = self.edge_conf_dir.to(adj.device)

        indices = adj.indices()
        values = adj.values()

        self.src_nodes = indices[0, :]
        self.dst_nodes = indices[1, :]
        if soft_sign:
            if not hasattr(self, "edge_pplus_dir"):
                v = values.float()
                if v.min().item() < 0 or v.max().item() > 1:
                    self.edge_pplus_dir = (v + 1.0) * 0.5
                else:
                    self.edge_pplus_dir = v
            self.edge_pplus_dir = self.edge_pplus_dir.clamp(1e-8, 1 - 1e-8)
            if self.conf_weight and not hasattr(self, "edge_conf_dir"):
                conf = 2.0 * (self.edge_pplus_dir - 0.5).abs()
                self.edge_conf_dir = conf.clamp(0.0, 1.0)
        else:
            if values.min().item() >= 0 and values.max().item() <= 1:
                values = torch.where(values >= 0.5, torch.ones_like(values), -torch.ones_like(values))
            self.edge_sign_dir = (values > 0).long()

        self.num_nodes = adj.size(0)
        self.num_edges = adj._nnz() // 2
        self.rev_edges = self.set_rev_edges(adj)

        # --- Two potentials: positive-edge prefers same, negative-edge prefers different ---
        p = float(potential)
        pot_pos = torch.tensor([[p, 1 - p],
                                [1 - p, p]], dtype=torch.float32, device=adj.device)
        pot_neg = torch.tensor([[1 - p, p],
                                [p, 1 - p]], dtype=torch.float32, device=adj.device)

        self.register_buffer("pot_pos", pot_pos)
        self.register_buffer("pot_neg", pot_neg)
        self.last_messages = None
        self.last_beliefs = None

    def set_rev_edges(self, edges):
        degrees = torch.bincount(self.src_nodes.cpu(), minlength=self.num_nodes).to(edges.device).int()
        zero = torch.zeros(1, dtype=torch.int64, device=edges.device)
        indices = torch.cat([zero, degrees.cumsum(dim=0)[:-1]])
        counts = torch.zeros(self.num_nodes, dtype=torch.int64, device=edges.device)
        rev_edges = torch.zeros(edges._nnz(), dtype=torch.int64, device=edges.device)

        edge_idx = 0
        for dst, degree in enumerate(degrees):
            for _ in range(int(degree.item())):
                src = self.dst_nodes[edge_idx]
                rev_edges[indices[src] + counts[src]] = edge_idx
                edge_idx += 1
                counts[src] += 1
        return rev_edges

    def update_messages(self, messages, beliefs):
        # beliefs: (N,2), messages: (2E,2)
        new_beliefs = beliefs[self.src_nodes]                     # (2E,2)
        rev_messages = messages[self.rev_edges].clamp_min(1e-12)   # avoid div0

        msg_in = (new_beliefs / rev_messages)                     # (2E,2)

        # pick per-edge potential: (2E,2,2)
        if self.soft_sign:
            p = self.edge_pplus_dir[:, None, None]
            pot = p * self.pot_pos[None, :, :] + (1.0 - p) * self.pot_neg[None, :, :]
        else:
            pot = torch.where(
                self.edge_sign_dir[:, None, None].bool(),
                self.pot_pos[None, :, :],
                self.pot_neg[None, :, :]
            )

        # batched matmul: (2E,1,2) x (2E,2,2) -> (2E,1,2) -> (2E,2)
        new_msgs = torch.bmm(msg_in.unsqueeze(1), pot).squeeze(1)
        new_msgs = new_msgs / new_msgs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if self.soft_sign and self.conf_weight and hasattr(self, "edge_conf_dir"):
            conf = self.edge_conf_dir[:, None]
            msg = new_msgs.clamp_min(1e-12)
            msg = msg.pow(conf)
            new_msgs = msg / msg.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return new_msgs

    def compute_beliefs(self, priors, messages):
        beliefs = priors.log()
        beliefs.index_add_(0, self.dst_nodes, messages.log())
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
        self.last_messages = messages
        self.last_beliefs = beliefs
        return beliefs

    def _directed_potentials(self):
        if self.soft_sign:
            p = self.edge_pplus_dir[:, None, None]
            return p * self.pot_pos[None, :, :] + (1.0 - p) * self.pot_neg[None, :, :]
        return torch.where(
            self.edge_sign_dir[:, None, None].bool(),
            self.pot_pos[None, :, :],
            self.pot_neg[None, :, :],
        )

    def pairwise_beliefs(self, beliefs=None, messages=None):
        if beliefs is None:
            beliefs = self.last_beliefs
        if messages is None:
            messages = self.last_messages
        if beliefs is None or messages is None:
            raise ValueError("Run forward() before requesting pairwise beliefs.")

        beliefs = beliefs.clamp_min(1e-12)
        messages = messages.clamp_min(1e-12)

        cavity_src = beliefs[self.src_nodes] / messages[self.rev_edges]
        cavity_dst = beliefs[self.dst_nodes] / messages
        cavity_src = cavity_src / cavity_src.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cavity_dst = cavity_dst / cavity_dst.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pot = self._directed_potentials()
        pairwise = cavity_src.unsqueeze(2) * pot * cavity_dst.unsqueeze(1)
        pairwise = pairwise / pairwise.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        return pairwise

    def pairwise_pplus(self, beliefs=None, messages=None):
        pairwise = self.pairwise_beliefs(beliefs=beliefs, messages=messages)
        keep = self.src_nodes < self.dst_nodes
        if not bool(keep.any().item()):
            empty_i = np.zeros((0,), dtype=np.int64)
            empty_f = np.zeros((0,), dtype=np.float32)
            return empty_i, empty_i, empty_f
        src = self.src_nodes[keep].detach().cpu().numpy().astype(np.int64, copy=False)
        dst = self.dst_nodes[keep].detach().cpu().numpy().astype(np.int64, copy=False)
        p_plus = (pairwise[keep, 0, 0] + pairwise[keep, 1, 1]).detach().cpu().numpy().astype(np.float32, copy=False)
        return src, dst, p_plus


class SignedBeliefRiskEstimator(nn.Module):
    def __init__(self, edges, priors, potential=0.9, bre_loss="sigmoid", recompute=False,
                 labels=None, soft_sign=False, conf_weight=False):
        super().__init__()
        if isinstance(priors, float):
            self.pi = priors
            assert labels is not None
        edges = self._normalize_signed_edges(edges, recompute, "SignedBeliefRiskEstimator")
        if recompute:
            if isinstance(priors, torch.Tensor):
                priors_t = priors
            else:
                priors_t = self.to_priors(labels, priors)
            model = InferenceModel(
                edges,
                potential,
                device=priors_t.device,
                soft_sign=soft_sign,
                conf_weight=conf_weight,
            )
            self.marginals = nn.Parameter(model(priors_t), requires_grad=False)
        else:
            if isinstance(priors, torch.Tensor):
                priors_t = priors
            else:
                priors_t = self.to_initial_priors(labels)
            self.marginals = priors_t
        if bre_loss == "sigmoid":
            self.loss = SigmoidLoss(reduction=False)
        elif bre_loss == "bce-logits":
            self.loss = LogitsBCE(reduction=False)
        else:
            raise ValueError(f"Unsupported bre_loss: {bre_loss}")

    @staticmethod
    def _normalize_signed_edges(edges, recompute, name):
        if edges is None:
            if recompute:
                raise ValueError(f"{name} requires signed_edges when recompute=True.")
            return None
        if hasattr(edges, "as_signed_edges"):
            return edges.as_signed_edges()
        return edges

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
        all_nodes = torch.arange(predictions.size(0), device=predictions.device)
        pos_nodes = all_nodes[labels == 1]
        unl_nodes = all_nodes[labels == 0]
        r_hat_plus_p = self.loss(predictions[pos_nodes], 1).mean()
        pred_unl = predictions[unl_nodes]
        unl_nodes_cpu = unl_nodes.detach().cpu()
        m_unl_pos = self.marginals[unl_nodes_cpu, 1].to(pred_unl.device)
        m_unl_neg = self.marginals[unl_nodes_cpu, 0].to(pred_unl.device)

        loss_pos = self.loss(pred_unl, 1)
        loss_neg = self.loss(pred_unl, 0)

        r_hat_u = (loss_pos * m_unl_pos + loss_neg * m_unl_neg).mean()
        return r_hat_plus_p + r_hat_u



class LSPLoss(nn.Module):
    """Link Sign Prediction loss (binary cross-entropy on edge logits)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, edge_logits, edge_target, edge_weight=None):
        loss = self.loss(edge_logits, edge_target)
        if edge_weight is not None and edge_weight.numel() == loss.numel():
            loss = loss * edge_weight
        return loss.mean()



class _EdgeSupervisionMixin:
    @staticmethod
    def _compute_confidence_np(p_plus):
        if p_plus is None or getattr(p_plus, "size", 0) == 0:
            return None
        p_plus = np.asarray(p_plus, dtype=np.float32)
        conf = 2.0 * np.abs(p_plus - 0.5)
        return conf.astype(np.float32, copy=False)

    def _init_edge_supervision(self, signed_edges, conf_weight):
        self._edge_supervision_enabled = False
        self._edge_weight_enabled = False

        def _register_empty():
            self.register_buffer("edge_u", torch.empty((0,), dtype=torch.long))
            self.register_buffer("edge_v", torch.empty((0,), dtype=torch.long))
            self.register_buffer("edge_target", torch.empty((0,), dtype=torch.float32))
            self.register_buffer("edge_weight", torch.empty((0,), dtype=torch.float32))

        if signed_edges is None:
            _register_empty()
            return

        if hasattr(signed_edges, "u") and hasattr(signed_edges, "v") and hasattr(signed_edges, "p_plus"):
            u = np.asarray(signed_edges.u, dtype=np.int64)
            v = np.asarray(signed_edges.v, dtype=np.int64)
            p_plus = np.asarray(signed_edges.p_plus, dtype=np.float32)
            conf = getattr(signed_edges, "conf", None)
        else:
            se = np.asarray(signed_edges)
            if se.size == 0:
                return
            u = se[:, 0].astype(np.int64, copy=False)
            v = se[:, 1].astype(np.int64, copy=False)
            s = se[:, 2].astype(np.float32, copy=False)
            if s.size == 0:
                return
            if s.min() < 0 or s.max() > 1:
                p_plus = (s + 1.0) * 0.5
            else:
                p_plus = s
            conf = None

        if p_plus.size == 0:
            _register_empty()
            return

        edge_u = torch.from_numpy(u).long()
        edge_v = torch.from_numpy(v).long()
        edge_target = torch.from_numpy(p_plus).float()
        edge_weight = torch.empty((0,), dtype=torch.float32)
        self._edge_supervision_enabled = True

        if conf_weight:
            if conf is None:
                conf = self._compute_confidence_np(p_plus)
            if conf is not None:
                edge_weight = torch.from_numpy(conf).float()
                self._edge_weight_enabled = True

        self.register_buffer("edge_u", edge_u)
        self.register_buffer("edge_v", edge_v)
        self.register_buffer("edge_target", edge_target)
        self.register_buffer("edge_weight", edge_weight)

    def get_edge_supervision(self):
        if not self._edge_supervision_enabled:
            return None
        weight = self.edge_weight if self._edge_weight_enabled else None
        return {
            "u": self.edge_u,
            "v": self.edge_v,
            "target": self.edge_target,
            "weight": weight,
        }


class SignedBeliefRiskWithLSP(_EdgeSupervisionMixin, nn.Module):
    """Joint loss: SignedBeliefRiskEstimator + LSPLoss."""

    def __init__(
        self,
        edges,
        priors,
        potential=0.9,
        bre_loss="sigmoid",
        recompute=False,
        labels=None,
        soft_sign=False,
        conf_weight=False,
        lsp_signed_edges=None,
        lsp_conf_weight=False,
        lsp_lambda=1.0,
    ):
        super().__init__()
        self.lsp_lambda = float(lsp_lambda)
        self.bre = SignedBeliefRiskEstimator(
            edges,
            priors,
            potential=potential,
            bre_loss=bre_loss,
            recompute=recompute,
            labels=labels,
            soft_sign=soft_sign,
            conf_weight=conf_weight,
        )
        self.marginals = self.bre.marginals
        self.lsp = LSPLoss()
        self._init_edge_supervision(lsp_signed_edges, lsp_conf_weight)

    def forward(self, predictions, labels, edge_logits=None, edge_target=None, edge_weight=None):
        loss = self.bre(predictions, labels)
        if edge_target is None and self._edge_supervision_enabled:
            edge_target = self.edge_target
            if edge_weight is None and self._edge_weight_enabled:
                edge_weight = self.edge_weight
        if edge_logits is None or edge_target is None:
            return loss
        return loss + self.lsp_lambda * self.lsp(edge_logits, edge_target, edge_weight)
