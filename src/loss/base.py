from torch import nn
import torch


class CrossEntropyLoss(nn.Module):
    """
    Typical cross entropy loss that balances the positive and negative classes.
    """

    def __init__(self, weighted=False):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, predictions, labels):
        labels = (labels > 0).float()
        loss = self.loss(predictions, labels)
        num_nodes = labels.size(0)
        loss_weight = torch.ones(num_nodes, dtype=torch.float, device=predictions.device)
        if self.weighted==True:
            loss_weight[labels == 0] *= (labels == 1).float().sum() / (labels == 0).float().sum()
            loss_weight *= num_nodes / loss_weight.sum()
        return (loss * loss_weight).mean()


class WCELoss(nn.Module):
    """Compute CE as mean of positive and negative group losses."""

    def __init__(self):
        super().__init__()
        # self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.loss = SigmoidLoss(reduction='none')

    def forward(self, predictions, labels):
        predictions = predictions.reshape(-1)
        bin_labels = (labels > 0).float().reshape(-1)
        loss = self.loss(predictions, bin_labels)
        if loss.ndim == 0:
            loss = loss.reshape(1)
        pos_mask = bin_labels == 1
        neg_mask = bin_labels == 0
        has_pos = bool(pos_mask.any().item())
        has_neg = bool(neg_mask.any().item())
        if has_pos and has_neg:
            return loss[pos_mask].mean() + loss[neg_mask].mean()
        if has_pos:
            return loss[pos_mask].mean()
        if has_neg:
            return loss[neg_mask].mean()
        return loss.mean()


class SigmoidLoss(nn.Module):
    """
    Sigmoid loss for the non-negative risk estimator.
    """

    def __init__(self, reduction=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction

    def forward(self, predictions, labels):
        predictions = predictions.reshape(-1)
        if torch.is_tensor(labels):
            labels = labels.to(predictions.device).reshape(-1)
        labels = labels * 2 - 1  # changes {+1, 0} labels to {+1, -1} labels.
        loss = self.sigmoid(-predictions * labels)
        if self.reduction is True or self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction is False or self.reduction == 'none' or self.reduction is None:
            pass
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        return loss


class LogitsBCE(nn.Module):
    """
    BCEWithLogits-based loss for PU/SBRE (labels in {0,1}).
    """

    def __init__(self, reduction=True):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, predictions, labels):
        if not torch.is_tensor(labels):
            labels = torch.full(predictions.size(), labels, device=predictions.device)
        loss = self.loss(predictions, labels.float())
        if self.reduction:
            loss = loss.mean()
        return loss

class RiskEstimator(nn.Module):
    """
    Risk estimator loss designed for PU learning.

    This is the same as Equation (6) of [3] if non_negative is True. Otherwise, this is the same as
    Equation (2) of [2], which was proposed in [1, 2].

    Refer to the following papers for more information:
    [1] Analysis of Learning from Positive and Unlabeled Data (NIPS 2014)
    [2] Convex Formulation for Learning from Positive and Unlabeled Data (ICML 2015)
    [3] Positive-Unlabeled Learning with Non-Negative Risk Estimator (NIPS 2017)
    [4] Long-short Distance Aggregation Networks for Positive Unlabeled Graph Learning (CIKM 2019)
    """

    def __init__(self, prior, nonnegative=False, stabilized=False):
        super().__init__()
        self.prior = prior  # \pi_\mathrm{p} in [1]
        self.loss = SigmoidLoss()
        self.nonnegative = nonnegative
        self.stabilized = stabilized

    def forward(self, predictions, labels):
        labels = (labels > 0).float()
        all_nodes = torch.arange(predictions.size(0), device=predictions.device)
        pos_nodes = all_nodes[labels == 1]
        unl_nodes = all_nodes[labels == 0]
        r_hat_plus_p = self.loss(predictions[pos_nodes], 1)
        r_hat_minus_p = self.loss(predictions[pos_nodes], 0)
        r_hat_minus_u = self.loss(predictions[unl_nodes], 0)

        loss1 = self.prior * r_hat_plus_p
        loss2 = r_hat_minus_u - self.prior * r_hat_minus_p

        if self.nonnegative:
            loss2 = loss2.clamp_min(0)
        if self.stabilized:
            loss2 = loss2.clamp_min(self.prior * r_hat_minus_u.item())

        return loss1 + loss2


