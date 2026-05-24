import argparse
import os
import random
import sys
import time

import pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

from models import (
    GAT,
    GCN,
    LSDAN,
    MLP,
    SignedGCN,
    BeliefRiskEstimator,
    SignedBeliefRiskEstimator,
    SignedBeliefRiskWithStructureLoss,
    build_lsdan_edges,
    build_sgcn_kwargs,
    train_edge_weights,
)
from loss.base import CrossEntropyLoss, RiskEstimator, WCELoss
from loss.distpu import DistPULoss
from loss.sbp import InferenceModel as SignedInferenceModel
from train import _forward_model, evaluate_model, train_model
from utils import (
    SignedGraphConstruction,
    compute_observed_prior,
    compute_prior,
    estimate_prior_argmax,
    estimate_prior_cdf,
)


def format_structure_lambda(value):
    return f"{float(value):g}"


def signedpu_structure_suffix(args):
    suffix = ""
    if float(getattr(args, "structure_lambda", 0.0)) != 0.0:
        suffix += f"_slsp{format_structure_lambda(args.structure_lambda)}"
    if bool(getattr(args, "sgcn_use_hard_sign", False)):
        suffix += "_hsgcn1"
    if bool(getattr(args, "edge_predictor_prop_lbp_target", False)):
        suffix += "_eplbpt1"
    if bool(getattr(args, "edge_predictor_hard_target", False)):
        suffix += "_epht1"
    return suffix


def signedpu_main_suffix(args):
    suffix = f"_conf{int(bool(getattr(args, 'conf', True)))}"
    suffix += signedpu_structure_suffix(args)
    return suffix


def format_sample_ratio(value):
    return f"{float(value):g}"


def _sync_device(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_block_start(device):
    _sync_device(device)
    return time.perf_counter()


def _time_block_end(device, start_time):
    _sync_device(device)
    return float(time.perf_counter() - start_time)


def to_device(gpu):
    """
    make torch to use GPU
    :param gpu: gpu to use
    :return:
    """
    if gpu is None or gpu < 0 or not torch.cuda.is_available():
        return torch.device('cpu')

    arch_list = torch.cuda.get_arch_list()
    if arch_list:
        supported_caps = set()
        for arch in arch_list:
            if not arch.startswith("sm_"):
                continue
            cap = arch.replace("sm_", "")
            if len(cap) < 2 or not cap.isdigit():
                continue
            supported_caps.add((int(cap[:-1]), int(cap[-1])))
        device_cap = torch.cuda.get_device_capability(gpu)
        if device_cap not in supported_caps:
            device = torch.device('cuda:{}'.format(gpu))
            try:
                # Some CUDA builds can execute on minor variants not listed
                # verbatim in get_arch_list() (e.g. sm_61 with sm_60 support).
                torch.empty(1, device=device)
            except Exception:
                print(
                    f"[Warn] GPU cuda:{gpu} capability sm_{device_cap[0]}{device_cap[1]} "
                    "is not supported by the current PyTorch build. Falling back to CPU."
                )
                return torch.device('cpu')
            print(
                f"[Warn] GPU cuda:{gpu} capability sm_{device_cap[0]}{device_cap[1]} "
                "is not listed in torch.cuda.get_arch_list(), but a CUDA smoke test passed. "
                "Using GPU."
            )

    return torch.device('cuda:{}'.format(gpu))

def define_model(
    model,
    num_features,
    args,
    device,
):
    model_name = str(model).lower()
    if model_name in {"gcn", "pulp", "grab", "gpl", "pugnn"}:
        model = GCN(num_features, args.units, args.layers, args.dropout).to(device)
    elif model_name == "gat":
        model = GAT(num_features, args.units, args.layers, args.dropout).to(device)
    elif model_name == "lsdan":
        model = LSDAN(num_features, args.units, args.layers, args.dropout).to(device)
    elif model_name == "signedpu":
        model = SignedGCN(
            num_features, args.units, args.layers,
            dropout=args.dropout,
            link_sign_dropout=args.pair_mlp_dropout,
            use_attention=False,
            ego_channel=bool(args.ego),
            link_sign_classifier=True,
        ).to(device)
    elif model_name == "mlp":
        model = MLP(num_features, args.units, args.layers, args.dropout).to(device)
    else:
        raise ValueError(f"Unknown model: {model}")
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


def build_optimizer(model, loss_func=None):
    params = []
    seen = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        params.append(param)
        seen.add(id(param))
    if loss_func is not None:
        for param in loss_func.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            params.append(param)
            seen.add(id(param))
    return optim.Adam(params)


def resolve_pair_input_dim(model, args):
    if hasattr(model, "hidden_dim"):
        return int(model.hidden_dim)
    if hasattr(model, "linear") and hasattr(model.linear, "in_features"):
        return int(model.linear.in_features)
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        return int(model.head.in_features)
    if getattr(args, "layers", 1) <= 1:
        return 1
    return int(args.units)


def build_pairwise_belief_signed_edges(
    edges,
    node_probs,
    prior,
    labels,
    args,
    timing_device=None,
    potential_signed_edges=None,
):
    edge_sign_pre_s = 0.0
    if potential_signed_edges is None:
        edge_sign_pre_start = _time_block_start(timing_device)
        base_signed_edges = SignedGraphConstruction.link_sign_estimation(
            edges,
            labels,
            node_probs=node_probs,
        )
        edge_sign_pre_s = _time_block_end(timing_device, edge_sign_pre_start)
    else:
        base_signed_edges = potential_signed_edges
    if base_signed_edges.u.size == 0:
        return base_signed_edges, None, {
            "edge_sign_pre_s": float(edge_sign_pre_s),
            "edge_belief_s": 0.0,
        }

    prob_pos = node_probs.detach().reshape(-1).clamp(1e-6, 1 - 1e-6)
    priors = torch.stack([1.0 - prob_pos, prob_pos], dim=1)
    edge_belief_start = _time_block_start(timing_device)
    sbp_model = SignedInferenceModel(
        base_signed_edges.as_signed_edges(),
        potential=args.potential,
        device=priors.device,
        soft_sign=True,
        conf_weight=bool(args.conf),
    )
    with torch.no_grad():
        beliefs = sbp_model(priors)
    u, v, p_plus = sbp_model.pairwise_pplus(beliefs=beliefs)
    edge_belief_s = _time_block_end(timing_device, edge_belief_start)
    conf = base_signed_edges.conf
    return SignedGraphConstruction(u=u, v=v, p_plus=p_plus, conf=conf), sbp_model, {
        "edge_sign_pre_s": float(edge_sign_pre_s),
        "edge_belief_s": float(edge_belief_s),
    }


def build_shared_node_edge_beliefs(
    edges,
    node_probs,
    prior,
    labels,
    args,
    *,
    potential_signed_edges=None,
    timing_device=None,
):
    """Run one prior-aware LBP and return both node/edge beliefs.

    This aligns node-belief and edge-belief sources:
      - node beliefs are computed from PU priors (SBRE-style),
      - edge beliefs are extracted from the same LBP result.
    """
    edge_sign_pre_s = 0.0
    if potential_signed_edges is None:
        edge_sign_pre_start = _time_block_start(timing_device)
        potential_signed_edges = SignedGraphConstruction.link_sign_estimation(
            edges,
            labels,
            node_probs=node_probs,
        )
        edge_sign_pre_s = _time_block_end(timing_device, edge_sign_pre_start)

    if isinstance(prior, torch.Tensor):
        priors = prior
        if priors.device != node_probs.device:
            priors = priors.to(node_probs.device)
    else:
        priors = SignedBeliefRiskEstimator.to_priors(labels, prior)

    edge_belief_start = _time_block_start(timing_device)
    sbp_model = SignedInferenceModel(
        potential_signed_edges.as_signed_edges(),
        potential=args.potential,
        device=priors.device,
        soft_sign=True,
        conf_weight=bool(args.conf),
    )
    with torch.no_grad():
        beliefs = sbp_model(priors)
    u, v, p_plus = sbp_model.pairwise_pplus(beliefs=beliefs)
    edge_belief_s = _time_block_end(timing_device, edge_belief_start)
    # Keep confidence semantics aligned with the upstream potential source.
    # Legacy path also carries confidence from pre-LBP signed edges.
    conf = None
    if (
        potential_signed_edges is not None
        and getattr(potential_signed_edges, "conf", None) is not None
        and len(potential_signed_edges.conf) == len(p_plus)
    ):
        conf = np.asarray(potential_signed_edges.conf, dtype=np.float32)
    else:
        conf = SignedGraphConstruction.compute_sign_confidence(p_plus)
    edge_signs = SignedGraphConstruction(u=u, v=v, p_plus=p_plus, conf=conf)
    return edge_signs, sbp_model, beliefs.detach(), {
        "edge_sign_pre_s": float(edge_sign_pre_s),
        "edge_belief_s": float(edge_belief_s),
    }


def build_predictor_signed_edges(
    model,
    features,
    model_edges,
    model_kwargs,
    graph_edges,
    timing_device=None,
    node_probs_fallback=None,
):
    """Build signed edges from the current model's edge-sign predictor output.

    This is used to feed SignedGCN propagation weights (p_uv) directly from
    the model's link-sign head. If the head is unavailable, it can fall back
    to node-posterior-derived p_uv when node_probs_fallback is provided;
    otherwise dot-product logits are used.
    """
    start = _time_block_start(timing_device)
    with torch.no_grad():
        out = _forward_model(model, features, model_edges, model_kwargs, return_embedding=True)
        if isinstance(out, tuple):
            _, embeddings = out
        else:
            embeddings = out

    edge_u = graph_edges[0]
    edge_v = graph_edges[1]
    keep = edge_u < edge_v
    if not bool(keep.any().item()):
        elapsed = _time_block_end(timing_device, start)
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float32)
        return SignedGraphConstruction(u=empty_i, v=empty_i, p_plus=empty_f, conf=empty_f), {"edge_pred_sign_s": float(elapsed)}

    u = edge_u[keep].long()
    v = edge_v[keep].long()
    if hasattr(model, "link_sign_logits"):
        logits = model.link_sign_logits(embeddings, u, v)
        p_plus = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
    elif node_probs_fallback is not None:
        probs = node_probs_fallback
        if not torch.is_tensor(probs):
            probs = torch.as_tensor(probs, device=u.device)
        else:
            probs = probs.to(u.device)
        if probs.ndim > 1:
            probs = probs.view(-1)
        p_u = probs[u]
        p_v = probs[v]
        p_plus_t = p_u * p_v + (1.0 - p_u) * (1.0 - p_v)
        p_plus = p_plus_t.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        logits = (embeddings[u] * embeddings[v]).sum(dim=1)
        p_plus = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
    u_np = u.detach().cpu().numpy().astype(np.int64, copy=False)
    v_np = v.detach().cpu().numpy().astype(np.int64, copy=False)
    conf = SignedGraphConstruction.compute_sign_confidence(p_plus)
    elapsed = _time_block_end(timing_device, start)
    return SignedGraphConstruction(u=u_np, v=v_np, p_plus=p_plus, conf=conf), {"edge_pred_sign_s": float(elapsed)}


def attach_confidence_to_hard_signed_edges(hard_edges, conf_source):
    """Keep SBRE signs hard while reusing confidence from the soft signed graph."""
    if hard_edges is None or conf_source is None or hard_edges.u.size == 0:
        return hard_edges
    if conf_source.conf is None or conf_source.u.size == 0:
        return hard_edges

    conf_by_edge = {
        (int(min(u, v)), int(max(u, v))): float(c)
        for u, v, c in zip(conf_source.u, conf_source.v, conf_source.conf)
    }
    conf = np.ones_like(hard_edges.p_plus, dtype=np.float32)
    for idx, (u, v) in enumerate(zip(hard_edges.u, hard_edges.v)):
        key = (int(min(u, v)), int(max(u, v)))
        conf[idx] = conf_by_edge.get(key, 1.0)
    return SignedGraphConstruction(
        u=hard_edges.u,
        v=hard_edges.v,
        p_plus=hard_edges.p_plus,
        conf=conf,
    )


def soften_hard_signed_edges(hard_edges, positive_prob=0.9):
    """Convert hard 0/1 edge signs to soft probabilities (e.g., 0.9/0.1)."""
    if hard_edges is None or hard_edges.u.size == 0:
        return hard_edges
    p = float(positive_prob)
    p = max(0.5, min(1.0, p))
    p_neg = 1.0 - p
    hard_pos = np.asarray(hard_edges.p_plus, dtype=np.float32) >= 0.5
    p_plus = np.where(hard_pos, p, p_neg).astype(np.float32)
    conf = SignedGraphConstruction.compute_sign_confidence(p_plus)
    return SignedGraphConstruction(
        u=hard_edges.u,
        v=hard_edges.v,
        p_plus=p_plus,
        conf=conf,
    )


def define_loss(
    loss_key,
    prior,
    args,
    *,
    edges=None,
    trn_labels=None,
    recompute=False,
    signed_edges=None,
    unary_signed_edges=None,
    signed_conf_weight=False,
    signed_soft_sign=False,
    unary_conf_weight=False,
    pair_input_dim=None,
    line_graph_topology=None,
    path_inference_model=None,
):
    if loss_key == "ce":
        loss_func = CrossEntropyLoss()
    elif loss_key == "wce":
        loss_func = WCELoss()
    elif loss_key == "ure":
        loss_func = RiskEstimator(prior)
    elif loss_key == "nre":
        loss_func = RiskEstimator(prior, nonnegative=True)
    elif loss_key == "distpu":
        loss_func = DistPULoss(edges, prior)
    elif loss_key == "bre":
        edges_np = edges.detach().cpu().t().numpy()
        loss_func = BeliefRiskEstimator(
            edges_np, prior, args.potential, args.BRELoss, recompute, trn_labels
        )
    elif loss_key == "sbre":
        loss_func = SignedBeliefRiskEstimator(
            signed_edges, prior, args.potential, args.BRELoss, recompute, trn_labels,
            soft_sign=bool(signed_soft_sign),
            conf_weight=bool(signed_conf_weight),
        )
    elif loss_key == "sbre-lsp":
        loss_func = SignedBeliefRiskWithStructureLoss(
            signed_edges, prior, args.potential, args.BRELoss, recompute, trn_labels,
            soft_sign=bool(signed_soft_sign),
            conf_weight=bool(signed_conf_weight),
            unary_signed_edges=unary_signed_edges,
            unary_conf_weight=bool(unary_conf_weight),
            structure_lambda=float(args.structure_lambda),
            pair_input_dim=pair_input_dim,
            pair_mlp_hidden=int(args.pair_mlp_hidden),
            pair_mlp_dropout=float(args.pair_mlp_dropout),
            max_line_pairs=args.max_line_pairs,
            line_graph_topology=line_graph_topology,
            path_inference_model=path_inference_model,
        )
    else:
        raise ValueError(loss_key)
    return loss_func


def _sample_node_subgraph(features, labels, edges, trn_nodes, test_nodes, ratio, seed):
    num_nodes = int(features.size(0))
    target_nodes = max(1, int(round(num_nodes * float(ratio))))
    target_nodes = min(target_nodes, num_nodes)
    if target_nodes >= num_nodes:
        return features, labels, edges, trn_nodes, test_nodes

    rng = np.random.default_rng(int(seed))
    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    all_nodes = np.arange(num_nodes, dtype=np.int64)
    pos_nodes = all_nodes[labels_np == 1]
    neg_nodes = all_nodes[labels_np == 0]
    min_pos = min(2, len(pos_nodes), target_nodes)
    min_neg = 1 if len(neg_nodes) > 0 and target_nodes > min_pos else 0

    kept_nodes = None
    for _ in range(64):
        candidate = rng.choice(all_nodes, size=target_nodes, replace=False)
        cand_labels = labels_np[candidate]
        if int((cand_labels == 1).sum()) >= min_pos and int((cand_labels == 0).sum()) >= min_neg:
            kept_nodes = np.sort(candidate)
            break
    if kept_nodes is None:
        kept_nodes = np.sort(rng.choice(all_nodes, size=target_nodes, replace=False))

    remap = -np.ones(num_nodes, dtype=np.int64)
    remap[kept_nodes] = np.arange(len(kept_nodes), dtype=np.int64)

    edge_u = edges[0].detach().cpu().numpy().astype(np.int64, copy=False)
    edge_v = edges[1].detach().cpu().numpy().astype(np.int64, copy=False)
    edge_mask = np.isin(edge_u, kept_nodes) & np.isin(edge_v, kept_nodes)
    sampled_edges = edges[:, torch.from_numpy(edge_mask).bool()]
    sampled_edges_np = sampled_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    sampled_edges = torch.from_numpy(remap[sampled_edges_np]).long()

    sampled_features = features[torch.from_numpy(kept_nodes).long()]
    sampled_labels = labels[torch.from_numpy(kept_nodes).long()]
    full_pos_count = max(1, int((labels == 1).sum().item()))
    sampled_trn_ratio = float(len(trn_nodes)) / float(full_pos_count)
    sampled_trn_nodes, sampled_test_nodes = data.split_nodes(
        sampled_labels,
        trn_ratio=sampled_trn_ratio,
        seed=seed,
    )
    return sampled_features, sampled_labels, sampled_edges, sampled_trn_nodes, sampled_test_nodes


def _sample_edge_subgraph(features, labels, edges, trn_nodes, test_nodes, ratio, seed):
    edge_u = edges[0].detach().cpu().numpy().astype(np.int64, copy=False)
    edge_v = edges[1].detach().cpu().numpy().astype(np.int64, copy=False)
    undirected_mask = edge_u < edge_v
    undirected_u = edge_u[undirected_mask]
    undirected_v = edge_v[undirected_mask]
    num_pairs = int(undirected_u.shape[0])
    target_pairs = max(1, int(round(num_pairs * float(ratio))))
    target_pairs = min(target_pairs, num_pairs)
    if target_pairs >= num_pairs:
        return features, labels, edges, trn_nodes, test_nodes
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_pairs)[:target_pairs]
    perm.sort()
    chosen_u = undirected_u[perm]
    chosen_v = undirected_v[perm]
    sampled_edges_np = np.stack(
        [np.concatenate([chosen_u, chosen_v]), np.concatenate([chosen_v, chosen_u])],
        axis=0,
    ).astype(np.int64, copy=False)
    sampled_edges = torch.from_numpy(sampled_edges_np).long()
    return features, labels, sampled_edges, trn_nodes, test_nodes


def _apply_scalability_sampling(features, labels, edges, trn_nodes, test_nodes, args):
    mode = str(getattr(args, "scalability_mode", "none"))
    ratio = float(getattr(args, "sample_ratio", 1.0))
    if mode == "none" or ratio >= 1.0:
        return features, labels, edges, trn_nodes, test_nodes
    if mode == "node":
        return _sample_node_subgraph(features, labels, edges, trn_nodes, test_nodes, ratio, args.seed)
    if mode == "edge":
        return _sample_edge_subgraph(features, labels, edges, trn_nodes, test_nodes, ratio, args.seed)
    raise ValueError(f"Unknown scalability_mode: {mode}")


def save_runtime_logs(runtime_rows, model_name, args):
    if not (bool(getattr(args, "save_runtime_log", False)) or str(getattr(args, "study", "")) == "scalability"):
        return
    mode = str(getattr(args, "scalability_mode", "none"))
    if mode not in {"node", "edge"}:
        return

    out_dir = os.path.join(args.out, "log", "scalability", mode)
    os.makedirs(out_dir, exist_ok=True)

    suffix = ""
    if model_name == "signedpu" and args.loss == "sbre-lsp":
        suffix = signedpu_main_suffix(args)
    ratio_suffix = f"_scal{mode}_r{format_sample_ratio(getattr(args, 'sample_ratio', 1.0))}"
    filename = (
        f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}"
        f"{suffix}{ratio_suffix}_seed{int(args.seed)}.csv"
    )
    pd.DataFrame(runtime_rows).to_csv(os.path.join(out_dir, filename), index=False)



def parse_args():
    """
    parser arguments to run program in cmd
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '--dataset', dest='data', type=str, default='blogcatalog')
    parser.add_argument(
        '--study',
        type=str,
        default='main',
        choices=['main', 'main_unknown', 'ablation', 'hyperparameter', 'main_tmp', 'main_unknown_tmp', 'ablation_tmp', 'scalability'],
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='../out')

    # Hyperparameters for training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--trn-ratio', type=float, default=0.5)

    # Hyperparameters for models
    parser.add_argument('--model', type=str, default='signedpu')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--units', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--potential', type=float, default=0.9)
    parser.add_argument('--em-iters', type=int, default=5)
    parser.add_argument('--prior', type=str, default='estimated', choices=['estimated', 'given'])
    parser.add_argument('--loss', type=str, default='sbre-lsp')
    parser.add_argument('--structure-lambda', '--lsp-lambda', dest='structure_lambda', type=float, default=1.0)

    parser.add_argument('--edge-predictor-prop-lbp-target', dest='edge_predictor_prop_lbp_target', action='store_true')
    parser.add_argument('--no-edge-predictor-prop-lbp-target', dest='edge_predictor_prop_lbp_target', action='store_false')
    parser.set_defaults(edge_predictor_prop_lbp_target=True)


    parser.add_argument('--pair-mlp-hidden', type=int, default=64)
    parser.add_argument('--pair-mlp-dropout', type=float, default=0.5)
    parser.add_argument('--max-line-pairs', type=int, default=2000)
    parser.add_argument('--BRELoss', type=str, default='sigmoid')
    parser.add_argument('--debug', type=str, default='y')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--data-stats', dest='data_stats', action='store_true')
    parser.add_argument('--no-data-stats', dest='data_stats', action='store_false')
    parser.set_defaults(data_stats=False)


    # Heterophily graph preprocess
    parser.add_argument('--preprocess', dest='preprocess_homophily', action='store_true')
    parser.add_argument('--wo-preprocess', dest='preprocess_homophily', action='store_false')
    parser.set_defaults(preprocess_homophily=True)

    # Confidence
    parser.add_argument('--conf', dest='conf', action='store_true')
    parser.add_argument('--wo-conf', dest='conf', action='store_false')
    parser.set_defaults(conf=False)

    # Ego embedding
    parser.add_argument('--ego', dest='ego', action='store_true')
    parser.add_argument('--wo-ego', dest='ego', action='store_false')
    parser.set_defaults(ego=True)
    
    # Marginalization
    parser.add_argument('--marg', dest='marg', action='store_true')
    parser.add_argument('--wo-marg', dest='marg', action='store_false')
    parser.set_defaults(marg=True)

    # Hard sign
    parser.add_argument('--sgcn-use-hard-sign', dest='sgcn_use_hard_sign', action='store_true')
    parser.add_argument('--no-sgcn-use-hard-sign', dest='sgcn_use_hard_sign', action='store_false')
    parser.set_defaults(sgcn_use_hard_sign=False)

    # Edge predictor hard pseudo-label target
    parser.add_argument('--edge-predictor-hard-target', dest='edge_predictor_hard_target', action='store_true')
    parser.add_argument('--no-edge-predictor-hard-target', dest='edge_predictor_hard_target', action='store_false')
    parser.set_defaults(edge_predictor_hard_target=False)



    # SIPU analysis logs
    parser.add_argument('--save-em-log', dest='save_em_log', action='store_true')
    parser.add_argument('--no-save-em-log', dest='save_em_log', action='store_false')
    parser.set_defaults(save_em_log=False)
    parser.add_argument('--save-epoch-log', dest='save_epoch_log', action='store_true')
    parser.add_argument('--no-save-epoch-log', dest='save_epoch_log', action='store_false')
    parser.set_defaults(save_epoch_log=False)
    parser.add_argument('--save-runtime-log', dest='save_runtime_log', action='store_true')
    parser.add_argument('--no-save-runtime-log', dest='save_runtime_log', action='store_false')
    parser.set_defaults(save_runtime_log=False)
    parser.add_argument(
        '--scalability-mode',
        type=str,
        default='none',
        choices=['none', 'node', 'edge'],
    )
    parser.add_argument('--sample-ratio', type=float, default=1.0)
    parser.add_argument('--disable-early-stopping', dest='disable_early_stopping', action='store_true')
    parser.add_argument('--enable-early-stopping', dest='disable_early_stopping', action='store_false')
    parser.set_defaults(disable_early_stopping=False)



    args = parser.parse_args()
    if str(args.study) == "scalability":
        # Scalability runs are runtime-only by design.
        args.save_runtime_log = True
        # Match the full training budget without relying on a special early-stop mode.
        args.patience = max(int(args.patience), int(args.epochs))
        # Disable homophily preprocessing to avoid full-graph drop_auto costs.
        args.preprocess_homophily = False
    return args



def print_stage_table(rows, header=True):
    if header:
        header_line = "stage     model   epoch    loss   prior  prior_mode     f1    acc    roc    pr"
        print(header_line)
        print("-" * len(header_line))
    for r in rows:
        print(
            f"{r['stage']:<9}"
            f"{r['model']:<7}"
            f"{r['epoch']:>7d}"
            f"{r['loss']:>8.3f}"
            f"{r['prior']:>8.3f} "
            f"{r['prior_mode']:<10} "
            f"{r['f1']:>7.3f}"
            f"{r['acc']:>7.3f}"
            f"{r['roc_auc']:>7.3f}"
            f"{r['pr_auc']:>7.3f}"
        )


def compute_positive_prior(best_model, features, model_edges, model_kwargs, labels, trn_labels, trn_nodes, model_name):
    with torch.no_grad():
        predictions = torch.sigmoid(_forward_model(best_model, features, model_edges, model_kwargs))
        trn_mask = torch.zeros(len(trn_labels), dtype=torch.bool, device=predictions.device)
        trn_mask[torch.as_tensor(trn_nodes, device=predictions.device)] = True
        labels_dev = labels.to(predictions.device)
        unlabeled_mask = (~trn_mask) & (labels_dev >= 0)
        real_pn = labels_dev[unlabeled_mask].float().mean().item() if bool(unlabeled_mask.any().item()) else float("nan")

    if model_name in {"gpl", "signedpu"}:
        expected_pn, _ = estimate_prior_cdf(predictions, trn_nodes)
        if not np.isfinite(expected_pn):
            expected_pn, _ = estimate_prior_argmax(predictions, trn_labels, trn_nodes)
    else:
        expected_pn, _ = estimate_prior_argmax(predictions, trn_labels, trn_nodes)
    return {"expected_PN": float(expected_pn), "real_PN": float(real_pn)}


def save_log(
    logs,
    best_epoch,
    best_model,
    features,
    model_edges,
    model_kwargs,
    labels,
    test_nodes,
    trn_labels,
    trn_nodes,
    model_name,
    args,
    summary_rows=None,
    em_iter=None,
):
    pn = compute_positive_prior(
        best_model, features, model_edges, model_kwargs, labels, trn_labels, trn_nodes, model_name
    )
    best_stage_row = None
    if summary_rows:
        best_stage_row = max(
            summary_rows,
            key=lambda r: (float(r["f1"]), float(r["acc"]), -int(r["epoch"])),
        )
    test_f1, test_acc, test_roc_auc, test_pr_auc = evaluate_model(
        best_model, features, model_edges, labels, test_nodes, model_kwargs
    )
    result = pd.DataFrame([{
        "epoch": int(best_epoch),
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }])
    result["expected_PN"] = pn["expected_PN"]
    result["real_PN"] = pn["real_PN"]

    if args.debug == "y":
        print(f"After EM Done, em iter n: {em_iter}")
        print(f"dataset: {args.data}")
        print(result.to_string(index=False, float_format="%.3f"))
        if summary_rows:
            print("\nSummary:")
            print_stage_table(summary_rows, header=True)

    if args.study in {"ablation", "ablation_tmp"}:
        out_dir = os.path.join(args.out, "log", "ablation")
        if args.study == "ablation_tmp":
            out_dir = os.path.join(args.out, "log", "ablation_tmp")
        suffix_parts = []
        if model_name == "signedpu":
            suffix_parts.extend([
                f"marg{int(bool(args.marg))}",
                f"ego{int(bool(args.ego))}",
                f"conf{int(bool(args.conf))}",
            ])
            if args.loss == "sbre-lsp":
                struct_suffix = signedpu_structure_suffix(args).lstrip("_")
                if struct_suffix:
                    suffix_parts.append(struct_suffix)
        if not bool(args.preprocess_homophily):
            suffix_parts.append("preprocess0")
        if args.BRELoss != "sigmoid":
            suffix_parts.append(f"bre_{args.BRELoss}")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        filename = (
            f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}"
            f"{suffix}.csv"
        )
    elif args.study == "hyperparameter":
        out_dir = os.path.join(args.out, "log", "hyperparameter")
        suffix = ""
        if model_name == "signedpu" and args.loss == "sbre-lsp":
            suffix = signedpu_main_suffix(args)
        filename = f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}{suffix}.csv"
    elif args.study in {"main_unknown", "main_unknown_tmp"}:
        out_dir = os.path.join(args.out, "log", "main_unknown")
        if args.study == "main_unknown_tmp":
            out_dir = os.path.join(args.out, "log", "main_unknown_tmp")
        suffix = ""
        if model_name == "signedpu" and args.loss == "sbre-lsp":
            suffix = signedpu_main_suffix(args)
        filename = f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}{suffix}.csv"
    elif args.study == "main_tmp":
        out_dir = os.path.join(args.out, "log", "main_tmp")
        suffix = ""
        if model_name == "signedpu" and args.loss == "sbre-lsp":
            suffix = signedpu_main_suffix(args)
        filename = f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}{suffix}.csv"
    else:
        out_dir = os.path.join(args.out, "log", "main")
        suffix = ""
        if model_name == "signedpu" and args.loss == "sbre-lsp":
            suffix = signedpu_main_suffix(args)
        filename = f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}{suffix}.csv"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    best_row = logs.loc[best_epoch].to_dict()
    best_row.update({
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    })
    row = {
        **best_row,
        **pn,
        "model": str(model_name),
        "loss": str(args.loss),
        "data": str(args.data),
        "trn_ratio": float(args.trn_ratio),
        "units": int(args.units),
        "layers": int(args.layers),
        "seed": int(args.seed),
        "prior_mode": "cpe" if model_name in {"gpl", "signedpu"} else args.prior,
        "study": str(args.study),
    }
    if model_name == "signedpu" and args.loss == "sbre-lsp":
        row["conf"] = int(bool(args.conf))
        row["structure_lambda"] = float(args.structure_lambda)
        row["sgcn_use_hard_sign"] = int(bool(args.sgcn_use_hard_sign))
        row["edge_predictor_prop_lbp_target"] = int(bool(args.edge_predictor_prop_lbp_target))
        row["edge_predictor_hard_target"] = int(bool(args.edge_predictor_hard_target))
        row["pair_mlp_hidden"] = int(args.pair_mlp_hidden)
        row["pair_mlp_dropout"] = float(args.pair_mlp_dropout)
        row["max_line_pairs"] = int(args.max_line_pairs)
    if best_stage_row is not None:
        row.update({
            "best_stage": str(best_stage_row["stage"]),
            "best_stage_f1": float(best_stage_row["f1"]),
            "best_stage_acc": float(best_stage_row["acc"]),
            "best_stage_roc_auc": float(best_stage_row["roc_auc"]),
            "best_stage_pr_auc": float(best_stage_row["pr_auc"]),
            "best_stage_expected_pn": float(best_stage_row["expected_pn"]),
        })
    if args.study == "ablation":
        row.update({
            "marginalization": int(bool(args.marg)),
            "sgcn_ego_channel": int(bool(args.ego)),
            "conf": int(bool(args.conf)),
        })
    df_new = pd.DataFrame([row])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        df_new.to_csv(path, index=False)
        return
    try:
        df_old = pd.read_csv(path)
    except Exception:
        df_new.to_csv(path, index=False)
        return
    cols = list(df_old.columns)
    for c in df_new.columns:
        if c not in cols:
            cols.append(c)
    df_old = df_old.reindex(columns=cols)
    df_new = df_new.reindex(columns=cols)
    df = pd.concat([df_old, df_new], ignore_index=True)
    df.to_csv(path, index=False)


def save_raw_logs(stage_logs, summary_rows, model_name, args):
    if not (bool(args.save_em_log) or bool(args.save_epoch_log)):
        return
    if str(args.study) == "scalability":
        return
    if model_name != "signedpu":
        return

    raw_root = os.path.join(args.out, "log", "raw", "signedpu")

    base_name = (
        f"{model_name}_{args.loss}_{args.data}_{args.trn_ratio}_{args.units}_{args.layers}"
        f"_seed{int(args.seed)}"
    )
    if args.loss == "sbre-lsp":
        base_name = f"{base_name}{signedpu_main_suffix(args)}"
    base_name = f"{base_name}_study{args.study}"

    if bool(args.save_epoch_log) and stage_logs:
        epoch_dir = os.path.join(raw_root, "epoch")
        os.makedirs(epoch_dir, exist_ok=True)
        epoch_frames = []
        for stage_log in stage_logs:
            frame = stage_log["logs"].copy()
            frame["stage"] = str(stage_log["stage"])
            frame["best_epoch"] = int(stage_log["best_epoch"])
            frame["best_loss"] = float(stage_log["best_loss"])
            frame["model"] = str(model_name)
            frame["loss_name"] = str(args.loss)
            frame["data"] = str(args.data)
            frame["seed"] = int(args.seed)
            frame["study"] = str(args.study)
            if args.loss == "sbre-lsp":
                frame["structure_lambda"] = float(args.structure_lambda)
                frame["sgcn_use_hard_sign"] = int(bool(args.sgcn_use_hard_sign))
                frame["edge_predictor_prop_lbp_target"] = int(bool(args.edge_predictor_prop_lbp_target))
                frame["pair_mlp_hidden"] = int(args.pair_mlp_hidden)
                frame["pair_mlp_dropout"] = float(args.pair_mlp_dropout)
                frame["max_line_pairs"] = int(args.max_line_pairs)
            epoch_frames.append(frame)
        epoch_df = pd.concat(epoch_frames, ignore_index=True)
        epoch_path = os.path.join(epoch_dir, f"{base_name}.csv")
        epoch_df.to_csv(epoch_path, index=False)

    if bool(args.save_em_log) and summary_rows:
        em_dir = os.path.join(raw_root, "em")
        os.makedirs(em_dir, exist_ok=True)
        em_df = pd.DataFrame(summary_rows).copy()
        em_df["model"] = str(model_name)
        em_df["loss_name"] = str(args.loss)
        em_df["data"] = str(args.data)
        em_df["seed"] = int(args.seed)
        em_df["study"] = str(args.study)
        if args.loss == "sbre-lsp":
            em_df["structure_lambda"] = float(args.structure_lambda)
            em_df["sgcn_use_hard_sign"] = int(bool(args.sgcn_use_hard_sign))
            em_df["edge_predictor_prop_lbp_target"] = int(bool(args.edge_predictor_prop_lbp_target))
            em_df["pair_mlp_hidden"] = int(args.pair_mlp_hidden)
            em_df["pair_mlp_dropout"] = float(args.pair_mlp_dropout)
            em_df["max_line_pairs"] = int(args.max_line_pairs)
        em_path = os.path.join(em_dir, f"{base_name}.csv")
        em_df.to_csv(em_path, index=False)



def main(seed=None):
    args = parse_args()
    model = args.model.lower()
    model_name = model
    if seed == None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    device = to_device(args.gpu)
    runtime_rows = []
    run_wall_start = time.perf_counter()

    # Call the designated data
    load_start = _time_block_start(device)
    features, labels, edges, trn_nodes, test_nodes = data.read_data(
        args.data, args.trn_ratio,
        verbose=bool(getattr(args, "verbose", False)),
        drop_auto=args.preprocess_homophily,
        print_stats=bool(getattr(args, "data_stats", True)),
        seed=args.seed
    )
    orig_num_nodes = int(features.size(0))
    orig_num_edges = int(edges.size(1))
    features, labels, edges, trn_nodes, test_nodes = _apply_scalability_sampling(
        features, labels, edges, trn_nodes, test_nodes, args
    )
    data_load_s = _time_block_end(device, load_start)
    graph_edges = edges
    num_nodes = features.size(0)
    num_features = features.size(1)
    num_edges = graph_edges.size(1)
    trn_labels = torch.zeros(num_nodes, dtype=torch.float)
    trn_labels[trn_nodes] = 1
    features = features.to(device)
    graph_edges = graph_edges.to(device)
    trn_labels = trn_labels.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    effective_patience = int(args.patience)
    if bool(getattr(args, "disable_early_stopping", False)):
        effective_patience = max(effective_patience, int(args.epochs))


    # Initialize a prior
    if args.prior == "given":
        prior = compute_prior(labels, trn_labels)
    else:
        prior = compute_observed_prior(trn_labels)

    # LSDAN: multiple hop graphs construction
    if model_name == "lsdan":
        model_edges = build_lsdan_edges(graph_edges, num_nodes, args.layers, device)
    else:
        model_edges = graph_edges
    model_kwargs = None

    # Initialize a model
    init_model = "gcn" if model_name == "signedpu" else model_name
    model, optimizer = define_model(
        init_model,
        num_features,
        args,
        device,
    )
    
    # Define loss
    loss_func = define_loss(
        args.loss,
        prior,
        args,
        edges=graph_edges,
        trn_labels=trn_labels,
        recompute=False,
        pair_input_dim=resolve_pair_input_dim(model, args),
    )
    loss_func = loss_func.to(device)
    optimizer = build_optimizer(model, loss_func)

    # Train the model
    init_train_start = _time_block_start(device)
    best_epoch, logs, best_loss, best_model = train_model(
        model, features, model_edges, labels, test_nodes, loss_func, optimizer, trn_labels,
        args.epochs, effective_patience, model_kwargs, eval_each_epoch=bool(args.save_epoch_log))
    init_train_s = _time_block_end(device, init_train_start)

    the_best_loss = best_loss
    old_logs = logs
    old_epoch = best_epoch
    old_model = best_model
    old_model_kwargs = model_kwargs
    raw_stage_logs = [{
        "stage": "init",
        "logs": logs.copy(),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
    }]

    summary_rows = []
    prior_mode = "cpe" if model_name in {"gpl", "signedpu"} else args.prior
    prior_disp = float(prior)
    if model_name in {"gpl", "signedpu"}:
        with torch.no_grad():
            p_init = torch.sigmoid(_forward_model(best_model, features, model_edges, model_kwargs))
        prior_cpe_init, _ = estimate_prior_cdf(p_init, trn_nodes)
        if prior_cpe_init < prior:
            prior_cpe_init = prior
        if np.isfinite(prior_cpe_init):
            prior_disp = float(prior_cpe_init)
    f1, acc, roc_auc, pr_auc = evaluate_model(best_model, features, model_edges, labels, test_nodes, model_kwargs)
    init_pn = compute_positive_prior(
        best_model, features, model_edges, model_kwargs, labels, trn_labels, trn_nodes, init_model
    )
    summary_rows.append({
        "stage": "init",
        "model": init_model,
        "epoch": int(best_epoch),
        "loss": float(best_loss),
        "prior": float(prior_disp),
        "prior_mode": prior_mode,
        "f1": float(f1),
        "acc": float(acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "expected_pn": float(init_pn["expected_PN"]),
        "real_pn": float(init_pn["real_PN"]),
    })
    if args.debug == 'y':
        print_stage_table(summary_rows, header=True)
    runtime_rows.append({
        "stage": "init",
        "iter_index": 0,
        "dataset": str(args.data),
        "seed": int(args.seed),
        "study": str(args.study),
        "model": str(model_name),
        "loss": str(args.loss),
        "prior_mode": str(prior_mode),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "orig_num_nodes": int(orig_num_nodes),
        "orig_num_edges": int(orig_num_edges),
        "num_features": int(num_features),
        "sample_mode": str(getattr(args, "scalability_mode", "none")),
        "sample_ratio": float(getattr(args, "sample_ratio", 1.0)),
        "data_load_s": float(data_load_s),
        "prior_update_s": 0.0,
        "signed_graph_s": 0.0,
        "node_belief_s": 0.0,
        "edge_sign_pre_s": 0.0,
        "edge_belief_s": 0.0,
        "edge_pred_sign_s": 0.0,
        "hard_edge_sign_s": 0.0,
        "edge_conf_attach_s": 0.0,
        "sgcn_kwargs_s": 0.0,
        "line_graph_topology_s": 0.0,
        "gpl_edge_weight_s": 0.0,
        "model_setup_s": 0.0,
        "train_s": float(init_train_s),
        "eval_s": 0.0,
        "iter_total_s": float(init_train_s),
        "iter_core_s": float(init_train_s),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "best_stage_f1": float(f1),
        "peak_gpu_mem_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda" else float("nan")
        ),
    })
    i = 0
    graph_edges = graph_edges.to(device)
    # Begin iteration
    if model_name in {"grab", "gpl", "signedpu"}:
        for i in range(args.em_iters):
            iter_start = _time_block_start(device)
            # Estimate prior
            prior_start = _time_block_start(device)
            with torch.no_grad():
                predictions = torch.sigmoid(_forward_model(model, features, model_edges, model_kwargs))
                if model_name == "gpl" or model_name == "signedpu":
                    prior, expected_trn_labels = estimate_prior_cdf(predictions, trn_nodes)
                else:
                    prior, expected_trn_labels = estimate_prior_argmax(predictions, trn_labels, trn_nodes)
                em_trn_labels = expected_trn_labels
                node_probs_edges = predictions
            prior_update_s = _time_block_end(device, prior_start)

            loss_key = args.loss
            edge_signs = None
            edge_signs_sbp = None
            target_edge_signs_from_node_lbp = None
            path_inference_model = None

            # SIPU: signed graph construction
            signed_graph_start = _time_block_start(device)
            node_belief_s = 0.0
            edge_sign_pre_s = 0.0
            edge_belief_s = 0.0
            edge_pred_sign_s = 0.0
            hard_edge_sign_s = 0.0
            edge_conf_attach_s = 0.0
            sgcn_kwargs_s = 0.0
            line_graph_topology_s = 0.0
            gpl_edge_weight_s = 0.0
            if model_name == "signedpu":
                hard_edge_sign_start = _time_block_start(device)
                hard_edge_signs = SignedGraphConstruction.link_sign_estimation(
                    graph_edges, expected_trn_labels, node_probs=None
                )
                hard_edge_sign_s = _time_block_end(device, hard_edge_sign_start)

                if args.marg:
                    edge_signs, path_inference_model, pair_timings = build_pairwise_belief_signed_edges(
                        graph_edges,
                        node_probs_edges,
                        prior,
                        expected_trn_labels,
                        args,
                        timing_device=device,
                    )
                    edge_sign_pre_s = float(pair_timings["edge_sign_pre_s"])
                    edge_belief_s = float(pair_timings["edge_belief_s"])
                else:
                    edge_sign_pre_start = _time_block_start(device)
                    edge_signs = SignedGraphConstruction.link_sign_estimation(
                        graph_edges, expected_trn_labels, node_probs=node_probs_edges
                    )
                    edge_sign_pre_s = _time_block_end(device, edge_sign_pre_start)

                edge_conf_attach_start = _time_block_start(device)
                edge_signs_sbp = attach_confidence_to_hard_signed_edges(hard_edge_signs, edge_signs)
                edge_conf_attach_s = _time_block_end(device, edge_conf_attach_start)

                if bool(getattr(args, "edge_predictor_prop_lbp_target", False)):
                    # Build unary edge-sign targets from the same prior-aware LBP
                    # used for node-belief marginals (SBRE input signed edges fixed
                    # to edge_signs_sbp), while keeping propagation sign source
                    # controlled by --edge-predictor-prop-lbp-target.
                    target_edge_signs_from_node_lbp, _, _, target_pair_timings = build_shared_node_edge_beliefs(
                        graph_edges,
                        node_probs_edges,
                        prior,
                        expected_trn_labels,
                        args,
                        potential_signed_edges=edge_signs_sbp,
                        timing_device=device,
                    )
                    edge_belief_s += float(target_pair_timings.get("edge_belief_s", 0.0))

                # Confidence is a shared SIPU component: it weights the
                # SignedGCN encoder, hard-sign SBRE belief propagation, and
                # LSP loss. SBRE keeps hard signs because that was stronger in
                # prior experiments; confidence only damps uncertain messages.
                sgcn_kwargs_start = _time_block_start(device)
                if bool(getattr(args, "edge_predictor_prop_lbp_target", False)):
                    sgcn_signed_edges_for_propagation, pred_timing = build_predictor_signed_edges(
                        model,
                        features,
                        model_edges,
                        model_kwargs,
                        graph_edges,
                        timing_device=device,
                        node_probs_fallback=node_probs_edges,
                    )
                    edge_pred_sign_s += float(pred_timing.get("edge_pred_sign_s", 0.0))
                elif bool(getattr(args, "sgcn_use_hard_sign", False)):
                    sgcn_signed_edges_for_propagation = hard_edge_signs
                else:
                    sgcn_signed_edges_for_propagation = edge_signs
                model_kwargs = build_sgcn_kwargs(
                    sgcn_signed_edges_for_propagation, device, use_conf=bool(args.conf)
                )
                sgcn_kwargs_s = _time_block_end(device, sgcn_kwargs_start)
            # GPL: Weighted graph construction
            if model_name == "gpl":
                gpl_edge_weight_start = _time_block_start(device)
                with torch.enable_grad():
                    gpl_edge_weights = train_edge_weights(predictions, expected_trn_labels, graph_edges)
                model_kwargs = {"edge_weight": gpl_edge_weights}
                gpl_edge_weight_s = _time_block_end(device, gpl_edge_weight_start)
            signed_graph_s = _time_block_end(device, signed_graph_start)

            # Initialize a model
            model_setup_start = _time_block_start(device)
            model, optimizer = define_model(
                model_name,
                num_features,
                args,
                device,
            )
            model_setup_s = _time_block_end(device, model_setup_start)

            # Define loss
            if bool(getattr(args, "edge_predictor_hard_target", False)):
                unary_signed_edges_for_loss = hard_edge_signs
            else:
                unary_signed_edges_for_loss = (
                    target_edge_signs_from_node_lbp
                    if target_edge_signs_from_node_lbp is not None
                    else edge_signs
                )
            loss_func = define_loss(
                loss_key,
                prior,
                args,
                edges=graph_edges,
                trn_labels=expected_trn_labels,
                recompute=True,
                signed_edges=edge_signs_sbp if model_name == "signedpu" else None,
                unary_signed_edges=unary_signed_edges_for_loss if model_name == "signedpu" else None,
                signed_conf_weight=bool(args.conf) if model_name == "signedpu" else False,
                signed_soft_sign=True if model_name == "signedpu" else False,
                # Keep confidence for SignedGCN aggregation/SBRE, but do not
                # use confidence as unary-loss weighting for this experiment.
                unary_conf_weight=False,
                pair_input_dim=resolve_pair_input_dim(model, args),
                path_inference_model=path_inference_model if model_name == "signedpu" else None,
            )
            loss_func = loss_func.to(device)
            optimizer = build_optimizer(model, loss_func)

            # Train model
            train_start = _time_block_start(device)
            best_epoch, logs, best_loss, best_model = train_model(
                model, features, model_edges, labels, test_nodes, loss_func, optimizer, em_trn_labels,
                args.epochs, effective_patience, model_kwargs, eval_each_epoch=bool(args.save_epoch_log))
            train_s = _time_block_end(device, train_start)
            raw_stage_logs.append({
                "stage": f"em{i+1}",
                "logs": logs.copy(),
                "best_epoch": int(best_epoch),
                "best_loss": float(best_loss),
            })

            eval_start = _time_block_start(device)
            f1, acc, roc_auc, pr_auc = evaluate_model(best_model, features, model_edges, labels, test_nodes, model_kwargs)
            stage_pn = compute_positive_prior(
                best_model, features, model_edges, model_kwargs, labels, trn_labels, trn_nodes, model_name
            )
            eval_s = _time_block_end(device, eval_start)
            iter_total_s = _time_block_end(device, iter_start)
            summary_rows.append({
                "stage": f"em{i+1}",
                "model": model_name,
                "epoch": int(best_epoch),
                "loss": float(best_loss),
                "prior": float(prior),
                "prior_mode": "cpe" if model_name in {"gpl", "signedpu"} else args.prior,
                "f1": float(f1),
                "acc": float(acc),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "expected_pn": float(stage_pn["expected_PN"]),
                "real_pn": float(stage_pn["real_PN"]),
            })
            if args.debug == 'y':
                print_stage_table([summary_rows[-1]], header=False)
            runtime_rows.append({
                "stage": f"em{i+1}",
                "iter_index": int(i + 1),
                "dataset": str(args.data),
                "seed": int(args.seed),
                "study": str(args.study),
                "model": str(model_name),
                "loss": str(args.loss),
                "prior_mode": "cpe" if model_name in {"gpl", "signedpu"} else str(args.prior),
                "num_nodes": int(num_nodes),
                "num_edges": int(num_edges),
                "orig_num_nodes": int(orig_num_nodes),
                "orig_num_edges": int(orig_num_edges),
                "num_features": int(num_features),
                "sample_mode": str(getattr(args, "scalability_mode", "none")),
                "sample_ratio": float(getattr(args, "sample_ratio", 1.0)),
                "data_load_s": 0.0,
                "prior_update_s": float(prior_update_s),
                "signed_graph_s": float(signed_graph_s),
                "node_belief_s": float(node_belief_s),
                "edge_sign_pre_s": float(edge_sign_pre_s),
                "edge_belief_s": float(edge_belief_s),
                "edge_pred_sign_s": float(edge_pred_sign_s),
                "hard_edge_sign_s": float(hard_edge_sign_s),
                "edge_conf_attach_s": float(edge_conf_attach_s),
                "sgcn_kwargs_s": float(sgcn_kwargs_s),
                "line_graph_topology_s": float(line_graph_topology_s),
                "gpl_edge_weight_s": float(gpl_edge_weight_s),
                "model_setup_s": float(model_setup_s),
                "train_s": float(train_s),
                "eval_s": float(eval_s),
                "iter_total_s": float(iter_total_s),
                "iter_core_s": float(prior_update_s + signed_graph_s + model_setup_s + train_s),
                "best_epoch": int(best_epoch),
                "best_loss": float(best_loss),
                "best_stage_f1": float(f1),
                "peak_gpu_mem_mb": (
                    float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                    if device.type == "cuda" else float("nan")
                ),
            })

            # if best_loss > the_best_loss:
            #     logs = old_logs
            #     best_epoch = old_epoch
            #     best_model = old_model
            #     model_kwargs = old_model_kwargs
            #     model = best_model
            #     if args.debug != "y":
            #         break
            # else:
            #     the_best_loss = best_loss
            #     old_logs = logs
            #     old_epoch = best_epoch
            #     old_model = best_model
            #     old_model_kwargs = model_kwargs
            if i == 0:
                # force accept em1
                the_best_loss = best_loss
                old_logs = logs
                old_epoch = best_epoch
                old_model = best_model
                old_model_kwargs = model_kwargs
            else:
                if best_loss > the_best_loss:
                    logs = old_logs
                    best_epoch = old_epoch
                    best_model = old_model
                    model_kwargs = old_model_kwargs
                    model = best_model
                    if args.debug != "y":
                        break
                else:
                    the_best_loss = best_loss
                    old_logs = logs
                    old_epoch = best_epoch
                    old_model = best_model
                    old_model_kwargs = model_kwargs


    if runtime_rows:
        runtime_rows.append({
            "stage": "run_total",
            "iter_index": -1,
            "dataset": str(args.data),
            "seed": int(args.seed),
            "study": str(args.study),
            "model": str(model_name),
            "loss": str(args.loss),
            "prior_mode": str(prior_mode),
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
            "orig_num_nodes": int(orig_num_nodes),
            "orig_num_edges": int(orig_num_edges),
            "num_features": int(num_features),
            "sample_mode": str(getattr(args, "scalability_mode", "none")),
            "sample_ratio": float(getattr(args, "sample_ratio", 1.0)),
            "data_load_s": float(data_load_s),
            "prior_update_s": 0.0,
            "signed_graph_s": 0.0,
            "node_belief_s": float(sum(r.get("node_belief_s", 0.0) for r in runtime_rows)),
            "edge_sign_pre_s": float(sum(r.get("edge_sign_pre_s", 0.0) for r in runtime_rows)),
            "edge_belief_s": float(sum(r.get("edge_belief_s", 0.0) for r in runtime_rows)),
            "edge_pred_sign_s": float(sum(r.get("edge_pred_sign_s", 0.0) for r in runtime_rows)),
            "hard_edge_sign_s": float(sum(r.get("hard_edge_sign_s", 0.0) for r in runtime_rows)),
            "edge_conf_attach_s": float(sum(r.get("edge_conf_attach_s", 0.0) for r in runtime_rows)),
            "sgcn_kwargs_s": float(sum(r.get("sgcn_kwargs_s", 0.0) for r in runtime_rows)),
            "line_graph_topology_s": float(sum(r.get("line_graph_topology_s", 0.0) for r in runtime_rows)),
            "gpl_edge_weight_s": float(sum(r.get("gpl_edge_weight_s", 0.0) for r in runtime_rows)),
            "model_setup_s": 0.0,
            "train_s": float(sum(r["train_s"] for r in runtime_rows if "train_s" in r)),
            "eval_s": float(sum(r["eval_s"] for r in runtime_rows if "eval_s" in r)),
            "iter_total_s": float(time.perf_counter() - run_wall_start),
            "iter_core_s": float(sum(r["iter_core_s"] for r in runtime_rows if "iter_core_s" in r)),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "best_stage_f1": float(summary_rows[-1]["f1"]) if summary_rows else float("nan"),
            "peak_gpu_mem_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda" else float("nan")
            ),
        })

    save_runtime_logs(runtime_rows, model_name, args)
    if str(args.study) == "scalability":
        return
    save_raw_logs(raw_stage_logs, summary_rows, model_name, args)
    save_log(
        logs,
        best_epoch,
        best_model,
        features,
        model_edges,
        model_kwargs,
        labels,
        test_nodes,
        trn_labels,
        trn_nodes,
        model_name,
        args,
        summary_rows=summary_rows,
        em_iter=i + 1,
    )

    return

if __name__ == '__main__':
    main()
