import os
import json
import csv
import io
import zipfile
import urllib.request
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import wikipedia_network, Actor, AttributedGraphDataset, Coauthor, Amazon, Planetoid

try:
    from torch_geometric.datasets import HeterophilousGraphDataset
except Exception:
    HeterophilousGraphDataset = None

try:
    from torch_geometric.typing import SparseTensor
except Exception:
    SparseTensor = None


def to_undirected_edges(edges):
    """
    Preprocess edges to make sure the following:
    1) No self-loops.
    2) Each pair (a, b) and (b, a) exists exactly once.
    """
    m = defaultdict(set)
    for src, dst in edges.t():
        src = int(src)
        dst = int(dst)
        if src != dst:
            m[src].add(dst)
            m[dst].add(src)

    undirected = []
    for src in sorted(m):
        for dst in sorted(m[src]):
            undirected.append((src, dst))
    return np.array(undirected, dtype=np.int64).transpose()


def compute_heterophily_ratio(edges, labels):
    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    labels_np = labels.cpu().numpy().astype(np.int64, copy=False)
    undirected_mask = src < dst
    if undirected_mask.size == 0:
        return float("nan")
    src_u = src[undirected_mask]
    dst_u = dst[undirected_mask]
    if src_u.size == 0:
        return float("nan")
    same = labels_np[src_u] == labels_np[dst_u]
    return float((~same).mean())


def preprocess_edges(edges: torch.Tensor, labels: torch.Tensor,
                           drop_pp: float = 0.0, drop_nn: float = 0.0,
                           seed: int = 0, drop_auto: bool = False,
                           log_stats: bool = False):
    """
    Drop fractions of P-P, N-N, and P-N edges without creating dangling nodes.
    edges: (2, M) with both directions present.
    drop_pp/drop_nn: 0..1 fractions of undirected P-P / N-N edges to drop.
    drop_auto: if True, compute drop_pp/drop_nn from current graph stats.
    """
    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    labels_np = labels.cpu().numpy().astype(np.int64, copy=False)
    undirected_mask = src < dst
    undirected_edges = np.stack([src[undirected_mask], dst[undirected_mask]], axis=1)
    if undirected_edges.shape[0] == 0:
        return edges

    pp_mask = (labels_np[undirected_edges[:, 0]] == 1) & (labels_np[undirected_edges[:, 1]] == 1)
    nn_mask = (labels_np[undirected_edges[:, 0]] == 0) & (labels_np[undirected_edges[:, 1]] == 0)
    pp_edges = undirected_edges[pp_mask]
    nn_edges = undirected_edges[nn_mask]
    pn_mask = labels_np[undirected_edges[:, 0]] != labels_np[undirected_edges[:, 1]]
    if drop_auto:
        total_ = int(undirected_edges.shape[0])
        pp_ = int(pp_edges.shape[0])
        nn_ = int(nn_edges.shape[0])
        pn_ = int(pn_mask.sum())
        hom_ = pp_ + nn_
        het_ = total_ - hom_
        if log_stats:
            print(f"[Info] edges(undirected)[before_drop]: total={total_} homophilic={hom_} heterophilic={het_}")
            print(f"[Info] edges(undirected)[before_drop]: p-p={pp_} p-n={pn_} n-n={nn_}")
        target = 1.0 * float(pn_)
        if nn_ <= target or nn_ <= 0:
            drop_nn = 0.0
        else:
            drop_nn = 1.0 - (target / float(nn_))
            drop_nn = max(0.0, min(1.0, drop_nn))
        if pp_ <= target or pp_ <= 0:
            drop_pp = 0.0
        else:
            drop_pp = 1.0 - (target / float(pp_))
            drop_pp = max(0.0, min(1.0, drop_pp))
        if log_stats:
            print(
                f"[Info] drop_auto: target=1.0*pn={target:.1f} "
                f"drop_nn={drop_nn:.3f} drop_pp={drop_pp:.3f}"
            )

    if drop_pp <= 0 and drop_nn <= 0:
        return edges
    if pp_edges.shape[0] == 0 and nn_edges.shape[0] == 0:
        return edges

    target_pp = int(np.floor(pp_edges.shape[0] * float(drop_pp))) if drop_pp > 0 else 0
    target_nn = int(np.floor(nn_edges.shape[0] * float(drop_nn))) if drop_nn > 0 else 0
    if target_pp <= 0 and target_nn <= 0:
        return edges

    num_nodes = labels_np.shape[0]
    deg = np.zeros(num_nodes, dtype=np.int64)
    for u, v in undirected_edges:
        deg[u] += 1
        deg[v] += 1

    rng = np.random.RandomState(seed)
    removed = []
    if target_pp > 0 and pp_edges.shape[0] > 0:
        order_pp = rng.permutation(pp_edges.shape[0])
        for idx in order_pp:
            if len(removed) >= target_pp:
                break
            u, v = pp_edges[idx]
            if deg[u] > 1 and deg[v] > 1:
                deg[u] -= 1
                deg[v] -= 1
                removed.append((int(u), int(v)))

    if target_nn > 0 and nn_edges.shape[0] > 0:
        order_nn = rng.permutation(nn_edges.shape[0])
        removed_nn = 0
        for idx in order_nn:
            if removed_nn >= target_nn:
                break
            u, v = nn_edges[idx]
            if deg[u] > 1 and deg[v] > 1:
                deg[u] -= 1
                deg[v] -= 1
                removed.append((int(u), int(v)))
                removed_nn += 1

    if not removed:
        return edges

    rem = np.array(removed, dtype=np.int64)
    rem_key = rem[:, 0] * num_nodes + rem[:, 1]
    pair_min = np.minimum(src, dst)
    pair_max = np.maximum(src, dst)
    pair_key = pair_min * num_nodes + pair_max
    keep_mask = ~np.isin(pair_key, rem_key)
    return edges[:, keep_mask]


def to_pu_setting(labels):
    """
    Make PU labels by selecting the most frequent class as positive.
    If binary, select the smaller class as positive.
    """
    labels = np.asarray(labels)
    valid = labels >= 0
    if not np.any(valid):
        return np.zeros_like(labels)
    count = np.bincount(labels[valid])
    if count.size == 2:
        pos_cls = int(count.argmin())
    else:
        pos_cls = int(count.argmax())
    positive_nodes = labels == pos_cls
    pu_labels = np.zeros_like(labels)
    pu_labels[positive_nodes] = 1
    return pu_labels


def to_bgp_pu_setting(labels):
    """
    Convert BGP labels to PU labels while keeping the original unlabeled majority class as -1.
    The second most frequent class becomes positive, other labeled classes become negative.
    """
    labels = np.asarray(labels)
    valid = labels >= 0
    if not np.any(valid):
        return np.full_like(labels, -1)

    count = np.bincount(labels[valid])
    sorted_classes = np.argsort(count)[::-1]
    if sorted_classes.size == 0:
        return np.full_like(labels, -1)

    unlabeled_cls = int(sorted_classes[0])
    pos_cls = int(sorted_classes[1]) if sorted_classes.size >= 2 else unlabeled_cls

    pu_labels = np.full(labels.shape, -1, dtype=np.int64)
    labeled_mask = valid & (labels != unlabeled_cls)
    pu_labels[labeled_mask] = 0
    pu_labels[labels == pos_cls] = 1
    return pu_labels


def _load_bgp_raw(dir_path: str):
    """Load BGP dataset from raw files in dir_path."""
    feat_path = os.path.join(dir_path, "as-feats.npy")
    class_map_path = os.path.join(dir_path, "as-class_map.json")
    edge_path = os.path.join(dir_path, "as-edge_list")
    if not (os.path.exists(feat_path) and os.path.exists(class_map_path) and os.path.exists(edge_path)):
        raise FileNotFoundError("Missing BGP raw files (as-feats.npy, as-class_map.json, as-edge_list).")

    x = np.load(feat_path)
    if x.ndim != 2:
        raise ValueError(f"Invalid as-feats.npy shape: {x.shape}")
    x = x.astype(np.float32, copy=False)

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    num_nodes = x.shape[0]
    y = np.full((num_nodes,), -1, dtype=np.int64)
    for k, v in class_map.items():
        idx = int(k)
        if isinstance(v, (list, tuple, np.ndarray)):
            vv = np.asarray(v)
            if vv.sum() > 0:
                y[idx] = int(vv.argmax())
        else:
            y[idx] = int(v)

    edges = np.loadtxt(edge_path, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Invalid as-edge_list shape: {edges.shape}")
    edges_rev = edges[:, [1, 0]]
    edges_all = np.concatenate([edges, edges_rev], axis=0)
    edges_all = np.unique(edges_all, axis=0)
    edge_index = torch.from_numpy(edges_all.T).to(torch.long)

    return Data(x=torch.from_numpy(x), y=torch.from_numpy(y), edge_index=edge_index)


def _load_filtered_npz(root_dir: str, dataset: str):
    """Load filtered chameleon/squirrel npz files from root_dir/npz."""
    filename = f"{dataset.replace('-', '_')}.npz"
    path = os.path.join(root_dir, "npz", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing filtered npz file: {path}")

    with np.load(path, allow_pickle=True) as data:
        x = data["node_features"].astype(np.float32, copy=False)
        y = data["node_labels"].astype(np.int64, copy=False)
        edges = data["edges"].astype(np.int64, copy=False)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Invalid edges shape in {path}: {edges.shape}")

    edge_index = torch.from_numpy(edges.T).to(torch.long)
    return Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=edge_index,
    )


def _twitch_name(dataset: str):
    """Map CLI dataset names like `twitch-de` to PyG's uppercase country codes."""
    twitch_map = {
        "twitch-de": "DE",
        "twitch-en": "ENGB",
        "twitch-es": "ES",
        "twitch-fr": "FR",
        "twitch-pt": "PTBR",
        "twitch-ru": "RU",
    }
    return twitch_map.get(dataset)


def _download_and_extract_twitch(root_dir: str, twitch_name: str):
    url = "https://snap.stanford.edu/data/twitch.zip"
    twitch_root = os.path.join(root_dir, "twitch")
    dataset_dir = os.path.join(twitch_root, twitch_name)
    edges_path = os.path.join(dataset_dir, f"musae_{twitch_name}_edges.csv")
    target_path = os.path.join(dataset_dir, f"musae_{twitch_name}_target.csv")
    feature_json_path = os.path.join(dataset_dir, f"musae_{twitch_name}_features.json")
    legacy_json_path = os.path.join(dataset_dir, f"musae_{twitch_name}.json")

    if os.path.exists(edges_path) and os.path.exists(target_path) and (
        os.path.exists(feature_json_path) or os.path.exists(legacy_json_path)
    ):
        return dataset_dir

    os.makedirs(twitch_root, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        prefix = f"twitch/{twitch_name}/"
        members = [name for name in zf.namelist() if name.startswith(prefix)]
        if not members:
            raise FileNotFoundError(f"Twitch archive does not contain split: {twitch_name}")
        zf.extractall(path=root_dir, members=members)
    return dataset_dir


def _load_twitch_raw(root_dir: str, dataset: str):
    twitch_name = _twitch_name(dataset)
    if twitch_name is None:
        raise ValueError(f"Unknown Twitch dataset: {dataset}")

    dataset_dir = _download_and_extract_twitch(root_dir, twitch_name)
    edges_path = os.path.join(dataset_dir, f"musae_{twitch_name}_edges.csv")
    target_path = os.path.join(dataset_dir, f"musae_{twitch_name}_target.csv")
    feature_json_path = os.path.join(dataset_dir, f"musae_{twitch_name}_features.json")
    legacy_json_path = os.path.join(dataset_dir, f"musae_{twitch_name}.json")
    features_path = feature_json_path if os.path.exists(feature_json_path) else legacy_json_path

    target_rows = []
    with open(target_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_rows.append(row)
    if not target_rows:
        raise ValueError(f"No target rows found in {target_path}")

    num_nodes = max(int(row["new_id"]) for row in target_rows) + 1
    y = np.zeros((num_nodes,), dtype=np.int64)
    for row in target_rows:
        node_id = int(row["new_id"])
        y[node_id] = 1 if str(row["mature"]).strip().lower() == "true" else 0

    with open(features_path, "r", encoding="utf-8") as f:
        feature_map = json.load(f)
    max_feat = -1
    for feat_ids in feature_map.values():
        if feat_ids:
            max_feat = max(max_feat, max(int(fid) for fid in feat_ids))
    feat_dim = max_feat + 1
    x = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    for node_id_str, feat_ids in feature_map.items():
        node_id = int(node_id_str)
        if not feat_ids:
            continue
        x[node_id, np.asarray(feat_ids, dtype=np.int64)] = 1.0

    edge_list = []
    with open(edges_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge_list.append((int(row["from"]), int(row["to"])))
    if not edge_list:
        raise ValueError(f"No edges found in {edges_path}")

    edge_index = torch.from_numpy(np.asarray(edge_list, dtype=np.int64).T).to(torch.long)
    return Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=edge_index,
    )


def split_nodes(labels, trn_ratio, seed=0):
    state = np.random.RandomState(seed)
    pos_nodes = np.argwhere(labels.cpu().numpy() == 1).reshape(-1)
    neg_nodes = np.argwhere(labels.cpu().numpy() == 0).reshape(-1)

    n_pos_nodes = len(pos_nodes)
    if n_pos_nodes == 0:
        return np.zeros((0,), dtype=np.int64), neg_nodes.astype(np.int64, copy=False)

    if n_pos_nodes == 1:
        trn_nodes = pos_nodes.astype(np.int64, copy=False)
        test_nodes = neg_nodes.astype(np.int64, copy=False)
        return trn_nodes, test_nodes

    n_trn_nodes = int(n_pos_nodes * trn_ratio)
    n_trn_nodes = max(1, min(n_pos_nodes - 1, n_trn_nodes))
    n_test_pos_nodes = n_pos_nodes - n_trn_nodes

    trn_nodes = state.choice(pos_nodes, size=n_trn_nodes, replace=False).astype(np.int64, copy=False)

    test_pos_candidates = np.array(list(set(pos_nodes).difference(set(trn_nodes))), dtype=np.int64)
    if n_test_pos_nodes > 0:
        test_pos_nodes = state.choice(test_pos_candidates, size=n_test_pos_nodes, replace=False).astype(np.int64, copy=False)
        test_nodes = np.concatenate([test_pos_nodes, neg_nodes.astype(np.int64, copy=False)])
    else:
        test_nodes = neg_nodes.astype(np.int64, copy=False)

    return trn_nodes, test_nodes


def read_data(dataset, trn_ratio, verbose=False, drop_auto=False,
              print_stats=False,
              seed=0):
    root = '../data'
    root_cached = os.path.join(root, 'cached', dataset)
    if dataset == "bgp":
        bgp_y_cache = os.path.join(root_cached, 'y.npy')
        bgp_raw_cache = os.path.join(root_cached, 'y_raw.npy')
        if os.path.exists(bgp_y_cache) and not os.path.exists(bgp_raw_cache):
            os.remove(bgp_y_cache)
    if not os.path.exists(root_cached):
        data_obj = None
        if dataset in ("cora", "citeseer", "pubmed"):
            data = Planetoid(os.path.join(root, "planetoid"), name=dataset)
        elif dataset in ('flickr', 'blogcatalog', 'facebook', 'ppi'):
            data = AttributedGraphDataset(os.path.join(root, "attributed"), dataset)
        elif dataset == 'bgp':
            bgp_dir = os.path.join(root, 'bgp')
            data_obj = _load_bgp_raw(bgp_dir)
        elif dataset in ('chameleon', 'squirrel'):
            data = wikipedia_network.WikipediaNetwork(os.path.join(root, 'wikics'), dataset)
        elif dataset in ('chameleon-filtered', 'squirrel-filtered'):
            data_obj = _load_filtered_npz(root, dataset)
        elif dataset == 'actor':
            data = Actor(os.path.join(root, 'actor'))
        elif dataset in ('tolokers', 'questions', 'amazon-ratings', 'roman-empire'):
            data = HeterophilousGraphDataset(os.path.join(root, 'heterophilous'), dataset)
        elif dataset in ("coauthor-cs", "coauthor-physics"):
            co_name = "CS" if dataset == "coauthor-cs" else "Physics"
            data = Coauthor(os.path.join(root, "coauthor"), name=co_name) 
        elif dataset in ("amazon-computers", "amazon-photo"):
            amz_name = "Computers" if dataset == "amazon-computers" else "Photo"
            data = Amazon(os.path.join(root, "amazon"), name=amz_name)
        elif _twitch_name(dataset) is not None:
            data_obj = _load_twitch_raw(root, dataset)
        else:
            raise ValueError(dataset)

        if data_obj is None:
            data_obj = data.data
        node_x = data_obj.x
        if node_x is None:
            num_nodes = int(data_obj.y.size(0))
            node_x = torch.eye(num_nodes, dtype=torch.float32)
        else:
            if SparseTensor is not None and isinstance(node_x, SparseTensor):
                node_x = node_x.to_dense()
            elif not torch.is_tensor(node_x):
                if hasattr(node_x, "todense"):
                    node_x = torch.from_numpy(np.asarray(node_x.todense()))
                else:
                    node_x = torch.from_numpy(np.asarray(node_x))
            node_x = node_x.to(torch.float)
        node_x[node_x.sum(dim=1) == 0] = 1
        node_x = node_x / node_x.sum(dim=1, keepdim=True)

        y_np = data_obj.y.detach().cpu().numpy().astype(np.int64, copy=False)
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.reshape(-1)
        elif y_np.ndim == 2 and y_np.shape[1] > 1:
            # Multi-label: pick the most frequent class as positive.
            class_counts = y_np.sum(axis=0)
            top_class = int(class_counts.argmax())
            node_y = (y_np[:, top_class] > 0).astype(np.int64, copy=False)
        else:
            node_y = None
        if dataset == "bgp":
            node_y = to_bgp_pu_setting(y_np)
        elif node_y is None:
            node_y = to_pu_setting(y_np)
        edges = to_undirected_edges(data_obj.edge_index)
        os.makedirs(root_cached, exist_ok=True)
        np.save(os.path.join(root_cached, 'x'), node_x)
        np.save(os.path.join(root_cached, 'y'), node_y)
        if dataset == "bgp":
            np.save(os.path.join(root_cached, 'y_raw'), y_np.astype(np.int64, copy=False))
        np.save(os.path.join(root_cached, 'edges'), edges)

    node_x = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'x.npy'), allow_pickle=True), dtype=np.float32))
    node_y = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'y.npy'), allow_pickle=True), dtype=np.int64))
    if node_y.ndim == 2 and node_y.size(1) == 1:
        node_y = node_y.view(-1)
    elif node_y.ndim == 2 and node_y.size(1) > 1:
        # Handle legacy cached multi-label targets (e.g., facebook).
        class_counts = node_y.sum(dim=0)
        top_class = int(class_counts.argmax().item())
        node_y = (node_y[:, top_class] > 0).to(torch.int64)
        np.save(os.path.join(root_cached, 'y'), node_y.numpy().astype(np.int64, copy=False))
    # If cached labels are not binary PU labels, convert them now.
    if dataset == "bgp":
        raw_path = os.path.join(root_cached, 'y_raw.npy')
        if os.path.exists(raw_path):
            raw_y_np = np.array(np.load(raw_path, allow_pickle=True), dtype=np.int64)
            node_y_np = to_bgp_pu_setting(raw_y_np)
            node_y = torch.from_numpy(node_y_np.astype(np.int64, copy=False))
            np.save(os.path.join(root_cached, 'y'), node_y_np.astype(np.int64, copy=False))
    elif node_y.ndim == 1 and (int(node_y.max().item()) > 1 or int(node_y.min().item()) < 0):
        node_y_np = to_pu_setting(node_y.cpu().numpy())
        node_y = torch.from_numpy(node_y_np.astype(np.int64, copy=False))
        np.save(os.path.join(root_cached, 'y'), node_y_np.astype(np.int64, copy=False))
    edges = torch.from_numpy(np.load(os.path.join(root_cached, 'edges.npy'), allow_pickle=True))
    edges_before_drop = edges
    if drop_auto:
        edges = preprocess_edges(edges, node_y, seed=0, drop_auto=True, log_stats=print_stats)

    def summarize_edges(edge_index, labels):
        src_local = edge_index[0]
        dst_local = edge_index[1]
        undirected_mask_local = src_local < dst_local
        if undirected_mask_local.numel() > 0 and bool(undirected_mask_local.any().item()):
            u_local = src_local[undirected_mask_local]
            v_local = dst_local[undirected_mask_local]
            valid_local = (labels[u_local] >= 0) & (labels[v_local] >= 0)
            total_local = int(valid_local.sum().item())
            if total_local > 0:
                same_local = labels[u_local[valid_local]] == labels[v_local[valid_local]]
                hetero_local = float((~same_local).float().mean().item())
                return total_local, hetero_local
        return 0, float("nan")

    total_before, hetero_before = summarize_edges(edges_before_drop, node_y)
    total_after, hetero_after = summarize_edges(edges, node_y)
    pos_cnt = int((node_y == 1).sum().item())
    neg_cnt = int((node_y == 0).sum().item())
    unk_cnt = int((node_y < 0).sum().item())
    if print_stats:
        print('Number of nodes: total={}, pos={}, neg={}, unknown={}'.format(node_x.size(0), pos_cnt, neg_cnt, unk_cnt))
        print('Number of features:', node_x.size(1))
        print('Number of edges:', total_before)
        print('Edge heterophily ratio:', hetero_before)
        if drop_auto:
            print('Number of edges [after_drop]:', total_after)
            print('Edge heterophily ratio [after_drop]:', hetero_after)

    trn_nodes, test_nodes = split_nodes(node_y, trn_ratio, seed=0)
    if verbose:
        print('Number of nodes:', node_x.size(0))
        print('Number of features:', node_x.size(1))
        print('Number of edges:', edges.size())
        print('Edge heterophily ratio:', compute_heterophily_ratio(edges_before_drop, node_y))
        print('Number of positive nodes:', (node_y == 1).sum().item())
        print('Number of negative nodes:', (node_y == 0).sum().item())
    return node_x, node_y, edges, trn_nodes, test_nodes


def main():
    pass


if __name__ == '__main__':
    main()
