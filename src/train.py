import io

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def _forward_model(model, features, edges, model_kwargs=None, return_embedding: bool = False):
    if model_kwargs:
        if "pos_edge_index" in model_kwargs:
            return model(
                features,
                model_kwargs["pos_edge_index"],
                model_kwargs["neg_edge_index"],
                model_kwargs["pos_weight"],
                model_kwargs["neg_weight"],
                return_embedding=return_embedding,
            )
        return model(features, edges, return_embedding=return_embedding, **model_kwargs)
    return model(features, edges, return_embedding=return_embedding)


def _compute_unary_edge_logits(model, embeddings, edge_supervision):
    u = edge_supervision["u"]
    v = edge_supervision["v"]
    if hasattr(model, "link_sign_logits"):
        return model.link_sign_logits(embeddings, u, v)
    return (embeddings[u] * embeddings[v]).sum(dim=1)


def _safe_roc_auc(y_true, y_score):
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true, y_score):
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def train_model(model, features, edges, labels, test_nodes, loss_func, optimizer,
                trn_labels, epochs, patience, model_kwargs=None,
                encoder=None, cont_pairs=None, cont_weight=0.0, cont_margin=1.0,
                eval_each_epoch=False):
    """
    train the model.
    :param model: model to use.
    :param features: features of data to use
    :param edges: edge information of data
    :param labels: true labels of data (transductive)
    :param test_nodes: index list of test nodes.
    :param loss_func: loss function for model
    :param optimizer: optimizer for model
    :param trn_labels: train labels of data which is actually used to train model
    :param epochs: number of epochs to run
    :param patience: number of patience
    :return:
    """
    logs = []
    saved_model, best_epoch = io.BytesIO(), -1
    best_loss = np.inf
    edge_supervision = None
    if hasattr(loss_func, "get_edge_supervision"):
        edge_supervision = loss_func.get_edge_supervision()

    for epoch in range(epochs + 1):
        model.train()
        if edge_supervision is not None:
            out, embeddings = _forward_model(model, features, edges, model_kwargs, return_embedding=True)
            out = out[:len(labels)]
            edge_logits = _compute_unary_edge_logits(model, embeddings, edge_supervision)
            loss = loss_func(
                out,
                trn_labels,
                edge_logits=edge_logits,
                edge_target=edge_supervision["target"],
                edge_weight=edge_supervision["weight"],
                embeddings=embeddings,
            )
        else:
            out = _forward_model(model, features, edges, model_kwargs)
            out = out[:len(labels)]
            loss = loss_func(out, trn_labels)
        loss_components = {}
        if hasattr(loss_func, "get_last_components"):
            loss_components = loss_func.get_last_components()
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss encountered at epoch {epoch}: {loss.item()}. "
                "No valid checkpoint was saved."
            )
        if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if eval_each_epoch:
            test_f1, test_acc, test_roc_auc, test_pr_auc = evaluate_model(
                model, features, edges, labels, test_nodes, model_kwargs
            )
        else:
            test_f1 = test_acc = test_roc_auc = test_pr_auc = float("nan")
        row = {
            "epoch": epoch,
            "trn_loss": float(loss.item()),
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "test_roc_auc": float(test_roc_auc),
            "test_pr_auc": float(test_pr_auc),
        }
        row.update(loss_components)
        logs.append(row)
        if loss.item() < best_loss:
            best_epoch = epoch
            best_loss = loss.item()
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)

        if patience > 0 and epoch >= best_epoch + patience:
            break

    if best_epoch < 0:
        raise RuntimeError("Training finished without saving a valid checkpoint.")

    saved_model.seek(0)
    try:
        state = torch.load(saved_model, weights_only=True)
    except TypeError:
        state = torch.load(saved_model)
    model.load_state_dict(state)
    return best_epoch, pd.DataFrame(logs), best_loss, model

def evaluate_model(model, features, edges, labels, test_nodes, model_kwargs=None):
    model.eval()
    with torch.no_grad():
        out = _forward_model(model, features, edges, model_kwargs).cpu()
        out_labels = (out > 0).int()
        out_score = torch.sigmoid(out)

    y_true = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    test_idx = np.asarray(test_nodes, dtype=np.int64)
    y_test = y_true[test_idx]
    y_pred = out_labels.numpy()[test_idx]
    y_score = out_score.numpy()[test_idx]
    test_f1 = float(f1_score(y_test, y_pred)) if y_test.size > 0 else float("nan")
    acc = float(accuracy_score(y_test, y_pred)) if y_test.size > 0 else float("nan")
    roc_auc = _safe_roc_auc(y_test, y_score)
    pr_auc = _safe_pr_auc(y_test, y_score)
    return test_f1, acc, roc_auc, pr_auc
