# src/metrics.py
import torch
import numpy as np


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for a batch.
    Args:
        output: logits or probabilities (B, C)
        target: ground truth labels (B,)
    Returns:
        accuracy in [0, 1]
    """
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct / target.size(0)


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """
    Compute top-k accuracy for a batch.
    Args:
        output: logits or probabilities (B, C)
        target: ground truth labels (B,)
        k: number of top predictions to consider
    Returns:
        accuracy in [0, 1]
    """
    _, topk_idx = output.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_idx.eq(target.view(-1, 1).expand_as(topk_idx)).any(dim=1).sum().item()
    return correct / target.size(0)


def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray, all_probs: np.ndarray,
                    idx_to_label: dict = None, k: int = 5) -> dict:
    """
    Compute evaluation metrics on the full dataset.
    Args:
        all_labels: true labels (N,)
        all_preds: predicted labels (N,)
        all_probs: probability matrix (N, C)
        idx_to_label: optional index-to-class-name mapping
        k: top-k value
    Returns:
        dict with keys: 'top1_acc', 'topk_acc', 'confusion_matrix', 'per_class_acc'
    """
    top1 = (all_labels == all_preds).mean()
    topk_indices = np.argsort(all_probs, axis=1)[:, -k:]
    topk_correct = np.any(topk_indices == all_labels.reshape(-1, 1), axis=1).mean()

    # Confusion matrix
    num_classes = all_probs.shape[1]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Per-class accuracy
    per_class = np.zeros(num_classes)
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[i] = (all_preds[mask] == i).mean()

    result = {
        "top1_acc": top1,
        f"top{k}_acc": topk_correct,
        "confusion_matrix": cm,
        "per_class_acc": per_class,
    }
    if idx_to_label:
        result["per_class_dict"] = {idx_to_label[i]: per_class[i] for i in range(num_classes)}
    return result