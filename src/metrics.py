"""Classification metrics for binary and multi-class training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def calculate_multiclass_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy from raw logits and integer class labels."""
    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.numel()

    if total == 0:
        return 0.0

    return correct / total


def compute_multiclass_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """
    Compute macro F1, per-class F1, and a confusion matrix.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Human-readable names for each class index.

    Returns:
        Dictionary of scalar metrics and the confusion matrix array.
    """
    labels = list(range(len(class_names)))
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    accuracy = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0
    f1_macro = float(
        f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0)
    )
    f1_per_class = f1_score(
        y_true_arr, y_pred_arr, average=None, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
    }

    for class_index, class_name in enumerate(class_names):
        metrics[f"f1/{class_name}"] = float(f1_per_class[class_index])

    return metrics
