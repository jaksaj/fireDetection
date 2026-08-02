"""Classification metrics for binary and multi-class training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


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


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute the metric set a fire detector should actually be judged on.

    Accuracy is retained for continuity but is not the headline: the D-Fire
    test split is ~74% non-fire, so a detector that never fires scores 0.74.
    What matters operationally is:

    - **recall** -- the fraction of real fires caught. Misses are the costly error.
    - **false_alarm_rate** -- FP / (FP + TN). Alarm fatigue is what gets a
      deployed detector switched off.
    - **pr_auc / roc_auc** -- threshold-free summaries, so the comparison does
      not hinge on the arbitrary 0.5 cutoff.
    - **best_f1_threshold** -- the operating point that maximises F1, which is
      the number a deployment would actually be tuned to.

    ``y_score`` is optional; without it the threshold-dependent metrics are
    omitted rather than faked.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()

    if not len(y_true_arr):
        return {}

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    true_neg, false_pos, false_neg, true_pos = cm.ravel()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics: dict[str, Any] = {
        "accuracy": float((y_true_arr == y_pred_arr).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "false_alarm_rate": float(1.0 - specificity),
        "miss_rate": float(1.0 - recall),
        "true_positives": int(true_pos),
        "false_positives": int(false_pos),
        "true_negatives": int(true_neg),
        "false_negatives": int(false_neg),
        "confusion_matrix": cm,
    }

    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=np.float64).ravel()
        if len(score_arr) == len(y_true_arr) and len(np.unique(y_true_arr)) > 1:
            metrics["pr_auc"] = float(average_precision_score(y_true_arr, score_arr))
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, score_arr))

            # Operating-point sweep: report the threshold maximising F1, so the
            # Discussion can argue about where to sit on the precision/recall
            # trade-off rather than silently accepting 0.5.
            precisions, recalls, thresholds = precision_recall_curve(y_true_arr, score_arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_curve = np.nan_to_num(
                    2 * precisions * recalls / (precisions + recalls), nan=0.0
                )
            best_index = int(np.argmax(f1_curve[:-1])) if len(thresholds) else 0
            if len(thresholds):
                metrics["best_f1"] = float(f1_curve[best_index])
                metrics["best_f1_threshold"] = float(thresholds[best_index])
                metrics["best_f1_precision"] = float(precisions[best_index])
                metrics["best_f1_recall"] = float(recalls[best_index])

    return metrics
