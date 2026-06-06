"""Shared utilities for training, evaluation, and checkpoint management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format for CLI scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute binary classification accuracy from raw logits and ground-truth labels.

    Args:
        logits: Model output of shape (N, 1) or (N,) before sigmoid.
        labels: Ground-truth labels of shape (N, 1) or (N,) with values 0 or 1.

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    predictions = (torch.sigmoid(logits) > 0.5).float()
    labels = labels.float()

    correct = (predictions == labels).sum().item()
    total = labels.numel()

    if total == 0:
        logger.warning("calculate_accuracy called with an empty batch.")
        return 0.0

    return correct / total


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str = "best_model.pt",
) -> Path:
    """
    Persist model weights, optimizer state, and training metadata to disk.

    Args:
        model: Trained PyTorch module.
        optimizer: Optimizer instance used during training.
        epoch: Current epoch index (1-based or 0-based; stored as given).
        metrics: Dictionary of scalar metrics to archive (e.g., val_loss, val_acc).
        checkpoint_dir: Directory where the checkpoint file will be written.
        filename: Checkpoint file name.

    Returns:
        Absolute path to the saved checkpoint file.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    destination = checkpoint_path / filename
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "device": str(DEVICE),
    }

    torch.save(payload, destination)
    logger.info("Checkpoint saved to %s (epoch=%d)", destination, epoch)
    return destination.resolve()


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """
    Restore model (and optionally optimizer) weights from a checkpoint file.

    Args:
        checkpoint_path: Path to a checkpoint created by ``save_checkpoint``.
        model: Module whose parameters will be loaded.
        optimizer: Optional optimizer to restore.

    Returns:
        Checkpoint metadata dictionary excluding state dicts.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Checkpoint loaded from %s (epoch=%d)", path, checkpoint.get("epoch", -1))

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }
