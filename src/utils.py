"""Shared utilities for training, evaluation, and checkpoint management."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """
    Resolve a device, falling back to CPU when CUDA is unavailable.

    The project originally hardcoded ``torch.device("cuda")`` at module scope in
    six files, which made it impossible to instantiate a model on a CPU-only
    machine -- and therefore impossible to benchmark CPU or ARM inference, which
    is the point of the edge-deployment analysis. Callers should pass a device
    explicitly; this helper only supplies the default.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available — falling back to CPU.")
    return torch.device("cpu")


# Default device. Kept as a module attribute for backward compatibility with
# existing call sites, but it is a default, not a constant: every model and
# trainer accepts an explicit device argument that overrides it.
DEVICE = resolve_device()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed every source of randomness that affects a training run.

    Covers Python's ``random``, NumPy, PyTorch CPU and all CUDA devices, plus
    the ``PYTHONHASHSEED`` environment variable. DataLoader worker processes are
    seeded separately via :func:`seed_worker` and :func:`make_generator`, which
    must both be passed to every ``DataLoader`` for shuffling to be reproducible.

    Args:
        seed: The seed value.
        deterministic: If True, request deterministic cuDNN kernels. This costs
            some throughput but makes a run bit-reproducible. Benchmarking runs
            should pass False, since determinism perturbs kernel selection and
            therefore latency.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logger.info("Seeded all RNGs with %d (deterministic=%s).", seed, deterministic)


def seed_worker(worker_id: int) -> None:
    """
    Seed a DataLoader worker process.

    Each worker inherits a distinct ``torch.initial_seed()`` derived from the
    base seed, so augmentation randomness differs across workers but is
    reproducible across runs. Pass as ``worker_init_fn=seed_worker``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Build a seeded generator for DataLoader shuffling (``generator=``)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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
    device: str | torch.device | None = None,
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

    checkpoint = torch.load(path, map_location=resolve_device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Checkpoint loaded from %s (epoch=%d)", path, checkpoint.get("epoch", -1))

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }
