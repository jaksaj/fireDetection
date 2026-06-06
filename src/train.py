"""Training loop for Iteration 1 binary fire classification."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.utils import calculate_accuracy, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


@dataclass
class EpochMetrics:
    """Container for aggregated metrics from a single training or validation epoch."""

    loss: float
    accuracy: float
    duration_sec: float


class Trainer:
    """
    Object-oriented training orchestrator for Iteration 1.

    Handles forward/backward passes, ``BCEWithLogitsLoss``, optimizer stepping,
    checkpoint persistence, and Weights & Biases logging. All batches and the
    model are explicitly moved to CUDA.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints/iteration1",
        wandb_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.wandb_config = wandb_config or {}
        self._wandb_run: Optional[Any] = None
        self.best_val_loss = float("inf")

        logger.info("Trainer initialized on device: %s", DEVICE)

    def _init_wandb(self, config: dict[str, Any]) -> None:
        """Initialize a Weights & Biases run with the merged experiment config."""
        init_kwargs = {
            "project": self.wandb_config.get("project", "smoke-fire-detection"),
            "config": config,
            "tags": self.wandb_config.get("tags", ["iteration1"]),
        }

        if self.wandb_config.get("entity"):
            init_kwargs["entity"] = self.wandb_config["entity"]
        if self.wandb_config.get("run_name"):
            init_kwargs["name"] = self.wandb_config["run_name"]

        self._wandb_run = wandb.init(**init_kwargs)
        wandb.watch(self.model, log="gradients", log_freq=100)
        logger.info("W&B run started: %s", self._wandb_run.name)

    def _run_epoch(self, loader: DataLoader, training: bool) -> EpochMetrics:
        """Execute one epoch of training or validation."""
        self.model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_time = time.perf_counter()

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for images, labels in loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += int(
                    calculate_accuracy(logits.detach(), labels) * batch_size
                )
                total_samples += batch_size

        duration_sec = time.perf_counter() - start_time
        avg_loss = total_loss / max(total_samples, 1)
        avg_accuracy = total_correct / max(total_samples, 1)

        return EpochMetrics(
            loss=avg_loss,
            accuracy=avg_accuracy,
            duration_sec=duration_sec,
        )

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Run a single training epoch and log metrics."""
        metrics = self._run_epoch(self.train_loader, training=True)
        logger.info(
            "Epoch %03d [train] loss=%.4f acc=%.4f time=%.2fs",
            epoch,
            metrics.loss,
            metrics.accuracy,
            metrics.duration_sec,
        )
        return metrics

    def validate_epoch(self, epoch: int) -> EpochMetrics:
        """Run a single validation epoch and log metrics."""
        metrics = self._run_epoch(self.val_loader, training=False)
        logger.info(
            "Epoch %03d [val]   loss=%.4f acc=%.4f time=%.2fs",
            epoch,
            metrics.loss,
            metrics.accuracy,
            metrics.duration_sec,
        )
        return metrics

    def evaluate(self, loader: DataLoader, split_name: str = "test") -> EpochMetrics:
        """Run inference on a held-out DataLoader (e.g. test split)."""
        metrics = self._run_epoch(loader, training=False)
        logger.info(
            "[%s] loss=%.4f acc=%.4f time=%.2fs",
            split_name,
            metrics.loss,
            metrics.accuracy,
            metrics.duration_sec,
        )
        return metrics

    def fit(
        self,
        epochs: int,
        experiment_config: Optional[dict[str, Any]] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> dict[str, Any]:
        """
        Full training loop across ``epochs`` with W&B logging and checkpointing.

        Args:
            epochs: Number of epochs to train.
            experiment_config: Hyperparameters and metadata logged to W&B.

        Returns:
            Summary dictionary with best validation metrics and checkpoint path.
        """
        config = experiment_config or {}
        self._init_wandb(config)

        best_checkpoint_path: Optional[str] = None
        test_metrics: Optional[EpochMetrics] = None

        try:
            for epoch in range(1, epochs + 1):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate_epoch(epoch)

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics.loss,
                        "train/accuracy": train_metrics.accuracy,
                        "train/epoch_time_sec": train_metrics.duration_sec,
                        "val/loss": val_metrics.loss,
                        "val/accuracy": val_metrics.accuracy,
                        "val/epoch_time_sec": val_metrics.duration_sec,
                    },
                    step=epoch,
                )

                if val_metrics.loss < self.best_val_loss:
                    self.best_val_loss = val_metrics.loss
                    checkpoint = save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics={
                            "val_loss": val_metrics.loss,
                            "val_accuracy": val_metrics.accuracy,
                            "train_loss": train_metrics.loss,
                            "train_accuracy": train_metrics.accuracy,
                        },
                        checkpoint_dir=self.checkpoint_dir,
                        filename="best_model.pt",
                    )
                    best_checkpoint_path = str(checkpoint)

                    wandb.log(
                        {
                            "best/val_loss": val_metrics.loss,
                            "best/val_accuracy": val_metrics.accuracy,
                        },
                        step=epoch,
                    )

            if test_loader is not None:
                if best_checkpoint_path is not None:
                    load_checkpoint(best_checkpoint_path, self.model, self.optimizer)
                test_metrics = self.evaluate(test_loader, split_name="test")
                wandb.log(
                    {
                        "test/loss": test_metrics.loss,
                        "test/accuracy": test_metrics.accuracy,
                        "test/epoch_time_sec": test_metrics.duration_sec,
                    },
                    step=epochs,
                )

        finally:
            if self._wandb_run is not None:
                wandb.finish()

        summary: dict[str, Any] = {
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": best_checkpoint_path,
            "epochs_completed": epochs,
        }

        if test_metrics is not None:
            summary["test_loss"] = test_metrics.loss
            summary["test_accuracy"] = test_metrics.accuracy

        logger.info("Training complete: %s", summary)
        return summary
