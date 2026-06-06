"""Shared training loop infrastructure."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.utils import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


@dataclass
class EpochMetrics:
    """Container for aggregated metrics from a single training or validation epoch."""

    loss: float
    accuracy: float
    duration_sec: float
    extras: dict[str, Any] = field(default_factory=dict)


class BaseTrainer:
    """
    Reusable training orchestrator that accepts any model, criterion, and optimizer.

    Subclasses provide task-specific accuracy computation and optional extended
    validation metrics (e.g. per-class F1, confusion matrices).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str = "checkpoints",
        wandb_config: Optional[dict[str, Any]] = None,
        log_every_n_batches: int = 25,
        wandb_tags: Optional[list[str]] = None,
    ) -> None:
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.log_every_n_batches = max(1, log_every_n_batches)

        self.wandb_config = wandb_config or {}
        self.wandb_tags = wandb_tags or []
        self._wandb_run: Optional[Any] = None
        self.best_val_loss = float("inf")

        logger.info("BaseTrainer initialized on device: %s", DEVICE)

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Replace the optimizer (used when switching training phases)."""
        self.optimizer = optimizer

    def compute_batch_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        raise NotImplementedError

    def compute_epoch_extras(
        self,
        loader: DataLoader,
        training: bool,
    ) -> dict[str, Any]:
        """Optional hook for extended validation metrics."""
        return {}

    def _init_wandb(self, config: dict[str, Any]) -> None:
        init_kwargs = {
            "project": self.wandb_config.get("project", "smoke-fire-detection"),
            "config": config,
            "tags": self.wandb_tags,
        }

        if self.wandb_config.get("entity"):
            init_kwargs["entity"] = self.wandb_config["entity"]
        if self.wandb_config.get("run_name"):
            init_kwargs["name"] = self.wandb_config["run_name"]

        self._wandb_run = wandb.init(**init_kwargs)
        wandb.watch(self.model, log="gradients", log_freq=100)
        logger.info("W&B run started: %s", self._wandb_run.name)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: Optional[int] = None,
        collect_predictions: bool = False,
    ) -> EpochMetrics:
        self.model.train(training)
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        start_time = time.perf_counter()
        total_batches = len(loader)

        y_true: list[int] = []
        y_pred: list[int] = []

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_index, (images, labels) in enumerate(loader, start=1):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                batch_accuracy = self.compute_batch_accuracy(logits.detach(), labels)

                total_loss += loss.item() * batch_size
                total_correct += batch_accuracy * batch_size
                total_samples += batch_size

                if collect_predictions and not training:
                    predictions = logits.argmax(dim=1)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(predictions.cpu().tolist())

                if training and batch_index % self.log_every_n_batches == 0:
                    elapsed_sec = time.perf_counter() - start_time
                    batches_per_sec = batch_index / max(elapsed_sec, 1e-6)
                    running_loss = total_loss / max(total_samples, 1)
                    running_accuracy = total_correct / max(total_samples, 1)
                    epoch_prefix = f"Epoch {epoch:03d} " if epoch is not None else ""
                    logger.info(
                        "%s[train] batch %04d/%04d loss=%.4f acc=%.4f speed=%.2f it/s",
                        epoch_prefix,
                        batch_index,
                        total_batches,
                        running_loss,
                        running_accuracy,
                        batches_per_sec,
                    )

        duration_sec = time.perf_counter() - start_time
        extras = self.build_epoch_extras(y_true, y_pred) if collect_predictions else {}

        return EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
            duration_sec=duration_sec,
            extras=extras,
        )

    def build_epoch_extras(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, Any]:
        """Hook for subclasses to derive extended metrics from collected predictions."""
        return {}

    def train_epoch(self, epoch: int) -> EpochMetrics:
        metrics = self._run_epoch(self.train_loader, training=True, epoch=epoch)
        logger.info(
            "Epoch %03d [train] loss=%.4f acc=%.4f time=%.2fs",
            epoch,
            metrics.loss,
            metrics.accuracy,
            metrics.duration_sec,
        )
        return metrics

    def validate_epoch(self, epoch: int) -> EpochMetrics:
        metrics = self._run_epoch(
            self.val_loader,
            training=False,
            epoch=epoch,
            collect_predictions=True,
        )
        logger.info(
            "Epoch %03d [val]   loss=%.4f acc=%.4f time=%.2fs",
            epoch,
            metrics.loss,
            metrics.accuracy,
            metrics.duration_sec,
        )
        for key, value in metrics.extras.items():
            if key.startswith("f1/") or key in {"f1_macro"}:
                logger.info("Epoch %03d [val]   %s=%.4f", epoch, key, value)
        return metrics

    def evaluate(self, loader: DataLoader, split_name: str = "test") -> EpochMetrics:
        metrics = self._run_epoch(loader, training=False, collect_predictions=True)
        logger.info(
            "[%s] loss=%.4f acc=%.4f time=%.2fs",
            split_name,
            metrics.loss,
            metrics.accuracy,
            metrics.duration_sec,
        )
        return metrics

    @staticmethod
    def _phase_prefix(phase: str) -> str:
        return f"{phase}/" if phase else ""

    def _log_epoch_metrics(
        self,
        epoch: int,
        phase: str,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
    ) -> None:
        prefix = self._phase_prefix(phase)
        log_payload = {
            "epoch": epoch,
            f"{prefix}train/loss": train_metrics.loss,
            f"{prefix}train/accuracy": train_metrics.accuracy,
            f"{prefix}train/epoch_time_sec": train_metrics.duration_sec,
            f"{prefix}val/loss": val_metrics.loss,
            f"{prefix}val/accuracy": val_metrics.accuracy,
            f"{prefix}val/epoch_time_sec": val_metrics.duration_sec,
        }

        for key, value in val_metrics.extras.items():
            if key in {"confusion_matrix", "y_true", "y_pred", "class_names"}:
                continue
            log_payload[f"{prefix}val/{key}"] = value

        wandb.log(log_payload, step=epoch)

        y_true = val_metrics.extras.get("y_true")
        y_pred = val_metrics.extras.get("y_pred")
        class_names = val_metrics.extras.get("class_names")
        if y_true is not None and y_pred is not None and class_names is not None:
            wandb.log(
                {
                    f"{prefix}val/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=list(class_names),
                    )
                },
                step=epoch,
            )

    def _maybe_save_best(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        phase: str,
        filename: str = "best_model.pt",
    ) -> Optional[str]:
        if val_metrics.loss >= self.best_val_loss:
            return None

        self.best_val_loss = val_metrics.loss
        checkpoint = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics={
                "phase": phase,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                **{
                    key: value
                    for key, value in val_metrics.extras.items()
                    if key != "confusion_matrix"
                },
            },
            checkpoint_dir=self.checkpoint_dir,
            filename=filename,
        )
        checkpoint_path = str(checkpoint)

        wandb.log(
            {
                "best/val_loss": val_metrics.loss,
                "best/val_accuracy": val_metrics.accuracy,
                "best/phase": phase,
            },
            step=epoch,
        )
        return checkpoint_path

    def fit(
        self,
        epochs: int,
        experiment_config: Optional[dict[str, Any]] = None,
        test_loader: Optional[DataLoader] = None,
        phase_name: str = "train",
    ) -> dict[str, Any]:
        config = experiment_config or {}
        if self._wandb_run is None:
            self._init_wandb(config)

        best_checkpoint_path: Optional[str] = None
        global_epoch = config.get("_global_epoch_offset", 0)

        for _ in range(1, epochs + 1):
            global_epoch += 1
            train_metrics = self.train_epoch(global_epoch)
            val_metrics = self.validate_epoch(global_epoch)
            self._log_epoch_metrics(global_epoch, phase_name, train_metrics, val_metrics)

            saved = self._maybe_save_best(
                global_epoch,
                train_metrics,
                val_metrics,
                phase_name,
            )
            if saved is not None:
                best_checkpoint_path = saved

        config["_global_epoch_offset"] = global_epoch
        return {
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": best_checkpoint_path,
            "epochs_completed": epochs,
            "global_epoch_offset": global_epoch,
        }

    def fit_with_test(
        self,
        epochs: int,
        experiment_config: Optional[dict[str, Any]] = None,
        test_loader: Optional[DataLoader] = None,
        phase_name: str = "train",
    ) -> dict[str, Any]:
        summary = self.fit(epochs, experiment_config, test_loader, phase_name)

        if test_loader is not None:
            if summary.get("best_checkpoint"):
                load_checkpoint(summary["best_checkpoint"], self.model, self.optimizer)
            test_metrics = self.evaluate(test_loader, split_name="test")
            step = experiment_config.get("_global_epoch_offset", epochs) if experiment_config else epochs

            test_log = {
                "test/loss": test_metrics.loss,
                "test/accuracy": test_metrics.accuracy,
                "test/epoch_time_sec": test_metrics.duration_sec,
            }
            for key, value in test_metrics.extras.items():
                if key != "confusion_matrix":
                    test_log[f"test/{key}"] = value

            wandb.log(test_log, step=step)

            summary["test_loss"] = test_metrics.loss
            summary["test_accuracy"] = test_metrics.accuracy

        return summary

    def finish_wandb(self) -> None:
        if self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None
