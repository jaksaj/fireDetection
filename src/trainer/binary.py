"""Binary classification trainer for Iteration 1."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainer.base import BaseTrainer, EpochMetrics
from src.utils import calculate_accuracy, load_checkpoint


class BinaryTrainer(BaseTrainer):
    """Trainer for binary fire vs. normal classification with BCEWithLogitsLoss."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints/iteration1",
        wandb_config: Optional[dict[str, Any]] = None,
        log_every_n_batches: int = 25,
    ) -> None:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            wandb_config=wandb_config,
            log_every_n_batches=log_every_n_batches,
            wandb_tags=(wandb_config or {}).get("tags", ["iteration1"]),
        )

    def compute_batch_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        return calculate_accuracy(logits, labels)

    def extract_predictions(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Threshold the single sigmoid output at 0.5; score is P(fire)."""
        scores = torch.sigmoid(logits.squeeze(-1))
        return (scores > 0.5).long(), scores

    def build_epoch_extras(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Full binary metric set for a fire detector.

        Accuracy alone was the only metric this iteration reported. For a
        safety-critical detector that is the wrong headline: on this split 74%
        of images contain no fire, so a model that never predicts fire scores
        74% accuracy while missing every fire. Recall and false-alarm rate are
        the operationally meaningful quantities, and PR-AUC/ROC-AUC summarise
        performance across all thresholds rather than the arbitrary 0.5.
        """
        from src.metrics import compute_binary_metrics

        return compute_binary_metrics(y_true, y_pred, y_score)

    def fit(
        self,
        epochs: int,
        experiment_config: Optional[dict[str, Any]] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> dict[str, Any]:
        config = experiment_config or {}
        self._init_wandb(config)

        best_checkpoint_path: Optional[str] = None
        test_metrics: Optional[EpochMetrics] = None

        try:
            summary = super().fit(epochs, config, test_loader, phase_name="")
            best_checkpoint_path = summary.get("best_checkpoint")

            if test_loader is not None:
                if best_checkpoint_path is not None:
                    load_checkpoint(best_checkpoint_path, self.model, self.optimizer)
                test_metrics = self.evaluate(test_loader, split_name="test")

                import wandb

                wandb.log(
                    {
                        "test/loss": test_metrics.loss,
                        "test/accuracy": test_metrics.accuracy,
                        "test/epoch_time_sec": test_metrics.duration_sec,
                    },
                    step=epochs,
                )
                summary["test_loss"] = test_metrics.loss
                summary["test_accuracy"] = test_metrics.accuracy
        finally:
            self.finish_wandb()

        return summary


# Backward-compatible alias used by Iteration 1 scripts.
Trainer = BinaryTrainer
