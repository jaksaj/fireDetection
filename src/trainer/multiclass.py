"""Multi-class trainer with two-phase transfer learning for Iteration 2."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dfire_labels import MULTICLASS_CLASS_NAMES
from src.metrics import calculate_multiclass_accuracy, compute_multiclass_metrics
from src.model import MobileNetV3FireClassifier
from src.trainer.base import BaseTrainer, EpochMetrics
from src.utils import load_checkpoint

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


class MulticlassTrainer(BaseTrainer):
    """
    Trainer for 4-class D-Fire classification with CrossEntropyLoss.

    Supports a two-phase schedule:
        1. Freeze backbone, train classification head.
        2. Unfreeze top backbone blocks and fine-tune at a lower learning rate.
    """

    def __init__(
        self,
        model: MobileNetV3FireClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: tuple[str, ...] = MULTICLASS_CLASS_NAMES,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = "checkpoints/iteration2",
        wandb_config: Optional[dict[str, Any]] = None,
        log_every_n_batches: int = 25,
    ) -> None:
        self.class_names = class_names
        self.weight_decay = weight_decay
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
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
            wandb_tags=(wandb_config or {}).get("tags", ["iteration2"]),
        )

    def compute_batch_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        return calculate_multiclass_accuracy(logits, labels)

    def build_epoch_extras(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float] | None = None,
    ) -> dict[str, Any]:
        # y_score (max softmax probability) is accepted for interface
        # compatibility; multi-class threshold sweeps are not part of this
        # iteration's evaluation.
        if not y_true:
            return {}

        metrics = compute_multiclass_metrics(y_true, y_pred, self.class_names)
        return {
            **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": metrics["confusion_matrix"],
            "y_true": y_true,
            "y_pred": y_pred,
            "class_names": self.class_names,
        }

    def _build_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        differential: bool = False,
    ) -> torch.optim.Optimizer:
        if differential and isinstance(self.model, MobileNetV3FireClassifier):
            backbone_params, head_params = self.model.trainable_parameter_groups()
            param_groups = []
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": learning_rate})
            if head_params:
                param_groups.append(
                    {"params": head_params, "lr": learning_rate * 10.0}
                )
            return torch.optim.Adam(param_groups, weight_decay=weight_decay)

        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _run_phase(
        self,
        epochs: int,
        phase_name: str,
        experiment_config: dict[str, Any],
    ) -> dict[str, Any]:
        logger.info("Starting phase '%s' for %d epochs.", phase_name, epochs)
        experiment_config["active_phase"] = phase_name
        return self.fit(epochs, experiment_config, phase_name=phase_name)

    def fit_two_phase(
        self,
        head_epochs: int,
        finetune_epochs: int,
        head_learning_rate: float,
        finetune_learning_rate: float,
        unfreeze_blocks: int,
        experiment_config: Optional[dict[str, Any]] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> dict[str, Any]:
        config = experiment_config or {}
        self._init_wandb(config)

        summary: dict[str, Any] = {}

        try:
            self.model.freeze_backbone()
            self.set_optimizer(
                self._build_optimizer(head_learning_rate, weight_decay=self.weight_decay)
            )
            head_summary = self._run_phase(head_epochs, "head", config)
            summary["head"] = head_summary

            self.model.unfreeze_top_layers(num_blocks=unfreeze_blocks)
            self.set_optimizer(
                self._build_optimizer(
                    finetune_learning_rate,
                    weight_decay=self.weight_decay,
                    differential=True,
                )
            )
            finetune_summary = self._run_phase(finetune_epochs, "finetune", config)
            summary["finetune"] = finetune_summary

            best_checkpoint = finetune_summary.get("best_checkpoint") or head_summary.get(
                "best_checkpoint"
            )
            summary["best_checkpoint"] = best_checkpoint
            summary["best_val_loss"] = self.best_val_loss

            if test_loader is not None:
                if best_checkpoint is not None:
                    load_checkpoint(best_checkpoint, self.model, self.optimizer)
                test_metrics = self.evaluate(test_loader, split_name="test")

                import wandb

                test_log = {
                    "test/loss": test_metrics.loss,
                    "test/accuracy": test_metrics.accuracy,
                    "test/epoch_time_sec": test_metrics.duration_sec,
                }
                for key, value in test_metrics.extras.items():
                    if key not in {"confusion_matrix", "y_true", "y_pred", "class_names"}:
                        test_log[f"test/{key}"] = value

                y_true = test_metrics.extras.get("y_true")
                y_pred = test_metrics.extras.get("y_pred")
                class_names = test_metrics.extras.get("class_names")
                if y_true is not None and y_pred is not None and class_names is not None:
                    test_log["test/confusion_matrix"] = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=list(class_names),
                    )

                step = config.get("_global_epoch_offset", head_epochs + finetune_epochs)
                wandb.log(test_log, step=step)

                summary["test_loss"] = test_metrics.loss
                summary["test_accuracy"] = test_metrics.accuracy
                summary["test_f1_macro"] = test_metrics.extras.get("f1_macro")

        finally:
            self.finish_wandb()

        logger.info("Two-phase training complete: %s", summary)
        return summary
