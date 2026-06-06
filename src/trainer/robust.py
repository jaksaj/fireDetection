"""Iteration 3 trainer with LR scheduling and edge-deployment simulation."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from src.edge_simulation import run_edge_simulation as simulate_edge_simulation
from src.model import MobileNetV3FireClassifier
from src.trainer.multiclass import MulticlassTrainer
from src.utils import load_checkpoint

logger = logging.getLogger(__name__)


class RobustMulticlassTrainer(MulticlassTrainer):
    """
    Extends the Iteration 2 trainer with learning-rate scheduling and PTQ edge simulation.

    Training still follows the two-phase transfer-learning schedule, but each phase
    can use ``CosineAnnealingLR`` or ``ReduceLROnPlateau`` for improved convergence.
  """

    def __init__(
        self,
        *args: Any,
        scheduler_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scheduler_config = scheduler_config or {"type": "cosine"}

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        scheduler_type = self.scheduler_config.get("type", "cosine").lower()

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=self.scheduler_config.get("eta_min", 1e-6),
            )

        if scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_config.get("factor", 0.5),
                patience=self.scheduler_config.get("patience", 2),
                min_lr=self.scheduler_config.get("eta_min", 1e-6),
            )

        logger.warning("Unknown scheduler type '%s' — training without a scheduler.", scheduler_type)
        return None

    def _run_phase(
        self,
        epochs: int,
        phase_name: str,
        experiment_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.set_scheduler(self._build_scheduler(self.optimizer, epochs))
        return super()._run_phase(epochs, phase_name, experiment_config)

    def fit_two_phase_with_edge_sim(
        self,
        head_epochs: int,
        finetune_epochs: int,
        head_learning_rate: float,
        finetune_learning_rate: float,
        unfreeze_blocks: int,
        experiment_config: Optional[dict[str, Any]] = None,
        test_loader: Optional[DataLoader] = None,
        image_size: int = 224,
        run_edge_simulation_flag: bool = True,
    ) -> dict[str, Any]:
        summary = self.fit_two_phase(
            head_epochs=head_epochs,
            finetune_epochs=finetune_epochs,
            head_learning_rate=head_learning_rate,
            finetune_learning_rate=finetune_learning_rate,
            unfreeze_blocks=unfreeze_blocks,
            experiment_config=experiment_config,
            test_loader=test_loader,
        )

        if not run_edge_simulation_flag or test_loader is None:
            return summary

        best_checkpoint = summary.get("best_checkpoint")
        if best_checkpoint is not None:
            load_checkpoint(best_checkpoint, self.model, self.optimizer)

        import wandb

        edge_metrics = simulate_edge_simulation(
            model=self.model,
            test_loader=test_loader,
            class_names=self.class_names,
            checkpoint_dir=self.checkpoint_dir,
            image_size=image_size,
        )
        step = (experiment_config or {}).get("_global_epoch_offset", head_epochs + finetune_epochs)
        wandb.log(edge_metrics, step=step)
        summary["edge_simulation"] = edge_metrics
        logger.info("Edge simulation logged to W&B at step %d.", step)
        return summary
