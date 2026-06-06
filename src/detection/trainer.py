"""YOLO26 object-detection training orchestrator with Weights & Biases integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEVICE = 0  # CUDA GPU index — no CPU fallback


class YOLO26DetectionTrainer:
    """
    Object-oriented wrapper around the Ultralytics YOLO26 training pipeline.

    Maps D-Fire detection training into the established W&B project and stores
    checkpoints under a configurable project directory.
    """

    def __init__(
        self,
        model_weights: str = "yolo26n.pt",
        data_yaml: str | Path = "configs/dfire_detection.yaml",
        checkpoint_dir: str | Path = "checkpoints/iteration4",
        run_name: str = "yolo26-dfire",
        wandb_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_weights = model_weights
        self.data_yaml = str(data_yaml)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.wandb_config = wandb_config or {}
        self.model = YOLO(model_weights)
        self._results: Optional[Any] = None
        self._best_weights: Optional[Path] = None

        logger.info(
            "YOLO26DetectionTrainer initialized — model=%s device=CUDA:%d",
            model_weights,
            DEVICE,
        )

    def _train_kwargs(self, train_config: dict[str, Any]) -> dict[str, Any]:
        """Merge training config with CUDA and checkpoint defaults."""
        return {
            "data": self.data_yaml,
            "epochs": train_config.get("epochs", 50),
            "imgsz": train_config.get("imgsz", 640),
            "batch": train_config.get("batch", 16),
            "patience": train_config.get("patience", 10),
            "device": DEVICE,
            "workers": train_config.get("workers", 4),
            "project": str(self.checkpoint_dir),
            "name": self.run_name,
            "exist_ok": True,
            "pretrained": train_config.get("pretrained", True),
            "optimizer": train_config.get("optimizer", "auto"),
            "lr0": train_config.get("learning_rate", 0.01),
            "weight_decay": train_config.get("weight_decay", 0.0005),
            "plots": train_config.get("plots", True),
            "save": True,
            "verbose": True,
        }

    def _configure_wandb(self) -> None:
        """Point Ultralytics W&B integration at the established project."""
        import os

        if self.wandb_config.get("project"):
            os.environ.setdefault("WANDB_PROJECT", self.wandb_config["project"])
        if self.wandb_config.get("entity"):
            os.environ.setdefault("WANDB_ENTITY", str(self.wandb_config["entity"]))
        if self.wandb_config.get("run_name"):
            os.environ.setdefault("WANDB_NAME", self.wandb_config["run_name"])

    def train(self, train_config: dict[str, Any]) -> dict[str, Any]:
        """
        Run YOLO26 training on CUDA and return validation metrics.

        Ultralytics logs mAP, box loss, and training curves to W&B when installed.
        """
        kwargs = self._train_kwargs(train_config)
        self._configure_wandb()
        logger.info("Starting YOLO26 training with config: %s", kwargs)

        self._results = self.model.train(**kwargs)
        self._best_weights = Path(self.model.trainer.best) if self.model.trainer else None

        metrics = self._extract_validation_metrics(self._results)
        metrics["best_weights"] = str(self._best_weights) if self._best_weights else None
        metrics["save_dir"] = str(getattr(self._results, "save_dir", self.checkpoint_dir))

        logger.info("Training complete: %s", metrics)
        return metrics

    def validate(self, split: str = "val") -> dict[str, Any]:
        """Run validation and return detection metrics for a dataset split."""
        logger.info("Running YOLO26 validation on split '%s'.", split)
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            device=DEVICE,
            plots=False,
        )
        return self._extract_validation_metrics(results, prefix=split)

    def predict_sample(
        self,
        source: str | Path,
        conf: float = 0.25,
        save: bool = False,
    ) -> list[Any]:
        """Run inference on an image or directory."""
        return self.model.predict(
            source=str(source),
            device=DEVICE,
            conf=conf,
            save=save,
            verbose=False,
        )

    @property
    def best_weights_path(self) -> Optional[Path]:
        return self._best_weights

    @staticmethod
    def _extract_validation_metrics(
        results: Any,
        prefix: str = "val",
    ) -> dict[str, Any]:
        """Parse Ultralytics results into a flat metrics dictionary."""
        metrics: dict[str, Any] = {}

        if results is None:
            return metrics

        results_dict = getattr(results, "results_dict", None)
        if results_dict:
            for key, value in results_dict.items():
                if value is None:
                    continue
                try:
                    metrics[f"{prefix}/{key}"] = float(value)
                except (TypeError, ValueError):
                    metrics[f"{prefix}/{key}"] = value
            return metrics

        box = getattr(results, "box", None)
        if box is not None:
            metrics[f"{prefix}/mAP50"] = float(getattr(box, "map50", 0.0) or 0.0)
            metrics[f"{prefix}/mAP50-95"] = float(getattr(box, "map", 0.0) or 0.0)
            metrics[f"{prefix}/box_loss"] = float(getattr(box, "loss", 0.0) or 0.0)

        return metrics
