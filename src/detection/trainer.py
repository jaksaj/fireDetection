"""YOLO26 object-detection training orchestrator with Weights & Biases integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
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
            # Ultralytics defaults to seed=0/deterministic=True. Passing these
            # explicitly means a multi-seed sweep actually varies the seed, and
            # the value is recorded in the run's args.yaml rather than implied.
            "seed": train_config.get("seed", 0),
            "deterministic": train_config.get("deterministic", True),
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

    def validate(
        self,
        split: str = "val",
        conf: float | None = None,
    ) -> dict[str, Any]:
        """
        Run validation and return detection metrics for a dataset split.

        ``conf`` is the detection confidence threshold. It was previously
        declared in ``configs/iteration4.yaml`` as ``conf_threshold: 0.25`` and
        never passed to anything -- a dead config key. Note that mAP is computed
        by sweeping confidence, so this mainly affects the precision/recall
        operating point and the confusion matrix, not mAP itself; leaving it at
        the Ultralytics default (0.001) is the correct choice for mAP reporting.
        """
        logger.info("Running YOLO26 validation on split '%s' (conf=%s).", split, conf)
        kwargs: dict[str, Any] = {
            "data": self.data_yaml,
            "split": split,
            "device": DEVICE,
            "plots": False,
        }
        if conf is not None:
            kwargs["conf"] = conf

        results = self.model.val(**kwargs)
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

        box = getattr(results, "box", None)
        if box is not None:
            metrics.setdefault(f"{prefix}/mAP50", float(getattr(box, "map50", 0.0) or 0.0))
            metrics.setdefault(f"{prefix}/mAP50-95", float(getattr(box, "map", 0.0) or 0.0))

            # Per-class AP. The aggregate mAP hides the single most important
            # fact about this detector: on the D-Fire test split smoke reaches
            # mAP50 0.816 while fire reaches only 0.672, so fire -- not smoke --
            # is the harder class. Every write-up in this project previously
            # asserted the opposite, because nothing ever extracted these
            # numbers from the results object.
            names = getattr(results, "names", None)
            names = names if isinstance(names, dict) else {}

            # `ap_class_index` is a numpy array. `array or []` evaluates
            # `bool(array)`, which raises for any array with more than one
            # element -- and it raised here *after* a full 50-epoch training
            # run, so the process exited non-zero and the trained model was
            # never recorded. Never apply truthiness to an array.
            class_indices_raw = getattr(box, "ap_class_index", None)
            if class_indices_raw is None:
                class_indices: list = []
            else:
                class_indices = list(np.atleast_1d(np.asarray(class_indices_raw)))

            for position, class_index in enumerate(class_indices):
                class_name = names.get(int(class_index), str(class_index))
                for attribute, label in (
                    ("p", "precision"),
                    ("r", "recall"),
                    ("ap50", "mAP50"),
                    ("ap", "mAP50-95"),
                    ("f1", "f1"),
                ):
                    values = getattr(box, attribute, None)
                    if values is None:
                        continue
                    try:
                        metrics[f"{prefix}/{label}/{class_name}"] = float(values[position])
                    except (IndexError, TypeError, ValueError):
                        continue

        return metrics
