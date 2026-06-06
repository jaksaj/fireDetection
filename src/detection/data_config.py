"""D-Fire YOLO detection dataset configuration for Ultralytics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from src.dfire_labels import DFIRE_CLASS_FIRE, DFIRE_CLASS_SMOKE, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = ("smoke", "fire")


@dataclass
class DFireDetectionDataConfig:
    """
    Builds and validates the Ultralytics ``data.yaml`` for D-Fire object detection.

    Expects the standard split layout::

        root_dir/
            train/images, train/labels
            val/images,   val/labels
            test/images,  test/labels
    """

    root_dir: Path
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES

    def validate(self) -> None:
        """Ensure all split folders exist and contain paired images/labels."""
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        for split_name in (self.train_split, self.val_split, self.test_split):
            images_dir = self.root_dir / split_name / "images"
            labels_dir = self.root_dir / split_name / "labels"
            if not images_dir.is_dir():
                raise FileNotFoundError(f"Missing images directory: {images_dir}")
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

            image_stems = {
                path.stem
                for path in images_dir.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            }
            if not image_stems:
                raise FileNotFoundError(f"No images found in {images_dir}")

            missing_labels = [
                stem
                for stem in image_stems
                if not (labels_dir / f"{stem}.txt").exists()
            ]
            if missing_labels:
                logger.warning(
                    "Split '%s' has %d images without label files (first: %s).",
                    split_name,
                    len(missing_labels),
                    missing_labels[0],
                )

        logger.info(
            "D-Fire detection dataset validated at %s (classes: %s).",
            self.root_dir,
            ", ".join(self.class_names),
        )

    def to_dict(self) -> dict:
        """Return Ultralytics-compatible dataset metadata."""
        root = self.root_dir.resolve()
        return {
            "path": str(root),
            "train": f"{self.train_split}/images",
            "val": f"{self.val_split}/images",
            "test": f"{self.test_split}/images",
            "names": {index: name for index, name in enumerate(self.class_names)},
            "nc": len(self.class_names),
            "metadata": {
                "dataset": "D-Fire",
                "task": "object_detection",
                "smoke_class_id": DFIRE_CLASS_SMOKE,
                "fire_class_id": DFIRE_CLASS_FIRE,
            },
        }

    def write_yaml(self, destination: Path) -> Path:
        """Write ``data.yaml`` to disk and return its path."""
        self.validate()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()

        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

        logger.info("Wrote Ultralytics data config to %s", destination)
        return destination.resolve()
