"""Dataset integrity checks and summary statistics for D-Fire splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.dfire_labels import (
    DFIRE_CLASS_FIRE,
    DFIRE_CLASS_SMOKE,
    IMAGE_EXTENSIONS,
    derive_binary_label,
    parse_yolo_label_file,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    """Summary statistics for a single train/val/test split."""

    split_name: str
    image_count: int = 0
    label_count: int = 0
    fire_images: int = 0
    normal_images: int = 0
    smoke_only_images: int = 0
    fire_and_smoke_images: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    yolo_fire_boxes: int = 0
    yolo_smoke_boxes: int = 0

    @property
    def binary_fire_ratio(self) -> float:
        if self.image_count == 0:
            return 0.0
        return self.fire_images / self.image_count


@dataclass
class DatasetReport:
    """Aggregated report across all D-Fire splits."""

    root_dir: Path
    splits: dict[str, SplitStats] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(split.image_count for split in self.splits.values())


class DFireDatasetInspector:
    """Inspect D-Fire YOLO splits and produce class-balance statistics."""

    def __init__(
        self,
        root_dir: str | Path,
        fire_class_id: int = DFIRE_CLASS_FIRE,
        smoke_class_id: int = DFIRE_CLASS_SMOKE,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.fire_class_id = fire_class_id
        self.smoke_class_id = smoke_class_id

    def inspect_split(self, split_name: str) -> SplitStats:
        split_dir = self.root_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        stats = SplitStats(split_name=split_name)

        if not images_dir.is_dir():
            logger.warning("Split images directory missing: %s", images_dir)
            return stats
        if not labels_dir.is_dir():
            logger.warning("Split labels directory missing: %s", labels_dir)
            return stats

        image_stems = {
            path.stem
            for path in images_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        label_stems = {path.stem for path in labels_dir.glob("*.txt")}

        stats.image_count = len(image_stems)
        stats.label_count = len(label_stems)
        stats.missing_labels = len(image_stems - label_stems)
        stats.orphan_labels = len(label_stems - image_stems)

        for stem in sorted(image_stems):
            label_path = labels_dir / f"{stem}.txt"
            class_ids = parse_yolo_label_file(label_path)

            if not label_path.exists():
                continue

            if not class_ids:
                stats.empty_labels += 1

            has_fire = self.fire_class_id in class_ids
            has_smoke = self.smoke_class_id in class_ids

            if has_fire and has_smoke:
                stats.fire_and_smoke_images += 1
            elif has_smoke:
                stats.smoke_only_images += 1

            stats.yolo_fire_boxes += class_ids.count(self.fire_class_id)
            stats.yolo_smoke_boxes += class_ids.count(self.smoke_class_id)

            if derive_binary_label(label_path, self.fire_class_id) == 1:
                stats.fire_images += 1
            else:
                stats.normal_images += 1

        return stats

    def inspect(self, splits: tuple[str, ...] = ("train", "val", "test")) -> DatasetReport:
        report = DatasetReport(root_dir=self.root_dir)

        for split_name in splits:
            split_dir = self.root_dir / split_name
            if not split_dir.is_dir():
                logger.warning("Split directory not found, skipping: %s", split_dir)
                continue
            report.splits[split_name] = self.inspect_split(split_name)

        return report

    @staticmethod
    def format_report(report: DatasetReport) -> str:
        lines = [
            f"D-Fire dataset report: {report.root_dir}",
            f"Total images: {report.total_images}",
            "",
        ]

        for split_name, stats in report.splits.items():
            lines.extend(
                [
                    f"[{split_name}]",
                    f"  images:              {stats.image_count}",
                    f"  label files:         {stats.label_count}",
                    f"  binary fire:         {stats.fire_images}",
                    f"  binary normal:       {stats.normal_images}",
                    f"  fire ratio:          {stats.binary_fire_ratio:.2%}",
                    f"  smoke-only images:   {stats.smoke_only_images}",
                    f"  fire+smoke images:   {stats.fire_and_smoke_images}",
                    f"  empty label files:   {stats.empty_labels}",
                    f"  missing labels:      {stats.missing_labels}",
                    f"  orphan labels:       {stats.orphan_labels}",
                    f"  YOLO fire boxes:     {stats.yolo_fire_boxes}",
                    f"  YOLO smoke boxes:    {stats.yolo_smoke_boxes}",
                    "",
                ]
            )

        return "\n".join(lines).rstrip()
