"""Integrity checks, class-balance statistics, and visualization generators for COCO splits."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationSplitStats:
    """Summary statistics for a single COCO segmentation split."""

    split_name: str
    total_images_in_coco: int = 0
    images_found_on_disk: int = 0
    images_missing: int = 0
    smoke_images: int = 0
    fire_images: int = 0
    both_images: int = 0
    empty_images: int = 0
    smoke_annotations: int = 0
    fire_annotations: int = 0
    total_annotations: int = 0


@dataclass
class SegmentationDatasetReport:
    """Aggregated report across all COCO segmentation splits."""

    root_dir: Path
    splits: dict[str, SegmentationSplitStats] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(split.images_found_on_disk for split in self.splits.values())


class COCOSegmentationInspector:
    """Inspect COCO segmentation dataset splits and save visual overlay examples."""

    def __init__(
        self,
        root_dir: str | Path,
        coco_annotation_file: str = "_annotations.coco.json",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.coco_annotation_file = coco_annotation_file

    def inspect_split(self, split_name: str, save_examples_count: int = 5) -> SegmentationSplitStats:
        split_dir = self.root_dir / split_name
        stats = SegmentationSplitStats(split_name=split_name)

        if not split_dir.is_dir():
            logger.warning("Split directory missing: %s", split_dir)
            return stats

        # Try to locate the annotation file
        json_paths = [
            split_dir / self.coco_annotation_file,
            split_dir / f"{self.coco_annotation_file}.json",
            split_dir / "_annotations.coco.json",
            split_dir / "_annotations.coco",
        ]

        annotation_path = None
        for path in json_paths:
            if path.is_file():
                annotation_path = path
                break

        if annotation_path is None:
            logger.warning("No COCO annotation file found in split: %s", split_dir)
            return stats

        with annotation_path.open("r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Map categories
        cat_id_to_name = {}
        cat_id_to_type = {}  # "smoke", "fire", or "other"
        for cat in coco_data.get("categories", []):
            name = cat["name"].lower()
            cat_id = cat["id"]
            cat_id_to_name[cat_id] = cat["name"]
            if "smoke" in name:
                cat_id_to_type[cat_id] = "smoke"
            elif "fire" in name:
                cat_id_to_type[cat_id] = "fire"
            else:
                cat_id_to_type[cat_id] = "other"

        # Map image ID to info and annotations
        images_dict = {img["id"]: img for img in coco_data.get("images", [])}
        annotations_by_image = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            annotations_by_image.setdefault(img_id, []).append(ann)

        stats.total_images_in_coco = len(images_dict)
        stats.total_annotations = len(coco_data.get("annotations", []))

        example_images_saved = 0
        examples_dir = self.root_dir / "examples" / split_name

        for img_id, img_info in images_dict.items():
            img_filename = img_info["file_name"]
            img_path = split_dir / img_filename

            # Handle mismatch cases
            if not img_path.is_file():
                matching_files = list(split_dir.glob(Path(img_filename).name))
                if matching_files:
                    img_path = matching_files[0]
                else:
                    stats.images_missing += 1
                    continue

            stats.images_found_on_disk += 1
            anns = annotations_by_image.get(img_id, [])

            has_smoke = False
            has_fire = False

            for ann in anns:
                cat_id = ann["category_id"]
                cat_type = cat_id_to_type.get(cat_id, "other")
                if cat_type == "smoke":
                    has_smoke = True
                    stats.smoke_annotations += 1
                elif cat_type == "fire":
                    has_fire = True
                    stats.fire_annotations += 1

            if has_smoke and has_fire:
                stats.both_images += 1
                stats.smoke_images += 1
                stats.fire_images += 1
            elif has_smoke:
                stats.smoke_images += 1
            elif has_fire:
                stats.fire_images += 1
            else:
                stats.empty_images += 1

            # Save a few visual overlay examples for inspection
            if save_examples_count > 0 and example_images_saved < save_examples_count and (has_smoke or has_fire):
                examples_dir.mkdir(parents=True, exist_ok=True)
                save_path = examples_dir / f"overlay_{img_path.name}"
                try:
                    self._save_overlay(img_path, img_info, anns, cat_id_to_type, save_path)
                    example_images_saved += 1
                except Exception as e:
                    logger.error("Failed to save visualization for %s: %s", img_path.name, e)

        return stats

    def _save_overlay(
        self,
        img_path: Path,
        img_info: dict,
        anns: list[dict],
        cat_id_to_type: dict[int, str],
        save_path: Path,
    ) -> None:
        """Render polygons and save as a blended semi-transparent overlay image."""
        width = img_info["width"]
        height = img_info["height"]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # Create separate mask channels for drawing
        mask_np = np.zeros((height, width), dtype=np.uint8)
        mask = Image.fromarray(mask_np, mode="L")
        draw = ImageDraw.Draw(mask)

        # Sort so fire draws on top of smoke
        anns_sorted = sorted(anns, key=lambda a: (1 if cat_id_to_type.get(a["category_id"]) == "fire" else 0, -a.get("area", 0)))

        for ann in anns_sorted:
            cat_id = ann["category_id"]
            cat_type = cat_id_to_type.get(cat_id, "other")
            val = 0
            if cat_type == "smoke":
                val = 1
            elif cat_type == "fire":
                val = 2

            if val == 0:
                continue

            segmentations = ann.get("segmentation", [])
            if isinstance(segmentations, list):
                for seg in segmentations:
                    if len(seg) >= 6:
                        draw.polygon(seg, fill=val)

        mask_np = np.array(mask)

        # Create alpha mask: semi-transparent (128) where there's smoke or fire
        alpha_np = np.zeros_like(mask_np)
        alpha_np[mask_np > 0] = 128
        alpha_mask = Image.fromarray(alpha_np, mode="L")

        # Create colored overlay image
        overlay = Image.new("RGB", image.size, (0, 0, 0))
        
        # Paste solid colors using class masks as stencils
        smoke_mask = Image.fromarray((mask_np == 1).astype(np.uint8) * 255, mode="L")
        fire_mask = Image.fromarray((mask_np == 2).astype(np.uint8) * 255, mode="L")

        # Deep Sky Blue for smoke, Bright Red-Orange for fire
        overlay.paste((0, 191, 255), mask=smoke_mask)
        overlay.paste((255, 69, 0), mask=fire_mask)

        # Composite the overlay onto the original image
        blended = Image.composite(overlay, image, alpha_mask)
        blended.save(save_path)

    def inspect(
        self,
        splits: tuple[str, ...] = ("train", "valid", "test"),
        save_examples_count: int = 5,
    ) -> SegmentationDatasetReport:
        report = SegmentationDatasetReport(root_dir=self.root_dir)

        for split_name in splits:
            split_dir = self.root_dir / split_name
            if not split_dir.is_dir():
                logger.warning("Split directory not found, skipping: %s", split_dir)
                continue
            report.splits[split_name] = self.inspect_split(split_name, save_examples_count)

        return report

    @staticmethod
    def format_report(report: SegmentationDatasetReport) -> str:
        lines = [
            f"COCO Segmentation Dataset Report: {report.root_dir}",
            f"Total images found on disk: {report.total_images}",
            "",
        ]

        for split_name, stats in report.splits.items():
            lines.extend(
                [
                    f"[{split_name}]",
                    f"  images in COCO:      {stats.total_images_in_coco}",
                    f"  images on disk:      {stats.images_found_on_disk}",
                    f"  images missing:      {stats.images_missing}",
                    f"  smoke images:        {stats.smoke_images}",
                    f"  fire images:         {stats.fire_images}",
                    f"  both (smoke & fire): {stats.both_images}",
                    f"  empty images:        {stats.empty_images}",
                    f"  smoke annotations:   {stats.smoke_annotations}",
                    f"  fire annotations:    {stats.fire_annotations}",
                    f"  total annotations:   {stats.total_annotations}",
                    "",
                ]
            )

        return "\n".join(lines).rstrip()
