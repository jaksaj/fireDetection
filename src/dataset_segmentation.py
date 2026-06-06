"""Dataset and DataLoader factory for COCO-formatted semantic segmentation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from src.augmentations import build_robust_train_transforms, build_robust_eval_transforms

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


class COCOSegmentationDataset(Dataset):
    """
    Custom ``torch.utils.data.Dataset`` for fire and smoke semantic segmentation.

    Expects a folder layout where images and their COCO JSON annotations are in the same folder:
        <split_dir>/
            _annotations.coco.json
            *.jpg or *.png

    Target classes:
        0: Background
        1: Smoke
        2: Fire
    """

    def __init__(
        self,
        split_dir: str | Path,
        transform: Optional[Callable] = None,
        coco_annotation_file: str = "_annotations.coco.json",
    ) -> None:
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.coco_annotation_file = coco_annotation_file

        # Try both the provided filename and a potential .json-less version
        json_paths = [
            self.split_dir / self.coco_annotation_file,
            self.split_dir / f"{self.coco_annotation_file}.json",
            self.split_dir / "_annotations.coco.json",
            self.split_dir / "_annotations.coco",
        ]

        self.annotation_path = None
        for path in json_paths:
            if path.is_file():
                self.annotation_path = path
                break

        if self.annotation_path is None:
            raise FileNotFoundError(
                f"Could not find COCO annotation file in {self.split_dir}. "
                f"Looked for: {[p.name for p in json_paths]}"
            )

        logger.info("Loading COCO annotations from: %s", self.annotation_path)
        with self.annotation_path.open("r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        # Build mapping from category_id in COCO to target classes: smoke=1, fire=2
        self.cat_id_to_target = {}
        for cat in self.coco_data.get("categories", []):
            cat_name = cat["name"].lower()
            cat_id = cat["id"]
            if "smoke" in cat_name:
                self.cat_id_to_target[cat_id] = 1
            elif "fire" in cat_name:
                self.cat_id_to_target[cat_id] = 2
            elif "background" in cat_name:
                self.cat_id_to_target[cat_id] = 0

        # Build mapping from image_id to its annotations
        self.annotations_by_image = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            self.annotations_by_image.setdefault(img_id, []).append(ann)

        # Build final list of valid images
        self.samples = []
        for img in self.coco_data.get("images", []):
            img_filename = img["file_name"]
            img_path = self.split_dir / img_filename
            if img_path.is_file():
                self.samples.append((img_path, img))
            else:
                # Some exports prefix or change suffixes, let's try direct matches
                matching_files = list(self.split_dir.glob(Path(img_filename).name))
                if matching_files:
                    self.samples.append((matching_files[0], img))
                else:
                    logger.warning("Image file listed in COCO not found: %s", img_path)

        if not self.samples:
            raise FileNotFoundError(f"No valid image files found in {self.split_dir}")

        logger.info(
            "COCOSegmentationDataset [%s]: Loaded %d samples (Mapped classes: %s)",
            self.split_dir.name,
            len(self.samples),
            self.cat_id_to_target,
        )

    def _render_mask(self, img_info: dict) -> Image.Image:
        """Render COCO polygon segmentations into a single-channel PIL mask image."""
        width = img_info["width"]
        height = img_info["height"]
        # Create a blank black mask
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        anns = self.annotations_by_image.get(img_info["id"], [])
        # Sort annotations so that smaller areas are drawn last (on top) if they overlap,
        # or we draw fire (class 2) on top of smoke (class 1)
        anns = sorted(anns, key=lambda a: (self.cat_id_to_target.get(a["category_id"], 0), -a.get("area", 0)))

        for ann in anns:
            cat_id = ann["category_id"]
            target_class = self.cat_id_to_target.get(cat_id, 0)
            if target_class == 0:
                continue

            segmentations = ann.get("segmentation", [])
            if isinstance(segmentations, list):
                for seg in segmentations:
                    # seg is a flat list of coordinates: [x1, y1, x2, y2, ...]
                    if len(seg) >= 6:
                        # Draw filled polygon on mask
                        draw.polygon(seg, fill=target_class)

        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, img_info = self.samples[index]

        # Load image
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # Render mask
        mask = self._render_mask(img_info)

        # Convert to numpy arrays for Albumentations
        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            # Fallback if no transform is provided
            image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


class SegmentationDataModule:
    """
    Builds train, validation, and test DataLoaders for semantic segmentation.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
        coco_annotation_file: str = "_annotations.coco.json",
        train_split: str = "train",
        val_split: str = "valid",
        test_split: str = "test",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.coco_annotation_file = coco_annotation_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_transform = build_robust_train_transforms(image_size)
        self.eval_transform = build_robust_eval_transforms(image_size)

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None

    def _build_loader(
        self,
        split_name: str,
        transform: Callable,
        shuffle: bool,
    ) -> DataLoader:
        dataset = COCOSegmentationDataset(
            split_dir=self.root_dir / split_name,
            transform=transform,
            coco_annotation_file=self.coco_annotation_file,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def setup(self) -> None:
        """Build train, validation, and test DataLoaders."""
        self._train_loader = self._build_loader(
            self.train_split, self.train_transform, shuffle=True
        )
        self._val_loader = self._build_loader(
            self.val_split, self.eval_transform, shuffle=False
        )
        self._test_loader = self._build_loader(
            self.test_split, self.eval_transform, shuffle=False
        )

        logger.info(
            "SegmentationDataModule ready — train: %d | val: %d | test: %d | device: %s",
            len(self._train_loader.dataset),
            len(self._val_loader.dataset),
            len(self._test_loader.dataset),
            DEVICE,
        )

    @property
    def train_loader(self) -> DataLoader:
        if self._train_loader is None:
            raise RuntimeError("Call setup() before accessing train_loader.")
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        if self._val_loader is None:
            raise RuntimeError("Call setup() before accessing val_loader.")
        return self._val_loader

    @property
    def test_loader(self) -> DataLoader:
        if self._test_loader is None:
            raise RuntimeError("Call setup() before accessing test_loader.")
        return self._test_loader
