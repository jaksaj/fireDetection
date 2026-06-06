"""D-Fire 4-class dataset and DataLoader factory for Iteration 2."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dfire_labels import (
    DFIRE_CLASS_FIRE,
    DFIRE_CLASS_SMOKE,
    IMAGE_EXTENSIONS,
    MULTICLASS_CLASS_NAMES,
    NUM_MULTICLASS_CLASSES,
    derive_multiclass_label,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


class DFireMulticlassDataset(Dataset):
    """
    Custom ``torch.utils.data.Dataset`` for D-Fire 4-class classification.

    Expects the official pre-split YOLO layout::

        <split_dir>/
            images/
            labels/

    Image-level classes:
        0 Neither, 1 Only_Fire, 2 Only_Smoke, 3 Both
    """

    CLASS_NAMES = MULTICLASS_CLASS_NAMES
    NUM_CLASSES = NUM_MULTICLASS_CLASSES

    def __init__(
        self,
        split_dir: str | Path,
        transform: Optional[Callable] = None,
        fire_class_id: int = DFIRE_CLASS_FIRE,
        smoke_class_id: int = DFIRE_CLASS_SMOKE,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.transform = transform
        self.fire_class_id = fire_class_id
        self.smoke_class_id = smoke_class_id
        self.samples: list[Tuple[Path, Path, int]] = self._discover_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.images_dir}. "
                "Expected D-Fire layout: <split>/images/ and <split>/labels/."
            )

        class_counts = Counter(label for _, _, label in self.samples)
        logger.info(
            "DFireMulticlassDataset [%s]: %d samples %s",
            self.split_dir.name,
            len(self.samples),
            {self.CLASS_NAMES[k]: v for k, v in sorted(class_counts.items())},
        )

    def _discover_samples(self) -> list[Tuple[Path, Path, int]]:
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        samples: list[Tuple[Path, Path, int]] = []

        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = self.labels_dir / f"{image_path.stem}.txt"
            multiclass_label = derive_multiclass_label(
                label_path,
                fire_class_id=self.fire_class_id,
                smoke_class_id=self.smoke_class_id,
            )
            samples.append((image_path, label_path, multiclass_label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, _label_path, label = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor

    def class_counts(self) -> Counter:
        return Counter(label for _, _, label in self.samples)


class DFireMulticlassDataModule:
    """Builds train, validation, and test DataLoaders for 4-class D-Fire training."""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        fire_class_id: int = DFIRE_CLASS_FIRE,
        smoke_class_id: int = DFIRE_CLASS_SMOKE,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fire_class_id = fire_class_id
        self.smoke_class_id = smoke_class_id
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._train_dataset: Optional[DFireMulticlassDataset] = None

    def _build_loader(
        self,
        split_name: str,
        transform: Callable,
        shuffle: bool,
    ) -> DataLoader:
        dataset = DFireMulticlassDataset(
            split_dir=self.root_dir / split_name,
            transform=transform,
            fire_class_id=self.fire_class_id,
            smoke_class_id=self.smoke_class_id,
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
        self._train_dataset = DFireMulticlassDataset(
            split_dir=self.root_dir / self.train_split,
            transform=self.train_transform,
            fire_class_id=self.fire_class_id,
            smoke_class_id=self.smoke_class_id,
        )
        val_dataset = DFireMulticlassDataset(
            split_dir=self.root_dir / self.val_split,
            transform=self.eval_transform,
            fire_class_id=self.fire_class_id,
            smoke_class_id=self.smoke_class_id,
        )
        test_dataset = DFireMulticlassDataset(
            split_dir=self.root_dir / self.test_split,
            transform=self.eval_transform,
            fire_class_id=self.fire_class_id,
            smoke_class_id=self.smoke_class_id,
        )

        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        logger.info(
            "Multiclass DataModule ready — train: %d | val: %d | test: %d | device: %s",
            len(self._train_dataset),
            len(val_dataset),
            len(test_dataset),
            DEVICE,
        )

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from the training split."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before computing class_weights.")

        counts = self._train_dataset.class_counts()
        total = sum(counts.values())
        weights = [
            total / (NUM_MULTICLASS_CLASSES * counts.get(class_index, 1))
            for class_index in range(NUM_MULTICLASS_CLASSES)
        ]
        return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    NUM_CLASSES = NUM_MULTICLASS_CLASSES
    CLASS_NAMES = MULTICLASS_CLASS_NAMES

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
