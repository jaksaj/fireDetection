"""D-Fire binary classification dataset and DataLoader factory."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DFireBinaryDataset(Dataset):
    """
    Custom ``torch.utils.data.Dataset`` for D-Fire binary classification.

    Expects the following directory layout::

        root_dir/
            Fire/
                *.jpg
            Normal/
                *.jpg

    Labels are encoded as ``1`` (Fire) and ``0`` (Normal).
  """

    CLASS_TO_LABEL = {"Fire": 1, "Normal": 0}
    LABEL_TO_CLASS = {1: "Fire", 0: "Normal"}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = self._discover_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. "
                "Expected subdirectories 'Fire/' and 'Normal/'."
            )

        logger.info(
            "DFireBinaryDataset initialized: %d samples from %s",
            len(self.samples),
            self.root_dir,
        )

    def _discover_samples(self) -> list[Tuple[Path, int]]:
        samples: list[Tuple[Path, int]] = []

        for class_name, label in self.CLASS_TO_LABEL.items():
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Missing class directory: %s", class_dir)
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        return image, label_tensor


class DFireDataModule:
    """
    Encapsulates dataset creation, train/val splitting, and DataLoader construction.

    All tensors produced by the DataLoaders are intended for GPU training; the
    ``Trainer`` moves each batch to CUDA during the training loop.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.val_transform = transforms.Compose(
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

    def _split_indices(self, dataset_size: int) -> Tuple[list[int], list[int]]:
        indices = list(range(dataset_size))
        random.Random(self.seed).shuffle(indices)

        val_size = int(dataset_size * self.val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return train_indices, val_indices

    def setup(self) -> None:
        """Build train and validation DataLoaders."""
        full_dataset = DFireBinaryDataset(self.root_dir, transform=None)
        train_indices, val_indices = self._split_indices(len(full_dataset))

        train_dataset = DFireBinaryDataset(self.root_dir, transform=self.train_transform)
        val_dataset = DFireBinaryDataset(self.root_dir, transform=self.val_transform)

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        self._train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self._val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        logger.info(
            "DataModule ready — train: %d | val: %d | device target: %s",
            len(train_subset),
            len(val_subset),
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
