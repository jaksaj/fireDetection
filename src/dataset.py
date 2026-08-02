"""D-Fire binary classification dataset and DataLoader factory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dfire_labels import (
    DFIRE_CLASS_FIRE,
    IMAGE_EXTENSIONS,
    derive_binary_label,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")

SPLIT_NAMES = ("train", "val", "test")


class DFireBinaryDataset(Dataset):
    """
    Custom ``torch.utils.data.Dataset`` for D-Fire binary classification.

    Expects the official pre-split YOLO layout::

        <split_dir>/
            images/
                *.jpg
            labels/
                *.txt

    Binary labels are derived from YOLO annotations:
        - Fire (1): at least one fire bounding box in the label file.
        - Normal (0): empty label or smoke-only / background.
    """

    LABEL_TO_CLASS = {1: "Fire", 0: "Normal"}

    def __init__(
        self,
        split_dir: str | Path,
        transform: Optional[Callable] = None,
        fire_class_id: int = DFIRE_CLASS_FIRE,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.transform = transform
        self.fire_class_id = fire_class_id
        self.samples: list[Tuple[Path, Path, int]] = self._discover_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.images_dir}. "
                "Expected D-Fire layout: <split>/images/ and <split>/labels/."
            )

        fire_count = sum(label for _, _, label in self.samples)
        normal_count = len(self.samples) - fire_count
        logger.info(
            "DFireBinaryDataset [%s]: %d samples (fire=%d, normal=%d)",
            self.split_dir.name,
            len(self.samples),
            fire_count,
            normal_count,
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
            binary_label = derive_binary_label(label_path, self.fire_class_id)
            samples.append((image_path, label_path, binary_label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, _label_path, label = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        return image, label_tensor


class DFireDataModule:
    """
    Builds train, validation, and test DataLoaders from the D-Fire split folders.

    All tensors are consumed on CUDA inside the ``Trainer`` training loop.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        fire_class_id: int = DFIRE_CLASS_FIRE,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        seed: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fire_class_id = fire_class_id
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        # Shuffling order and per-worker augmentation randomness are only
        # reproducible if the loader is given an explicit generator and worker
        # seeding function; seeding torch alone is not enough.
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

    def _build_loader(
        self,
        split_name: str,
        transform: Callable,
        shuffle: bool,
    ) -> DataLoader:
        dataset = DFireBinaryDataset(
            split_dir=self.root_dir / split_name,
            transform=transform,
            fire_class_id=self.fire_class_id,
        )
        loader_kwargs: dict = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.seed is not None:
            from src.utils import make_generator, seed_worker

            loader_kwargs["worker_init_fn"] = seed_worker
            loader_kwargs["generator"] = make_generator(self.seed)

        # See DFireMulticlassDataModule.setup: this pipeline is I/O-latency
        # bound on many small JPEG reads, so keep workers alive between epochs
        # and prefetch deeper. Throughput-only change; sample order is unaffected.
        # Training loader only (`shuffle` identifies it) -- holding persistent
        # workers on val and test as well tripled the resident process count and
        # contributed to a host-RAM OOM on this 16 GB machine.
        if self.num_workers > 0 and shuffle:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        return DataLoader(dataset, **loader_kwargs)

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
            "DataModule ready — train: %d | val: %d | test: %d | device: %s",
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
