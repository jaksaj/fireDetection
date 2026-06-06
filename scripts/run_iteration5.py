"""Entry point for Iteration 5: lightweight custom U-Net semantic segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_segmentation import SegmentationDataModule
from src.model_segmentation import LightweightUNet
from src.losses import DiceFocalLoss
from src.trainer.segmentation import SegmentationTrainer
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Iteration 5 custom U-Net semantic segmentation model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration5.yaml",
        help="Path to YAML experiment configuration.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    # Enable cuDNN auto-tuner for optimal convolution performance
    torch.backends.cudnn.benchmark = True

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]

    # Initialize data module
    data_module = SegmentationDataModule(
        root_dir=PROJECT_ROOT / data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        coco_annotation_file=data_cfg.get("coco_annotation_file", "_annotations.coco.json"),
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )
    data_module.setup()

    # Initialize custom lightweight U-Net
    model = LightweightUNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 3),
        base_channels=model_cfg.get("base_channels", 32),
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Initialize learning rate scheduler
    scheduler_cfg = train_cfg.get("scheduler", {"type": "cosine", "eta_min": 1e-6})
    scheduler_type = scheduler_cfg.get("type", "cosine").lower()
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 2),
            min_lr=scheduler_cfg.get("eta_min", 1e-6),
        )

    # Initialize combined Dice + Focal loss
    criterion = DiceFocalLoss(
        dice_weight=train_cfg.get("dice_weight", 1.0),
        focal_weight=train_cfg.get("focal_weight", 1.0),
        alpha=train_cfg.get("focal_alpha", 0.25),
        gamma=train_cfg.get("focal_gamma", 2.0),
    )

    # Initialize training loop orchestrator
    trainer = SegmentationTrainer(
        use_amp=train_cfg.get("use_amp", True),
        class_names=tuple(data_cfg["classes"]),
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=str(PROJECT_ROOT / train_cfg["checkpoint_dir"]),
        wandb_config=wandb_cfg,
        log_every_n_batches=train_cfg.get("log_every_n_batches", 10),
        wandb_tags=wandb_cfg.get("tags", []),
    )

    if scheduler is not None:
        trainer.set_scheduler(scheduler)

    # Run training and testing
    experiment_config = {
        "iteration": 5,
        "task": "semantic_segmentation",
        **data_cfg,
        **model_cfg,
        **train_cfg,
    }

    test_loader = (
        data_module.test_loader
        if train_cfg.get("run_test_after_training", True)
        else None
    )

    trainer.fit_with_test(
        epochs=train_cfg["epochs"],
        experiment_config=experiment_config,
        test_loader=test_loader,
        phase_name="train",
    )

    trainer.finish_wandb()


if __name__ == "__main__":
    main()
