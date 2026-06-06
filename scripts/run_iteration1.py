"""Entry point for Iteration 1: Binary Fire vs. Normal classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DFireDataModule
from src.model import FireCNN
from src.train import Trainer
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Iteration 1 binary fire classifier on D-Fire."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration1.yaml",
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

    torch.backends.cudnn.benchmark = True

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]

    data_module = DFireDataModule(
        root_dir=PROJECT_ROOT / data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        fire_class_id=data_cfg["fire_class_id"],
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )
    data_module.setup()

    model = FireCNN(
        in_channels=model_cfg["in_channels"],
        num_conv_blocks=model_cfg["num_conv_blocks"],
        base_channels=model_cfg["base_channels"],
    )

    trainer = Trainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        checkpoint_dir=str(PROJECT_ROOT / train_cfg["checkpoint_dir"]),
        wandb_config=wandb_cfg,
        log_every_n_batches=train_cfg.get("log_every_n_batches", 25),
    )

    experiment_config = {
        "iteration": 1,
        "task": "binary_classification",
        **data_cfg,
        **model_cfg,
        **train_cfg,
    }

    test_loader = (
        data_module.test_loader
        if train_cfg.get("run_test_after_training", True)
        else None
    )

    trainer.fit(
        epochs=train_cfg["epochs"],
        experiment_config=experiment_config,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
