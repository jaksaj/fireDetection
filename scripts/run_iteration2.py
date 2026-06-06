"""Entry point for Iteration 2: 4-class transfer learning on D-Fire."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_multiclass import DFireMulticlassDataModule
from src.model import MobileNetV3FireClassifier
from src.trainer.multiclass import MulticlassTrainer
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Iteration 2 four-class fire/smoke classifier on D-Fire."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration2.yaml",
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

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]

    data_module = DFireMulticlassDataModule(
        root_dir=PROJECT_ROOT / data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        fire_class_id=data_cfg["fire_class_id"],
        smoke_class_id=data_cfg["smoke_class_id"],
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )
    data_module.setup()

    class_weights = None
    if data_cfg.get("use_class_weights", True):
        class_weights = data_module.class_weights()

    model = MobileNetV3FireClassifier(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )

    trainer = MulticlassTrainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        class_names=DFireMulticlassDataModule.CLASS_NAMES,
        learning_rate=train_cfg["head_learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        class_weights=class_weights,
        checkpoint_dir=str(PROJECT_ROOT / train_cfg["checkpoint_dir"]),
        wandb_config=wandb_cfg,
        log_every_n_batches=train_cfg.get("log_every_n_batches", 25),
    )

    experiment_config = {
        "iteration": 2,
        "task": "multiclass_classification",
        **data_cfg,
        **model_cfg,
        **train_cfg,
    }

    test_loader = (
        data_module.test_loader
        if train_cfg.get("run_test_after_training", True)
        else None
    )

    trainer.fit_two_phase(
        head_epochs=train_cfg["head_epochs"],
        finetune_epochs=train_cfg["finetune_epochs"],
        head_learning_rate=train_cfg["head_learning_rate"],
        finetune_learning_rate=train_cfg["finetune_learning_rate"],
        unfreeze_blocks=train_cfg["unfreeze_blocks"],
        experiment_config=experiment_config,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
