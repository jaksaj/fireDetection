"""Run edge PTQ simulation on a trained Iteration 3 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_multiclass import DFireMulticlassDataModule
from src.edge_simulation import run_edge_simulation
from src.model import MobileNetV3FireClassifier
from src.utils import configure_logging, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and benchmark an Iteration 3 checkpoint for edge deployment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration3.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "iteration3" / "best_model.pt",
        help="Path to a trained FP32 checkpoint.",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log edge simulation metrics to Weights & Biases.",
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
    wandb_cfg = config.get("wandb", {})

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
        augmentation=data_cfg.get("augmentation", "albumentations"),
    )
    data_module.setup()

    model = MobileNetV3FireClassifier(
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        dropout=model_cfg["dropout"],
    )
    load_checkpoint(args.checkpoint, model)

    edge_metrics = run_edge_simulation(
        model=model,
        test_loader=data_module.test_loader,
        class_names=DFireMulticlassDataModule.CLASS_NAMES,
        checkpoint_dir=args.checkpoint.parent,
        image_size=data_cfg["image_size"],
        fp32_checkpoint_name=args.checkpoint.name,
    )

    print("Edge simulation results:")
    for key, value in edge_metrics.items():
        print(f"  {key}: {value}")

    if args.log_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "smoke-fire-detection"),
            entity=wandb_cfg.get("entity"),
            name=f"{wandb_cfg.get('run_name', 'iteration3')}-edge-sim",
            tags=[*(wandb_cfg.get("tags", [])), "edge-simulation"],
            config=config,
        )
        wandb.log(edge_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
