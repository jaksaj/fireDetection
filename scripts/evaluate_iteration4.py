"""Validate a trained YOLO26 checkpoint on D-Fire val or test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.data_config import DFireDetectionDataConfig
from src.detection.trainer import YOLO26DetectionTrainer
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26 detector mAP on val or test split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration4.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained .pt weights.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate.",
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
    train_cfg = config["training"]

    data_yaml_path = PROJECT_ROOT / data_cfg["yaml_output"]
    if not data_yaml_path.exists():
        data_config = DFireDetectionDataConfig(
            root_dir=PROJECT_ROOT / data_cfg["root_dir"],
            train_split=data_cfg["train_split"],
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            class_names=tuple(data_cfg["class_names"]),
        )
        data_config.write_yaml(data_yaml_path)

    from ultralytics import YOLO

    trainer = YOLO26DetectionTrainer(
        model_weights=str(args.weights),
        data_yaml=data_yaml_path,
        checkpoint_dir=PROJECT_ROOT / train_cfg["checkpoint_dir"],
        run_name=train_cfg["run_name"],
    )
    trainer.model = YOLO(str(args.weights))

    metrics = trainer.validate(split=args.split)
    print(f"\n{args.split} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
