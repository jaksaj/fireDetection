"""Entry point for Iteration 4: YOLO26 fire and smoke object detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.data_config import DFireDetectionDataConfig
from src.detection.export import YOLOEdgeExporter
from src.detection.trainer import YOLO26DetectionTrainer
from src.results import record_run
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26 object detector on D-Fire with W&B logging."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration4.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX/TensorRT export and FPS benchmark after training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, passed through to Ultralytics. Overrides the config value.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for the Ultralytics run name, so multi-seed runs do not collide.",
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
    export_cfg = config.get("export", {})
    wandb_cfg = config["wandb"]

    # Ultralytics seeds itself (its default is seed=0, deterministic=True), so
    # unlike the PyTorch iterations this one was already reproducible. The seed
    # is threaded through explicitly here so a multi-seed sweep can vary it and
    # so the value lands in the recorded run config rather than being implicit.
    seed = args.seed if args.seed is not None else config.get("seed")
    if seed is not None:
        train_cfg = {**train_cfg, "seed": int(seed)}

    # Ultralytics writes to <checkpoint_dir>/<run_name>/, so the run name must
    # differ per seed or each run overwrites the previous one's weights.
    suffix = args.tag or (f"seed{seed}" if seed is not None else "")
    if suffix:
        train_cfg = {**train_cfg, "run_name": f"{train_cfg['run_name']}-{suffix}"}

    data_root = PROJECT_ROOT / data_cfg["root_dir"]
    data_yaml_path = PROJECT_ROOT / data_cfg["yaml_output"]

    data_config = DFireDetectionDataConfig(
        root_dir=data_root,
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        class_names=tuple(data_cfg["class_names"]),
    )
    data_config.write_yaml(data_yaml_path)

    trainer = YOLO26DetectionTrainer(
        model_weights=model_cfg["weights"],
        data_yaml=data_yaml_path,
        checkpoint_dir=PROJECT_ROOT / train_cfg["checkpoint_dir"],
        run_name=train_cfg["run_name"],
        wandb_config=wandb_cfg,
    )

    train_metrics = trainer.train(train_cfg)
    print("\nTraining metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value}")

    if train_cfg.get("run_test_eval", True):
        test_metrics = trainer.validate(split="test")
        print("\nTest metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value}")

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(test_metrics)
        except ImportError:
            pass
    else:
        test_metrics = {}

    # Persist to results/ like every other iteration. This was missing: the
    # patch that added `record_run` to the other four run scripts skipped this
    # one because iteration 4 does not use BaseTrainer. The consequence was
    # silent -- a seeded run trained for 2.7 h, exited 0, and left no row in
    # results/metrics.csv, so it looked like it had never run.
    record_run(
        "iteration4",
        {**train_metrics, **test_metrics},
        seed=seed,
        split="test" if test_metrics else "val",
        config={"iteration": 4, "task": "object_detection", **data_cfg, **model_cfg, **train_cfg},
        extra={"config_path": str(args.config), "run_name": train_cfg["run_name"]},
    )

    if args.skip_export or not export_cfg.get("enabled", True):
        return

    best_weights = trainer.best_weights_path
    if best_weights is None or not best_weights.exists():
        print("Best weights not found — skipping export pipeline.")
        return

    exporter = YOLOEdgeExporter(
        weights_path=best_weights,
        export_dir=PROJECT_ROOT / export_cfg.get("export_dir", "checkpoints/iteration4/exports"),
        imgsz=train_cfg.get("imgsz", 640),
    )

    benchmark_source = PROJECT_ROOT / export_cfg.get("benchmark_source", "data/test/images")
    export_summary = exporter.run_full_export_pipeline(
        export_formats=tuple(export_cfg.get("formats", ["onnx"])),
        benchmark_source=benchmark_source,
        target_edge_fps=export_cfg.get("target_edge_fps"),
    )

    print("\nEdge export summary:")
    for key, value in export_summary.items():
        print(f"  {key}: {value}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(export_summary)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
