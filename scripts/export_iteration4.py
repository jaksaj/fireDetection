"""Export a trained YOLO26 checkpoint to ONNX/TensorRT and benchmark edge FPS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.export import YOLOEdgeExporter
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 weights for edge deployment and benchmark FPS."
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
        default=None,
        help="Path to trained .pt weights (defaults to best in checkpoint dir).",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=None,
        help="Export formats: onnx engine openvino",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log export metrics to Weights & Biases.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config["training"]
    export_cfg = config.get("export", {})
    wandb_cfg = config.get("wandb", {})

    default_weights = (
        PROJECT_ROOT
        / train_cfg["checkpoint_dir"]
        / train_cfg["run_name"]
        / "weights"
        / "best.pt"
    )
    weights_path = args.weights or default_weights
    export_formats = tuple(args.format or export_cfg.get("formats", ["onnx"]))

    exporter = YOLOEdgeExporter(
        weights_path=weights_path,
        export_dir=PROJECT_ROOT / export_cfg.get("export_dir", "checkpoints/iteration4/exports"),
        imgsz=train_cfg.get("imgsz", 640),
    )

    benchmark_source = PROJECT_ROOT / export_cfg.get("benchmark_source", "data/test/images")
    summary = exporter.run_full_export_pipeline(
        export_formats=export_formats,
        benchmark_source=benchmark_source,
        target_edge_fps=export_cfg.get("target_edge_fps"),
    )

    print("Export summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.log_wandb:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "smoke-fire-detection"),
            entity=wandb_cfg.get("entity"),
            name=f"{wandb_cfg.get('run_name', 'iteration4')}-export",
            tags=[*(wandb_cfg.get("tags", [])), "export"],
            config=config,
        )
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
