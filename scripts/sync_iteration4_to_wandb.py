"""Reconstruct an Iteration 4 W&B run from the saved Ultralytics results.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a completed Iteration 4 YOLO run to Weights & Biases from results.csv."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "iteration4" / "yolo26-dfire" / "results.csv",
        help="Path to the Ultralytics results.csv file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "iteration4" / "yolo26-dfire" / "weights" / "best.pt",
        help="Optional best checkpoint to upload as an artifact.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration4.yaml",
        help="Path to the iteration 4 config used for the run.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional W&B run name override. Defaults to the training config run name.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="smoke-fire-detection",
        help="W&B project name.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--upload-artifacts",
        action="store_true",
        help="Upload the CSV and best weights as W&B artifacts.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _to_number(value: str):
    text = value.strip()
    if text == "":
        return None

    try:
        integer_value = int(text)
        if str(integer_value) == text:
            return integer_value
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not args.results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {args.results_csv}")

    wandb.init(
        project=args.project,
        entity=args.entity or config.get("wandb", {}).get("entity"),
        name=args.name or config.get("wandb", {}).get("run_name"),
        tags=config.get("wandb", {}).get("tags", ["iteration4", "yolo26", "object-detection", "d-fire", "edge-deployment"]),
        config=config,
    )

    with args.results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        last_row: dict[str, object] | None = None

        for row in reader:
            parsed_row = {
                key: _to_number(value)
                for key, value in row.items()
                if value is not None and value.strip() != ""
            }
            last_row = parsed_row
            step = int(parsed_row.get("epoch", 0) or 0)
            wandb.log(parsed_row, step=step)

    if last_row is not None:
        wandb.run.summary.update(last_row)

    if args.upload_artifacts:
        artifact = wandb.Artifact("iteration4-results", type="results")
        artifact.add_file(str(args.results_csv))
        if args.weights.exists():
            artifact.add_file(str(args.weights))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()