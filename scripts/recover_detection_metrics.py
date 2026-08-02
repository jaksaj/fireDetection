"""Recover test metrics from already-trained YOLO checkpoints.

Why this exists
---------------
Three seeded iteration-4 runs completed all 50 training epochs and then crashed
in the results-parsing step (`ap_class_index` truthiness on a numpy array), so
they exited non-zero and nothing was persisted. The trained weights were saved
by Ultralytics regardless.

Retraining would cost ~2.7 h per seed for work already done. This script instead
loads each saved `best.pt`, runs validation on the requested split, and persists
the metrics through the normal `record_run` path -- recovering the results in
about a minute per seed.

Use it whenever a detection run dies after training but before its metrics are
recorded.

Usage::

    python scripts/recover_detection_metrics.py
    python scripts/recover_detection_metrics.py --runs yolo26-dfire-seed43
    python scripts/recover_detection_metrics.py --split val
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.detection.data_config import DFireDetectionDataConfig
from src.detection.trainer import YOLO26DetectionTrainer
from src.results import record_run
from src.utils import configure_logging

logger = logging.getLogger("recover_detection_metrics")

CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "iteration4"


def discover_runs() -> list[Path]:
    """Every iteration-4 run directory that has a saved best checkpoint."""
    if not CHECKPOINT_ROOT.exists():
        return []
    return sorted(
        path
        for path in CHECKPOINT_ROOT.iterdir()
        if path.is_dir() and (path / "weights" / "best.pt").exists()
    )


def seed_from_name(name: str) -> int | None:
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover metrics from trained YOLO weights.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration4.yaml",
        help="Iteration 4 config, used for the dataset definition.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run directory names under checkpoints/iteration4 (default: all with weights).",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose seed is already recorded in results/metrics.csv.",
    )
    return parser.parse_args()


def already_recorded(seed: int | None) -> bool:
    if seed is None:
        return False
    metrics_csv = PROJECT_ROOT / "results" / "metrics.csv"
    if not metrics_csv.exists():
        return False
    import pandas as pd

    frame = pd.read_csv(metrics_csv)
    subset = frame[(frame["method"] == "iteration4") & (frame["seed"] == seed)]
    return not subset.empty


def main() -> None:
    configure_logging()
    args = parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]

    data_yaml_path = PROJECT_ROOT / data_cfg["yaml_output"]
    DFireDetectionDataConfig(
        root_dir=PROJECT_ROOT / data_cfg["root_dir"],
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        class_names=tuple(data_cfg["class_names"]),
    ).write_yaml(data_yaml_path)

    if args.runs:
        run_dirs = [CHECKPOINT_ROOT / name for name in args.runs]
    else:
        run_dirs = discover_runs()

    if not run_dirs:
        logger.error("No iteration-4 run directories with weights found.")
        return

    recovered: list[tuple[str, float, float]] = []

    for run_dir in run_dirs:
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            logger.warning("Skipping %s — no best.pt", run_dir.name)
            continue

        seed = seed_from_name(run_dir.name)
        if args.skip_existing and already_recorded(seed):
            logger.info("Skipping %s — seed %s already recorded.", run_dir.name, seed)
            continue

        logger.info("Recovering %s (seed=%s)", run_dir.name, seed)
        try:
            trainer = YOLO26DetectionTrainer(
                model_weights=str(weights),
                data_yaml=data_yaml_path,
                checkpoint_dir=CHECKPOINT_ROOT,
                run_name=run_dir.name,
                wandb_config={"mode": "disabled"},
            )
            metrics = trainer.validate(split=args.split)
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop recovery
            logger.error("Failed to recover %s: %s", run_dir.name, exc)
            continue

        metrics["recovered_from"] = str(weights)
        record_run(
            "iteration4",
            metrics,
            seed=seed,
            split=args.split,
            config={"iteration": 4, "task": "object_detection", **config.get("training", {})},
            extra={
                "recovered": True,
                "reason": "run crashed after training in metrics parsing; weights reused",
                "run_dir": str(run_dir),
            },
        )

        map50 = metrics.get(f"{args.split}/metrics/mAP50(B)", float("nan"))
        map5095 = metrics.get(f"{args.split}/metrics/mAP50-95(B)", float("nan"))
        recovered.append((run_dir.name, map50, map5095))
        logger.info("%s: mAP50=%.4f mAP50-95=%.4f", run_dir.name, map50, map5095)

    if not recovered:
        logger.warning("Nothing recovered.")
        return

    print(f"\n{'run':<28} {'mAP50':>9} {'mAP50-95':>10}")
    print("-" * 50)
    for name, map50, map5095 in recovered:
        print(f"{name:<28} {map50:>9.4f} {map5095:>10.4f}")


if __name__ == "__main__":
    main()
