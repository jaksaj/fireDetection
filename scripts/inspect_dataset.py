"""Print D-Fire split statistics and integrity checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_inspector import DFireDatasetInspector
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect D-Fire train/val/test splits.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root directory containing train/, val/, and test/ splits.",
    )
    parser.add_argument(
        "--fire-class-id",
        type=int,
        default=1,
        help="YOLO class ID for fire (D-Fire default: 1).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split folder names to inspect.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if missing/orphan labels are found.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    inspector = DFireDatasetInspector(
        root_dir=args.data_dir,
        fire_class_id=args.fire_class_id,
    )
    report = inspector.inspect(splits=tuple(args.splits))
    print(DFireDatasetInspector.format_report(report))

    if args.strict:
        issues = [
            f"{split_name}: {stats.missing_labels} missing, {stats.orphan_labels} orphan"
            for split_name, stats in report.splits.items()
            if stats.missing_labels or stats.orphan_labels
        ]
        if issues:
            print("\nDataset integrity errors:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)


if __name__ == "__main__":
    main()
