"""Print COCO segmentation split statistics, verify annotations, and save visual overlay examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_segmentation_inspector import COCOSegmentationInspector
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect COCO segmentation train/val/test splits.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "coco",
        help="Root directory containing train/, valid/, and test/ splits.",
    )
    parser.add_argument(
        "--coco-annotation-file",
        type=str,
        default="_annotations.coco.json",
        help="COCO annotation file name (default: _annotations.coco.json).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Split folder names to inspect.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of visualization example overlays to save per split.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    print(f"Inspecting dataset at: {args.data_dir}...")
    inspector = COCOSegmentationInspector(
        root_dir=args.data_dir,
        coco_annotation_file=args.coco_annotation_file,
    )
    
    report = inspector.inspect(
        splits=tuple(args.splits),
        save_examples_count=args.examples,
    )
    
    print("\n" + "=" * 50)
    print(COCOSegmentationInspector.format_report(report))
    print("=" * 50)
    
    if args.examples > 0:
        examples_path = args.data_dir / "examples"
        print(f"\n[Info] Saved overlay examples to: {examples_path.resolve()}")


if __name__ == "__main__":
    main()
