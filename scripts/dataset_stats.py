"""Persist D-Fire dataset statistics for the Dataset chapter.

``src/dataset_inspector.py`` already computes counts and class balance, but
``scripts/inspect_dataset.py`` only prints them to a terminal, so no file in the
repository records how large the splits are or how the classes are distributed.
That gap matters beyond bookkeeping: ``README.md`` attributes the low
``Only_Fire`` F1 (68.78%) to class imbalance, and until now nothing in the
project evidenced that imbalance.

This script writes:

- ``results/dataset_stats.json`` -- full nested record.
- ``results/dataset_stats.csv`` -- one row per (split, class) for tables.
- ``results/split_manifest.csv`` -- every filename with its split and label.

The manifest is the important one for defensibility: it pins down exactly which
images were in which split, so a result can be traced to the data that produced
it even if ``data/`` is later rebuilt.

It also runs a **cross-split integrity check**. D-Fire ships train/test only;
this project's ``val/`` was carved out of the official train pool. The check
verifies that no filename appears in two splits and reports the numeric ID
ranges per source prefix, which is what shows whether the official test split is
still intact.

Usage::

    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --hash-check     # also scan for duplicate image content
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dfire_labels import (
    DFIRE_CLASS_FIRE,
    DFIRE_CLASS_SMOKE,
    IMAGE_EXTENSIONS,
    MULTICLASS_CLASS_NAMES,
    derive_multiclass_label,
    parse_yolo_label_file,
)
from src.results import RESULTS_DIR, append_rows, environment_info
from src.utils import configure_logging

logger = logging.getLogger("dataset_stats")

DATA_DIR = PROJECT_ROOT / "data"
SPLITS = ("train", "val", "test")

NAME_PATTERN = re.compile(r"^([A-Za-z_]+?)(\d+)$")

STATS_FIELDS = [
    "split",
    "class_name",
    "class_index",
    "count",
    "percentage",
    "n_split",
]

MANIFEST_FIELDS = [
    "split",
    "filename",
    "prefix",
    "numeric_id",
    "label_index",
    "label_name",
    "n_smoke_boxes",
    "n_fire_boxes",
]


def collect_split(split: str) -> dict:
    """Gather per-image labels and box counts for one split."""
    image_dir = DATA_DIR / split / "images"
    label_dir = DATA_DIR / split / "labels"

    if not image_dir.exists():
        logger.warning("Missing split directory: %s", image_dir)
        return {}

    label_counts: Counter = Counter()
    box_counts: Counter = Counter()
    prefix_ids: dict[str, list[int]] = defaultdict(list)
    manifest: list[dict] = []
    missing_labels = 0

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1

        class_ids = parse_yolo_label_file(label_path)
        label_index = derive_multiclass_label(label_path)
        label_counts[label_index] += 1

        n_smoke = sum(1 for c in class_ids if c == DFIRE_CLASS_SMOKE)
        n_fire = sum(1 for c in class_ids if c == DFIRE_CLASS_FIRE)
        box_counts["smoke"] += n_smoke
        box_counts["fire"] += n_fire

        match = NAME_PATTERN.match(image_path.stem)
        prefix = match.group(1) if match else "OTHER"
        numeric_id = int(match.group(2)) if match else -1
        prefix_ids[prefix].append(numeric_id)

        manifest.append(
            {
                "split": split,
                "filename": image_path.name,
                "prefix": prefix,
                "numeric_id": numeric_id,
                "label_index": label_index,
                "label_name": MULTICLASS_CLASS_NAMES[label_index],
                "n_smoke_boxes": n_smoke,
                "n_fire_boxes": n_fire,
            }
        )

    total = len(images)
    return {
        "split": split,
        "n_images": total,
        "missing_label_files": missing_labels,
        "label_counts": {
            MULTICLASS_CLASS_NAMES[i]: label_counts.get(i, 0)
            for i in range(len(MULTICLASS_CLASS_NAMES))
        },
        "label_percentages": {
            MULTICLASS_CLASS_NAMES[i]: (
                100.0 * label_counts.get(i, 0) / total if total else 0.0
            )
            for i in range(len(MULTICLASS_CLASS_NAMES))
        },
        # Iteration 1's binary target: fire present (Only_Fire or Both).
        "binary_fire": label_counts.get(1, 0) + label_counts.get(3, 0),
        "binary_normal": label_counts.get(0, 0) + label_counts.get(2, 0),
        "box_counts": dict(box_counts),
        "prefix_ranges": {
            prefix: {
                "count": len(ids),
                "min_id": min(ids),
                "max_id": max(ids),
            }
            for prefix, ids in sorted(prefix_ids.items())
        },
        "_manifest": manifest,
        "_filenames": {entry["filename"] for entry in manifest},
    }


def collect_coco() -> dict:
    """
    Statistics for the Roboflow COCO segmentation set used by iteration 5.

    This was previously uncounted anywhere in results/, which mattered more than
    it sounds: the split turns out to be 99.16 / 0.56 / 0.28 percent, so the
    segmentation test metric is computed on **20 images**. A held-out set that
    small cannot support a precise accuracy claim, and the seed-to-seed standard
    deviation reported elsewhere reflects training variance only -- it says
    nothing about the sampling error of a 20-image evaluation.

    Also checks for RLE / non-polygon segmentations, which
    ``src/dataset_segmentation.py`` silently drops to background.
    """
    coco_dir = DATA_DIR / "coco"
    if not coco_dir.exists():
        logger.warning("No COCO dataset at %s", coco_dir)
        return {}

    splits: dict = {}
    total_images = 0
    for split in ("train", "valid", "test"):
        annotation_file = coco_dir / split / "_annotations.coco.json"
        if not annotation_file.exists():
            continue
        with annotation_file.open(encoding="utf-8") as handle:
            data = json.load(handle)

        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        per_category: Counter = Counter()
        non_polygon = 0
        for annotation in data.get("annotations", []):
            per_category[categories.get(annotation["category_id"], "?")] += 1
            if not isinstance(annotation.get("segmentation"), list):
                non_polygon += 1

        n_images = len(data.get("images", []))
        total_images += n_images
        splits[split] = {
            "n_images": n_images,
            "n_images_with_annotations": len({a["image_id"] for a in data.get("annotations", [])}),
            "n_annotations": len(data.get("annotations", [])),
            "annotations_per_category": dict(per_category),
            "non_polygon_annotations": non_polygon,
        }

    for split, info in splits.items():
        info["share_percent"] = (
            100.0 * info["n_images"] / total_images if total_images else 0.0
        )

    return {
        "source": "Roboflow COCO export (segmentation, iteration 5)",
        "total_images": total_images,
        "splits": splits,
        "notes": (
            "Split is extremely unbalanced; the test split holds 20 images. "
            "Segmentation accuracy figures must be reported with that caveat. "
            "All segmentations are polygons (no RLE), so nothing is dropped by "
            "the polygon-only mask renderer in src/dataset_segmentation.py."
        ),
    }


def integrity_check(per_split: dict[str, dict], hash_check: bool) -> dict:
    """
    Check for filename and (optionally) content overlap across splits.

    Filename overlap would mean an image is literally in two splits. Content
    hashing additionally catches the same image saved under two names, which a
    naive re-split can easily introduce.
    """
    report: dict = {"filename_overlaps": {}, "content_duplicates": {}}

    names = {split: data.get("_filenames", set()) for split, data in per_split.items()}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1 :]:
            overlap = names.get(first, set()) & names.get(second, set())
            report["filename_overlaps"][f"{first}&{second}"] = len(overlap)
            if overlap:
                report["filename_overlaps"][f"{first}&{second}_examples"] = sorted(overlap)[:10]

    if not hash_check:
        report["content_duplicates"]["status"] = "not run (pass --hash-check)"
        return report

    logger.info("Hashing image content across splits — this reads every file.")
    digests: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split in SPLITS:
        image_dir = DATA_DIR / split / "images"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = hashlib.md5(image_path.read_bytes()).hexdigest()
            digests[digest].append((split, image_path.name))

    cross_split = {
        digest: entries
        for digest, entries in digests.items()
        if len({split for split, _ in entries}) > 1
    }
    within_split = {
        digest: entries
        for digest, entries in digests.items()
        if len(entries) > 1 and len({split for split, _ in entries}) == 1
    }

    report["content_duplicates"] = {
        "status": "checked",
        "n_cross_split_duplicate_groups": len(cross_split),
        "n_within_split_duplicate_groups": len(within_split),
        "cross_split_examples": [entries for entries in list(cross_split.values())[:10]],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist D-Fire dataset statistics.")
    parser.add_argument(
        "--hash-check",
        action="store_true",
        help="Also MD5-hash every image to detect duplicate content across splits.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing the per-image split manifest.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_split = {split: collect_split(split) for split in SPLITS}
    per_split = {split: data for split, data in per_split.items() if data}

    if not per_split:
        logger.error("No splits found under %s", DATA_DIR)
        return

    integrity = integrity_check(per_split, args.hash_check)

    totals = Counter()
    for data in per_split.values():
        totals.update(data["label_counts"])
    grand_total = sum(totals.values())

    # CSV: one row per (split, class).
    rows: list[dict] = []
    for split, data in per_split.items():
        for index, name in enumerate(MULTICLASS_CLASS_NAMES):
            rows.append(
                {
                    "split": split,
                    "class_name": name,
                    "class_index": index,
                    "count": data["label_counts"][name],
                    "percentage": data["label_percentages"][name],
                    "n_split": data["n_images"],
                }
            )
    for index, name in enumerate(MULTICLASS_CLASS_NAMES):
        rows.append(
            {
                "split": "all",
                "class_name": name,
                "class_index": index,
                "count": totals[name],
                "percentage": 100.0 * totals[name] / grand_total if grand_total else 0.0,
                "n_split": grand_total,
            }
        )

    stats_path = RESULTS_DIR / "dataset_stats.csv"
    stats_path.unlink(missing_ok=True)
    append_rows("dataset_stats.csv", rows, STATS_FIELDS)

    if not args.no_manifest:
        manifest_path = RESULTS_DIR / "split_manifest.csv"
        manifest_path.unlink(missing_ok=True)
        manifest_rows = [row for data in per_split.values() for row in data["_manifest"]]
        append_rows("split_manifest.csv", manifest_rows, MANIFEST_FIELDS)
        logger.info("Wrote manifest with %d rows.", len(manifest_rows))

    coco = collect_coco()

    record = {
        "dataset": "D-Fire (YOLO format)",
        "class_mapping": {"0": "smoke", "1": "fire"},
        "image_level_classes": list(MULTICLASS_CLASS_NAMES),
        "environment": environment_info(),
        "splits": {
            split: {key: value for key, value in data.items() if not key.startswith("_")}
            for split, data in per_split.items()
        },
        "totals": {
            "n_images": grand_total,
            "label_counts": dict(totals),
            "label_percentages": {
                name: 100.0 * count / grand_total if grand_total else 0.0
                for name, count in totals.items()
            },
        },
        "integrity": integrity,
        "coco_segmentation": coco,
    }

    json_path = RESULTS_DIR / "dataset_stats.json"
    json_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print(f"\n{'split':<8} {'n':>7} " + " ".join(f"{n:>12}" for n in MULTICLASS_CLASS_NAMES))
    print("-" * 78)
    for split, data in per_split.items():
        counts = data["label_counts"]
        percentages = data["label_percentages"]
        cells = " ".join(
            f"{counts[n]:>6} {percentages[n]:>4.1f}%" for n in MULTICLASS_CLASS_NAMES
        )
        print(f"{split:<8} {data['n_images']:>7} {cells}")

    print("\nCross-split filename overlap:")
    for key, value in integrity["filename_overlaps"].items():
        if not key.endswith("_examples"):
            print(f"  {key}: {value}")

    print("\nSource-prefix ID ranges (contiguous, disjoint ranges in test indicate")
    print("the official D-Fire test split is intact):")
    for split, data in per_split.items():
        for prefix, info in data["prefix_ranges"].items():
            print(
                f"  {split:<6} {prefix:<14} n={info['count']:>6} "
                f"ids {info['min_id']}..{info['max_id']}"
            )

    if coco:
        print("\nRoboflow COCO segmentation set (iteration 5):")
        for split, info in coco["splits"].items():
            print(
                f"  {split:<6} {info['n_images']:>5} images "
                f"({info['share_percent']:>5.2f}%)  "
                f"{info['n_annotations']:>6} annotations  "
                f"non-polygon: {info['non_polygon_annotations']}"
            )
        print(f"  TOTAL  {coco['total_images']:>5} images")

    logger.info("Wrote %s and %s", stats_path.name, json_path.name)


if __name__ == "__main__":
    main()
