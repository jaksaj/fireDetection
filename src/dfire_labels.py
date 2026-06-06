"""Utilities for parsing D-Fire YOLO annotation files."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Official D-Fire data.yaml mapping: 0 = smoke, 1 = fire
DFIRE_CLASS_SMOKE = 0
DFIRE_CLASS_FIRE = 1

# Iteration 2 multi-class image-level labels
MULTICLASS_NEITHER = 0
MULTICLASS_ONLY_FIRE = 1
MULTICLASS_ONLY_SMOKE = 2
MULTICLASS_BOTH = 3

MULTICLASS_CLASS_NAMES = ("Neither", "Only_Fire", "Only_Smoke", "Both")
NUM_MULTICLASS_CLASSES = len(MULTICLASS_CLASS_NAMES)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_yolo_label_file(label_path: Path) -> list[int]:
    """
    Parse a YOLO-format label file and return detected class IDs.

    Args:
        label_path: Path to a ``.txt`` annotation file.

    Returns:
        List of integer class IDs (one per bounding box line).
    """
    if not label_path.exists():
        return []

    class_ids: list[int] = []
    content = label_path.read_text(encoding="utf-8").strip()

    if not content:
        return []

    for line_number, line in enumerate(content.splitlines(), start=1):
        tokens = line.strip().split()
        if not tokens:
            continue
        try:
            class_ids.append(int(tokens[0]))
        except ValueError:
            logger.warning(
                "Skipping malformed label line %d in %s: %r",
                line_number,
                label_path,
                line,
            )

    return class_ids


def has_fire_annotation(
    label_path: Path,
    fire_class_id: int = DFIRE_CLASS_FIRE,
) -> bool:
    """
    Return True when the label file contains at least one fire bounding box.

    For Iteration 1 binary classification:
        - Fire (1): image has a fire bbox (class ``fire_class_id``).
        - Normal (0): empty label, smoke-only, or background.
    """
    return fire_class_id in parse_yolo_label_file(label_path)


def derive_binary_label(
    label_path: Path,
    fire_class_id: int = DFIRE_CLASS_FIRE,
) -> int:
    """Map a YOLO label file to a binary target: 1 = Fire, 0 = Normal."""
    return 1 if has_fire_annotation(label_path, fire_class_id) else 0


def derive_multiclass_label(
    label_path: Path,
    fire_class_id: int = DFIRE_CLASS_FIRE,
    smoke_class_id: int = DFIRE_CLASS_SMOKE,
) -> int:
    """
    Map a YOLO label file to a 4-class image-level target.

    Classes:
        0 - Neither:   no fire or smoke annotations
        1 - Only_Fire: fire present, smoke absent
        2 - Only_Smoke: smoke present, fire absent
        3 - Both:      fire and smoke present
    """
    class_ids = parse_yolo_label_file(label_path)
    has_fire = fire_class_id in class_ids
    has_smoke = smoke_class_id in class_ids

    if has_fire and has_smoke:
        return MULTICLASS_BOTH
    if has_fire:
        return MULTICLASS_ONLY_FIRE
    if has_smoke:
        return MULTICLASS_ONLY_SMOKE
    return MULTICLASS_NEITHER
