"""Score every detection paradigm on one common task, on one test set.

The problem this solves
-----------------------
The project's five methods report 93.80% binary accuracy, 90.25% 4-class
accuracy, 74.40 mAP50, and 85.22% mIoU. Those are four different scales
measuring four different things, so "which method is better?" is not answerable
from them -- not for lack of measurements, but because there is no shared
yardstick. A thesis comparing detection paradigms needs one.

The protocol
------------
Every model's native output is collapsed to an **image-level presence
prediction** on the same D-Fire test images:

    method              native output        collapse rule
    ------------------  -------------------  ----------------------------------
    FireCNN (iter 1)    binary logit         sigmoid > 0.5 -> fire present
    MobileNetV3 (2, 3)  4-class softmax      argmax -> {fire, smoke} presence
    YOLO26n (iter 4)    boxes + scores       any box of class c, conf > tau
    U-Net (iter 5)      per-pixel mask       pixels of class c > tau * area

Ground truth comes from the same D-Fire YOLO annotations used by every
iteration (``src/dfire_labels.derive_multiclass_label``), so the labels are
identical across methods by construction.

Two axes are reported:

- **Binary "fire present"** -- the only question all five methods can answer,
  and therefore the primary common axis.
- **4-class Neither/Only_Fire/Only_Smoke/Both** -- for the four methods that
  distinguish smoke from fire. Iteration 1 cannot and is excluded there rather
  than being given a degenerate mapping that would flatter or penalise it
  arbitrarily.

Caveats recorded with the results
---------------------------------
Iteration 5 was trained on a separate Roboflow COCO dataset, so evaluating it
on D-Fire images measures it under domain shift. That is a real limitation, not
a bug: it is reported in the output and must be stated in the Discussion.

Usage::

    python scripts/evaluate_common.py
    python scripts/evaluate_common.py --methods iteration4 --sweep
    python scripts/evaluate_common.py --limit 200      # quick check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms

from src.dfire_labels import (
    IMAGE_EXTENSIONS,
    MULTICLASS_BOTH,
    MULTICLASS_CLASS_NAMES,
    MULTICLASS_NEITHER,
    MULTICLASS_ONLY_FIRE,
    MULTICLASS_ONLY_SMOKE,
    derive_multiclass_label,
)
from src.results import append_rows, record_run
from src.utils import configure_logging, resolve_device

logger = logging.getLogger("evaluate_common")

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Segmentation target classes (src/dataset_segmentation.py): fire is drawn over
# smoke, so class 2 wins any overlap.
SEG_CLASS_SMOKE = 1
SEG_CLASS_FIRE = 2

COMMON_FIELDS = [
    "method",
    "model_name",
    "paradigm",
    "axis",
    "threshold",
    "n_images",
    "accuracy",
    "f1_macro",
    "f1_fire",
    "precision_fire",
    "recall_fire",
    "f1_Neither",
    "f1_Only_Fire",
    "f1_Only_Smoke",
    "f1_Both",
    "domain_shift",
    "notes",
]


def presence_to_multiclass(has_fire: bool, has_smoke: bool) -> int:
    """Map (fire, smoke) presence flags to the 4-class D-Fire label."""
    if has_fire and has_smoke:
        return MULTICLASS_BOTH
    if has_fire:
        return MULTICLASS_ONLY_FIRE
    if has_smoke:
        return MULTICLASS_ONLY_SMOKE
    return MULTICLASS_NEITHER


def list_test_images(split: str = "test") -> list[Path]:
    """All images in the split, sorted so every method sees the same order."""
    image_dir = DATA_DIR / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {image_dir}")
    return sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ground_truth_labels(image_paths: list[Path], split: str = "test") -> np.ndarray:
    """Derive 4-class ground truth from the D-Fire YOLO annotations."""
    label_dir = DATA_DIR / split / "labels"
    return np.array(
        [derive_multiclass_label(label_dir / f"{path.stem}.txt") for path in image_paths],
        dtype=np.int64,
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """The deterministic eval transform used by every classification iteration."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def predict_classifier(
    model: torch.nn.Module,
    image_paths: list[Path],
    image_size: int,
    device: torch.device,
    binary: bool,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run a classification model over the split.

    Returns an array of 4-class labels. For a binary model, only the fire axis
    is meaningful: predictions are ``Only_Fire`` or ``Neither``, and the caller
    must score them on the binary axis alone.
    """
    transform = build_eval_transform(image_size)
    model = model.to(device).eval()
    predictions: list[int] = []

    for batch_paths in _batched(image_paths, batch_size):
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        logits = model(batch)

        if binary:
            fire = (torch.sigmoid(logits.squeeze(-1)) > 0.5).cpu().numpy()
            predictions.extend(
                presence_to_multiclass(bool(flag), False) for flag in fire
            )
        else:
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return np.array(predictions, dtype=np.int64)


@torch.no_grad()
def predict_segmentation(
    model: torch.nn.Module,
    image_paths: list[Path],
    image_size: int,
    device: torch.device,
    area_threshold: float,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Collapse per-pixel segmentation masks to image-level presence.

    A class counts as present when it occupies more than ``area_threshold`` of
    the image. A bare ``> 0 pixels`` rule would make the segmenter fire on
    single stray pixels and is not how a deployed alerting system would read a
    mask, so the threshold is swept and reported rather than assumed.
    """
    transform = build_eval_transform(image_size)
    model = model.to(device).eval()
    predictions: list[int] = []

    for batch_paths in _batched(image_paths, batch_size):
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        masks = model(batch).argmax(dim=1)

        total_pixels = masks.shape[-1] * masks.shape[-2]
        smoke_fraction = (masks == SEG_CLASS_SMOKE).sum(dim=(1, 2)).float() / total_pixels
        fire_fraction = (masks == SEG_CLASS_FIRE).sum(dim=(1, 2)).float() / total_pixels

        for smoke, fire in zip(smoke_fraction.cpu().numpy(), fire_fraction.cpu().numpy()):
            predictions.append(
                presence_to_multiclass(fire > area_threshold, smoke > area_threshold)
            )

    return np.array(predictions, dtype=np.int64)


def predict_detector(
    weights: Path,
    image_paths: list[Path],
    device: torch.device,
    conf_threshold: float,
    imgsz: int = 640,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Collapse detector boxes to image-level presence.

    A class is present when at least one box of that class survives
    ``conf_threshold``. The threshold is the detector's operating point and is
    swept, because a detector tuned for high recall and one tuned for high
    precision are different deployments of the same weights.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    predictions: list[int] = []

    for batch_paths in _batched(image_paths, batch_size):
        results = model.predict(
            source=[str(path) for path in batch_paths],
            conf=conf_threshold,
            imgsz=imgsz,
            device=str(device),
            verbose=False,
        )
        for result in results:
            classes = set()
            if result.boxes is not None and len(result.boxes):
                classes = {int(value) for value in result.boxes.cls.cpu().numpy()}
            # D-Fire detection classes: 0 = smoke, 1 = fire.
            predictions.append(presence_to_multiclass(1 in classes, 0 in classes))

    return np.array(predictions, dtype=np.int64)


def score(
    truth: np.ndarray,
    predicted: np.ndarray,
    axis: str,
) -> dict[str, float]:
    """Compute the metric set for one axis ('binary' or 'multiclass')."""
    if axis == "binary":
        truth_binary = np.isin(truth, [MULTICLASS_ONLY_FIRE, MULTICLASS_BOTH]).astype(int)
        predicted_binary = np.isin(
            predicted, [MULTICLASS_ONLY_FIRE, MULTICLASS_BOTH]
        ).astype(int)
        return {
            "accuracy": accuracy_score(truth_binary, predicted_binary),
            "f1_macro": f1_score(truth_binary, predicted_binary, average="macro", zero_division=0),
            "f1_fire": f1_score(truth_binary, predicted_binary, pos_label=1, zero_division=0),
            "precision_fire": precision_score(
                truth_binary, predicted_binary, pos_label=1, zero_division=0
            ),
            "recall_fire": recall_score(
                truth_binary, predicted_binary, pos_label=1, zero_division=0
            ),
        }

    labels = list(range(len(MULTICLASS_CLASS_NAMES)))
    per_class = f1_score(truth, predicted, labels=labels, average=None, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(truth, predicted),
        "f1_macro": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
    }
    for name, value in zip(MULTICLASS_CLASS_NAMES, per_class):
        metrics[f"f1_{name}"] = float(value)
    return metrics


def load_state(model: torch.nn.Module, path: Path, device: torch.device) -> bool:
    if not path.exists():
        logger.warning("Checkpoint missing: %s", path)
        return False
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score all methods on one common task.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["iteration1", "iteration2", "iteration3", "iteration4", "iteration5"],
        help="Methods to evaluate.",
    )
    parser.add_argument("--split", default="test", help="Data split to evaluate on.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep detector confidence and mask-area thresholds.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument(
        "--mask-area", type=float, default=0.005, help="Segmentation area threshold."
    )
    parser.add_argument("--output", default="common_eval.csv", help="CSV filename under results/.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    image_paths = list_test_images(args.split)
    if args.limit:
        image_paths = image_paths[: args.limit]
    truth = ground_truth_labels(image_paths, args.split)

    logger.info(
        "Common evaluation on %d %s images; ground-truth distribution: %s",
        len(image_paths),
        args.split,
        {
            name: int((truth == index).sum())
            for index, name in enumerate(MULTICLASS_CLASS_NAMES)
        },
    )

    rows: list[dict] = []

    def emit(method: str, model_name: str, paradigm: str, predicted: np.ndarray,
             axes: list[str], threshold: float = float("nan"),
             domain_shift: bool = False, notes: str = "") -> None:
        for axis in axes:
            metrics = score(truth, predicted, axis)
            row = {
                "method": method,
                "model_name": model_name,
                "paradigm": paradigm,
                "axis": axis,
                "threshold": threshold,
                "n_images": len(image_paths),
                "domain_shift": int(domain_shift),
                "notes": notes,
                **metrics,
            }
            rows.append(row)
            logger.info(
                "%-12s %-11s thr=%-6s acc=%.4f f1_macro=%.4f",
                method,
                axis,
                f"{threshold:.3f}" if threshold == threshold else "-",
                metrics["accuracy"],
                metrics["f1_macro"],
            )

    if "iteration1" in args.methods:
        from src.model import FireCNN

        model = FireCNN(device=device)
        if load_state(model, CHECKPOINTS / "iteration1" / "best_model.pt", device):
            predicted = predict_classifier(model, image_paths, 224, device, binary=True)
            emit(
                "iteration1",
                "FireCNN",
                "binary classification",
                predicted,
                ["binary"],
                notes="cannot distinguish smoke; binary axis only",
            )
        del model
        torch.cuda.empty_cache()

    for method in ("iteration2", "iteration3"):
        if method not in args.methods:
            continue
        from src.model import MobileNetV3FireClassifier

        model = MobileNetV3FireClassifier(num_classes=4, pretrained=False, device=device)
        if load_state(model, CHECKPOINTS / method / "best_model.pt", device):
            predicted = predict_classifier(model, image_paths, 224, device, binary=False)
            emit(
                method,
                "MobileNetV3-Small" + (" robust" if method == "iteration3" else ""),
                "multiclass classification",
                predicted,
                ["binary", "multiclass"],
            )
        del model
        torch.cuda.empty_cache()

    if "iteration4" in args.methods:
        weights = CHECKPOINTS / "iteration4" / "yolo26-dfire" / "weights" / "best.pt"
        if weights.exists():
            thresholds = [0.05, 0.10, 0.25, 0.40, 0.50, 0.70] if args.sweep else [args.conf]
            for threshold in thresholds:
                predicted = predict_detector(weights, image_paths, device, threshold)
                emit(
                    "iteration4",
                    "YOLO26n",
                    "object detection",
                    predicted,
                    ["binary", "multiclass"],
                    threshold=threshold,
                    notes="boxes collapsed to image-level presence",
                )
        else:
            logger.warning("YOLO weights missing: %s", weights)

    if "iteration5" in args.methods:
        from src.model_segmentation import LightweightUNet

        model = LightweightUNet(num_classes=3, device=device)
        if load_state(model, CHECKPOINTS / "iteration5" / "best_model.pt", device):
            thresholds = (
                [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05] if args.sweep else [args.mask_area]
            )
            for threshold in thresholds:
                predicted = predict_segmentation(model, image_paths, 256, device, threshold)
                emit(
                    "iteration5",
                    "LightweightUNet",
                    "semantic segmentation",
                    predicted,
                    ["binary", "multiclass"],
                    threshold=threshold,
                    domain_shift=True,
                    notes="trained on Roboflow COCO, evaluated on D-Fire (domain shift)",
                )
        del model
        torch.cuda.empty_cache()

    if not rows:
        logger.error("No methods produced predictions.")
        return

    path = append_rows(args.output, rows, COMMON_FIELDS)
    record_run(
        "common_evaluation",
        {"rows": len(rows), "n_images": len(image_paths)},
        split=args.split,
        extra={"methods": args.methods, "sweep": args.sweep},
    )
    logger.info("Wrote %d rows to %s", len(rows), path)

    print(f"\n{'method':<12} {'paradigm':<26} {'axis':<11} {'thr':>6} "
          f"{'acc':>8} {'F1 macro':>9}")
    print("-" * 78)
    for row in rows:
        threshold = row["threshold"]
        threshold_text = f"{threshold:.3f}" if threshold == threshold else "-"
        print(
            f"{row['method']:<12} {row['paradigm']:<26} {row['axis']:<11} "
            f"{threshold_text:>6} {row['accuracy']:>8.4f} {row['f1_macro']:>9.4f}"
        )


if __name__ == "__main__":
    main()
