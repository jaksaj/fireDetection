"""Measure accuracy under image corruption, for the robustness claim.

Evaluates any classification checkpoint on the D-Fire test split under the fixed
corruption suite in ``src/corruptions.py``, at every severity, and writes one
row per (method, corruption, severity) to ``results/robustness.csv``.

The intended comparison is iteration 2 (standard augmentation) against
iteration 3 (Albumentations robustness pipeline). Both share an architecture and
a parameter count, so they have *identical inference cost*. If iteration 3 holds
accuracy better under degradation, that is a free improvement at deployment
time -- which is a genuinely useful result for an edge thesis. If it does not,
the project's robustness claim is false and should be retracted. Either outcome
is worth having; the current state, where nobody knows, is not.

Results are grouped by whether the corruption resembles something iteration 3
trained on (``fog``, brightness/contrast shifts) or not (motion blur, JPEG,
noise). Only the second group is an honest generalization test.

Usage::

    python scripts/evaluate_robustness.py
    python scripts/evaluate_robustness.py --methods iteration2 iteration3 --limit 500
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
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.corruptions import CORRUPTIONS, SEVERITIES, apply_corruption, corruption_group
from src.dfire_labels import MULTICLASS_CLASS_NAMES
from src.results import append_rows
from src.utils import configure_logging, resolve_device

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_common import (  # noqa: E402
    build_eval_transform,
    ground_truth_labels,
    list_test_images,
    load_state,
)

logger = logging.getLogger("evaluate_robustness")

CHECKPOINTS = PROJECT_ROOT / "checkpoints"

ROBUSTNESS_FIELDS = [
    "method",
    "model_name",
    "corruption",
    "group",
    "severity",
    "n_images",
    "accuracy",
    "f1_macro",
    "f1_Neither",
    "f1_Only_Fire",
    "f1_Only_Smoke",
    "f1_Both",
    "accuracy_drop",
    "relative_drop_pct",
]


@torch.no_grad()
def predict_corrupted(
    model: torch.nn.Module,
    image_paths: list[Path],
    image_size: int,
    device: torch.device,
    corruption: str,
    severity: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict 4-class labels with a corruption applied before preprocessing."""
    transform = build_eval_transform(image_size)
    model = model.to(device).eval()
    predictions: list[int] = []

    for start in range(0, len(image_paths), batch_size):
        tensors = []
        for path in image_paths[start : start + batch_size]:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                # Corruption is applied to the raw image, before resize and
                # normalization, so a given severity means the same physical
                # degradation regardless of the model's input resolution.
                corrupted = apply_corruption(image, corruption, severity)
                tensors.append(transform(corrupted))
        batch = torch.stack(tensors).to(device)
        predictions.extend(model(batch).argmax(dim=1).cpu().numpy().tolist())

    return np.array(predictions, dtype=np.int64)


def build_model(method: str, device: torch.device) -> torch.nn.Module | None:
    """Instantiate and load a 4-class classification checkpoint."""
    from src.model import MobileNetV3FireClassifier

    model = MobileNetV3FireClassifier(num_classes=4, pretrained=False, device=device)
    if not load_state(model, CHECKPOINTS / method / "best_model.pt", device):
        return None
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness under corruption.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["iteration2", "iteration3"],
        help="4-class classification methods to compare.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=sorted(CORRUPTIONS),
        choices=sorted(CORRUPTIONS),
    )
    parser.add_argument("--output", default="robustness.csv")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    image_paths = list_test_images(args.split)
    if args.limit:
        image_paths = image_paths[: args.limit]
    truth = ground_truth_labels(image_paths, args.split)

    labels = list(range(len(MULTICLASS_CLASS_NAMES)))
    rows: list[dict] = []

    for method in args.methods:
        model = build_model(method, device)
        if model is None:
            logger.warning("Skipping %s — no checkpoint.", method)
            continue

        model_name = "MobileNetV3-Small" + (" robust" if method == "iteration3" else "")

        # Clean baseline first: every drop is expressed relative to this, so the
        # comparison is about degradation rather than absolute starting point.
        clean_predictions = predict_corrupted(model, image_paths, 224, device, "none", 1)
        clean_accuracy = accuracy_score(truth, clean_predictions)
        logger.info("%s clean accuracy: %.4f", method, clean_accuracy)

        combinations = [("none", 0)] + [
            (name, severity) for name in args.corruptions for severity in SEVERITIES
        ]

        for corruption, severity in combinations:
            if corruption == "none":
                predictions = clean_predictions
            else:
                predictions = predict_corrupted(
                    model, image_paths, 224, device, corruption, severity
                )

            accuracy = accuracy_score(truth, predictions)
            per_class = f1_score(
                truth, predictions, labels=labels, average=None, zero_division=0
            )
            row = {
                "method": method,
                "model_name": model_name,
                "corruption": corruption,
                "group": corruption_group(corruption),
                "severity": severity,
                "n_images": len(image_paths),
                "accuracy": accuracy,
                "f1_macro": f1_score(
                    truth, predictions, labels=labels, average="macro", zero_division=0
                ),
                "accuracy_drop": clean_accuracy - accuracy,
                "relative_drop_pct": (
                    100.0 * (clean_accuracy - accuracy) / clean_accuracy
                    if clean_accuracy
                    else 0.0
                ),
            }
            for name, value in zip(MULTICLASS_CLASS_NAMES, per_class):
                row[f"f1_{name}"] = float(value)
            rows.append(row)

            logger.info(
                "%-12s %-16s sev=%d acc=%.4f (drop %.4f, %.1f%%)",
                method,
                corruption,
                severity,
                accuracy,
                row["accuracy_drop"],
                row["relative_drop_pct"],
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        logger.error("No results produced.")
        return

    path = append_rows(args.output, rows, ROBUSTNESS_FIELDS)
    logger.info("Wrote %d rows to %s", len(rows), path)

    # Headline comparison: mean accuracy per method, split by whether the
    # corruption resembles something iteration 3 trained on.
    print(f"\n{'method':<12} {'group':<20} {'mean acc':>10} {'mean drop':>11}")
    print("-" * 56)
    for method in args.methods:
        for group in ("clean", "seen_in_training", "unseen_in_training"):
            subset = [r for r in rows if r["method"] == method and r["group"] == group]
            if not subset:
                continue
            mean_accuracy = sum(r["accuracy"] for r in subset) / len(subset)
            mean_drop = sum(r["accuracy_drop"] for r in subset) / len(subset)
            print(f"{method:<12} {group:<20} {mean_accuracy:>10.4f} {mean_drop:>11.4f}")


if __name__ == "__main__":
    main()
