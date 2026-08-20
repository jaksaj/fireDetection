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
    checkpoint_for,
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
    "seed",
    "checkpoint",
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



@torch.no_grad()
def predict_corrupted_multi(
    models: list[tuple[str, int | None, torch.nn.Module, Path]],
    image_paths: list[Path],
    image_size: int,
    device: torch.device,
    corruption: str,
    severity: int,
    batch_size: int = 64,
) -> dict[tuple[str, int | None], np.ndarray]:
    """
    Corrupt each batch ONCE and evaluate every model on it.

    Corruptions are deterministic functions of (image, severity), so the
    corrupted pixels are identical for every model and every seed. Corrupting
    per model -- as the single-model path does -- repeats the CPU-bound work
    once per checkpoint. With 2 methods x 5 seeds that is 10x the necessary
    corruption cost, and corruption, not inference, is the bottleneck here.

    Results are bit-identical to evaluating each model separately.
    """
    transform = build_eval_transform(image_size)
    for _, _, model, _ in models:
        model.eval()

    outputs: dict[tuple[str, int | None], list[int]] = {
        (method, seed): [] for method, seed, _, _ in models
    }

    for start in range(0, len(image_paths), batch_size):
        tensors = []
        for path in image_paths[start : start + batch_size]:
            with Image.open(path) as handle:
                corrupted = apply_corruption(handle.convert("RGB"), corruption, severity)
                tensors.append(transform(corrupted))
        batch = torch.stack(tensors).to(device)

        for method, seed, model, _ in models:
            preds = model(batch).argmax(dim=1).cpu().numpy().tolist()
            outputs[(method, seed)].extend(preds)

    return {key: np.array(value, dtype=np.int64) for key, value in outputs.items()}


def build_model(
    method: str, device: torch.device, seed: int | None = None
) -> tuple[torch.nn.Module | None, Path]:
    """
    Instantiate and load a 4-class classification checkpoint.

    Returns the checkpoint path as well as the model. Without a seed this loads
    the ORIGINAL single-run checkpoint; iteration 1's original run is a ~5 sigma
    outlier that does not replicate, and the same class of defect put an
    unreplicated checkpoint under the common-task comparison before it was
    caught. The path is written into every output row so the basis of a result
    is never in doubt.
    """
    from src.model import MobileNetV3FireClassifier

    path = checkpoint_for(method, seed)
    model = MobileNetV3FireClassifier(num_classes=4, pretrained=False, device=device)
    if not load_state(model, path, device):
        return None, path
    return model, path


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
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Evaluate several seeded checkpoints in one pass, sharing the "
             "corruption work between them (identical results, ~n-times faster).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Evaluate the checkpoint from this seeded run. Omit to use the "
             "original single-run checkpoint.",
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

    seeds = args.seeds if args.seeds else [args.seed]

    # Load every (method, seed) checkpoint up front so one corruption pass can
    # serve all of them.
    models: list[tuple[str, int | None, torch.nn.Module, Path]] = []
    for method in args.methods:
        for seed in seeds:
            model, checkpoint_path = build_model(method, device, seed)
            if model is None:
                logger.warning("Skipping %s seed=%s — no checkpoint at %s",
                               method, seed, checkpoint_path)
                continue
            models.append((method, seed, model, checkpoint_path))

    if not models:
        logger.error("No checkpoints loaded.")
        return
    logger.info("Loaded %d checkpoints; corruption work is shared between them.", len(models))

    def model_name_for(method: str) -> str:
        return "MobileNetV3-Small" + (" robust" if method == "iteration3" else "")

    # Clean baseline per model; every drop is relative to its own clean score.
    clean = predict_corrupted_multi(models, image_paths, 224, device, "none", 1)
    clean_accuracy = {
        key: accuracy_score(truth, preds) for key, preds in clean.items()
    }
    for (method, seed), value in sorted(clean_accuracy.items(), key=lambda kv: str(kv[0])):
        logger.info("%s seed=%s clean accuracy: %.4f", method, seed, value)

    combinations = [("none", 0)] + [
        (name, severity) for name in args.corruptions for severity in SEVERITIES
    ]

    for corruption, severity in combinations:
        if corruption == "none":
            predictions = clean
        else:
            predictions = predict_corrupted_multi(
                models, image_paths, 224, device, corruption, severity
            )

        for method, seed, _, checkpoint_path in models:
            preds = predictions[(method, seed)]
            accuracy = accuracy_score(truth, preds)
            per_class = f1_score(truth, preds, labels=labels, average=None, zero_division=0)
            base = clean_accuracy[(method, seed)]
            row = {
                "method": method,
                "model_name": model_name_for(method),
                "corruption": corruption,
                "group": corruption_group(corruption),
                "severity": severity,
                "n_images": len(image_paths),
                "accuracy": accuracy,
                "f1_macro": f1_score(
                    truth, preds, labels=labels, average="macro", zero_division=0
                ),
                "accuracy_drop": base - accuracy,
                "relative_drop_pct": (100.0 * (base - accuracy) / base) if base else 0.0,
                "seed": seed if seed is not None else "",
                "checkpoint": str(checkpoint_path),
            }
            for name, value in zip(MULTICLASS_CLASS_NAMES, per_class):
                row[f"f1_{name}"] = float(value)
            rows.append(row)

        logger.info("%-16s sev=%d done (%d models)", corruption, severity, len(models))

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
