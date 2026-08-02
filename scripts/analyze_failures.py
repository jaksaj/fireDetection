"""Export the worst failure cases per class, for the Discussion chapter.

``scripts/visualize_predictions.py`` samples images at random, which shows what
a model typically does but never what it gets wrong. A Discussion chapter needs
the opposite: the confident mistakes, ranked, with enough context to say *why*
they happen.

This script ranks every test image by the model's loss on it, then exports:

- ``results/failures/<method>/worst_<class>_NN.png`` -- the highest-loss images
  per true class, annotated with true label, prediction, and confidence.
- ``results/failures/<method>/confusion_<true>_as_<pred>.png`` -- montages of
  the most common confusion pairs, which is where the interesting errors are.
- ``results/failures/<method>_failures.csv`` -- every misclassified image with
  its loss and predicted distribution, so claims about failure modes can be
  counted rather than eyeballed.

The confident-and-wrong cases (high loss, high confidence) are the ones worth
discussing: they indicate a systematic representation problem rather than
ordinary boundary noise.

Usage::

    python scripts/analyze_failures.py --method iteration3
    python scripts/analyze_failures.py --method iteration2 --top-k 24
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dfire_labels import MULTICLASS_CLASS_NAMES
from src.results import RESULTS_DIR, append_rows
from src.utils import configure_logging, resolve_device

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_common import (  # noqa: E402
    build_eval_transform,
    ground_truth_labels,
    list_test_images,
    load_state,
)

logger = logging.getLogger("analyze_failures")

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
FAILURES_DIR = RESULTS_DIR / "failures"

FAILURE_FIELDS = [
    "method",
    "filename",
    "true_label",
    "pred_label",
    "loss",
    "confidence",
    "correct",
    "p_Neither",
    "p_Only_Fire",
    "p_Only_Smoke",
    "p_Both",
]


@torch.no_grad()
def score_all(
    model: torch.nn.Module,
    image_paths: list[Path],
    truth: np.ndarray,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (per-image loss, predicted label, full probability matrix)."""
    transform = build_eval_transform(image_size)
    model = model.to(device).eval()

    losses: list[float] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        tensors = []
        for path in chunk:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))

        batch = torch.stack(tensors).to(device)
        labels = torch.from_numpy(truth[start : start + len(chunk)]).to(device)

        logits = model(batch)
        # reduction="none" gives the per-image loss needed for ranking; a mean
        # would collapse exactly the signal this script exists to surface.
        batch_losses = F.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=1)

        losses.extend(batch_losses.cpu().numpy().tolist())
        predictions.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        probabilities.append(probs.cpu().numpy())

    return (
        np.array(losses),
        np.array(predictions, dtype=np.int64),
        np.concatenate(probabilities, axis=0),
    )


def montage(
    image_paths: list[Path],
    titles: list[str],
    output_path: Path,
    suptitle: str,
    columns: int = 4,
) -> None:
    if not image_paths:
        return
    rows = (len(image_paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows), squeeze=False)

    for index, (path, title) in enumerate(zip(image_paths, titles)):
        axis = axes[index // columns][index % columns]
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=8)
        axis.axis("off")

    for index in range(len(image_paths), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ranked failure cases.")
    parser.add_argument(
        "--method",
        default="iteration3",
        choices=["iteration2", "iteration3"],
        help="4-class classification checkpoint to analyse.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12, help="Images per montage.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    from src.model import MobileNetV3FireClassifier

    model = MobileNetV3FireClassifier(num_classes=4, pretrained=False, device=device)
    if not load_state(model, CHECKPOINTS / args.method / "best_model.pt", device):
        logger.error("No checkpoint for %s", args.method)
        return

    image_paths = list_test_images(args.split)
    if args.limit:
        image_paths = image_paths[: args.limit]
    truth = ground_truth_labels(image_paths, args.split)

    logger.info("Scoring %d images with %s", len(image_paths), args.method)
    losses, predictions, probabilities = score_all(model, image_paths, truth, device)

    correct = predictions == truth
    logger.info("Accuracy %.4f — %d errors", correct.mean(), int((~correct).sum()))

    output_dir = FAILURES_DIR / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, path in enumerate(image_paths):
        row = {
            "method": args.method,
            "filename": path.name,
            "true_label": MULTICLASS_CLASS_NAMES[truth[index]],
            "pred_label": MULTICLASS_CLASS_NAMES[predictions[index]],
            "loss": float(losses[index]),
            "confidence": float(probabilities[index].max()),
            "correct": int(correct[index]),
        }
        for class_index, name in enumerate(MULTICLASS_CLASS_NAMES):
            row[f"p_{name}"] = float(probabilities[index][class_index])
        rows.append(row)

    csv_path = RESULTS_DIR / f"{args.method}_failures.csv"
    csv_path.unlink(missing_ok=True)
    append_rows(csv_path.name, rows, FAILURE_FIELDS)

    # Worst failures per true class.
    for class_index, class_name in enumerate(MULTICLASS_CLASS_NAMES):
        mask = (truth == class_index) & (~correct)
        candidates = np.where(mask)[0]
        if not len(candidates):
            continue
        ranked = candidates[np.argsort(-losses[candidates])][: args.top_k]
        montage(
            [image_paths[i] for i in ranked],
            [
                f"pred {MULTICLASS_CLASS_NAMES[predictions[i]]}\n"
                f"conf {probabilities[i].max():.2f} loss {losses[i]:.2f}"
                for i in ranked
            ],
            output_dir / f"worst_{class_name}.png",
            f"{args.method}: highest-loss errors on true class '{class_name}' "
            f"({int(mask.sum())} errors of {int((truth == class_index).sum())} images)",
        )

    # Most frequent confusion pairs.
    pairs: dict[tuple[int, int], list[int]] = {}
    for index in np.where(~correct)[0]:
        pairs.setdefault((int(truth[index]), int(predictions[index])), []).append(int(index))

    for (true_index, pred_index), indices in sorted(
        pairs.items(), key=lambda item: -len(item[1])
    )[:4]:
        ranked = sorted(indices, key=lambda i: -probabilities[i].max())[: args.top_k]
        true_name = MULTICLASS_CLASS_NAMES[true_index]
        pred_name = MULTICLASS_CLASS_NAMES[pred_index]
        montage(
            [image_paths[i] for i in ranked],
            [f"conf {probabilities[i].max():.2f}" for i in ranked],
            output_dir / f"confusion_{true_name}_as_{pred_name}.png",
            f"{args.method}: '{true_name}' predicted as '{pred_name}' "
            f"({len(indices)} cases) — most confident first",
        )

    print(f"\nTop confusion pairs for {args.method}:")
    print(f"{'true':<12} -> {'predicted':<12} {'count':>6} {'mean conf':>10}")
    print("-" * 46)
    for (true_index, pred_index), indices in sorted(
        pairs.items(), key=lambda item: -len(item[1])
    )[:8]:
        mean_confidence = float(np.mean([probabilities[i].max() for i in indices]))
        print(
            f"{MULTICLASS_CLASS_NAMES[true_index]:<12} -> "
            f"{MULTICLASS_CLASS_NAMES[pred_index]:<12} {len(indices):>6} {mean_confidence:>10.3f}"
        )

    logger.info("Failure artifacts -> %s", output_dir)


if __name__ == "__main__":
    main()
