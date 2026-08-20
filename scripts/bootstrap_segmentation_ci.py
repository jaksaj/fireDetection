"""Bootstrap a confidence interval for the segmentation mIoU over its 20 test images.

Why this exists
---------------
The Roboflow COCO test split holds **20 images** (`results/dataset_stats.json` ->
`coco_segmentation`), so one image is 5% of every segmentation metric this
project reports. The interval quoted elsewhere -- mIoU 0.8547 +/- 0.0119 -- is
the **standard deviation across training seeds**. It answers "how much does
retraining move the number?" It says nothing about "how much would the number
move on a different 20 images?", which on a set this small is the larger
question and was previously unquantified.

This script answers the second question by resampling the test images with
replacement and recomputing the metric on each resample.

Two things to keep straight, and the output labels them:

* **Seed interval** -- variability of the training process, n = 5 checkpoints.
* **Sampling interval** -- variability of the evaluation set, from bootstrapping
  the 20 images. This is the one that belongs next to a claim about the true
  mIoU of the method.

They measure different things and must not be combined or substituted.

IoU is accumulated dataset-wide (summing intersections and unions across images)
rather than averaged per image, matching how `SegmentationTrainer` reports it, so
each bootstrap replicate re-sums the per-image counts of the resampled images.

Usage::

    python scripts/bootstrap_segmentation_ci.py
    python scripts/bootstrap_segmentation_ci.py --seeds 42 43 44 45 46 --resamples 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.augmentations import build_robust_eval_transforms
from src.dataset_segmentation import COCOSegmentationDataset
from src.results import RESULTS_DIR, append_rows
from src.utils import configure_logging, resolve_device

logger = logging.getLogger("bootstrap_segmentation_ci")

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
COCO_TEST = PROJECT_ROOT / "data" / "coco" / "test"
CLASS_NAMES = ("background", "smoke", "fire")

BOOTSTRAP_FIELDS = [
    "seed", "metric", "point_estimate",
    "ci_low", "ci_high", "ci_width",
    "n_images", "n_resamples", "interval_type", "checkpoint",
]


@torch.no_grad()
def per_image_counts(
    model: torch.nn.Module, dataset, device: torch.device, num_classes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-image intersection and union counts, shape (n_images, num_classes).

    Kept per image rather than pre-aggregated so a bootstrap replicate can re-sum
    whichever images it drew.
    """
    model = model.to(device).eval()
    intersections = np.zeros((len(dataset), num_classes), dtype=np.float64)
    unions = np.zeros((len(dataset), num_classes), dtype=np.float64)

    for index in range(len(dataset)):
        image, mask = dataset[index]
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
        target = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        for klass in range(num_classes):
            pred_k = prediction == klass
            true_k = target == klass
            intersections[index, klass] = np.logical_and(pred_k, true_k).sum()
            unions[index, klass] = np.logical_or(pred_k, true_k).sum()

    return intersections, unions


def miou_from(intersections: np.ndarray, unions: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Dataset-level mIoU over all classes, and over hazard classes only."""
    per_class = intersections.sum(axis=0) / (unions.sum(axis=0) + eps)
    return float(per_class.mean()), float(per_class[1:].mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CI for segmentation mIoU.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="segmentation_bootstrap.csv")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    device = resolve_device(args.device)

    if not COCO_TEST.exists():
        logger.error("Missing %s", COCO_TEST)
        return

    dataset = COCOSegmentationDataset(
        split_dir=COCO_TEST, transform=build_robust_eval_transforms(256)
    )
    n_images = len(dataset)
    logger.info("Segmentation test split: %d images", n_images)

    from src.model_segmentation import LightweightUNet

    rows: list[dict] = []
    point_estimates: dict[str, list[float]] = {"mIoU": [], "mIoU_hazard_only": []}
    rng = np.random.default_rng(42)

    for seed in args.seeds:
        checkpoint = CHECKPOINTS / f"iteration5-seed{seed}" / "best_model.pt"
        if not checkpoint.exists():
            logger.warning("Missing %s", checkpoint)
            continue

        model = LightweightUNet(num_classes=3, device=device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload))

        inter, union = per_image_counts(model, dataset, device)
        miou, hazard = miou_from(inter, union)
        point_estimates["mIoU"].append(miou)
        point_estimates["mIoU_hazard_only"].append(hazard)

        # Resample images with replacement; recompute the dataset-level metric.
        draws = rng.integers(0, n_images, size=(args.resamples, n_images))
        boot_miou = np.empty(args.resamples)
        boot_hazard = np.empty(args.resamples)
        for r in range(args.resamples):
            idx = draws[r]
            boot_miou[r], boot_hazard[r] = miou_from(inter[idx], union[idx])

        for name, values, point in (
            ("mIoU", boot_miou, miou),
            ("mIoU_hazard_only", boot_hazard, hazard),
        ):
            low, high = np.percentile(values, [100 * args.alpha / 2, 100 * (1 - args.alpha / 2)])
            rows.append({
                "seed": seed, "metric": name, "point_estimate": point,
                "ci_low": float(low), "ci_high": float(high),
                "ci_width": float(high - low),
                "n_images": n_images, "n_resamples": args.resamples,
                "interval_type": "bootstrap over test images (sampling error)",
                "checkpoint": str(checkpoint),
            })
        logger.info("seed %d: mIoU %.4f  95%% CI [%.4f, %.4f]",
                    seed, miou, rows[-2]["ci_low"], rows[-2]["ci_high"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        logger.error("No checkpoints evaluated.")
        return

    append_rows(args.output, rows, BOOTSTRAP_FIELDS)

    print(f"\nSegmentation test split: {n_images} images, {args.resamples} bootstrap resamples")
    print(f"{'metric':<20}{'seed mean':>11}{'seed std':>10}{'mean 95% CI':>26}{'CI width':>10}")
    print("-" * 80)
    for name in ("mIoU", "mIoU_hazard_only"):
        subset = [r for r in rows if r["metric"] == name]
        points = np.array(point_estimates[name])
        low = np.mean([r["ci_low"] for r in subset])
        high = np.mean([r["ci_high"] for r in subset])
        print(f"{name:<20}{points.mean():>11.4f}{points.std(ddof=1):>10.4f}"
              f"{f'[{low:.4f}, {high:.4f}]':>26}{high - low:>10.4f}")

    print("\nThe seed std and the bootstrap CI measure different things:")
    print("  seed std     -- how much retraining moves the number (n = %d checkpoints)"
          % len(point_estimates["mIoU"]))
    print("  bootstrap CI -- how much a different %d-image test set would move it" % n_images)
    print("Do not substitute one for the other, and do not combine them.")


if __name__ == "__main__":
    main()
