"""Reconcile the repaired detector's 0.8050 macro-F1 against the informal
200-image confidence probe from the Q-035 follow-up (Q-036).

Why this exists
----------------
``results/q035_detector_confidence.csv`` reports the repaired (head-excluded)
INT8 detector puts a box above confidence 0.10 on 37 of 200 images -- 18.5%.
``results/common_eval_q035b.csv`` reports the same model's recall on the fire
class as 0.5399 at the same threshold 0.10. Naive division (37/200) suggests
recall near 0.18, an order of magnitude off from 0.54. Both numbers cannot
describe the same operating point on the same population -- unless they are
not the same population, which is exactly what this script checks.

The 200-image probe (written ad hoc during the Q-035 follow-up, never
committed as a script) selected the FIRST 200 test images whose ground truth
was NOT "Neither" -- i.e. fire OR smoke, in whatever order ``sorted(Path.iterdir())``
produced -- and counted a box of ANY class above 0.10. The common-task recall
figure is computed by ``evaluate_onnx_detseg.predict_detector_onnx`` over the
FULL 4306-image test split, and counts only boxes of the FIRE class specifically
against the ~1115 genuinely fire-positive images. Different sample (first 200
of an unknown mix of fire/smoke images vs. all fire-positive images), different
class filter (any box vs. fire-class box specifically) -- two different
statistics, not one measured twice.

This script settles it directly: it re-runs the exact production function
(``predict_detector_onnx``, unmodified) over the full test split for both the
FP32 control and the repaired INT8 model, at the recorded threshold (0.10),
and separately extracts the per-image FIRE-class confidence so the true
confusion counts and the true confidence distribution (over the actual
1115-image fire-positive population, not a 200-image non-random slice) can be
reported rather than inferred.

Nothing here changes evaluate_onnx_detseg.py's logic -- it only calls into it
and additionally records the box list Ultralytics already computed, which
predict_detector_onnx discards after checking presence.

Usage::

    python scripts/reconcile_q036_confusion.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

from src.results import RESULTS_DIR, append_rows
from src.utils import configure_logging

from evaluate_common import (  # noqa: E402
    MULTICLASS_BOTH,
    MULTICLASS_ONLY_FIRE,
    ground_truth_labels,
    list_test_images,
)

logger = logging.getLogger("reconcile_q036")

CONF_THRESHOLD = 0.10  # the recorded operating point for both rows being reconciled
IMG_SIZE = 640

PER_IMAGE_FIELDS = [
    "model", "image", "truth_fire", "pred_fire_at_010",
    "top_fire_conf", "top_any_conf",
]


def score_all(onnx_path: Path, image_paths: list[Path], label: str) -> list[dict]:
    """
    Run every test image through the SAME Ultralytics ONNX path
    ``predict_detector_onnx`` uses, but keep the per-image box list instead of
    collapsing it to a presence flag, so a real confidence distribution and
    exact confusion counts can be recovered.
    """
    from ultralytics import YOLO

    model = YOLO(str(onnx_path), task="detect")
    truth = ground_truth_labels(image_paths, "test")
    truth_fire = np.isin(truth, [MULTICLASS_ONLY_FIRE, MULTICLASS_BOTH])

    rows = []
    for index, path in enumerate(image_paths):
        result = model.predict(
            source=str(path), conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False
        )[0]
        fire_conf = 0.0
        any_conf = 0.0
        if result.boxes is not None and len(result.boxes):
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            any_conf = float(confs.max())
            fire_mask = classes == 1  # D-Fire: 0 = smoke, 1 = fire
            if fire_mask.any():
                fire_conf = float(confs[fire_mask].max())
        rows.append({
            "model": label,
            "image": path.name,
            "truth_fire": bool(truth_fire[index]),
            "pred_fire_at_010": bool(fire_conf > CONF_THRESHOLD),
            "top_fire_conf": fire_conf,
            "top_any_conf": any_conf,
        })
        if (index + 1) % 500 == 0:
            logger.info("  %s: %d/%d", label, index + 1, len(image_paths))
    return rows


def summarize(rows: list[dict], label: str) -> None:
    arr_truth = np.array([r["truth_fire"] for r in rows])
    arr_pred = np.array([r["pred_fire_at_010"] for r in rows])
    arr_fire_conf = np.array([r["top_fire_conf"] for r in rows])

    tp = int(np.sum(arr_truth & arr_pred))
    fn = int(np.sum(arr_truth & ~arr_pred))
    fp = int(np.sum(~arr_truth & arr_pred))
    tn = int(np.sum(~arr_truth & ~arr_pred))
    n_fire = int(arr_truth.sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1_fire = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    logger.info("=== %s ===", label)
    logger.info("n_fire_positive_images = %d (of %d total)", n_fire, len(rows))
    logger.info("TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)
    logger.info("precision_fire=%.10f  recall_fire=%.10f  f1_fire=%.10f",
                precision, recall, f1_fire)

    fire_pos_conf = arr_fire_conf[arr_truth]
    pct = np.percentile(fire_pos_conf, [0, 50, 95, 100])
    logger.info(
        "Fire-class top-box confidence over the %d ACTUAL fire-positive images:\n"
        "    min=%.6f  median=%.6f  mean=%.6f  p95=%.6f  max=%.6f",
        n_fire, pct[0], pct[1], fire_pos_conf.mean(), pct[2], pct[3],
    )
    logger.info(
        "Fraction of fire-positive images with fire-class conf > %.2f: %.4f (== recall)",
        CONF_THRESHOLD, float((fire_pos_conf > CONF_THRESHOLD).mean()),
    )


SENSITIVITY_FIELDS = [
    "model", "threshold", "tp", "fp", "fn", "tn",
    "precision_fire", "recall_fire", "f1_fire", "macro_f1",
]


def threshold_sensitivity_from_stored_confidence(rows: list[dict]) -> None:
    """
    Recompute precision/recall/macro-F1 at other thresholds from the per-image
    confidences already collected above -- no new model inference, since the
    top fire-class confidence per image was recorded regardless of whether it
    cleared 0.10. This is what settles whether 0.8050 sits on a knife-edge or
    on a plateau of the threshold curve.
    """
    out_rows = []
    for model in {r["model"] for r in rows}:
        sub = [r for r in rows if r["model"] == model]
        truth = np.array([r["truth_fire"] for r in sub])
        conf = np.array([r["top_fire_conf"] for r in sub])
        for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            pred = conf > thr
            tp = int((truth & pred).sum())
            fn = int((truth & ~pred).sum())
            fp = int((~truth & pred).sum())
            tn = int((~truth & ~pred).sum())
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1_fire = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
            tn_precision = tn / (tn + fn) if (tn + fn) else float("nan")
            tn_recall = tn / (tn + fp) if (tn + fp) else float("nan")
            f1_neg = 2 * tn_precision * tn_recall / (tn_precision + tn_recall) if (tn_precision + tn_recall) else float("nan")
            out_rows.append({
                "model": model, "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision_fire": precision, "recall_fire": recall,
                "f1_fire": f1_fire, "macro_f1": (f1_fire + f1_neg) / 2,
            })
    path = append_rows("q036_threshold_sensitivity.csv", out_rows, SENSITIVITY_FIELDS)
    logger.info("Wrote %d threshold-sensitivity rows to %s", len(out_rows), path)


def main() -> None:
    configure_logging()
    image_paths = list_test_images("test")
    logger.info("Test split: %d images", len(image_paths))

    targets = [
        (RESULTS_DIR / "int8_models" / "iteration4_ultra.onnx", "fp32-control"),
        (RESULTS_DIR / "int8_models" / "iteration4_ultra_int8headfp32.onnx", "int8-head-fp32"),
    ]

    all_rows: list[dict] = []
    for onnx_path, label in targets:
        if not onnx_path.exists():
            logger.error("Missing %s -- stopping rather than substituting.", onnx_path)
            return
        logger.info("Scoring %s (%s)...", onnx_path.name, label)
        rows = score_all(onnx_path, image_paths, label)
        all_rows.extend(rows)
        summarize(rows, label)

    path = append_rows("q036_detector_confusion.csv", all_rows, PER_IMAGE_FIELDS)
    logger.info("Wrote %d per-image rows to %s", len(all_rows), path)


if __name__ == "__main__":
    main()
