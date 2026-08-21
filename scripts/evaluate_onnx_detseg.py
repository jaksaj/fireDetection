"""Score quantized ONNX detector and segmenter artifacts on the common task.

Why this exists
---------------
``scripts/evaluate_common.py --onnx-dir`` deliberately covers the classifiers
only::

    if method in {"iteration4", "iteration5"}:
        logger.info("Skipping %s -- ONNX scoring covers the classifiers only.", method)

The detector and the segmenter need different treatment: the detector's ONNX
graph is an Ultralytics export whose input is bound by the name ``images`` and
which expects letterboxed input, and the segmenter needs its per-pixel output
collapsed by an area threshold. So the INT8 rows for those two methods in
``results/common_eval_int8_detseg.csv`` were produced by a throwaway script that
was never committed -- which meant the single strongest quantization claim in the
project had no reproducible provenance, and its operating point could not be
recovered from the repository at all. This script is that missing piece, written
to be run again.

Two properties matter and are enforced here:

1. **The threshold is always recorded.** The rows in the original CSV carry
   ``threshold = nan``, so there was no way to tell whether the FP32 control and
   the INT8 measurement were scored at the same operating point -- and a
   detector's macro-F1 moves by 0.17 across its plausible threshold range, which
   is a third of the quantization drop being claimed.
2. **Detection postprocessing is Ultralytics' own.** Loading the ONNX through
   ``YOLO(path)`` reuses the same letterboxing and decoding as the PyTorch path
   in ``evaluate_common.predict_detector``, so an FP32 ONNX control isolates
   quantization from the export and evaluation path.

Usage::

    python scripts/evaluate_onnx_detseg.py --onnx results/int8_models/iteration4_ultra_int8.onnx \
        --method iteration4 --conf 0.10 --tag int8-minmax --output common_eval_int8_repro.csv
    python scripts/evaluate_onnx_detseg.py --onnx jetson/models/iteration5.onnx \
        --method iteration5 --mask-area 0.02 --tag fp32 --output common_eval_int8_repro.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
from PIL import Image

from src.results import append_rows
from src.utils import configure_logging

from evaluate_common import (  # noqa: E402
    COMMON_FIELDS,
    SEG_CLASS_FIRE,
    SEG_CLASS_SMOKE,
    build_eval_transform,
    ground_truth_labels,
    list_test_images,
    presence_to_multiclass,
    score,
)

logger = logging.getLogger("evaluate_onnx_detseg")


def predict_detector_onnx(
    onnx_path: Path, image_paths: list[Path], conf_threshold: float, imgsz: int = 640
) -> np.ndarray:
    """
    Collapse ONNX detector boxes to image-level presence.

    Deliberately routed through Ultralytics rather than a hand-rolled decoder.
    An earlier hand-rolled version -- plain resize, no letterbox -- produced a
    maximum confidence of 0.0034 across the whole split, i.e. no detections at
    all, and would have been indistinguishable from a catastrophic quantization
    failure. Using the same postprocessing as the PyTorch path is what makes the
    FP32 control meaningful.
    """
    from ultralytics import YOLO

    model = YOLO(str(onnx_path), task="detect")
    predictions: list[int] = []

    # Batch 1: the exports carry a static batch dimension.
    for path in image_paths:
        result = model.predict(
            source=str(path), conf=conf_threshold, imgsz=imgsz, verbose=False
        )[0]
        classes: set[int] = set()
        if result.boxes is not None and len(result.boxes):
            classes = {int(value) for value in result.boxes.cls.cpu().numpy()}
        # D-Fire detection classes: 0 = smoke, 1 = fire.
        predictions.append(presence_to_multiclass(1 in classes, 0 in classes))

    return np.array(predictions, dtype=np.int64)


def predict_segmentation_onnx(
    onnx_path: Path, image_paths: list[Path], area_threshold: float, image_size: int = 256
) -> np.ndarray:
    """Mirror ``evaluate_common.predict_segmentation`` against an ONNX graph."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    transform = build_eval_transform(image_size)
    predictions: list[int] = []

    for path in image_paths:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).numpy()[None, ...]
        logits = session.run(None, {input_name: tensor})[0]
        mask = logits.argmax(axis=1)[0]

        total = mask.size
        smoke = (mask == SEG_CLASS_SMOKE).sum() / total
        fire = (mask == SEG_CLASS_FIRE).sum() / total
        predictions.append(
            presence_to_multiclass(fire > area_threshold, smoke > area_threshold)
        )

    return np.array(predictions, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ONNX detector/segmenter artifacts.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=["iteration4", "iteration5"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Detector confidence threshold. Defaults to the project's common-task "
             "operating point (results/tables/threshold_sensitivity.md), NOT to "
             "evaluate_common.py's 0.25, so a matched FP32/INT8 pair is the default "
             "rather than something the caller has to remember.",
    )
    parser.add_argument(
        "--mask-area", type=float, default=0.02,
        help="Segmentation area threshold; the project's operating point.",
    )
    parser.add_argument("--tag", default="", help="Label recorded in the notes column.")
    parser.add_argument("--output", default="common_eval_onnx_detseg.csv")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if not args.onnx.exists():
        logger.error("Missing %s", args.onnx)
        raise SystemExit(1)

    image_paths = list_test_images(args.split)
    if args.limit:
        image_paths = image_paths[: args.limit]
    truth = ground_truth_labels(image_paths, args.split)
    logger.info("Scoring %s on %d %s images", args.onnx.name, len(image_paths), args.split)

    if args.method == "iteration4":
        threshold = args.conf
        predicted = predict_detector_onnx(args.onnx, image_paths, threshold)
        paradigm, axes, domain_shift = "quantized detection", ["binary", "multiclass"], 0
    else:
        threshold = args.mask_area
        predicted = predict_segmentation_onnx(args.onnx, image_paths, threshold)
        paradigm, axes, domain_shift = "quantized segmentation", ["binary", "multiclass"], 1

    label = args.tag or "onnx"
    rows = []
    for axis in axes:
        metrics = score(truth, predicted, axis)
        rows.append({
            "method": args.method,
            "model_name": f"{args.method} ({label})",
            "paradigm": paradigm,
            "axis": axis,
            "threshold": threshold,
            "n_images": len(image_paths),
            "domain_shift": domain_shift,
            "seed": "",
            "notes": f"ONNX {label}: {args.onnx.name}",
            **metrics,
        })
        logger.info(
            "%-12s %-11s thr=%-7.4f acc=%.4f f1_macro=%.4f",
            args.method, axis, threshold, metrics["accuracy"], metrics["f1_macro"],
        )

    path = append_rows(args.output, rows, COMMON_FIELDS)
    logger.info("Wrote %d rows to %s", len(rows), path)


if __name__ == "__main__":
    main()
