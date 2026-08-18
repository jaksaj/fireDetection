"""Static (calibrated) INT8 post-training quantization, and what it actually costs.

Why this exists
---------------
The project's original quantization step used
`torch.quantization.quantize_dynamic(model, {Linear, Conv2d})`, which was measured
converting **4 modules** of MobileNetV3-Small and **0** of the U-Net: PyTorch's
dynamic path covers Linear/RNN, not Conv2d, so the convolutional trunk holding
almost all the parameters stayed in FP32. The result was a no-op dressed as a
quantized model, and it is the basis of the retired "~1.1 MiB (INT8)" claim.

Static quantization is the real thing: it calibrates activation ranges on real
data and *does* quantize convolutions. This script

1. calibrates on a sample of the D-Fire **training** split (never test — the
   calibration set must not touch the data used to report accuracy),
2. writes an INT8 ONNX model per method,
3. reports the actual on-disk size change,
4. benchmarks INT8 vs FP32 latency on CPU under the standard protocol.

Accuracy of the quantized models is measured separately by
`scripts/evaluate_common.py --onnx-dir results/int8_models`, so the size/latency
win can be weighed against any accuracy loss rather than quoted alone.

Usage::

    python scripts/quantize_int8.py
    python scripts/quantize_int8.py --models iteration2 --calibration-images 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src.benchmark import BENCHMARK_FIELDS, benchmark_onnx_model
from src.dfire_labels import IMAGE_EXTENSIONS
from src.results import RESULTS_DIR, append_rows
from src.utils import configure_logging

logger = logging.getLogger("quantize_int8")

MODELS_DIR = PROJECT_ROOT / "jetson" / "models"
INT8_DIR = RESULTS_DIR / "int8_models"
DATA_DIR = PROJECT_ROOT / "data"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SPECS = {
    "iteration1": ("FireCNN (binary classification)", 224),
    "iteration2": ("MobileNetV3-Small (4-class)", 224),
    "iteration3": ("MobileNetV3-Small robust (4-class)", 224),
    "iteration4": ("YOLO26n (detection)", 640),
    "iteration5": ("LightweightUNet (segmentation)", 256),
}


def preprocess(path: Path, size: int) -> np.ndarray:
    """Match the eval transform used everywhere else: resize, scale, normalize."""
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


class CalibrationReader:
    """Feeds calibration batches to ONNX Runtime's static quantizer."""

    def __init__(self, image_paths: list[Path], size: int, input_name: str) -> None:
        self.image_paths = image_paths
        self.size = size
        self.input_name = input_name
        self._index = 0

    def get_next(self):  # noqa: D401 - interface required by onnxruntime
        if self._index >= len(self.image_paths):
            return None
        path = self.image_paths[self._index]
        self._index += 1
        if self._index % 50 == 0:
            logger.info("  calibrated on %d/%d images", self._index, len(self.image_paths))
        return {self.input_name: preprocess(path, self.size)}

    def rewind(self) -> None:
        self._index = 0


def calibration_images(count: int, seed: int = 42) -> list[Path]:
    """
    Sample from the TRAIN split.

    Calibrating on test data would leak the evaluation set into the model, and
    the resulting accuracy figures would be meaningless.
    """
    image_dir = DATA_DIR / "train" / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing {image_dir}")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(paths), size=min(count, len(paths)), replace=False)
    return [paths[i] for i in sorted(picked)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static INT8 PTQ via ONNX Runtime.")
    parser.add_argument("--models", nargs="+", default=sorted(SPECS), choices=sorted(SPECS))
    parser.add_argument("--calibration-images", type=int, default=200,
                        help="Images sampled from the train split for calibration.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--calibrate-method", default="MinMax",
                        choices=["MinMax", "Percentile", "Entropy"],
                        help="MinMax is fastest but sensitive to activation outliers, "
                             "which is exactly what breaks depthwise/hard-swish networks.")
    parser.add_argument("--suffix", default="_int8", help="Output filename suffix.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType
        from onnxruntime.quantization import quantize_static
        from onnxruntime.quantization.preprocess import quant_pre_process
    except ImportError as exc:
        logger.error("onnxruntime quantization tools unavailable: %s", exc)
        return

    INT8_DIR.mkdir(parents=True, exist_ok=True)
    images = calibration_images(args.calibration_images)
    logger.info("Calibration set: %d images from the train split", len(images))

    summary: list[dict] = []
    rows: list[dict] = []

    for model_key in args.models:
        name, size = SPECS[model_key]
        source = MODELS_DIR / f"{model_key}.onnx"
        if not source.exists():
            logger.warning("Missing %s — run scripts/export_for_jetson.py first.", source)
            continue

        logger.info("=== %s ===", model_key)
        prepared = INT8_DIR / f"{model_key}_prep.onnx"
        target = INT8_DIR / f"{model_key}{args.suffix}.onnx"

        try:
            # Shape inference + graph cleanup; the quantizer needs this to place
            # quantize/dequantize nodes correctly.
            quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
            input_name = ort.InferenceSession(
                str(prepared), providers=["CPUExecutionProvider"]
            ).get_inputs()[0].name

            reader = CalibrationReader(images, size, input_name)
            quantize_static(
                str(prepared), str(target), reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=getattr(CalibrationMethod, args.calibrate_method),
                per_channel=True,
            )
        except Exception as exc:  # noqa: BLE001 - one model must not stop the run
            logger.error("Quantization failed for %s: %s", model_key, exc)
            continue
        finally:
            prepared.unlink(missing_ok=True)

        fp32_mb = source.stat().st_size / (1024 * 1024)
        int8_mb = target.stat().st_size / (1024 * 1024)
        summary.append({
            "model": model_key, "fp32_mb": fp32_mb, "int8_mb": int8_mb,
            "ratio": fp32_mb / int8_mb if int8_mb else 0.0,
        })
        logger.info("  size %.2f MB -> %.2f MB (%.2fx smaller)",
                    fp32_mb, int8_mb, fp32_mb / int8_mb if int8_mb else 0)

        if args.skip_benchmark:
            continue

        result = benchmark_onnx_model(
            target, model_key=model_key, model_name=name,
            input_shape=(1, 3, size, size), providers=["CPUExecutionProvider"],
            bench_device="cpu", precision="int8-static",
            warmup=args.warmup, iters=args.iters,
            notes=f"static PTQ, QDQ, per-channel, {len(images)} calibration images from train",
        )
        if result is not None:
            rows.append(result.as_row())
            logger.info("  INT8 CPU latency: %.3f ms", result.latency_ms_median)

    if rows:
        append_rows("benchmarks.csv", rows, BENCHMARK_FIELDS)

    if not summary:
        logger.error("Nothing quantized.")
        return

    print(f"\n{'model':<12} {'FP32 MB':>9} {'INT8 MB':>9} {'shrink':>8}")
    print("-" * 42)
    for entry in summary:
        print(f"{entry['model']:<12} {entry['fp32_mb']:>9.2f} {entry['int8_mb']:>9.2f} "
              f"{entry['ratio']:>7.2f}x")
    print(f"\nINT8 models in {INT8_DIR}")
    print("Measure their accuracy with:")
    print("  python scripts/evaluate_common.py --onnx-dir results/int8_models --tag int8")


if __name__ == "__main__":
    main()
