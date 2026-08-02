"""Measure inference cost for every model in the project, on this host.

Runs the uniform protocol in ``src/benchmark.py`` across every
(model x device x precision x backend) combination this machine supports and
appends the rows to ``results/benchmarks.csv``.

The same script is meant to be run unchanged on the workstation and on a Jetson
Orin Nano; ``available_configurations()`` reports what the current host can
measure, and rows are tagged with a ``bench_device`` that distinguishes them.

Examples::

    python scripts/run_benchmarks.py                    # everything this host supports
    python scripts/run_benchmarks.py --models iteration1 iteration5
    python scripts/run_benchmarks.py --batch-sizes 1 4 8 16
    python scripts/run_benchmarks.py --quick            # smoke test, few iterations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.benchmark import (
    BENCHMARK_FIELDS,
    available_configurations,
    benchmark_onnx_model,
    benchmark_torch_model,
)
from src.results import append_rows
from src.utils import configure_logging

logger = logging.getLogger("run_benchmarks")

CHECKPOINTS = PROJECT_ROOT / "checkpoints"

#: Every model the thesis compares, with the input resolution it was trained at.
#: Resolution differs by design (224 for classification, 640 for detection, 256
#: for segmentation) and is recorded per row, because latency is meaningless
#: without it -- a detector is not "slower" than a classifier when it is also
#: processing 8x the pixels. The Results chapter reports both the native-
#: resolution number and a resolution-matched comparison.
MODEL_SPECS: dict[str, dict] = {
    "iteration1": {
        "name": "FireCNN (binary classification)",
        "builder": "firecnn",
        "input": (3, 224, 224),
        "checkpoint": CHECKPOINTS / "iteration1" / "best_model.pt",
    },
    "iteration2": {
        "name": "MobileNetV3-Small (4-class)",
        "builder": "mobilenet",
        "input": (3, 224, 224),
        "checkpoint": CHECKPOINTS / "iteration2" / "best_model.pt",
    },
    "iteration3": {
        "name": "MobileNetV3-Small robust (4-class)",
        "builder": "mobilenet",
        "input": (3, 224, 224),
        "checkpoint": CHECKPOINTS / "iteration3" / "best_model.pt",
    },
    "iteration4": {
        "name": "YOLO26n (detection)",
        "builder": "yolo",
        "input": (3, 640, 640),
        "checkpoint": CHECKPOINTS / "iteration4" / "yolo26-dfire" / "weights" / "best.pt",
        "onnx": CHECKPOINTS / "iteration4" / "yolo26-dfire" / "weights" / "best.onnx",
    },
    "iteration5": {
        "name": "LightweightUNet (segmentation)",
        "builder": "unet",
        "input": (3, 256, 256),
        "checkpoint": CHECKPOINTS / "iteration5" / "best_model.pt",
    },
}


def build_model(builder: str, device: str) -> torch.nn.Module:
    """Instantiate a model on ``device`` with its trained weights when present."""
    if builder == "firecnn":
        from src.model import FireCNN

        return FireCNN(device=device)
    if builder == "mobilenet":
        from src.model import MobileNetV3FireClassifier

        return MobileNetV3FireClassifier(num_classes=4, pretrained=False, device=device)
    if builder == "unet":
        from src.model_segmentation import LightweightUNet

        return LightweightUNet(num_classes=3, device=device)
    if builder == "yolo":
        from ultralytics import YOLO

        spec = MODEL_SPECS["iteration4"]
        yolo = YOLO(str(spec["checkpoint"]))
        # Benchmark the bare nn.Module. Ultralytics' predict() wraps the forward
        # pass in file loading, letterboxing, NMS and Results construction; that
        # end-to-end path is a different measurement and is reported separately.
        module = yolo.model.float().eval()
        return module.to(device)

    raise ValueError(f"Unknown builder {builder!r}")


def load_weights_if_available(model: torch.nn.Module, path: Path, device: str) -> bool:
    """Load a project checkpoint if it exists. Latency does not depend on the
    weight *values*, but loading them keeps the benchmarked artifact identical
    to the evaluated one, which removes a class of "did you measure the same
    model?" questions."""
    if not path.exists():
        logger.warning("No checkpoint at %s — benchmarking randomly initialised weights.", path)
        return False
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
        return True
    except Exception as exc:  # noqa: BLE001 - benchmark must not abort on this
        logger.warning("Could not load %s (%s) — using initialised weights.", path, exc)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all models on this host.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
        help="Subset of models to benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="Batch sizes to sweep. Batch 1 is the headline deployment number.",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Untimed warmup iterations.")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 5 warmup / 10 timed iterations.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX Runtime measurements.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks.csv",
        help="CSV filename under results/.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Repeat the whole sweep N times, tagging each pass. CPU timings drift "
            "across a long sequential sweep (warm allocator, page cache, thermal "
            "state), so a single pass cannot support a difference below ~30%%. "
            "Use >=3 for any CPU claim in the thesis."
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    warmup = 5 if args.quick else args.warmup
    iters = 10 if args.quick else args.iters

    # Autotuning on, determinism off: this is the configuration a deployed
    # system would run under, and deterministic kernel selection would perturb
    # exactly the quantity being measured. Input shapes are fixed per model, so
    # cuDNN's algorithm cache is warm after the warmup iterations.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    configs = available_configurations()
    logger.info("Host supports %d configurations: %s", len(configs), configs)

    rows: list[dict] = []

    for pass_index in range(args.repeat):
        if args.repeat > 1:
            logger.info("=== sweep pass %d/%d ===", pass_index + 1, args.repeat)
        rows.extend(run_sweep(args, configs, warmup, iters, pass_index))

    if not rows:
        logger.error("No benchmark rows produced.")
        return

    path = append_rows(args.output, rows, BENCHMARK_FIELDS)
    logger.info("Wrote %d benchmark rows to %s", len(rows), path)

    print(f"\n{'model':<12} {'device':<12} {'backend':<20} {'prec':<6} {'bs':>3} "
          f"{'median ms':>10} {'p95 ms':>9} {'FPS':>8} {'GFLOPs':>8}")
    print("-" * 102)
    for row in rows:
        print(
            f"{row['model_key']:<12} {row['bench_device']:<12} {row['backend']:<20} "
            f"{row['precision']:<6} {row['batch_size']:>3} {row['latency_ms_median']:>10.3f} "
            f"{row['latency_ms_p95']:>9.3f} {row['fps']:>8.1f} {row['gflops']:>8.2f}"
        )


def run_sweep(
    args: argparse.Namespace,
    configs: list[tuple[str, str, str]],
    warmup: int,
    iters: int,
    pass_index: int,
) -> list[dict]:
    """One full pass over every (model, device, precision, batch) combination."""
    rows: list[dict] = []

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        channels, height, width = spec["input"]

        for bench_device, torch_device, precision in configs:
            for batch_size in args.batch_sizes:
                # INT8 dynamic quantization only meaningfully applies to the
                # classification models; YOLO's fused module graph and the
                # U-Net's all-conv body are not covered by dynamic quant, and
                # measuring them anyway would produce a number that looks like
                # a quantized result but is not one.
                if precision == "int8" and spec["builder"] in {"yolo"}:
                    logger.info("Skipping int8 for %s (not covered by dynamic quant).", model_key)
                    continue

                try:
                    model = build_model(spec["builder"], torch_device)
                    if spec["builder"] != "yolo":
                        load_weights_if_available(model, spec["checkpoint"], torch_device)

                    result = benchmark_torch_model(
                        model,
                        model_key=model_key,
                        model_name=spec["name"],
                        input_shape=(batch_size, channels, height, width),
                        device=torch_device,
                        precision=precision,
                        bench_device=bench_device,
                        checkpoint_path=spec["checkpoint"],
                        warmup=warmup,
                        iters=iters,
                        notes=f"pass={pass_index}",
                    )
                    rows.append(result.as_row())
                except Exception as exc:  # noqa: BLE001 - one failure must not stop the sweep
                    logger.error(
                        "FAILED %s on %s/%s batch=%d: %s",
                        model_key,
                        bench_device,
                        precision,
                        batch_size,
                        exc,
                    )
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        onnx_path = spec.get("onnx")
        if onnx_path and not args.skip_onnx:
            measured_batches: set[int] = set()
            for batch_size in args.batch_sizes:
                result = benchmark_onnx_model(
                    onnx_path,
                    model_key=model_key,
                    model_name=spec["name"],
                    input_shape=(batch_size, channels, height, width),
                    providers=["CPUExecutionProvider"],
                    bench_device="cpu",
                    precision="fp32",
                    warmup=warmup,
                    iters=iters,
                    notes="exported artifact; ARM-CPU proxy",
                )
                if result is None:
                    continue
                # A statically-exported graph pins its batch dimension, so a
                # request for batch 8 silently runs at the exported size. Record
                # each realised batch once instead of emitting duplicate rows
                # that look like a sweep but are the same measurement repeated.
                if result.batch_size in measured_batches:
                    logger.info(
                        "ONNX graph for %s is static at batch %d — skipping requested batch %d.",
                        model_key,
                        result.batch_size,
                        batch_size,
                    )
                    continue
                measured_batches.add(result.batch_size)
                result.notes = f"{result.notes}; pass={pass_index}"
                rows.append(result.as_row())

    return rows


if __name__ == "__main__":
    main()
