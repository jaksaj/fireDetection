"""Benchmark the desktop GPU through TensorRT, matched to the Jetson protocol.

Why this exists
---------------
The Jetson GPU tier was measured with `trtexec --noDataTransfers`, which times
pure GPU compute. Measuring the desktop through ONNX Runtime's TensorRT provider
instead would include ORT dispatch and host<->device copies, making the desktop
look artificially slow and **inflating** the apparent edge-device advantage --
the wrong direction for an honest comparison.

`trtexec` is not shipped in the pip TensorRT package on Windows, so this script
reproduces its measurement directly against the TensorRT Python API:

* engine built from the same ONNX file used on the Jetson,
* device buffers allocated **once** and reused,
* no host<->device transfers inside the timed region,
* `torch.cuda.synchronize()` bracketing the timed region,
* same warmup/iteration counts and the same median/p95 reporting.

Device memory is managed with torch tensors (`.data_ptr()`) rather than pycuda,
so no extra dependency is needed.

The TensorRT libraries installed by pip are not on the default DLL search path,
so this script adds them before importing.

Usage::

    python scripts/benchmark_tensorrt_desktop.py
    python scripts/benchmark_tensorrt_desktop.py --models iteration4 --precisions fp16
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TensorRT and CUDA DLLs ship inside site-packages and are not on PATH by
# default on Windows; register them before anything imports tensorrt.
for _sub in ("tensorrt_libs", "torch/lib"):
    _dir = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / _sub
    if _dir.exists():
        os.environ["PATH"] = str(_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_dir))
            except OSError:
                pass

import torch

from src.benchmark import BENCHMARK_FIELDS, WARMUP_ITERS, TIMED_ITERS, _describe_device
from src.results import append_rows
from src.utils import configure_logging

logger = logging.getLogger("benchmark_tensorrt_desktop")

MODELS_DIR = PROJECT_ROOT / "jetson" / "models"
ENGINE_DIR = PROJECT_ROOT / "results" / "trt_engines_desktop"

SPECS = {
    "iteration1": ("FireCNN (binary classification)", (3, 224, 224)),
    "iteration2": ("MobileNetV3-Small (4-class)", (3, 224, 224)),
    "iteration3": ("MobileNetV3-Small robust (4-class)", (3, 224, 224)),
    "iteration4": ("YOLO26n (detection)", (3, 640, 640)),
    "iteration5": ("LightweightUNet (segmentation)", (3, 256, 256)),
}

TORCH_DTYPE = {"FLOAT": torch.float32, "HALF": torch.float16, "INT32": torch.int32,
               "INT64": torch.int64, "BOOL": torch.bool, "INT8": torch.int8}


def build_engine(onnx_path: Path, engine_path: Path, precision: str):
    """Build (or load) a serialized TensorRT engine for one ONNX graph."""
    import tensorrt as trt

    logger_trt = trt.Logger(trt.Logger.ERROR)

    if engine_path.exists():
        with trt.Runtime(logger_trt) as runtime:
            return runtime.deserialize_cuda_engine(engine_path.read_bytes())

    builder = trt.Builder(logger_trt)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger_trt)
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            logger.error("ONNX parse: %s", parser.get_error(index))
        return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("Platform reports no fast FP16.")
        config.set_flag(trt.BuilderFlag.FP16)

    logger.info("Building %s engine for %s (may take minutes)...", precision, onnx_path.stem)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error("Engine build failed for %s/%s", onnx_path.stem, precision)
        return None

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)

    with trt.Runtime(logger_trt) as runtime:
        return runtime.deserialize_cuda_engine(serialized)


def benchmark_engine(engine, warmup: int, iters: int,
                     warmup_ms: float = 2000.0) -> tuple[list[float], float]:
    """
    Time pure GPU execution with persistent device buffers.

    Warmup runs for at least `warmup_ms` of wall-clock time as well as `warmup`
    iterations. A fixed iteration count is not sufficient: 50 iterations of a
    0.14 ms model is 7 ms of load, which leaves the GPU at idle clocks and
    produced a 2.2x discrepancy between two runs of the *same* cached engine.
    Jetson's `trtexec` warms up by time (`--warmUp=2000`), so matching that is
    also what makes the two platforms comparable.
    """
    import tensorrt as trt

    context = engine.create_execution_context()
    buffers: dict[str, torch.Tensor] = {}

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        dtype = TORCH_DTYPE.get(engine.get_tensor_dtype(name).name, torch.float32)
        # Allocated once and reused: no host<->device traffic in the timed loop.
        tensor = torch.zeros(shape, dtype=dtype, device="cuda")
        buffers[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    handle = stream.cuda_stream

    def run() -> None:
        context.execute_async_v3(stream_handle=handle)

    with torch.cuda.stream(stream):
        deadline = time.perf_counter() + warmup_ms / 1000.0
        count = 0
        while count < warmup or time.perf_counter() < deadline:
            run()
            count += 1
            if count % 200 == 0:
                torch.cuda.synchronize()
    torch.cuda.synchronize()

    latencies: list[float] = []
    with torch.cuda.stream(stream):
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            run()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del context, buffers
    return latencies, peak_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desktop TensorRT benchmark.")
    parser.add_argument("--models", nargs="+", default=sorted(SPECS), choices=sorted(SPECS))
    parser.add_argument("--precisions", nargs="+", default=["fp16", "fp32"],
                        choices=["fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=TIMED_ITERS)
    parser.add_argument("--output", default="benchmarks.csv")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if not torch.cuda.is_available():
        logger.error("No CUDA device.")
        return

    try:
        import tensorrt as trt

        logger.info("TensorRT %s on %s", trt.__version__, torch.cuda.get_device_name(0))
    except ImportError:
        logger.error("tensorrt not installed: pip install tensorrt")
        return

    device = torch.device("cuda")
    rows: list[dict] = []

    for model_key in args.models:
        name, (channels, height, width) = SPECS[model_key]
        onnx_path = MODELS_DIR / f"{model_key}.onnx"
        if not onnx_path.exists():
            logger.warning("Missing %s — run scripts/export_for_jetson.py first.", onnx_path)
            continue

        for precision in args.precisions:
            engine_path = ENGINE_DIR / f"{model_key}_{precision}_b1.engine"
            try:
                engine = build_engine(onnx_path, engine_path, precision)
                if engine is None:
                    continue
                torch.cuda.reset_peak_memory_stats()
                latencies, peak_mb = benchmark_engine(engine, args.warmup, args.iters)
                del engine
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001 - one model must not stop the sweep
                logger.error("Failed %s/%s: %s", model_key, precision, exc)
                continue

            latencies.sort()
            median = statistics.median(latencies)
            rows.append({
                "model_key": model_key, "model_name": name,
                "bench_device": "cuda", "device_label": _describe_device(device),
                "backend": "tensorrt[python-api]", "precision": precision,
                "batch_size": 1, "input_h": height, "input_w": width,
                "params": 0, "gflops": 0.0,
                "size_mb": engine_path.stat().st_size / (1024 * 1024),
                "latency_ms_mean": statistics.fmean(latencies),
                "latency_ms_median": median,
                "latency_ms_p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
                "latency_ms_std": statistics.pstdev(latencies),
                "latency_ms_min": latencies[0],
                "fps": 1000.0 / median if median else 0.0,
                "throughput_ips": 1000.0 / median if median else 0.0,
                "peak_mem_mb": peak_mb,
                "warmup_iters": args.warmup, "timed_iters": args.iters,
                "host": os.environ.get("COMPUTERNAME", ""),
                "torch_version": torch.__version__,
                "notes": "pure TRT compute, persistent device buffers, no H2D/D2H "
                         "in timed region; matches Jetson trtexec --noDataTransfers",
            })
            logger.info("%s | %s | median %.3f ms | %.1f FPS",
                        model_key, precision, median, 1000.0 / median)

    if not rows:
        logger.error("No rows produced.")
        return

    append_rows(args.output, rows, BENCHMARK_FIELDS)
    print(f"\n{'model':<12} {'prec':<6} {'median ms':>10} {'p95 ms':>9} {'FPS':>9}")
    print("-" * 50)
    for row in rows:
        print(f"{row['model_key']:<12} {row['precision']:<6} "
              f"{row['latency_ms_median']:>10.3f} {row['latency_ms_p95']:>9.3f} "
              f"{row['fps']:>9.1f}")


if __name__ == "__main__":
    main()
