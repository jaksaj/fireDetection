"""Standalone inference benchmark for NVIDIA Jetson. Run this ON the device.

Design constraints
------------------
This runs on borrowed hardware, so it is deliberately conservative:

* **Installs nothing.** No pip, no apt, no venv. It uses whatever is already on
  the device and skips any backend that is missing.
* **No sudo, no system changes.** It never touches ``nvpmodel``, ``jetson_clocks``,
  or any device setting. Power mode is *read* if possible and recorded, never
  changed. Switching power modes is a manual step documented in README.md.
* **Writes only inside this folder.** All output goes to ``./results/``.
* **No network access.**

Measurement protocol -- identical to the desktop harness (``src/benchmark.py``)
so the numbers are directly comparable:

1. Inputs are random tensors created **once** in memory. No disk I/O, no image
   decoding, no preprocessing inside the timed region.
2. ``--warmup`` untimed iterations first (default 50), to absorb CUDA context
   creation, TensorRT engine build, cuDNN algorithm selection and cache warming.
3. ``--iters`` timed iterations (default 200), each recorded individually.
4. On CUDA, ``torch.cuda.synchronize()`` brackets the timed region. CUDA launches
   are asynchronous; timing without synchronising measures launch overhead, not
   execution.
5. **Median and p95** are reported, not just the mean. Latency on a thermally
   constrained board is right-skewed, and p95 is what a real-time system must
   budget for.

Backends attempted, each skipped cleanly if unavailable:

* ONNX Runtime -- TensorRT, CUDA and CPU execution providers
* PyTorch -- CUDA FP32/FP16 and CPU FP32

Usage::

    python3 benchmark_jetson.py                 # everything available
    python3 benchmark_jetson.py --quick         # smoke test, ~1 minute
    python3 benchmark_jetson.py --models iteration1 iteration4
    python3 benchmark_jetson.py --tag 15W       # label the run with the power mode
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"

WARMUP_ITERS = 50
TIMED_ITERS = 200

CSV_FIELDS = [
    "model_key",
    "model_name",
    "bench_device",
    "device_label",
    "backend",
    "precision",
    "batch_size",
    "input_h",
    "input_w",
    "params",
    "gflops",
    "size_mb",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p95",
    "latency_ms_std",
    "latency_ms_min",
    "fps",
    "throughput_ips",
    "peak_mem_mb",
    "warmup_iters",
    "timed_iters",
    "host",
    "torch_version",
    "power_mode",
    "notes",
]


# ---------------------------------------------------------------------------
# Device description (all read-only)
# ---------------------------------------------------------------------------


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").strip().replace("\x00", "")
    except OSError:
        return ""


def device_model() -> str:
    """Board name from the device tree, e.g. 'NVIDIA Jetson Orin Nano'."""
    return read_text("/proc/device-tree/model") or platform.machine()


def jetpack_version() -> str:
    """L4T / JetPack version, if the release file is present."""
    release = read_text("/etc/nv_tegra_release")
    if release:
        return release.splitlines()[0]
    return ""


def power_mode() -> str:
    """
    Current nvpmodel power mode, read-only and best effort.

    `nvpmodel -q` often needs root; if it does, this returns "" rather than
    prompting or escalating. The mode can also be passed explicitly with --tag.
    """
    try:
        output = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        for line in output.splitlines():
            if "NV Power Mode" in line:
                return line.split(":")[-1].strip()
        return output.strip().splitlines()[-1] if output.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def describe_environment() -> dict:
    info = {
        "device_model": device_model(),
        "jetpack": jetpack_version(),
        "power_mode": power_mode(),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_properties(0).name
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"unavailable ({type(exc).__name__})"

    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        info["onnxruntime"] = f"unavailable ({type(exc).__name__})"

    return info


# ---------------------------------------------------------------------------
# Timing core (mirrors src/benchmark.py on the desktop)
# ---------------------------------------------------------------------------


def summarise(latencies_ms: list[float], batch_size: int) -> dict:
    latencies_ms = sorted(latencies_ms)
    median = statistics.median(latencies_ms)
    index_p95 = min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))
    return {
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_median": median,
        "latency_ms_p95": latencies_ms[index_p95],
        "latency_ms_std": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "latency_ms_min": latencies_ms[0],
        "fps": 1000.0 / median if median else 0.0,
        "throughput_ips": (1000.0 / median if median else 0.0) * batch_size,
    }


def time_callable(fn, sync=None, warmup=WARMUP_ITERS, iters=TIMED_ITERS) -> list[float]:
    """Run `fn` warmup times untimed, then `iters` times with per-call timing."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    latencies = []
    for _ in range(iters):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


# ---------------------------------------------------------------------------
# Model registry -- must match scripts/export_for_jetson.py on the desktop
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "iteration1": {"name": "FireCNN (binary classification)", "input": (3, 224, 224)},
    "iteration2": {"name": "MobileNetV3-Small (4-class)", "input": (3, 224, 224)},
    "iteration3": {"name": "MobileNetV3-Small robust (4-class)", "input": (3, 224, 224)},
    "iteration4": {"name": "YOLO26n (detection)", "input": (3, 640, 640)},
    "iteration5": {"name": "LightweightUNet (segmentation)", "input": (3, 256, 256)},
}


def onnx_path(model_key: str) -> Path:
    return MODELS_DIR / f"{model_key}.onnx"


def weights_path(model_key: str) -> Path:
    return MODELS_DIR / f"{model_key}_weights.pt"


# ---------------------------------------------------------------------------
# ONNX Runtime benchmarking
# ---------------------------------------------------------------------------


def benchmark_onnx(model_key: str, provider: str, batch_size: int,
                   warmup: int, iters: int, env: dict, tag: str) -> dict | None:
    import numpy as np
    import onnxruntime as ort

    path = onnx_path(model_key)
    if not path.exists():
        return None

    available = ort.get_available_providers()
    if provider not in available:
        return None

    spec = MODEL_SPECS[model_key]
    channels, height, width = spec["input"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    provider_options = [{}]
    if provider == "TensorrtExecutionProvider":
        # Cache built engines inside the bundle so a repeat run does not pay the
        # (multi-minute) build cost again. Nothing is written outside ./results.
        cache_dir = RESULTS_DIR / "trt_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        provider_options = [{
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_fp16_enable": True,
        }]

    try:
        session = ort.InferenceSession(
            str(path), options, providers=[provider], provider_options=provider_options
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {model_key} / {provider}: session failed ({exc})")
        return None

    meta = session.get_inputs()[0]
    shape = [batch_size, channels, height, width]
    # A statically exported graph pins its batch dimension; honour it rather
    # than failing or silently mislabelling the batch size.
    for index, dim in enumerate(meta.shape):
        if isinstance(dim, int) and dim > 0 and index < len(shape):
            shape[index] = dim

    inputs = {meta.name: np.random.randn(*shape).astype(np.float32)}

    try:
        latencies = time_callable(lambda: session.run(None, inputs),
                                  warmup=warmup, iters=iters)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {model_key} / {provider}: run failed ({exc})")
        return None

    precision = "fp16" if provider == "TensorrtExecutionProvider" else "fp32"
    short = provider.replace("ExecutionProvider", "")
    row = {
        "model_key": model_key,
        "model_name": spec["name"],
        "bench_device": "jetson-cuda" if short in {"Tensorrt", "CUDA"} else "jetson-cpu",
        "device_label": env.get("device_model", ""),
        "backend": f"onnxruntime[{short}]",
        "precision": precision,
        "batch_size": shape[0],
        "input_h": shape[2],
        "input_w": shape[3],
        "params": 0,
        "gflops": 0.0,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "peak_mem_mb": 0.0,
        "warmup_iters": warmup,
        "timed_iters": iters,
        "host": env.get("hostname", ""),
        "torch_version": str(env.get("torch", "")),
        "power_mode": tag or env.get("power_mode", ""),
        "notes": "TRT fp16 enabled" if precision == "fp16" else "",
    }
    row.update(summarise(latencies, shape[0]))
    return row


# ---------------------------------------------------------------------------
# PyTorch benchmarking (optional -- only if torch is installed on the device)
# ---------------------------------------------------------------------------


def benchmark_torch(model_key: str, device: str, precision: str, batch_size: int,
                    warmup: int, iters: int, env: dict, tag: str) -> dict | None:
    try:
        import torch
    except ImportError:
        return None

    path = weights_path(model_key)
    if not path.exists():
        return None
    if device == "cuda" and not torch.cuda.is_available():
        return None

    spec = MODEL_SPECS[model_key]
    channels, height, width = spec["input"]

    try:
        # TorchScript modules carry their own architecture, so the device needs
        # no torchvision/ultralytics and no matching model source.
        model = torch.jit.load(str(path), map_location=device).eval()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {model_key} / torch-{device}: load failed ({exc})")
        return None

    dtype = torch.float16 if precision == "fp16" else torch.float32
    if precision == "fp16":
        model = model.half()

    inputs = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    sync = torch.cuda.synchronize if device == "cuda" else None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.no_grad():
            latencies = time_callable(lambda: model(inputs), sync=sync,
                                      warmup=warmup, iters=iters)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {model_key} / torch-{device}/{precision}: run failed ({exc})")
        return None

    peak_mem = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device == "cuda" else 0.0

    row = {
        "model_key": model_key,
        "model_name": spec["name"],
        "bench_device": f"jetson-{device}",
        "device_label": env.get("device_model", ""),
        "backend": "pytorch",
        "precision": precision,
        "batch_size": batch_size,
        "input_h": height,
        "input_w": width,
        "params": sum(p.numel() for p in model.parameters()),
        "gflops": 0.0,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "peak_mem_mb": peak_mem,
        "warmup_iters": warmup,
        "timed_iters": iters,
        "host": env.get("hostname", ""),
        "torch_version": str(env.get("torch", "")),
        "power_mode": tag or env.get("power_mode", ""),
        "notes": "torchscript",
    }
    row.update(summarise(latencies, batch_size))
    return row


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TensorRT via trtexec
# ---------------------------------------------------------------------------


def find_trtexec() -> str | None:
    """Locate trtexec, which ships with the `tensorrt` package."""
    from shutil import which

    found = which("trtexec")
    if found:
        return found
    for candidate in (
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/opt/nvidia/tensorrt/bin/trtexec",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def benchmark_trtexec(model_key: str, precision: str, batch_size: int,
                      iters: int, env: dict, tag: str, trtexec: str) -> dict | None:
    """
    Benchmark an ONNX graph through TensorRT using the trtexec CLI.

    On a freshly flashed JetPack, prebuilt `onnxruntime-gpu` and PyTorch wheels
    matching the CUDA/Python combination often do not exist yet, which would
    leave the GPU tier unmeasured. `trtexec` ships with the `tensorrt` package
    itself, so it is the most dependable route to a GPU number.

    trtexec reports its own GPU-compute latency distribution (median, mean,
    percentile), measured around the enqueue/execute cycle. Those figures are
    parsed here rather than timed externally, so the numbers are TensorRT's own
    and not confounded by Python overhead.
    """
    path = onnx_path(model_key)
    if not path.exists():
        return None

    spec = MODEL_SPECS[model_key]
    channels, height, width = spec["input"]

    cache_dir = RESULTS_DIR / "trt_engines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    engine = cache_dir / f"{model_key}_{precision}_b{batch_size}.engine"

    common = [
        trtexec,
        f"--iterations={iters}",
        "--avgRuns=1",
        "--warmUp=2000",          # ms of warmup before timing
        "--duration=0",           # honour --iterations rather than a time budget
        "--noDataTransfers",      # time compute only, not host<->device copies
    ]

    if engine.exists() and engine.stat().st_size > 0:
        # Reuse a previously built engine. This matters for the power-mode
        # sweep: TensorRT picks kernel tactics by timing them at *build* time,
        # so rebuilding under each power budget would confound the measurement
        # with build-time tactic differences. Building once and replaying the
        # same engine isolates the effect of the power cap on inference, and
        # also matches real deployment, where an engine is built once and
        # shipped.
        command = common + [f"--loadEngine={engine}"]
        print(f"  reusing cached TensorRT {precision} engine...")
    else:
        command = common + [f"--onnx={path}", f"--saveEngine={engine}"]
        if precision == "fp16":
            command.append("--fp16")
        elif precision == "int8":
            command.append("--int8")
        print(f"  building/running TensorRT {precision} engine (can take minutes)...")
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  ! trtexec failed for {model_key}/{precision}: {exc}")
        return None

    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = [ln for ln in output.splitlines() if ln.strip()][-3:]
        print(f"  ! trtexec exit {proc.returncode} for {model_key}/{precision}: {' | '.join(tail)}")
        return None

    # Parse the "GPU Compute Time" block, e.g.:
    #   GPU Compute Time: min = 1.2 ms, max = 2.1 ms, mean = 1.4 ms,
    #                     median = 1.3 ms, percentile(99%) = 2.0 ms
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        if "GPU Compute Time" not in line:
            continue
        for part in line.split(":", 1)[1].split(","):
            if "=" not in part:
                continue
            name, _, value = part.partition("=")
            name = name.strip().lower()
            try:
                number = float(value.strip().split()[0])
            except (ValueError, IndexError):
                continue
            if name.startswith("min"):
                metrics["min"] = number
            elif name.startswith("mean"):
                metrics["mean"] = number
            elif name.startswith("median"):
                metrics["median"] = number
            elif name.startswith("percentile"):
                metrics["p95"] = number  # trtexec default percentile is 99%
        break

    if "median" not in metrics:
        print(f"  ! could not parse trtexec output for {model_key}/{precision}")
        return None

    median = metrics["median"]
    return {
        "model_key": model_key,
        "model_name": spec["name"],
        "bench_device": "jetson-cuda",
        "device_label": env.get("device_model", ""),
        "backend": "tensorrt[trtexec]",
        "precision": precision,
        "batch_size": batch_size,
        "input_h": height,
        "input_w": width,
        "params": 0,
        "gflops": 0.0,
        "size_mb": engine.stat().st_size / (1024 * 1024) if engine.exists() else 0.0,
        "latency_ms_mean": metrics.get("mean", median),
        "latency_ms_median": median,
        "latency_ms_p95": metrics.get("p95", median),
        "latency_ms_std": 0.0,
        "latency_ms_min": metrics.get("min", median),
        "fps": 1000.0 / median if median else 0.0,
        "throughput_ips": (1000.0 / median if median else 0.0) * batch_size,
        "peak_mem_mb": 0.0,
        "warmup_iters": 0,
        "timed_iters": iters,
        "host": env.get("hostname", ""),
        "torch_version": str(env.get("torch", "")),
        "power_mode": tag or env.get("power_mode", ""),
        # trtexec's percentile is 99% by default, not 95% -- recorded so the
        # column is not silently compared against the desktop's p95.
        "notes": "trtexec GPU-compute time; p95 column holds percentile(99%); --noDataTransfers",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fire-detection models on Jetson.")
    parser.add_argument("--models", nargs="+", default=sorted(MODEL_SPECS),
                        choices=sorted(MODEL_SPECS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=TIMED_ITERS)
    parser.add_argument("--quick", action="store_true",
                        help="5 warmup / 10 timed iterations, for a fast sanity check.")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-tensorrt", action="store_true",
                        help="TensorRT engine builds can take several minutes per model.")
    parser.add_argument("--tag", default="",
                        help="Label for this run, e.g. the power mode ('15W', 'MAXN').")
    parser.add_argument("--output", default="jetson_benchmarks.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warmup = 5 if args.quick else args.warmup
    iters = 10 if args.quick else args.iters

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = describe_environment()

    print("=" * 68)
    print("Jetson benchmark")
    print("=" * 68)
    for key, value in env.items():
        print(f"  {key:26s} {value}")
    if args.tag:
        print(f"  {'run tag':26s} {args.tag}")
    print("=" * 68)

    (RESULTS_DIR / "environment.json").write_text(
        json.dumps(env, indent=2, default=str), encoding="utf-8"
    )

    providers: list[str] = []
    if not args.skip_onnx:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            if "TensorrtExecutionProvider" in available and not args.skip_tensorrt:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
        except ImportError:
            print("\n! onnxruntime not installed — skipping ONNX backends.\n")

    torch_configs: list[tuple[str, str]] = []
    if not args.skip_torch:
        try:
            import torch

            if torch.cuda.is_available():
                torch_configs += [("cuda", "fp32"), ("cuda", "fp16")]
            torch_configs.append(("cpu", "fp32"))
        except ImportError:
            print("\n! torch not installed — skipping PyTorch backends.\n")

    trtexec = None if args.skip_tensorrt else find_trtexec()
    trt_precisions = ["fp16", "fp32"] if trtexec else []
    if trtexec:
        print(f"\nUsing trtexec at {trtexec} for the TensorRT GPU tier.")
    elif not args.skip_tensorrt:
        print("\n! trtexec not found — GPU tier will rely on ONNX Runtime, if present.")

    rows: list[dict] = []

    for model_key in args.models:
        print(f"\n--- {model_key} ---")
        for batch_size in args.batch_sizes:
            for provider in providers:
                row = benchmark_onnx(model_key, provider, batch_size, warmup, iters, env, args.tag)
                if row:
                    rows.append(row)
                    print(f"  {row['backend']:<28} {row['precision']:<5} "
                          f"median {row['latency_ms_median']:8.2f} ms  {row['fps']:7.1f} FPS")
            for device, precision in torch_configs:
                row = benchmark_torch(model_key, device, precision, batch_size,
                                      warmup, iters, env, args.tag)
                if row:
                    rows.append(row)
                    print(f"  {'pytorch-' + device:<28} {precision:<5} "
                          f"median {row['latency_ms_median']:8.2f} ms  {row['fps']:7.1f} FPS")
            for precision in trt_precisions:
                row = benchmark_trtexec(model_key, precision, batch_size,
                                        iters, env, args.tag, trtexec)
                if row:
                    rows.append(row)
                    print(f"  {'tensorrt[trtexec]':<28} {precision:<5} "
                          f"median {row['latency_ms_median']:8.2f} ms  {row['fps']:7.1f} FPS")

    if not rows:
        print("\nNo measurements produced. Run check_env.py to see what is available.")
        sys.exit(1)

    output_path = RESULTS_DIR / args.output
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")
    print("Copy the whole results/ folder back to the workstation.")


if __name__ == "__main__":
    main()
