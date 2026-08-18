"""Unified inference-cost measurement for every model in the project.

This module replaces two broken measurement paths:

- ``src/edge_simulation.py:53-94`` timed CUDA forward passes with
  ``time.perf_counter()`` and never called ``torch.cuda.synchronize()``. CUDA
  launches are asynchronous, so that loop measured kernel-launch overhead, not
  execution time, and every ratio derived from it was meaningless.
- ``src/detection/export.py:99-123`` called ``model.predict(source=<path>)`` in
  a loop, so each "inference" re-read and re-decoded a JPEG from disk and
  rebuilt an Ultralytics ``Results`` object. That is a disk-I/O benchmark.

The protocol here is deliberately uniform across every model, device, and
precision so that numbers are comparable:

1. Inputs are random tensors materialised **once** in memory. No disk access,
   no decoding, no preprocessing inside the timed region.
2. ``WARMUP_ITERS`` untimed iterations first, to pay one-off costs: CUDA
   context creation, cuDNN algorithm selection, JIT, and CPU cache warming.
3. ``TIMED_ITERS`` timed iterations, each individually recorded.
4. On CUDA, ``torch.cuda.synchronize()`` is called before starting the clock
   and before stopping it, so the measurement brackets actual execution.
5. **Median and p95** are reported, not just mean. Latency distributions on a
   shared desktop are right-skewed; a mean silently absorbs scheduler noise
   while the median does not, and p95 is what a real-time system must budget.

Static cost (parameters, FLOPs, on-disk size) is measured alongside latency,
because the thesis argues about deployability, and a model that fits in cache
but takes 40 ms is a different proposition from one that is 4x larger and
takes 4 ms.

A caveat this harness surfaced about itself
-------------------------------------------
In the first full sweep the U-Net measured 107.8 ms at CPU FP32 and 76.6 ms at
CPU "INT8" -- a 29% improvement from a quantization step that converted
**zero** modules (PyTorch dynamic quantization covers Linear/RNN, not Conv2d,
and the U-Net is entirely convolutional, so the two runs executed identical
graphs). The difference is therefore measurement drift, not precision: CPU runs
executed later in a long sweep benefit from a warmed allocator and page cache,
and possibly differ in thermal/turbo state.

Within-run spread is visible in the p95/median ratio (~1.10 for that config);
between-run drift was ~1.3x, i.e. larger. So on CPU, a single sequential pass is
not sufficient to attribute a difference of less than roughly 30% to anything.
``--repeat`` re-runs the whole sweep so between-run variance can be estimated
rather than assumed away, and any CPU claim in the thesis should cite a repeated
measurement. GPU measurements, which are synchronized and far more tightly
distributed, do not show this behaviour.

The module deliberately has no project-specific imports at module scope beyond
torch, so it can be copied to a Jetson and run there unchanged.
"""

from __future__ import annotations

import gc
import logging
import platform
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

WARMUP_ITERS = 50
TIMED_ITERS = 200

#: Row schema for results/benchmarks.csv.
BENCHMARK_FIELDS = [
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
    "notes",
]


@dataclass
class BenchmarkResult:
    """One measured (model, device, backend, precision, batch size) point."""

    model_key: str
    model_name: str
    bench_device: str
    device_label: str
    backend: str
    precision: str
    batch_size: int
    input_h: int
    input_w: int
    params: int = 0
    gflops: float = 0.0
    size_mb: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_median: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_std: float = 0.0
    latency_ms_min: float = 0.0
    fps: float = 0.0
    throughput_ips: float = 0.0
    peak_mem_mb: float = 0.0
    warmup_iters: int = WARMUP_ITERS
    timed_iters: int = TIMED_ITERS
    host: str = field(default_factory=platform.node)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def count_parameters(model: nn.Module) -> int:
    """Total parameter count, including frozen parameters."""
    return sum(p.numel() for p in model.parameters())


def measure_gflops(model: nn.Module, input_shape: tuple[int, int, int, int]) -> float:
    """
    Multiply-accumulate cost in GFLOPs for one forward pass, via ``thop``.

    Returns 0.0 if thop is unavailable or the model contains ops it cannot
    profile -- a missing FLOP count should never abort a latency measurement.

    Note on convention: thop reports MACs. This function doubles them to get
    FLOPs, which is the convention used throughout the thesis. Some papers
    report the MAC number and call it FLOPs, so figures may differ by 2x from
    published values; the convention is stated wherever the number appears.
    """
    try:
        from thop import profile
    except ImportError:
        logger.warning("thop not installed — skipping FLOP measurement.")
        return 0.0

    try:
        device = next(model.parameters()).device
        dummy = torch.randn(*input_shape, device=device)
        model_copy_mode = model.training
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        model.train(model_copy_mode)
        return float(macs) * 2.0 / 1e9
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.warning("FLOP measurement failed: %s", exc)
        return 0.0


def measure_size_mb(model: nn.Module, path: Path | None = None) -> float:
    """
    On-disk size of the model's state dict in MiB.

    If ``path`` points at an existing file (a real checkpoint or exported
    artifact) its actual size is used; otherwise the state dict is serialised to
    a temporary file. Reporting the real artifact size matters because a
    PyTorch checkpoint also carries optimizer state and metadata, which is what
    actually has to be shipped to a device.
    """
    if path is not None and Path(path).exists():
        return Path(path).stat().st_size / (1024 * 1024)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        return temp_path.stat().st_size / (1024 * 1024)
    finally:
        temp_path.unlink(missing_ok=True)


def _synchronize(device: torch.device) -> None:
    """Block until all queued work on the device has completed."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_memory_mb(device: torch.device) -> float:
    """
    Peak memory in MiB.

    On CUDA this is exact (``max_memory_allocated``). On CPU there is no
    equivalent per-op counter, so process RSS is reported instead -- a coarser
    figure that includes the interpreter and loaded libraries. The distinction
    is recorded in the ``notes`` column rather than silently conflated.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _summarise(latencies_ms: list[float], batch_size: int, result: BenchmarkResult) -> None:
    """Fill the timing fields of ``result`` from raw per-iteration samples."""
    latencies_ms.sort()
    result.latency_ms_mean = statistics.fmean(latencies_ms)
    result.latency_ms_median = statistics.median(latencies_ms)
    result.latency_ms_p95 = latencies_ms[min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))]
    result.latency_ms_std = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    result.latency_ms_min = latencies_ms[0]
    # FPS is the reciprocal of per-batch latency (batch=1 => frames per second).
    # Throughput additionally credits the batch, which is the fair number when
    # comparing batch sweeps.
    result.fps = 1000.0 / result.latency_ms_median if result.latency_ms_median else 0.0
    result.throughput_ips = result.fps * batch_size


def benchmark_callable(
    fn: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> list[float]:
    """
    Time an arbitrary zero-argument callable, returning per-iteration ms.

    The callable must perform exactly the work being measured and nothing else:
    no data loading, no allocation of new host tensors, no postprocessing that
    would not run in deployment.
    """
    for _ in range(warmup):
        fn()
    _synchronize(device)

    latencies: list[float] = []
    for _ in range(iters):
        _synchronize(device)
        start = time.perf_counter()
        fn()
        _synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)

    return latencies


def benchmark_torch_model(
    model: nn.Module,
    *,
    model_key: str,
    model_name: str,
    input_shape: tuple[int, int, int, int],
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    bench_device: str | None = None,
    device_label: str = "",
    checkpoint_path: Path | None = None,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
    notes: str = "",
) -> BenchmarkResult:
    """
    Measure one PyTorch model at one precision on one device.

    Args:
        model: The module to measure. Moved to ``device`` and set to eval mode.
        input_shape: ``(batch, channels, height, width)``.
        precision: ``fp32``, ``fp16`` (CUDA only), or ``int8`` (CPU dynamic
            quantization only).
        bench_device: Key used to group rows in the CSV, e.g. ``cuda``, ``cpu``,
            ``jetson-cuda``. Defaults to the torch device type.

    Returns:
        A populated :class:`BenchmarkResult`.
    """
    device = torch.device(device)
    bench_device = bench_device or device.type
    batch_size = input_shape[0]

    model = model.to(device).eval()

    extra_notes = [notes] if notes else []

    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 benchmarking requires a CUDA device.")
        model = model.half()
    elif precision == "int8":
        if device.type != "cpu":
            raise ValueError("Dynamic INT8 quantization runs on CPU only.")
        # Report what was actually converted. PyTorch's dynamic quantization is
        # built around Linear/RNN; Conv2d coverage varies by version and is
        # frequently a no-op. Claiming an "INT8 model" without checking is how
        # unverifiable size figures end up in a thesis.
        before = count_parameters(model)
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        converted = [
            name
            for name, module in model.named_modules()
            if "quantized" in type(module).__module__
        ]
        extra_notes.append(
            f"dynamic-int8 converted {len(converted)} modules; params_before={before}"
        )
        logger.info("INT8 conversion touched %d modules: %s", len(converted), converted[:8])

    dtype = torch.float16 if precision == "fp16" else torch.float32
    inputs = torch.randn(*input_shape, device=device, dtype=dtype)

    result = BenchmarkResult(
        model_key=model_key,
        model_name=model_name,
        bench_device=bench_device,
        device_label=device_label or _describe_device(device),
        backend="pytorch",
        precision=precision,
        batch_size=batch_size,
        input_h=input_shape[2],
        input_w=input_shape[3],
        warmup_iters=warmup,
        timed_iters=iters,
    )

    result.params = count_parameters(model)
    result.size_mb = measure_size_mb(model, checkpoint_path)
    if precision == "fp32":
        result.gflops = measure_gflops(model, input_shape)

    if device.type == "cpu":
        extra_notes.append("peak_mem is process RSS, not per-op")

    gc.collect()
    _reset_peak_memory(device)

    with torch.no_grad():
        latencies = benchmark_callable(
            lambda: model(inputs), device, warmup=warmup, iters=iters
        )

    result.peak_mem_mb = _peak_memory_mb(device)
    _summarise(latencies, batch_size, result)
    result.notes = "; ".join(extra_notes)

    logger.info(
        "%s | %s | %s | %s | median %.2f ms | p95 %.2f ms | %.1f FPS",
        model_key,
        bench_device,
        result.backend,
        precision,
        result.latency_ms_median,
        result.latency_ms_p95,
        result.fps,
    )
    return result


def benchmark_onnx_model(
    onnx_path: Path,
    *,
    model_key: str,
    model_name: str,
    input_shape: tuple[int, int, int, int],
    providers: list[str] | None = None,
    bench_device: str = "cpu",
    device_label: str = "",
    precision: str = "fp32",
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
    notes: str = "",
) -> BenchmarkResult | None:
    """
    Measure an exported ONNX artifact through ONNX Runtime.

    This benchmarks the **exported file**, not the PyTorch model it came from.
    The previous export pipeline recorded only artifact file sizes and then
    benchmarked the original ``.pt``, so no latency evidence existed for any
    export format.

    ONNX Runtime on CPU is also the closest proxy available on an x86
    workstation for ARM CPU deployment, and is labelled as a proxy wherever it
    is used as one.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX benchmark.")
        return None

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        logger.warning("ONNX artifact not found: %s", onnx_path)
        return None

    providers = providers or ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        logger.warning("None of the requested ONNX providers are available.")
        return None

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), session_options, providers=providers)

    # ONNX Runtime lists a provider in `get_available_providers()` even when the
    # underlying libraries are absent, and then **silently falls back to CPU** at
    # session creation. Without this check a CPU measurement would be recorded
    # and labelled "CUDA" or "TensorRT" -- exactly the kind of mislabelled number
    # this project exists to eliminate. Verify what actually got bound.
    actually_used = session.get_providers()
    if actually_used and actually_used[0] != providers[0]:
        logger.warning(
            "Requested %s but ONNX Runtime bound %s — skipping rather than "
            "recording a mislabelled row.",
            providers[0],
            actually_used[0],
        )
        return None

    input_meta = session.get_inputs()[0]
    # Honour the exported graph's fixed spatial dims when it has them; a static
    # export will reject a differently-shaped input.
    shape = list(input_shape)
    for index, dim in enumerate(input_meta.shape):
        if isinstance(dim, int) and dim > 0 and index < len(shape):
            shape[index] = dim

    inputs = {input_meta.name: np.random.randn(*shape).astype(np.float32)}
    cpu_device = torch.device("cpu")

    result = BenchmarkResult(
        model_key=model_key,
        model_name=model_name,
        bench_device=bench_device,
        device_label=device_label or platform.processor() or platform.machine(),
        backend=f"onnxruntime[{providers[0].replace('ExecutionProvider', '')}]",
        precision=precision,
        batch_size=shape[0],
        input_h=shape[2],
        input_w=shape[3],
        size_mb=onnx_path.stat().st_size / (1024 * 1024),
        warmup_iters=warmup,
        timed_iters=iters,
        notes="; ".join(filter(None, [notes, f"providers={providers}"])),
    )

    latencies = benchmark_callable(
        lambda: session.run(None, inputs), cpu_device, warmup=warmup, iters=iters
    )
    _summarise(latencies, shape[0], result)

    logger.info(
        "%s | %s | %s | median %.2f ms | %.1f FPS",
        model_key,
        result.backend,
        precision,
        result.latency_ms_median,
        result.fps,
    )
    return result


def _describe_device(device: torch.device) -> str:
    """Human-readable device name for the results table."""
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device.index or 0)
        return f"{props.name} ({props.total_memory / 1e9:.1f} GB, SM {props.major}.{props.minor})"
    return platform.processor() or platform.machine() or "cpu"


def available_configurations() -> list[tuple[str, str, str]]:
    """
    Enumerate (bench_device, torch_device, precision) triples this host supports.

    Running the same script on the workstation and on a Jetson therefore
    produces the rows that host can actually measure, with no per-device
    branching in the caller.
    """
    is_jetson = _is_jetson()
    prefix = "jetson-" if is_jetson else ""

    configs: list[tuple[str, str, str]] = [
        (f"{prefix}cpu", "cpu", "fp32"),
        (f"{prefix}cpu", "cpu", "int8"),
    ]
    if torch.cuda.is_available():
        configs.insert(0, (f"{prefix}cuda", "cuda", "fp32"))
        configs.insert(1, (f"{prefix}cuda", "cuda", "fp16"))
    return configs


def _is_jetson() -> bool:
    """Detect NVIDIA Tegra hardware so rows are labelled correctly."""
    model_path = Path("/proc/device-tree/model")
    try:
        if model_path.exists():
            model = model_path.read_text(errors="ignore").lower()
            return "jetson" in model or "orin" in model or "tegra" in model
    except OSError:
        pass
    return "tegra" in platform.platform().lower()
