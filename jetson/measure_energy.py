"""Measure energy per inference on Jetson using tegrastats power rails.

Latency alone does not answer the edge-deployment question. A model that runs in
4 ms but draws 15 W is a different proposition on a battery or PoE budget than
one that runs in 6 ms at 7 W. This script measures what that costs, in
millijoules per frame.

Method
------
1. **Idle baseline.** Sample power for `--idle-seconds` with nothing running.
   This is the board's fixed overhead: SoC, memory, I/O, fans.
2. **Under load.** Run one model in a continuous loop for `--load-seconds`,
   sampling power throughout, and take the mean over the steady-state window
   (the first `--settle-seconds` are discarded so ramp-up is excluded).
3. **Energy.** Two figures are reported, because they answer different questions:

   * ``energy_total_mj``  = mean_power_under_load x latency
     What one frame costs including the board's fixed overhead. This is what a
     battery actually sees.
   * ``energy_marginal_mj`` = (mean_power_under_load - idle_power) x latency
     What the *computation* costs. This is the fair figure for comparing models,
     since it removes overhead the deployment pays regardless.

Rails reported by tegrastats on Orin Nano:
  VDD_IN          -- total board input power (used for the headline figures)
  VDD_CPU_GPU_CV  -- CPU + GPU + compute-vision complex
  VDD_SOC         -- rest of the SoC

Runs read-only: no sudo, no system changes, nothing written outside ./results.

Usage::

    python3 measure_energy.py --tag 25W
    python3 measure_energy.py --tag 25W --models iteration4 --load-seconds 30
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"

RAIL_PATTERN = re.compile(r"(VDD[_A-Z0-9]*)\s+(\d+)mW")

ENERGY_FIELDS = [
    "model_key", "backend", "precision", "power_mode",
    "latency_ms_median", "fps",
    "idle_power_w", "load_power_w", "delta_power_w",
    "energy_total_mj", "energy_marginal_mj",
    "cpu_gpu_cv_w", "soc_w",
    "samples", "load_seconds", "host",
]

MODEL_SPECS = {
    "iteration1": ("FireCNN (binary classification)", (3, 224, 224)),
    "iteration2": ("MobileNetV3-Small (4-class)", (3, 224, 224)),
    "iteration3": ("MobileNetV3-Small robust (4-class)", (3, 224, 224)),
    "iteration4": ("YOLO26n (detection)", (3, 640, 640)),
    "iteration5": ("LightweightUNet (segmentation)", (3, 256, 256)),
}


class PowerSampler:
    """Background tegrastats reader. Collects per-rail milliwatt samples."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict[str, int]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            rails = {m.group(1): int(m.group(2)) for m in RAIL_PATTERN.finditer(line)}
            if rails:
                rails["_t"] = time.perf_counter()
                self.samples.append(rails)

    def __enter__(self) -> "PowerSampler":
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def mean_since(self, start_time: float, rail: str = "VDD_IN") -> tuple[float, int]:
        """Mean watts on `rail` over samples taken after `start_time`."""
        values = [s[rail] for s in self.samples if s.get("_t", 0) >= start_time and rail in s]
        if not values:
            return 0.0, 0
        return statistics.fmean(values) / 1000.0, len(values)


def find_trtexec() -> str | None:
    from shutil import which

    found = which("trtexec")
    if found:
        return found
    for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec"):
        if Path(candidate).exists():
            return candidate
    return None


def load_latency(model_key: str, backend: str, precision: str, mode: str) -> float:
    """Reuse the latency already measured by benchmark_jetson.py at this mode."""
    csv_path = RESULTS_DIR / "jetson_benchmarks.csv"
    if not csv_path.exists():
        return 0.0
    best = 0.0
    for row in csv.DictReader(csv_path.open()):
        if (row["model_key"] == model_key and row["power_mode"] == mode
                and row["precision"] == precision and backend in row["backend"]):
            try:
                best = float(row["latency_ms_median"])
            except (ValueError, KeyError):
                continue
    return best


def run_trt_load(trtexec: str, engine: Path, seconds: int) -> subprocess.Popen:
    return subprocess.Popen(
        [trtexec, f"--loadEngine={engine}", f"--duration={seconds}",
         "--noDataTransfers", "--iterations=0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_cpu_load(model_key: str, seconds: int) -> subprocess.Popen:
    """Continuous ONNX Runtime CPU inference in a child process."""
    channels, height, width = MODEL_SPECS[model_key][1]
    script = (
        "import time,numpy as np,onnxruntime as ort;"
        f"s=ort.InferenceSession(r'{MODELS_DIR / (model_key + '.onnx')}',"
        "providers=['CPUExecutionProvider']);"
        "n=s.get_inputs()[0].name;"
        f"x=np.random.randn(1,{channels},{height},{width}).astype(np.float32);"
        f"end=time.perf_counter()+{seconds};"
        "\nwhile time.perf_counter()<end: s.run(None,{n:x})"
    )
    return subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure energy per inference.")
    parser.add_argument("--models", nargs="+", default=sorted(MODEL_SPECS),
                        choices=sorted(MODEL_SPECS))
    parser.add_argument("--tag", default="", help="Power mode label, e.g. 25W.")
    parser.add_argument("--idle-seconds", type=int, default=20)
    parser.add_argument("--load-seconds", type=int, default=20)
    parser.add_argument("--settle-seconds", type=float, default=5.0,
                        help="Discard this much of each load window as ramp-up.")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--output", default="jetson_energy.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = args.tag
    if not mode:
        try:
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                                 text=True, timeout=10).stdout
            for line in out.splitlines():
                if "NV Power Mode" in line:
                    mode = line.split(":")[-1].strip()
        except (OSError, subprocess.SubprocessError):
            mode = "unknown"

    import socket

    host = socket.gethostname()
    trtexec = find_trtexec()

    print("=" * 70)
    print(f"Energy per inference — power mode {mode}")
    print("=" * 70)

    # ---- idle baseline -------------------------------------------------
    print(f"\nMeasuring idle baseline for {args.idle_seconds}s (keep the board quiet)...")
    with PowerSampler() as sampler:
        start = time.perf_counter()
        time.sleep(args.settle_seconds)
        window = time.perf_counter()
        time.sleep(args.idle_seconds)
        idle_w, idle_n = sampler.mean_since(window, "VDD_IN")
        idle_cpu_gpu, _ = sampler.mean_since(window, "VDD_CPU_GPU_CV")
        idle_soc, _ = sampler.mean_since(window, "VDD_SOC")
    print(f"  idle VDD_IN {idle_w:.2f} W  (CPU_GPU_CV {idle_cpu_gpu:.2f} W, "
          f"SOC {idle_soc:.2f} W) over {idle_n} samples")

    configs: list[tuple[str, str, str]] = []
    if trtexec:
        configs += [("tensorrt[trtexec]", "fp16", "trt"), ("tensorrt[trtexec]", "fp32", "trt")]
    if not args.skip_cpu:
        configs.append(("onnxruntime[CPU]", "fp32", "cpu"))

    rows: list[dict] = []

    for model_key in args.models:
        print(f"\n--- {model_key} ---")
        for backend, precision, kind in configs:
            latency = load_latency(model_key, backend.split("[")[0], precision, mode)
            if latency <= 0:
                print(f"  ! no latency recorded for {backend}/{precision} at {mode}; "
                      "run benchmark_jetson.py first — skipping")
                continue

            if kind == "trt":
                engine = RESULTS_DIR / "trt_engines" / f"{model_key}_{precision}_b1.engine"
                if not engine.exists():
                    print(f"  ! engine missing: {engine.name} — skipping")
                    continue

            with PowerSampler() as sampler:
                proc = (run_trt_load(trtexec, engine, args.load_seconds) if kind == "trt"
                        else run_cpu_load(model_key, args.load_seconds))
                time.sleep(args.settle_seconds)
                window = time.perf_counter()
                proc.wait()
                load_w, n = sampler.mean_since(window, "VDD_IN")
                cpu_gpu_w, _ = sampler.mean_since(window, "VDD_CPU_GPU_CV")
                soc_w, _ = sampler.mean_since(window, "VDD_SOC")

            if n == 0:
                print(f"  ! no power samples for {backend}/{precision} — skipping")
                continue

            delta_w = load_w - idle_w
            energy_total = load_w * (latency / 1000.0) * 1000.0      # mJ
            energy_marginal = max(delta_w, 0.0) * (latency / 1000.0) * 1000.0

            rows.append({
                "model_key": model_key, "backend": backend, "precision": precision,
                "power_mode": mode, "latency_ms_median": latency,
                "fps": 1000.0 / latency if latency else 0.0,
                "idle_power_w": round(idle_w, 3), "load_power_w": round(load_w, 3),
                "delta_power_w": round(delta_w, 3),
                "energy_total_mj": round(energy_total, 3),
                "energy_marginal_mj": round(energy_marginal, 3),
                "cpu_gpu_cv_w": round(cpu_gpu_w, 3), "soc_w": round(soc_w, 3),
                "samples": n, "load_seconds": args.load_seconds, "host": host,
            })
            print(f"  {backend:<20} {precision:<5} {load_w:5.2f} W "
                  f"(+{delta_w:4.2f} W)  {latency:7.2f} ms  ->  "
                  f"{energy_total:7.2f} mJ total, {energy_marginal:6.2f} mJ marginal")

    if not rows:
        print("\nNo energy measurements produced.")
        sys.exit(1)

    path = RESULTS_DIR / args.output
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENERGY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
