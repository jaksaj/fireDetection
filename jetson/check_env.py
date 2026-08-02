"""Read-only pre-flight check. Run this on the Jetson FIRST.

Reports what the device already has so you know what the benchmark will be able
to measure before committing to a long run. Changes nothing, installs nothing,
needs no sudo, and makes no network calls.

Usage::

    python3 check_env.py
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

EXPECTED_MODELS = [
    "iteration1",
    "iteration2",
    "iteration3",
    "iteration4",
    "iteration5",
]


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").strip().replace("\x00", "")
    except OSError:
        return ""


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    ok = True

    print("=" * 66)
    print("Jetson pre-flight check (read-only)")
    print("=" * 66)

    section("Device")
    model = read_text("/proc/device-tree/model")
    print(f"  board            : {model or '(not a Tegra device?)'}")
    print(f"  arch             : {platform.machine()}")
    print(f"  kernel           : {platform.release()}")
    print(f"  cpu cores        : {os.cpu_count()}")
    release = read_text("/etc/nv_tegra_release")
    print(f"  L4T release      : {release.splitlines()[0] if release else '(unknown)'}")

    # Total RAM, so you can judge whether batch sizes above 1 are sensible.
    meminfo = read_text("/proc/meminfo")
    for line in meminfo.splitlines():
        if line.startswith("MemTotal"):
            kb = int(line.split()[1])
            print(f"  total RAM        : {kb / 1024 / 1024:.1f} GB")
            break

    section("Power mode (read-only)")
    if shutil.which("nvpmodel"):
        try:
            result = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                                    text=True, timeout=10, check=False)
            output = (result.stdout or result.stderr).strip()
            print("  " + (output.replace("\n", "\n  ") if output else "(no output)"))
            if "permission" in output.lower() or result.returncode != 0:
                print("  note: reading the mode needs sudo on some images.")
                print("        You can pass it manually: benchmark_jetson.py --tag 15W")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  could not query nvpmodel ({exc})")
    else:
        print("  nvpmodel not found on PATH")

    section("Python packages")
    for name, required in [
        ("numpy", True),
        ("onnxruntime", False),
        ("torch", False),
        ("tensorrt", False),
    ]:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print(f"  {name:16s} {version}")
        except ImportError:
            marker = "MISSING (required)" if required else "not installed (optional)"
            print(f"  {name:16s} {marker}")
            if required:
                ok = False

    section("Backends the benchmark can use")
    backends = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print(f"  onnxruntime providers: {providers}")
        if "TensorrtExecutionProvider" in providers:
            backends.append("ONNX Runtime + TensorRT (fp16)")
        if "CUDAExecutionProvider" in providers:
            backends.append("ONNX Runtime + CUDA (fp32)")
        if "CPUExecutionProvider" in providers:
            backends.append("ONNX Runtime + ARM CPU (fp32)")
    except ImportError:
        print("  onnxruntime not available")

    try:
        import torch

        print(f"  torch CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  torch CUDA device    : {torch.cuda.get_device_name(0)}")
            backends += ["PyTorch CUDA fp32", "PyTorch CUDA fp16"]
        backends.append("PyTorch CPU fp32")
    except ImportError:
        print("  torch not available")

    if backends:
        print("\n  Will measure:")
        for backend in backends:
            print(f"    - {backend}")
    else:
        print("\n  No usable backend found. Install nothing — report back instead.")
        ok = False

    section("Model files")
    if not MODELS_DIR.exists():
        print(f"  MISSING: {MODELS_DIR}")
        print("  Did the bundle copy across completely?")
        ok = False
    else:
        for key in EXPECTED_MODELS:
            onnx = MODELS_DIR / f"{key}.onnx"
            weights = MODELS_DIR / f"{key}_weights.pt"
            parts = []
            if onnx.exists():
                parts.append(f"onnx {onnx.stat().st_size / 1048576:.1f} MB")
            if weights.exists():
                parts.append(f"torchscript {weights.stat().st_size / 1048576:.1f} MB")
            status = ", ".join(parts) if parts else "MISSING"
            print(f"  {key:14s} {status}")
            if not parts:
                ok = False

    section("Disk space in this folder")
    try:
        usage = shutil.disk_usage(HERE)
        print(f"  free: {usage.free / 2**30:.1f} GB of {usage.total / 2**30:.1f} GB")
        if usage.free < 2 * 2**30:
            print("  warning: TensorRT engine caches can need ~1 GB.")
    except OSError:
        pass

    print()
    print("=" * 66)
    if ok:
        print("READY. Next: python3 benchmark_jetson.py --quick")
    else:
        print("NOT READY — see the items marked MISSING above.")
    print("=" * 66)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
