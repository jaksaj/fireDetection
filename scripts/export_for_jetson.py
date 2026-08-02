"""Build a self-contained Jetson benchmark bundle. Run this on the WORKSTATION.

Exports every trained model into two device-portable formats and assembles them
with the device-side scripts into ``jetson/`` plus a tarball ready to copy.

Why two formats
---------------
* **ONNX** -- consumed by ONNX Runtime with the TensorRT, CUDA and CPU execution
  providers. This is the primary path and needs no PyTorch on the device.
* **TorchScript** -- a self-describing archive that carries its own architecture,
  so PyTorch benchmarking on the Jetson needs neither ``torchvision`` nor
  ``ultralytics`` nor this repository's source. That matters because a borrowed
  device should not have packages installed on it.

Neither format requires the checkpoint files, the training code, or the dataset,
so the bundle stays small and the device stays untouched.

Usage::

    python scripts/export_for_jetson.py
    python scripts/export_for_jetson.py --models iteration1 iteration4
    python scripts/export_for_jetson.py --no-archive
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils import configure_logging

logger = logging.getLogger("export_for_jetson")

CHECKPOINTS = PROJECT_ROOT / "checkpoints"
BUNDLE_DIR = PROJECT_ROOT / "jetson"
MODELS_DIR = BUNDLE_DIR / "models"

SPECS = {
    "iteration1": {
        "name": "FireCNN (binary classification)",
        "builder": "firecnn",
        "input": (1, 3, 224, 224),
        "checkpoint": CHECKPOINTS / "iteration1" / "best_model.pt",
    },
    "iteration2": {
        "name": "MobileNetV3-Small (4-class)",
        "builder": "mobilenet",
        "input": (1, 3, 224, 224),
        "checkpoint": CHECKPOINTS / "iteration2" / "best_model.pt",
    },
    "iteration3": {
        "name": "MobileNetV3-Small robust (4-class)",
        "builder": "mobilenet",
        "input": (1, 3, 224, 224),
        "checkpoint": CHECKPOINTS / "iteration3" / "best_model.pt",
    },
    "iteration4": {
        "name": "YOLO26n (detection)",
        "builder": "yolo",
        "input": (1, 3, 640, 640),
        "checkpoint": CHECKPOINTS / "iteration4" / "yolo26-dfire" / "weights" / "best.pt",
    },
    "iteration5": {
        "name": "LightweightUNet (segmentation)",
        "builder": "unet",
        "input": (1, 3, 256, 256),
        "checkpoint": CHECKPOINTS / "iteration5" / "best_model.pt",
    },
}


def build_model(builder: str, checkpoint: Path) -> torch.nn.Module:
    """Instantiate on CPU with trained weights loaded."""
    if builder == "firecnn":
        from src.model import FireCNN

        model = FireCNN(device="cpu")
    elif builder == "mobilenet":
        from src.model import MobileNetV3FireClassifier

        model = MobileNetV3FireClassifier(num_classes=4, pretrained=False, device="cpu")
    elif builder == "unet":
        from src.model_segmentation import LightweightUNet

        model = LightweightUNet(num_classes=3, device="cpu")
    elif builder == "yolo":
        from ultralytics import YOLO

        # `.model` is the bare nn.Module. Export that rather than the Ultralytics
        # wrapper, so the device measures the network and not the surrounding
        # file-loading, letterboxing and NMS pipeline.
        yolo = YOLO(str(checkpoint))
        return yolo.model.float().eval().cpu()
    else:
        raise ValueError(f"Unknown builder {builder!r}")

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
    else:
        logger.warning("Checkpoint missing (%s) — exporting untrained weights.", checkpoint)

    return model.eval()


def export_onnx(model: torch.nn.Module, shape: tuple, destination: Path) -> bool:
    dummy = torch.randn(*shape)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            do_constant_folding=True,
            # Static batch. A fixed shape lets TensorRT build one optimised
            # engine; the benchmark measures batch 1, which is the deployment
            # case for a live camera feed.
            dynamic_axes=None,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("ONNX export failed: %s", exc)
        return False


class FirstOutput(torch.nn.Module):
    """Return only the first output tensor of a wrapped model.

    Ultralytics detection models return a mixed structure (a tensor plus a list
    of feature-map tensors). `torch.jit.trace` rejects that with "Dictionary
    inputs to traced functions must have consistent type", so the raw model
    cannot be scripted. Only the first element is the detection output, and
    latency is unaffected by discarding the rest -- the full forward pass still
    executes.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out


def export_torchscript(model: torch.nn.Module, shape: tuple, destination: Path) -> bool:
    dummy = torch.randn(*shape)
    for candidate, label in ((model, "direct"), (FirstOutput(model).eval(), "first-output")):
        try:
            with torch.no_grad():
                traced = torch.jit.trace(candidate, dummy, strict=False)
            traced.save(str(destination))
            if label != "direct":
                logger.info("  (traced via %s wrapper)", label)
            return True
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    logger.error("TorchScript export failed: %s", last_error)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Jetson benchmark bundle.")
    parser.add_argument("--models", nargs="+", default=sorted(SPECS), choices=sorted(SPECS))
    parser.add_argument("--no-archive", action="store_true", help="Skip the .tar.gz.")
    parser.add_argument("--skip-torchscript", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[str, str, float]] = []

    for key in args.models:
        spec = SPECS[key]
        logger.info("=== %s ===", key)
        try:
            model = build_model(spec["builder"], spec["checkpoint"])
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not build %s: %s", key, exc)
            continue

        onnx_file = MODELS_DIR / f"{key}.onnx"
        if export_onnx(model, spec["input"], onnx_file):
            size = onnx_file.stat().st_size / 1048576
            summary.append((key, "onnx", size))
            logger.info("  ONNX        %.2f MB", size)

        if not args.skip_torchscript:
            ts_file = MODELS_DIR / f"{key}_weights.pt"
            if export_torchscript(model, spec["input"], ts_file):
                size = ts_file.stat().st_size / 1048576
                summary.append((key, "torchscript", size))
                logger.info("  TorchScript %.2f MB", size)

        del model

    if not summary:
        logger.error("Nothing exported.")
        return

    # A manifest lets the device-side scripts (and you, later) confirm the bundle
    # matches the models the thesis reports.
    manifest = MODELS_DIR / "MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Fire detection models exported for Jetson benchmarking\n")
        handle.write(f"Exported by: {Path(__file__).name}\n")
        handle.write(f"torch: {torch.__version__}\n\n")
        for key, fmt, size in summary:
            handle.write(f"{key:14s} {fmt:12s} {size:8.2f} MB  {SPECS[key]['name']}\n")

    total = sum(size for _, _, size in summary)
    logger.info("Bundle models: %d files, %.1f MB total", len(summary), total)

    if not args.no_archive:
        archive = PROJECT_ROOT / "jetson_bundle.tar.gz"
        # Exclude results/ so a rebuild never ships stale measurements.
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(BUNDLE_DIR.rglob("*")):
                if not path.is_file():
                    # `tar.add` recurses into directories by default, so adding
                    # both the directory and each file underneath it stores
                    # every payload twice and doubles the archive.
                    continue
                if "results" in path.relative_to(BUNDLE_DIR).parts:
                    continue
                if path.name.endswith(".pyc") or "__pycache__" in path.parts:
                    continue
                tar.add(
                    path,
                    arcname=str(Path("jetson") / path.relative_to(BUNDLE_DIR)),
                    recursive=False,
                )
        logger.info("Archive: %s (%.1f MB)", archive, archive.stat().st_size / 1048576)

    print(f"\n{'model':<14} {'format':<13} {'size MB':>9}")
    print("-" * 40)
    for key, fmt, size in summary:
        print(f"{key:<14} {fmt:<13} {size:>9.2f}")
    print(f"\nBundle ready: {BUNDLE_DIR}")
    if not args.no_archive:
        print("Copy to the Jetson with:")
        print("  scp jetson_bundle.tar.gz <user>@<jetson-host>:~/")


if __name__ == "__main__":
    main()
