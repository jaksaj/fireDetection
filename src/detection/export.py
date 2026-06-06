"""ONNX / TensorRT export and edge FPS benchmarking for YOLO26 detectors."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEVICE = 0


class YOLOEdgeExporter:
    """Export trained YOLO weights to edge formats and benchmark inference FPS."""

    SUPPORTED_FORMATS = ("onnx", "engine", "openvino")

    def __init__(
        self,
        weights_path: str | Path,
        export_dir: str | Path = "checkpoints/iteration4/exports",
        imgsz: int = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.export_dir = Path(export_dir)
        self.imgsz = imgsz
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(self.weights_path))

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        logger.info("YOLOEdgeExporter loaded weights from %s", self.weights_path)

    def export(self, export_format: str) -> Path:
        """
        Export the model to ``onnx``, ``engine`` (TensorRT), or ``openvino``.

        Returns:
            Path to the exported artifact.
        """
        export_format = export_format.lower()
        if export_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported export format '{export_format}'. "
                f"Choose from {self.SUPPORTED_FORMATS}."
            )

        logger.info("Exporting YOLO26 weights to %s...", export_format)
        export_path = self.model.export(
            format=export_format,
            imgsz=self.imgsz,
            device=DEVICE,
            simplify=True,
        )
        destination = Path(export_path)
        logger.info("Export complete: %s (%.2f MB)", destination, self._file_size_mb(destination))
        return destination

    def benchmark_fps(
        self,
        source: str | Path,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        conf: float = 0.25,
    ) -> dict[str, Any]:
        """
        Measure end-to-end inference throughput on a CUDA device.

        Args:
            source: Image, directory, or video path for benchmarking.
            warmup_iterations: Warmup passes before timing.
            benchmark_iterations: Timed inference passes.
            conf: Detection confidence threshold.

        Returns:
            Dictionary with FPS, latency_ms, and iteration counts.
        """
        source_path = Path(source)
        if source_path.is_dir():
            from src.dfire_labels import IMAGE_EXTENSIONS

            image_files = [
                path
                for path in source_path.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise FileNotFoundError(f"No images found for FPS benchmark in {source_path}")
            benchmark_source = str(image_files[0])
        else:
            benchmark_source = str(source_path)

        for _ in range(warmup_iterations):
            self.model.predict(
                source=benchmark_source,
                device=DEVICE,
                conf=conf,
                verbose=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            self.model.predict(
                source=benchmark_source,
                device=DEVICE,
                conf=conf,
                verbose=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_sec = time.perf_counter() - start
        latency_ms = (elapsed_sec / benchmark_iterations) * 1000.0
        fps = benchmark_iterations / elapsed_sec

        summary = {
            "edge/inference_ms": latency_ms,
            "edge/fps": fps,
            "edge/benchmark_iterations": benchmark_iterations,
            "edge/benchmark_source": benchmark_source,
        }
        logger.info(
            "FPS benchmark [CUDA:%d]: %.1f FPS (%.2f ms/frame)",
            DEVICE,
            fps,
            latency_ms,
        )
        return summary

    def run_full_export_pipeline(
        self,
        export_formats: tuple[str, ...],
        benchmark_source: str | Path,
        target_edge_fps: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Export to all requested formats and benchmark the native PyTorch weights.

        Logs model size, FPS, and optional target FPS comparison for W&B.
        """
        summary: dict[str, Any] = {
            "fp32/weights_path": str(self.weights_path),
            "fp32/model_size_mb": self._file_size_mb(self.weights_path),
        }

        fps_metrics = self.benchmark_fps(benchmark_source)
        summary.update(fps_metrics)

        if target_edge_fps is not None:
            summary["edge/target_fps"] = target_edge_fps
            summary["edge/meets_target_fps"] = fps_metrics["edge/fps"] >= target_edge_fps

        for export_format in export_formats:
            try:
                artifact = self.export(export_format)
                key_prefix = export_format
                summary[f"{key_prefix}/path"] = str(artifact)
                summary[f"{key_prefix}/model_size_mb"] = self._file_size_mb(artifact)
            except Exception as exc:
                logger.error("Export to %s failed: %s", export_format, exc)
                summary[f"{export_format}/error"] = str(exc)

        return summary

    @staticmethod
    def _file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)
