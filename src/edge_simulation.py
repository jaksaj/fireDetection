"""Post-training quantization and edge deployment simulation utilities."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import compute_multiclass_metrics

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")
CPU = torch.device("cpu")


def checkpoint_size_mb(checkpoint_path: Path) -> float:
    """Return on-disk checkpoint size in megabytes."""
    return checkpoint_path.stat().st_size / (1024 * 1024)


def save_state_dict_size_mb(model: nn.Module, destination: Path) -> float:
    """Persist ``state_dict`` and return file size in megabytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return checkpoint_size_mb(destination)


def apply_dynamic_ptq(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic post-training quantization (FP32 -> INT8).

    Quantized inference runs on CPU, simulating typical edge deployment targets.
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    logger.info("Dynamic PTQ applied — INT8 weights ready for CPU inference.")
    return quantized_model


def benchmark_inference_ms(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 1,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
) -> float:
    """
    Measure average inference latency in milliseconds for a single forward pass.

    Args:
        model: Eval-mode module.
        device: Target device for the benchmark.
        image_size: Spatial input resolution.
        batch_size: Batch dimension used for timing.
        warmup_iterations: Warmup runs before timing.
        benchmark_iterations: Timed runs averaged for the result.

    Returns:
        Mean latency in milliseconds per batch.
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(dummy_input)

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            model(dummy_input)
        elapsed_sec = time.perf_counter() - start

    latency_ms = (elapsed_sec / benchmark_iterations) * 1000.0
    logger.info(
        "Inference benchmark [%s, batch=%d]: %.3f ms/batch",
        device,
        batch_size,
        latency_ms,
    )
    return latency_ms


def evaluate_on_device(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Run classification metrics for a model on an arbitrary device."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    metrics = compute_multiclass_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["accuracy"] = metrics["accuracy"]
    return metrics


def run_edge_simulation(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: tuple[str, ...],
    checkpoint_dir: str | Path,
    image_size: int = 224,
    fp32_checkpoint_name: str = "best_model.pt",
) -> dict[str, Any]:
    """
    Compare FP32 GPU vs dynamically quantized INT8 CPU deployment characteristics.

    Logs model footprint, inference latency, and test-set accuracy for both variants.
    """
    checkpoint_dir = Path(checkpoint_dir)
    fp32_path = checkpoint_dir / fp32_checkpoint_name

    fp32_latency_ms = benchmark_inference_ms(model, DEVICE, image_size=image_size)
    fp32_metrics = evaluate_on_device(model, test_loader, DEVICE, class_names)
    fp32_size_mb = checkpoint_size_mb(fp32_path) if fp32_path.exists() else save_state_dict_size_mb(
        model, checkpoint_dir / "fp32_export.pt"
    )

    quantized_model = apply_dynamic_ptq(model)
    int8_path = checkpoint_dir / "quantized_int8.pt"
    int8_size_mb = save_state_dict_size_mb(quantized_model, int8_path)
    int8_latency_ms = benchmark_inference_ms(
        quantized_model, CPU, image_size=image_size
    )
    int8_metrics = evaluate_on_device(quantized_model, test_loader, CPU, class_names)

    compression_ratio = fp32_size_mb / max(int8_size_mb, 1e-6)
    speedup_ratio = fp32_latency_ms / max(int8_latency_ms, 1e-6)
    accuracy_delta = int8_metrics["accuracy"] - fp32_metrics["accuracy"]

    summary = {
        "fp32/model_size_mb": fp32_size_mb,
        "fp32/inference_ms": fp32_latency_ms,
        "fp32/test_accuracy": fp32_metrics["accuracy"],
        "fp32/test_loss": fp32_metrics["loss"],
        "fp32/test_f1_macro": fp32_metrics["f1_macro"],
        "int8/model_size_mb": int8_size_mb,
        "int8/inference_ms": int8_latency_ms,
        "int8/test_accuracy": int8_metrics["accuracy"],
        "int8/test_loss": int8_metrics["loss"],
        "int8/test_f1_macro": int8_metrics["f1_macro"],
        "edge/size_compression_ratio": compression_ratio,
        "edge/latency_speedup_ratio": speedup_ratio,
        "edge/accuracy_delta_int8_vs_fp32": accuracy_delta,
    }

    logger.info("Edge simulation complete: %s", summary)
    return summary
