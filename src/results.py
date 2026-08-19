"""Machine-readable result persistence.

Every number that reaches the thesis must come from a file that can be
regenerated, diffed, and traced back to the exact code that produced it.
Before this module existed, results lived only in the Weights & Biases web UI
and were transcribed into Markdown by hand.

Two sinks are written for every recorded result:

- ``results/metrics.csv`` -- one flat row per (run, split, metric). Long format
  rather than wide, because the metric set differs per task (accuracy/F1 for
  classification, mAP for detection, IoU/Dice for segmentation) and a wide
  table would be mostly empty columns.
- ``results/runs/<run_id>.json`` -- the full record for one run: config,
  environment, git state, and every metric, nested.

Both are append-only with respect to runs: re-running an experiment adds rows,
it never silently overwrites an earlier result.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
RUNS_DIR = RESULTS_DIR / "runs"

CSV_FIELDS = [
    "run_id",
    "method",
    "backbone",
    "seed",
    "split",
    "metric",
    "value",
    "timestamp",
    "git_sha",
    "git_dirty",
]


def git_state() -> tuple[str, bool]:
    """Return (short SHA, dirty flag) for the working tree, or ("unknown", True)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", True


def environment_info() -> dict[str, Any]:
    """Capture the hardware/software environment for the Methodology chapter."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        pass

    return info


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Flatten nested metric dicts into dotted keys, keeping only numbers."""
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub, out)
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    # Strings and everything else are kept in the JSON record but are not
    # valid CSV metric rows, so they are dropped here on purpose.


def record_run(
    method: str,
    metrics: dict[str, Any],
    *,
    seed: int | None = None,
    backbone: str = "",
    split: str = "test",
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Persist one run's metrics to ``results/metrics.csv`` and a JSON record.

    Args:
        method: Identifier for the experiment (e.g. ``"iteration2"``).
        metrics: Metric dict; may be nested. Non-numeric leaves are kept in the
            JSON record but omitted from the CSV.
        seed: Random seed used for the run, if any.
        backbone: Architecture identifier, for the backbone-comparison axis.
        split: Which data split these metrics describe.
        config: The resolved experiment config, archived into the JSON record.
        extra: Any additional metadata to archive.
        run_id: Explicit run identifier; generated from method/seed/time if omitted.

    Returns:
        Path to the JSON record that was written.
    """
    # Smoke tests run real pipelines for one epoch to prove they execute. Their
    # metrics are meaningless and must never reach results/metrics.csv: a
    # 1-epoch iteration-1 run scored 0.869 and, mixed into the seeded rows,
    # moved the reported mean from 92.80% +/- 0.20% to 91.8% +/- 2.5%. The
    # smoke-test harness sets this variable so its runs are diverted.
    if os.environ.get("FIREDETECT_SMOKE_RUN") == "1":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        marker = RESULTS_DIR / "smoke_runs.log"
        with marker.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {method} seed={seed}\n")
        logger.info("Smoke run — metrics NOT recorded to results/metrics.csv.")
        return marker

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha, dirty = git_state()

    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        seed_part = f"-seed{seed}" if seed is not None else ""
        run_id = f"{method}{seed_part}-{stamp}"

    flat: dict[str, float] = {}
    _flatten("", metrics, flat)

    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for metric_name, value in sorted(flat.items()):
            writer.writerow(
                {
                    "run_id": run_id,
                    "method": method,
                    "backbone": backbone,
                    "seed": "" if seed is None else seed,
                    "split": split,
                    "metric": metric_name,
                    "value": value,
                    "timestamp": timestamp,
                    "git_sha": sha,
                    "git_dirty": int(dirty),
                }
            )

    record = {
        "run_id": run_id,
        "method": method,
        "backbone": backbone,
        "seed": seed,
        "split": split,
        "timestamp": timestamp,
        "git_sha": sha,
        "git_dirty": dirty,
        "environment": environment_info(),
        "config": config or {},
        "metrics": metrics,
        "extra": extra or {},
    }

    json_path = RUNS_DIR / f"{run_id}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)

    logger.info(
        "Recorded %d metrics for %s (seed=%s, split=%s) -> %s",
        len(flat),
        method,
        seed,
        split,
        json_path.name,
    )
    return json_path


def append_rows(filename: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    """
    Append arbitrary rows to a CSV under ``results/``, writing a header if new.

    Used by the benchmark harness and the corruption/common-evaluation scripts,
    which have their own row schemas rather than the long metric format.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logger.info("Appended %d rows to %s", len(rows), path)
    return path
