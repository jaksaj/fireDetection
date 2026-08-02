"""Merge measurements returned from the Jetson into the workstation results.

Run this on the WORKSTATION after copying `results/` back from the device:

    scp -r USER@JETSON_HOST:~/jetson/results ./jetson_results
    python scripts/merge_jetson_results.py --input jetson_results
    python scripts/make_tables.py

The device CSV carries one extra column (`power_mode`) that the workstation
schema does not have. Rather than dropping it — it is exactly the axis that makes
several operating points out of one board — it is folded into the `notes` column
and also used to disambiguate `bench_device`, so `jetson-cuda @ 15W` and
`jetson-cuda @ MAXN` stay separate rows in every downstream table.

The merge is idempotent: re-running with the same input will not duplicate rows,
because each row is keyed on
(model, bench_device, backend, precision, batch size, host, power mode).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.benchmark import BENCHMARK_FIELDS
from src.results import RESULTS_DIR
from src.utils import configure_logging

logger = logging.getLogger("merge_jetson_results")

KEY_COLUMNS = [
    "model_key",
    "bench_device",
    "backend",
    "precision",
    "batch_size",
    "host",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Jetson results into results/.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "jetson_results",
        help="Folder copied back from the device (contains jetson_benchmarks.csv).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=RESULTS_DIR / "benchmarks.csv",
        help="Workstation benchmark CSV to merge into.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    source_csv = args.input / "jetson_benchmarks.csv"
    if not source_csv.exists():
        logger.error("Not found: %s", source_csv)
        logger.error("Expected the folder copied back from the Jetson.")
        return

    jetson = pd.read_csv(source_csv)
    logger.info("Read %d rows from %s", len(jetson), source_csv)

    if jetson.empty:
        logger.error("The Jetson CSV is empty — nothing to merge.")
        return

    # Preserve the power mode: append it to bench_device so downstream grouping
    # treats each operating point separately, and keep it in notes for the table.
    if "power_mode" in jetson.columns:
        mode = jetson["power_mode"].fillna("").astype(str).str.strip()
        has_mode = mode.ne("")
        jetson.loc[has_mode, "bench_device"] = (
            jetson.loc[has_mode, "bench_device"] + "@" + mode[has_mode]
        )
        jetson["notes"] = (
            jetson.get("notes", "").fillna("").astype(str)
            + jetson["power_mode"].fillna("").astype(str).radd("; power_mode=").where(has_mode, "")
        ).str.strip("; ")
        jetson = jetson.drop(columns=["power_mode"])

    for column in BENCHMARK_FIELDS:
        if column not in jetson.columns:
            jetson[column] = "" if jetson.dtypes.get(column) == object else 0
    jetson = jetson[BENCHMARK_FIELDS]

    if args.target.exists():
        existing = pd.read_csv(args.target)
        before = len(existing)
        combined = pd.concat([existing, jetson], ignore_index=True)
    else:
        before = 0
        combined = jetson

    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    added = len(combined) - before

    devices = sorted(jetson["bench_device"].unique())
    logger.info("Jetson devices/modes merged: %s", devices)

    print(f"\n{'model':<12} {'device':<22} {'backend':<26} {'prec':<6} {'median ms':>10} {'FPS':>8}")
    print("-" * 92)
    for _, row in jetson.sort_values(["model_key", "bench_device"]).iterrows():
        print(
            f"{row['model_key']:<12} {str(row['bench_device']):<22} {str(row['backend']):<26} "
            f"{str(row['precision']):<6} {row['latency_ms_median']:>10.3f} {row['fps']:>8.1f}"
        )

    if args.dry_run:
        print(f"\n[dry run] would write {len(combined)} rows ({added:+d}) to {args.target}")
        return

    if args.target.exists():
        backup = args.target.with_suffix(".csv.bak")
        shutil.copy2(args.target, backup)
        logger.info("Backed up existing results to %s", backup.name)

    combined.to_csv(args.target, index=False)
    logger.info("Wrote %d rows (%+d) to %s", len(combined), added, args.target)

    environment = args.input / "environment.json"
    if environment.exists():
        destination = RESULTS_DIR / "jetson_environment.json"
        shutil.copy2(environment, destination)
        info = json.loads(environment.read_text(encoding="utf-8"))
        logger.info(
            "Device: %s | JetPack: %s | torch: %s",
            info.get("device_model", "?"),
            info.get("jetpack", "?"),
            info.get("torch", "?"),
        )
        logger.info("Saved device environment to %s (needed for Methodology).", destination.name)
    else:
        logger.warning("No environment.json found — the thesis needs the device spec.")

    print("\nNext: python scripts/make_tables.py")


if __name__ == "__main__":
    main()
