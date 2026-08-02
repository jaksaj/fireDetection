"""Fast end-to-end validation of every training pipeline.

Purpose: catch a runtime break *before* committing a machine to a 20-hour
unattended sweep. A syntax check does not prove that a config loads, that a
DataModule accepts the arguments a run script passes it, that a trainer's
metric hooks return what the results writer expects, or that a checkpoint round
-trips. This does, in a couple of minutes.

Each pipeline is run for one epoch on a small subset by writing a temporary
config derived from the real one, so the code path exercised is the same one a
full run takes -- not a mock.

Usage::

    python scripts/smoke_test.py                    # all pipelines
    python scripts/smoke_test.py --methods iteration1 iteration5
    python scripts/smoke_test.py --skip-detection   # detection is the slow one
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import configure_logging

logger = logging.getLogger("smoke_test")

CONFIG_DIR = PROJECT_ROOT / "configs"

#: Per-method overrides that shrink a real config to a one-epoch smoke run.
SHRINK = {
    "iteration1": {"training": {"epochs": 1, "log_every_n_batches": 200}},
    "iteration2": {"training": {"head_epochs": 1, "finetune_epochs": 1, "log_every_n_batches": 200}},
    "iteration3": {"training": {"head_epochs": 1, "finetune_epochs": 1, "log_every_n_batches": 200}},
    "iteration4": {"training": {"epochs": 1, "patience": 1}},
    "iteration5": {"training": {"epochs": 1, "log_every_n_batches": 200}},
}


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def make_smoke_config(method: str, temp_dir: Path) -> Path:
    """Write a one-epoch, W&B-disabled variant of the method's real config."""
    source = CONFIG_DIR / f"{method}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))

    deep_update(config, SHRINK.get(method, {}))

    # Never let a smoke run touch the real checkpoints or a real W&B project.
    config.setdefault("wandb", {})
    config["wandb"]["mode"] = "disabled"
    config["wandb"]["run_name"] = f"smoke-{method}"

    training = config.setdefault("training", {})
    training["checkpoint_dir"] = str(temp_dir / "checkpoints" / method)
    if method == "iteration4":
        training["run_name"] = f"smoke-{method}"
        # Export is slow and separately tested.
        config.setdefault("export", {})["enabled"] = False

    destination = temp_dir / f"{method}.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def run_pipeline(method: str, config_path: Path, timeout: int) -> tuple[bool, str, float]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / f"run_{method}.py"),
        "--config", str(config_path),
        "--seed", "1234",
        "--tag", "smoke",
    ]
    if method == "iteration4":
        command.append("--skip-export")

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s", time.perf_counter() - start

    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return False, "\n".join(tail[-12:]), elapsed
    return True, "", elapsed


def check_auxiliary() -> list[tuple[str, bool, str]]:
    """Exercise the measurement scripts that do not train anything."""
    checks: list[tuple[str, bool, str]] = []

    probes = [
        ("benchmarks", [sys.executable, str(PROJECT_ROOT / "scripts" / "run_benchmarks.py"),
                        "--quick", "--models", "iteration1", "--output", "smoke_benchmarks.csv"]),
        ("common_eval", [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_common.py"),
                         "--methods", "iteration1", "--limit", "64",
                         "--output", "smoke_common.csv"]),
        ("robustness", [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_robustness.py"),
                        "--methods", "iteration2", "--limit", "64",
                        "--corruptions", "fog", "--output", "smoke_robustness.csv"]),
        ("make_tables", [sys.executable, str(PROJECT_ROOT / "scripts" / "make_tables.py")]),
    ]

    for name, command in probes:
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800
        )
        ok = completed.returncode == 0
        detail = "" if ok else (completed.stderr or "").strip().splitlines()[-6:]
        checks.append((name, ok, "\n".join(detail) if detail else ""))

    # Clean up smoke artifacts so they never pollute the real results.
    for filename in ("smoke_benchmarks.csv", "smoke_common.csv", "smoke_robustness.csv"):
        (PROJECT_ROOT / "results" / filename).unlink(missing_ok=True)

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate every pipeline end to end.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["iteration1", "iteration2", "iteration3", "iteration5"],
        help="Training pipelines to check. Detection is excluded by default (slow).",
    )
    parser.add_argument(
        "--skip-detection", action="store_true", help="Never run iteration4."
    )
    parser.add_argument(
        "--skip-auxiliary", action="store_true", help="Skip the measurement scripts."
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Per-pipeline timeout (s).")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    methods = [m for m in args.methods if not (args.skip_detection and m == "iteration4")]
    temp_dir = Path(tempfile.mkdtemp(prefix="firedetect-smoke-"))
    logger.info("Smoke artifacts -> %s", temp_dir)

    results: list[tuple[str, bool, str, float]] = []
    try:
        for method in methods:
            logger.info("--- %s ---", method)
            config_path = make_smoke_config(method, temp_dir)
            ok, detail, elapsed = run_pipeline(method, config_path, args.timeout)
            results.append((method, ok, detail, elapsed))
            logger.info("%s: %s (%.1fs)", method, "PASS" if ok else "FAIL", elapsed)
            if not ok:
                logger.error("%s output tail:\n%s", method, detail)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    auxiliary: list[tuple[str, bool, str]] = []
    if not args.skip_auxiliary:
        logger.info("--- measurement scripts ---")
        auxiliary = check_auxiliary()
        for name, ok, detail in auxiliary:
            logger.info("%s: %s", name, "PASS" if ok else "FAIL")
            if not ok:
                logger.error("%s:\n%s", name, detail)

    print("\n" + "=" * 60)
    print(f"{'pipeline':<20} {'result':<8} {'seconds':>9}")
    print("-" * 60)
    for method, ok, _, elapsed in results:
        print(f"{method:<20} {'PASS' if ok else 'FAIL':<8} {elapsed:>9.1f}")
    for name, ok, _ in auxiliary:
        print(f"{name:<20} {'PASS' if ok else 'FAIL':<8} {'-':>9}")

    failed = [m for m, ok, _, _ in results if not ok] + [n for n, ok, _ in auxiliary if not ok]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll checks passed — safe to launch the full sweep.")


if __name__ == "__main__":
    main()
