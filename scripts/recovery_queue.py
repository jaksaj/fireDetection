"""Wait for the main sweep to finish, then run the recovery work sequentially.

Replaces an earlier shell-based watcher. Detached `nohup ... &` shell scripts
did not survive reliably here, and a mis-evaluated wait condition in a shell
loop risks the opposite failure -- starting the queue *immediately* and running
jobs concurrently, which is what caused a host-RAM OOM that killed a 2.6 h
training run. This version is explicit about both conditions and refuses to
start while the sweep is still alive.

Work performed, strictly one job at a time:

1. Recover iteration-4 seeds 43/44 from their saved weights. Both completed all
   50 training epochs and then died in metrics parsing, so the weights exist and
   retraining would waste ~2.7 h per seed.
2. Re-run the five iteration-3 seeds lost to the post-`wandb.finish()` logging
   crash.
3. Retrain iteration-4 seed 42, which was killed mid-training by the OOM and
   therefore has no trustworthy final checkpoint.

Usage::

    python scripts/recovery_queue.py                # wait, then run
    python scripts/recovery_queue.py --no-wait      # run immediately
    python scripts/recovery_queue.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import configure_logging

logger = logging.getLogger("recovery_queue")

PYTHON = sys.executable
SCRIPTS = PROJECT_ROOT / "scripts"

#: Minimum free RAM before starting a job. The machine has 15.9 GB total and a
#: YOLO run with 10 dataloader workers has already OOM-ed once here.
MIN_FREE_GB = 3.0


def sweep_running() -> bool:
    """True while any run_seeds.py / run_iteration*.py process is alive."""
    try:
        import psutil
    except ImportError:
        return False

    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if process.info["name"] not in {"python.exe", "python"}:
                continue
            cmdline = " ".join(process.info["cmdline"] or [])
            if "recovery_queue" in cmdline:
                continue
            if "run_seeds" in cmdline or "run_iteration" in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def wait_for_free_ram(minimum_gb: float = MIN_FREE_GB, timeout_s: int = 900) -> None:
    """Block until enough RAM is free, so a job does not start into an OOM."""
    import psutil

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free_gb = psutil.virtual_memory().available / 2**30
        if free_gb >= minimum_gb:
            return
        logger.info("Only %.1f GB free — waiting for %.1f GB.", free_gb, minimum_gb)
        time.sleep(30)
    logger.warning("Proceeding despite low free RAM after %ds.", timeout_s)


def build_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = [
        (
            "recover iteration4 seeds 43/44 from saved weights",
            [
                PYTHON, str(SCRIPTS / "recover_detection_metrics.py"),
                "--runs", "yolo26-dfire-seed43", "yolo26-dfire-seed44",
                "--split", "test",
            ],
        )
    ]
    for seed in (42, 43, 44, 45, 46):
        jobs.append(
            (
                f"iteration3 seed {seed}",
                [
                    PYTHON, str(SCRIPTS / "run_iteration3.py"),
                    "--seed", str(seed), "--tag", f"seed{seed}",
                ],
            )
        )
    jobs.append(
        (
            "iteration4 seed 42 retrain",
            [
                PYTHON, str(SCRIPTS / "run_iteration4.py"),
                "--seed", "42", "--tag", "seed42", "--skip-export",
            ],
        )
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential recovery queue.")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for the sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument(
        "--poll-seconds", type=int, default=60, help="Sweep polling interval."
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    jobs = build_jobs()
    logger.info("Recovery queue: %d jobs", len(jobs))
    for name, _ in jobs:
        logger.info("  - %s", name)
    if args.dry_run:
        return

    if not args.no_wait:
        logger.info("Waiting for the main sweep to finish...")
        waited = 0
        while sweep_running():
            time.sleep(args.poll_seconds)
            waited += args.poll_seconds
            if waited % 1800 == 0:
                logger.info("Still waiting (%.1f h).", waited / 3600)
        logger.info("Main sweep finished after %.1f h of waiting.", waited / 3600)

    results: list[tuple[str, int, float]] = []
    for index, (name, command) in enumerate(jobs, start=1):
        wait_for_free_ram()
        logger.info("[%d/%d] %s", index, len(jobs), name)
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        elapsed = (time.perf_counter() - start) / 60.0
        results.append((name, completed.returncode, elapsed))
        level = logging.INFO if completed.returncode == 0 else logging.ERROR
        logger.log(
            level,
            "%s -> exit %d (%.1f min)",
            name,
            completed.returncode,
            elapsed,
        )

    print(f"\n{'job':<50} {'exit':>5} {'minutes':>9}")
    print("-" * 68)
    for name, code, elapsed in results:
        print(f"{name:<50} {code:>5} {elapsed:>9.1f}")

    failed = [name for name, code, _ in results if code != 0]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\nRecovery complete. Regenerate tables: python scripts/make_tables.py")


if __name__ == "__main__":
    main()
