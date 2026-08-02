"""Run every method across multiple seeds, unattended.

Every headline number in this project came from a single unrepeated run. With no
variance estimate, differences like Iteration 2's 89.16% against Iteration 3's
90.25% cannot be called real -- they may be entirely within run-to-run noise,
and "did you repeat it?" is the first question anyone asks of a one-point claim.

Measured single-seed wall-clock on the project's RTX 3060:

    iteration1   10.3 min
    iteration2   16.5 min
    iteration3   24.2 min
    iteration4  164.7 min
    iteration5   72.5 min
    ------------------------
    total         4.8 h per seed

So repeats are cheap in wall-clock terms and expensive only in patience. The
default schedule runs the four fast methods at more seeds than the slow detector,
which is the right allocation of a fixed compute budget; the asymmetry is
recorded per run in results/metrics.csv via the `seed` column and must be stated
in the thesis rather than glossed over.

Each run is launched as a **separate process** so that no state leaks between
runs: no accumulated CUDA fragmentation, no RNG advanced by a previous run, no
cuDNN autotuning cache warmed on a different architecture.

Usage::

    python scripts/run_seeds.py                                  # default schedule
    python scripts/run_seeds.py --seeds 42 43 44 45 46
    python scripts/run_seeds.py --methods iteration1 --seeds 1 2 3
    python scripts/run_seeds.py --dry-run
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

logger = logging.getLogger("run_seeds")

#: Approximate single-run cost, from the recorded W&B runtimes. Used only to
#: print a schedule estimate before committing to an overnight sweep.
APPROX_MINUTES = {
    "iteration1": 10.3,
    "iteration2": 16.5,
    "iteration3": 24.2,
    "iteration4": 164.7,
    "iteration5": 72.5,
}

FAST_METHODS = ["iteration1", "iteration2", "iteration3", "iteration5"]
ALL_METHODS = ["iteration1", "iteration2", "iteration3", "iteration4", "iteration5"]


def build_command(method: str, seed: int) -> list[str]:
    script = PROJECT_ROOT / "scripts" / f"run_{method}.py"
    command = [sys.executable, str(script), "--seed", str(seed)]
    if method != "iteration4":
        command += ["--tag", f"seed{seed}"]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed sweep across methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        help="Methods to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seeds for the fast methods.",
    )
    parser.add_argument(
        "--detector-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for iteration 4 (defaults to the first 3 of --seeds, since it is ~2.7 h/run).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Keep going if one run fails (default).",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    detector_seeds = args.detector_seeds or args.seeds[:3]

    plan: list[tuple[str, int]] = []
    for method in args.methods:
        seeds = detector_seeds if method == "iteration4" else args.seeds
        plan.extend((method, seed) for seed in seeds)

    estimate = sum(APPROX_MINUTES.get(method, 30.0) for method, _ in plan)
    logger.info(
        "Planned: %d runs across %d methods. Estimated %.1f h of GPU time.",
        len(plan),
        len(args.methods),
        estimate / 60.0,
    )
    for method, seed in plan:
        logger.info("  %-12s seed=%d  (~%.0f min)", method, seed, APPROX_MINUTES.get(method, 30))

    if args.dry_run:
        return

    started = time.perf_counter()
    failures: list[tuple[str, int, int]] = []

    for index, (method, seed) in enumerate(plan, start=1):
        command = build_command(method, seed)
        logger.info("[%d/%d] %s", index, len(plan), " ".join(command))
        run_start = time.perf_counter()
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        elapsed = (time.perf_counter() - run_start) / 60.0

        if completed.returncode != 0:
            logger.error("FAILED %s seed=%d (exit %d)", method, seed, completed.returncode)
            failures.append((method, seed, completed.returncode))
            if not args.continue_on_error:
                break
        else:
            logger.info("OK %s seed=%d in %.1f min", method, seed, elapsed)

    total_hours = (time.perf_counter() - started) / 3600.0
    logger.info("Sweep finished in %.2f h — %d/%d succeeded.",
                total_hours, len(plan) - len(failures), len(plan))

    if failures:
        logger.warning("Failures:")
        for method, seed, code in failures:
            logger.warning("  %s seed=%d exit=%d", method, seed, code)

    logger.info("Aggregate with: python scripts/make_tables.py")


if __name__ == "__main__":
    main()
