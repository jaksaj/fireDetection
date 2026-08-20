"""Draw the PRISMA 2020 flow diagram for the literature review.

Counts are fixed inputs, verified against the review record, and every
arithmetic relation between them is asserted before anything is drawn: if a
count is ever edited inconsistently the script raises rather than emitting a
figure that quietly disagrees with itself.

Labels are Croatian, matching the thesis.

Usage::

    python scripts/make_prisma_figure.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.results import RESULTS_DIR
from src.utils import configure_logging

logger = logging.getLogger("make_prisma_figure")

IDENTIFIED = 461
SCREENED = 461
SOUGHT = 96
ASSESSED = 65
INCLUDED = 25

REMOVED_BEFORE_SCREENING = 0
EXCLUDED_SCREENING = 365
NOT_RETRIEVED = 31
EXCLUDED_FULLTEXT = 40

EXCLUSION_REASONS = [
    ("nema brojčanog pokazatelja učinkovitosti", 10),
    ("navodi brzinu, ali ne imenuje sklopovlje", 8),
    ("navodi samo jedan pokazatelj", 7),
    ("snimke iz zraka ili sa satelita", 5),
    ("detekcija vatre nije glavni predmet rada", 4),
    ("navodi samo pokazatelje neovisne o sklopovlju", 4),
    ("mjerenje nije pripisivo opisanom postavu", 1),
    ("nije isključivo vizualni pristup", 1),
]

MAIN_FILL = "#DCE6F1"
MAIN_EDGE = "#2E4A6B"
EXCL_FILL = "#F2E3D5"
EXCL_EDGE = "#9C5A28"


def check_arithmetic() -> None:
    """Refuse to draw a diagram whose counts do not reconcile."""
    assert EXCLUDED_SCREENING + SOUGHT == SCREENED, (
        f"{EXCLUDED_SCREENING} + {SOUGHT} != {SCREENED}"
    )
    assert NOT_RETRIEVED + ASSESSED == SOUGHT, f"{NOT_RETRIEVED} + {ASSESSED} != {SOUGHT}"
    assert EXCLUDED_FULLTEXT + INCLUDED == ASSESSED, (
        f"{EXCLUDED_FULLTEXT} + {INCLUDED} != {ASSESSED}"
    )
    assert IDENTIFIED - REMOVED_BEFORE_SCREENING == SCREENED, (
        f"{IDENTIFIED} - {REMOVED_BEFORE_SCREENING} != {SCREENED}"
    )
    total_reasons = sum(n for _, n in EXCLUSION_REASONS)
    assert total_reasons == EXCLUDED_FULLTEXT, (
        f"exclusion reasons sum to {total_reasons}, expected {EXCLUDED_FULLTEXT}"
    )
    logger.info("Arithmetic checks passed.")


def box(axis, x, y, w, h, text, fill, edge, fontsize=9, weight="normal"):
    axis.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2,
        )
    )
    axis.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, zorder=3,
        color="#1A1A1A", fontweight=weight, linespacing=1.45,
    )


def arrow(axis, start, end, colour="#2E4A6B"):
    axis.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.3, color=colour, zorder=1,
            shrinkA=0, shrinkB=0,
        )
    )


def main() -> None:
    configure_logging()
    check_arithmetic()

    fig, axis = plt.subplots(figsize=(11.5, 10))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.06)
    axis.axis("off")

    lx, lw = 0.04, 0.44           # left column
    rx, rw = 0.545, 0.415         # right column
    h = 0.105

    rows = [0.865, 0.685, 0.505, 0.325, 0.055]

    box(axis, lx, rows[0], lw, h,
        f"Zapisi identificirani pretraživanjem\nbaza podataka\n(n = {IDENTIFIED})",
        MAIN_FILL, MAIN_EDGE, weight="bold")
    box(axis, lx, rows[1], lw, h,
        f"Zapisi probrani po naslovu i sažetku\n(n = {SCREENED})", MAIN_FILL, MAIN_EDGE)
    box(axis, lx, rows[2], lw, h,
        f"Radovi zatraženi za dohvat\n(n = {SOUGHT})", MAIN_FILL, MAIN_EDGE)
    box(axis, lx, rows[3], lw, h,
        f"Radovi ocijenjeni čitanjem\ncjelovitog teksta\n(n = {ASSESSED})", MAIN_FILL, MAIN_EDGE)
    box(axis, lx, rows[4], lw, h,
        f"Radovi uključeni u pregled\n(n = {INCLUDED})", MAIN_FILL, MAIN_EDGE, weight="bold")

    box(axis, rx, rows[0], rw, h,
        f"Uklonjeni prije probira\n(n = {REMOVED_BEFORE_SCREENING})", EXCL_FILL, EXCL_EDGE)
    box(axis, rx, rows[1], rw, h,
        f"Isključeni pri probiru\n(n = {EXCLUDED_SCREENING})", EXCL_FILL, EXCL_EDGE)
    box(axis, rx, rows[2], rw, h,
        f"Radovi koji nisu dohvaćeni\n(n = {NOT_RETRIEVED})", EXCL_FILL, EXCL_EDGE)

    reasons = "\n".join(f"• {name}: {n}" for name, n in EXCLUSION_REASONS)
    reason_h = 0.30
    box(axis, rx, rows[3] + h - reason_h, rw, reason_h,
        f"Isključeni nakon čitanja cjelovitog teksta\n(n = {EXCLUDED_FULLTEXT})\n\n{reasons}",
        EXCL_FILL, EXCL_EDGE, fontsize=7.6)

    # vertical flow
    for upper, lower in zip(rows[:-1], rows[1:]):
        arrow(axis, (lx + lw / 2, upper), (lx + lw / 2, lower + h))

    # horizontal exclusions, each leaving the box it belongs to
    for row in rows[:3]:
        arrow(axis, (lx + lw, row + h / 2), (rx, row + h / 2), colour=EXCL_EDGE)
    arrow(axis, (lx + lw, rows[3] + h / 2), (rx, rows[3] + h / 2), colour=EXCL_EDGE)

    axis.text(0.5, 1.05, "PRISMA 2020 dijagram toka", ha="center", va="top",
              fontsize=13, fontweight="bold", color="#1A1A1A")

    FIGURES = RESULTS_DIR / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"prisma_flow.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", FIGURES / "prisma_flow.png")

    print(f"identified {IDENTIFIED} -> screened {SCREENED} "
          f"-> sought {SOUGHT} -> assessed {ASSESSED} -> included {INCLUDED}")
    print(f"exclusions: {EXCLUDED_SCREENING} / {NOT_RETRIEVED} / {EXCLUDED_FULLTEXT} "
          f"({len(EXCLUSION_REASONS)} reasons summing to {sum(n for _, n in EXCLUSION_REASONS)})")


if __name__ == "__main__":
    main()
