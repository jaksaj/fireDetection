"""Generate every thesis table and figure from the CSVs under results/.

Nothing in this script computes a result. It only reads what the measurement
scripts wrote and renders it. That separation is the point: every number in the
thesis becomes traceable to a file that a measurement script produced, and
regenerating the whole figure set after a rerun is one command instead of a
round of manual transcription.

Inputs (each optional -- missing ones are skipped with a warning):
    results/benchmarks.csv     -- scripts/run_benchmarks.py
    results/common_eval.csv    -- scripts/evaluate_common.py
    results/robustness.csv     -- scripts/evaluate_robustness.py
    results/metrics.csv        -- every training run, via src/results.record_run
    results/dataset_stats.csv  -- scripts/dataset_stats.py

Outputs land in results/figures/ (PNG + PDF) and results/tables/ (Markdown +
LaTeX).

Usage::

    python scripts/make_tables.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.results import RESULTS_DIR
from src.utils import configure_logging

logger = logging.getLogger("make_tables")

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Consistent identity per method across every figure, so a reader can track one
# method through the whole Results chapter by colour and marker alone.
METHOD_STYLE = {
    "iteration1": {"label": "FireCNN (binary cls)", "color": "#4C72B0", "marker": "o"},
    "iteration2": {"label": "MobileNetV3-S (4-cls)", "color": "#DD8452", "marker": "s"},
    "iteration3": {"label": "MobileNetV3-S robust", "color": "#55A868", "marker": "^"},
    "iteration4": {"label": "YOLO26n (detection)", "color": "#C44E52", "marker": "D"},
    "iteration5": {"label": "U-Net (segmentation)", "color": "#8172B3", "marker": "v"},
}

REALTIME_FPS = 25.0
MONITORING_FPS = 10.0


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if not path.exists():
        logger.warning("Missing %s — skipping the outputs that depend on it.", name)
        return None
    frame = pd.read_csv(path)
    logger.info("Loaded %s (%d rows)", name, len(frame))
    return frame


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s", stem)


def save_table(frame: pd.DataFrame, stem: str, caption: str, float_format: str = "%.4f") -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"{stem}.md").write_text(
        f"# {caption}\n\n" + frame.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    try:
        (TABLES_DIR / f"{stem}.tex").write_text(
            frame.to_latex(index=False, float_format=float_format, caption=caption, escape=True),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - LaTeX export is a convenience
        logger.warning("LaTeX export failed for %s: %s", stem, exc)
    logger.info("Wrote table %s", stem)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_outputs(benchmarks: pd.DataFrame) -> None:
    batch1 = benchmarks[benchmarks["batch_size"] == 1].copy()

    table = batch1[
        [
            "model_key",
            "bench_device",
            "backend",
            "precision",
            "params",
            "gflops",
            "size_mb",
            "latency_ms_median",
            "latency_ms_p95",
            "fps",
            "peak_mem_mb",
        ]
    ].sort_values(["model_key", "bench_device", "precision"])
    save_table(table, "benchmark_matrix", "Inference cost, batch size 1")

    # Static cost per model: parameters, FLOPs, artifact size. Independent of
    # device, so it is reported once rather than repeated in every latency row.
    static = (
        batch1[batch1["precision"] == "fp32"]
        .groupby("model_key")
        .agg(
            model_name=("model_name", "first"),
            params=("params", "max"),
            gflops=("gflops", "max"),
            size_mb=("size_mb", "max"),
        )
        .reset_index()
    )
    save_table(static, "model_static_cost", "Static model cost")

    # Latency by device and precision.
    fig, axis = plt.subplots(figsize=(10, 5))
    devices = sorted(batch1["bench_device"].unique())
    models = [m for m in METHOD_STYLE if m in set(batch1["model_key"])]
    width = 0.8 / max(len(models), 1)

    for index, model in enumerate(models):
        subset = batch1[(batch1["model_key"] == model) & (batch1["precision"] == "fp32")]
        values = [
            subset[subset["bench_device"] == device]["latency_ms_median"].min()
            if len(subset[subset["bench_device"] == device])
            else 0.0
            for device in devices
        ]
        positions = [i + index * width for i in range(len(devices))]
        axis.bar(
            positions,
            values,
            width,
            label=METHOD_STYLE[model]["label"],
            color=METHOD_STYLE[model]["color"],
        )

    axis.set_xticks([i + 0.4 - width / 2 for i in range(len(devices))])
    axis.set_xticklabels(devices)
    axis.set_ylabel("Median latency (ms, batch=1)")
    axis.set_yscale("log")
    axis.set_title("Inference latency by device (FP32, lower is better)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    save_figure(fig, "latency_by_device")

    # Batch sweep, if more than one batch size was measured.
    if benchmarks["batch_size"].nunique() > 1:
        fig, axis = plt.subplots(figsize=(8, 5))
        for model in models:
            subset = benchmarks[
                (benchmarks["model_key"] == model)
                & (benchmarks["precision"] == "fp32")
                & (benchmarks["backend"] == "pytorch")
            ]
            for device in subset["bench_device"].unique():
                device_subset = subset[subset["bench_device"] == device].sort_values("batch_size")
                if len(device_subset) < 2:
                    continue
                axis.plot(
                    device_subset["batch_size"],
                    device_subset["throughput_ips"],
                    marker=METHOD_STYLE[model]["marker"],
                    color=METHOD_STYLE[model]["color"],
                    linestyle="-" if "cuda" in device else "--",
                    label=f"{METHOD_STYLE[model]['label']} ({device})",
                )
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Throughput (images/s)")
        axis.set_title("Throughput vs batch size")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
        save_figure(fig, "throughput_vs_batch")


# ---------------------------------------------------------------------------
# The Pareto figure -- the thesis's central result
# ---------------------------------------------------------------------------


def pareto_outputs(common: pd.DataFrame, benchmarks: pd.DataFrame) -> None:
    """
    Accuracy against inference cost, one point per (method, device, precision).

    This is the figure the whole project exists to produce: it is the only view
    in which "which way of detecting fire should I deploy, and where?" has a
    visible answer, because it is the only one where accuracy and cost share an
    axis pair.
    """
    binary = common[common["axis"] == "binary"].copy()
    # Where a threshold was swept, take the operating point that maximises
    # macro-F1 -- the best each method can do, which is the fair comparison.
    best = binary.loc[binary.groupby("method")["f1_macro"].idxmax()]

    batch1 = benchmarks[(benchmarks["batch_size"] == 1) & (benchmarks["backend"] == "pytorch")]

    fig, axis = plt.subplots(figsize=(9, 6))
    plotted = []

    for _, row in best.iterrows():
        method = row["method"]
        if method not in METHOD_STYLE:
            continue
        style = METHOD_STYLE[method]
        for device in sorted(batch1["bench_device"].unique()):
            subset = batch1[
                (batch1["model_key"] == method)
                & (batch1["bench_device"] == device)
                & (batch1["precision"] == "fp32")
            ]
            if subset.empty:
                continue
            latency = float(subset["latency_ms_median"].min())
            axis.scatter(
                latency,
                row["f1_macro"],
                s=140 if "cuda" in device else 90,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.95 if "cuda" in device else 0.55,
            )
            axis.annotate(
                device.replace("jetson-", "J-"),
                (latency, row["f1_macro"]),
                fontsize=6,
                xytext=(4, 4),
                textcoords="offset points",
            )
            plotted.append(
                {
                    "method": method,
                    "label": style["label"],
                    "device": device,
                    "latency_ms": latency,
                    "fps": 1000.0 / latency if latency else 0.0,
                    "f1_macro": row["f1_macro"],
                    "accuracy": row["accuracy"],
                    "threshold": row["threshold"],
                }
            )

    axis.axvline(1000.0 / REALTIME_FPS, color="grey", linestyle=":", linewidth=1)
    axis.text(
        1000.0 / REALTIME_FPS, axis.get_ylim()[0], f" {REALTIME_FPS:.0f} FPS", fontsize=7,
        color="grey", rotation=90, va="bottom",
    )
    axis.axvline(1000.0 / MONITORING_FPS, color="grey", linestyle="--", linewidth=1)
    axis.text(
        1000.0 / MONITORING_FPS, axis.get_ylim()[0], f" {MONITORING_FPS:.0f} FPS", fontsize=7,
        color="grey", rotation=90, va="bottom",
    )

    handles = [
        plt.Line2D(
            [], [], color=style["color"], marker=style["marker"], linestyle="",
            markersize=8, label=style["label"],
        )
        for key, style in METHOD_STYLE.items()
        if key in set(best["method"])
    ]
    axis.legend(handles=handles, fontsize=8, loc="lower left")
    axis.set_xscale("log")
    axis.set_xlabel("Median inference latency (ms, batch=1, log scale)")
    axis.set_ylabel("Macro-F1 on the common binary 'fire present' task")
    axis.set_title("Accuracy vs inference cost across detection paradigms")
    axis.grid(alpha=0.3)
    save_figure(fig, "pareto_accuracy_vs_latency")

    if plotted:
        frame = pd.DataFrame(plotted).sort_values(["method", "device"])
        save_table(frame, "pareto_points", "Accuracy vs cost, per method and device")


def common_eval_outputs(common: pd.DataFrame) -> None:
    for axis_name in common["axis"].unique():
        subset = common[common["axis"] == axis_name].copy()
        best = subset.loc[subset.groupby("method")["f1_macro"].idxmax()]
        columns = [
            c
            for c in [
                "method", "model_name", "paradigm", "threshold", "n_images",
                "accuracy", "f1_macro", "f1_fire", "precision_fire", "recall_fire",
                "f1_Neither", "f1_Only_Fire", "f1_Only_Smoke", "f1_Both",
                "domain_shift",
            ]
            if c in best.columns
        ]
        save_table(
            best[columns].sort_values("f1_macro", ascending=False),
            f"common_eval_{axis_name}",
            f"Common-task comparison ({axis_name} axis), best operating point per method",
        )

    # Threshold sensitivity, where a sweep exists.
    swept = common[common["threshold"].notna() & (common["axis"] == "binary")]
    if swept["method"].nunique() and len(swept) > swept["method"].nunique():
        fig, axis = plt.subplots(figsize=(8, 5))
        for method in swept["method"].unique():
            subset = swept[swept["method"] == method].sort_values("threshold")
            if len(subset) < 2:
                continue
            style = METHOD_STYLE.get(method, {"label": method, "color": None, "marker": "o"})
            axis.plot(
                subset["threshold"], subset["f1_macro"],
                marker=style["marker"], color=style["color"], label=style["label"],
            )
        axis.set_xlabel("Operating threshold (detector confidence / mask area fraction)")
        axis.set_ylabel("Macro-F1 (binary axis)")
        axis.set_title("Sensitivity to the collapse threshold")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        save_figure(fig, "threshold_sensitivity")


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def robustness_outputs(robustness: pd.DataFrame) -> None:
    summary = (
        robustness.groupby(["method", "group"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_drop=("accuracy_drop", "mean"),
            mean_relative_drop_pct=("relative_drop_pct", "mean"),
            n=("accuracy", "size"),
        )
        .reset_index()
    )
    save_table(summary, "robustness_summary", "Accuracy under corruption, grouped")

    pivot = robustness.pivot_table(
        index=["corruption", "severity"], columns="method", values="accuracy"
    ).reset_index()
    save_table(pivot, "robustness_per_corruption", "Accuracy by corruption and severity")

    corruptions = [c for c in robustness["corruption"].unique() if c != "none"]
    if not corruptions:
        return

    columns = 4
    rows = (len(corruptions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)

    for index, corruption in enumerate(sorted(corruptions)):
        axis = axes[index // columns][index % columns]
        for method in sorted(robustness["method"].unique()):
            subset = robustness[
                (robustness["method"] == method) & (robustness["corruption"] == corruption)
            ].sort_values("severity")
            clean = robustness[
                (robustness["method"] == method) & (robustness["corruption"] == "none")
            ]["accuracy"]
            style = METHOD_STYLE.get(method, {"label": method, "color": None, "marker": "o"})
            severities = [0] + subset["severity"].tolist()
            accuracies = [float(clean.iloc[0]) if len(clean) else None] + subset[
                "accuracy"
            ].tolist()
            axis.plot(
                severities, accuracies,
                marker=style["marker"], color=style["color"], label=style["label"],
            )
        group = robustness[robustness["corruption"] == corruption]["group"].iloc[0]
        axis.set_title(f"{corruption}\n({group.replace('_', ' ')})", fontsize=9)
        axis.set_xlabel("Severity")
        axis.set_ylabel("Accuracy")
        axis.grid(alpha=0.3)
        if index == 0:
            axis.legend(fontsize=7)

    for index in range(len(corruptions), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle("Accuracy degradation under corruption: standard vs robust training", y=1.0)
    fig.tight_layout()
    save_figure(fig, "robustness_curves")


# ---------------------------------------------------------------------------
# Training metrics across seeds
# ---------------------------------------------------------------------------


def seed_outputs(metrics: pd.DataFrame) -> None:
    seeded = metrics[metrics["seed"].notna()].copy()
    if seeded.empty:
        logger.warning("No seeded runs in metrics.csv — skipping the variance table.")
        return

    # Two spellings exist for the same quantity. `fit_with_test` (iterations 1
    # and 5, and FireCNN in the backbone comparison) nests extras under
    # `test_metrics.*`, while `fit_two_phase` (iterations 2 and 3, and the
    # pretrained backbones) flattens them to `test_*`. Both are accepted and
    # normalised to one name so a table never silently drops half its rows.
    ALIASES = {
        "test_metrics.f1_macro": "test_f1_macro",
        "test_metrics.accuracy": "test_accuracy",
        "test_metrics.mIoU": "test_mIoU",
        "test_metrics.mIoU_hazard_only": "test_mIoU_hazard_only",
        "test_metrics.mDice": "test_mDice",
        "test_metrics.pr_auc": "test_pr_auc",
        "test_metrics.roc_auc": "test_roc_auc",
        "test_metrics.recall": "test_recall",
        "test_metrics.precision": "test_precision",
        "test_metrics.false_alarm_rate": "test_false_alarm_rate",
    }
    seeded = seeded.copy()
    seeded["metric"] = seeded["metric"].replace(ALIASES)
    # A run can carry both spellings (e.g. `test_accuracy` and
    # `test_metrics.accuracy`), which after aliasing would count the same run
    # twice and shrink the reported std. Keep one value per (run, metric).
    seeded = seeded.drop_duplicates(subset=["run_id", "metric"], keep="last")

    headline = seeded[
        seeded["metric"].isin(
            [
                "test_accuracy", "test_loss", "test_f1_macro",
                "test_mIoU", "test_mIoU_hazard_only", "test_mDice",
                "test_pr_auc", "test_roc_auc", "test_recall",
                "test_precision", "test_false_alarm_rate",
            ]
        )
    ]
    if headline.empty:
        logger.warning("No headline metrics found among seeded runs.")
        return

    summary = (
        headline.groupby(["method", "backbone", "metric"])["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values(["method", "metric"])
    )
    summary["mean_pm_std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if r["count"] > 1 else f"{r['mean']:.4f}",
        axis=1,
    )
    save_table(summary, "seed_variance", "Headline metrics across seeds (mean ± std)")


def dataset_outputs(stats: pd.DataFrame) -> None:
    pivot = stats.pivot_table(
        index="split", columns="class_name", values="count", aggfunc="sum"
    ).reset_index()
    save_table(pivot, "dataset_distribution", "D-Fire image-level class distribution")

    per_split = stats[stats["split"] != "all"]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    splits = sorted(per_split["split"].unique())
    classes = ["Neither", "Only_Fire", "Only_Smoke", "Both"]
    bottom = [0.0] * len(splits)

    for class_name in classes:
        values = [
            float(
                per_split[
                    (per_split["split"] == s) & (per_split["class_name"] == class_name)
                ]["percentage"].sum()
            )
            for s in splits
        ]
        axis.bar(splits, values, bottom=bottom, label=class_name)
        bottom = [b + v for b, v in zip(bottom, values)]

    axis.set_ylabel("Share of split (%)")
    axis.set_title("Class distribution is stable across splits (Only_Fire is 5.4% overall)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    save_figure(fig, "dataset_distribution")


def main() -> None:
    configure_logging()
    argparse.ArgumentParser(description="Generate thesis tables and figures.").parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = load("benchmarks.csv")
    common = load("common_eval.csv")
    robustness = load("robustness.csv")
    metrics = load("metrics.csv")
    dataset = load("dataset_stats.csv")

    if benchmarks is not None:
        benchmark_outputs(benchmarks)
    if common is not None:
        common_eval_outputs(common)
    if common is not None and benchmarks is not None:
        pareto_outputs(common, benchmarks)
    if robustness is not None:
        robustness_outputs(robustness)
    if metrics is not None:
        seed_outputs(metrics)
    if dataset is not None:
        dataset_outputs(dataset)

    logger.info("Figures -> %s", FIGURES_DIR)
    logger.info("Tables  -> %s", TABLES_DIR)


if __name__ == "__main__":
    main()
