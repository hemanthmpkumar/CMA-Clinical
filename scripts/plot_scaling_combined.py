#!/usr/bin/env python3
"""
scripts/plot_scaling_combined.py

Combine the per-scale results of the scaling study into cumulative,
cross-scale result figures for all four arms (TF-IDF control, BM25, CMA, GDT).

For every session scale S that has a `results.csv` under outputs/scaling/<S>/
this script recomputes the aggregate comparison metrics directly from the raw
session rows (mean/median time-to-info, accuracy, cognitive load, latency,
queries, and per-arm % change vs control). It then renders:

  1. scaling_trends.png       - one panel per metric; lines per arm across
                                session scale S (log x-axis) so the cumulative
                                scaling behaviour of TF-IDF / BM25 / CMA / GDT
                                is visible together.
  2. scaling_pct_change.png   - % change vs control (BM25, CMA, GDT) across S
                                for time, cognitive load and latency.
  3. scaling_accuracy.png     - task accuracy across S per arm.

CSV summaries are written alongside (combined_results.csv and
combined_pct_change.csv) for the manuscript tables.

Metrics are always recomputed from results.csv, so this works even when the
per-scale statistics.json is stale (missing GDT contrasts) or a scale was run
before all four arms existed - missing arms are simply skipped on that scale.

Usage:
  python scripts/plot_scaling_combined.py
  python scripts/plot_scaling_combined.py --scales 10 100 1000
  python scripts/plot_scaling_combined.py --out-root outputs/scaling --out outputs/scaling/combined
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz.plots import add_timestamp, ensure_dirs  # noqa: E402

CONDITIONS = ["control", "bm25", "cma", "gdt"]
ARM_LABELS = {"control": "TF-IDF", "bm25": "BM25", "cma": "CMA", "gdt": "GDT"}
ARM_COLORS = {"control": "#E74C3C", "bm25": "#F39C12", "cma": "#27AE60",
              "gdt": "#8E44AD"}

METRICS = {
    "time_to_info": "Mean time-to-correct-information (s)",
    "cognitive_load": "Mean cognitive load",
    "latency_ms": "Mean latency (ms)",
    "n_queries_issued": "Mean queries per session",
}

DEFAULT_SCALES = [10, 100, 1000]


def available_scales(out_root: Path, scales: list[int]) -> list[int]:
    found = []
    for s in sorted(scales):
        if (out_root / str(s) / "results.csv").exists():
            found.append(s)
    return found


def per_scale_stats(results_path: Path) -> pd.DataFrame:
    """Compute, per condition, the aggregate metrics from one scale's results."""
    df = pd.read_csv(results_path)
    rows = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        rows.append({
            "condition": cond,
            "arm": ARM_LABELS[cond],
            "n_sessions": int(len(sub)),
            "time_to_info_mean": round(sub["time_to_info"].mean(), 3),
            "time_to_info_median": round(sub["time_to_info"].median(), 3),
            "accuracy": round(sub["accuracy"].mean(), 4),
            "cognitive_load_mean": round(sub["cognitive_load"].mean(), 3),
            "latency_ms_mean": round(sub["latency_ms"].mean(), 3),
            "n_queries_issued_mean": round(sub["n_queries_issued"].mean(), 3),
        })
    return pd.DataFrame(rows)


def pct_change_vs_control(scale_rows: pd.DataFrame) -> pd.DataFrame:
    """% change of each arm vs control for the key metrics."""
    ctrl = scale_rows[scale_rows["condition"] == "control"]
    if ctrl.empty:
        return pd.DataFrame()
    rows = []
    for _, arm in scale_rows[scale_rows["condition"] != "control"].iterrows():
        cond = arm["condition"]
        for metric in ["time_to_info", "cognitive_load", "latency_ms"]:
            col = f"{metric}_mean"
            cv = ctrl[col].iloc[0]
            av = arm[col]
            if cv == 0:
                continue
            rows.append({
                "condition": cond,
                "arm": ARM_LABELS[cond],
                "metric": metric,
                "control_mean": round(cv, 3),
                "arm_mean": round(av, 3),
                "pct_change": round((av - cv) / cv * 100, 2),
            })
    return pd.DataFrame(rows)


def plot_trends(all_stats: pd.DataFrame, scales: list[int], out_dir: Path):
    """Cumulative metric trends across session scale for all four arms."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = np.array(scales, dtype=float)
    for ax, (metric, ylabel) in zip(axes.flat, METRICS.items()):
        col = f"{metric}_mean"
        for cond in CONDITIONS:
            ys = [np.nan] * len(scales)
            for i, s in enumerate(scales):
                row = all_stats[(all_stats["scale"] == s) &
                                (all_stats["condition"] == cond)]
                if not row.empty:
                    ys[i] = row[col].iloc[0]
            ax.plot(x, ys, marker="o", linestyle="-",
                    color=ARM_COLORS[cond], label=ARM_LABELS[cond])
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(s)) for s in x])
        ax.set_xlabel("Session scale S")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Scaling study — cumulative results (all four arms)", y=0.99,
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    add_timestamp(axes.flat[-1])
    fig.savefig(out_dir / "scaling_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pct_change(pct: pd.DataFrame, scales: list[int], out_dir: Path):
    """% change vs control across scales for BM25, CMA, GDT."""
    metrics = ["time_to_info", "cognitive_load", "latency_ms"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.array(scales, dtype=float)
    for ax, metric in zip(axes, metrics):
        for cond in CONDITIONS:
            if cond == "control":
                continue
            ys = []
            for s in scales:
                row = pct[(pct["scale"] == s) & (pct["condition"] == cond) &
                          (pct["metric"] == metric)]
                ys.append(row["pct_change"].iloc[0] if not row.empty else np.nan)
            ax.plot(x, ys, marker="o", linestyle="-",
                    color=ARM_COLORS[cond], label=ARM_LABELS[cond])
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(s)) for s in x])
        ax.set_xlabel("Session scale S")
        ax.set_ylabel("Mean % change vs TF-IDF control")
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Scaling study — arm performance vs TF-IDF control", y=1.02,
                 fontsize=14)
    fig.tight_layout()
    add_timestamp(axes[-1])
    fig.savefig(out_dir / "scaling_pct_change.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(all_stats: pd.DataFrame, scales: list[int], out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.array(scales, dtype=float)
    for cond in CONDITIONS:
        ys = []
        for s in scales:
            row = all_stats[(all_stats["scale"] == s) &
                            (all_stats["condition"] == cond)]
            ys.append(row["accuracy"].iloc[0] if not row.empty else np.nan)
        ax.plot(x, ys, marker="o", linestyle="-",
                color=ARM_COLORS[cond], label=ARM_LABELS[cond])
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in x])
    ax.set_xlabel("Session scale S")
    ax.set_ylabel("Task accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Scaling study — task accuracy per arm")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "scaling_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Combine scaling-study scale results into cumulative figures")
    ap.add_argument("--scales", type=int, nargs="*", default=None,
                    help=f"Session scales to combine (default: all found under "
                         f"--out-root; canonical set {' '.join(map(str, DEFAULT_SCALES))})")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs/scaling",
                    help="Directory containing outputs/scaling/<S>/results.csv.")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/scaling/combined",
                    help="Output directory for the combined figures + CSVs.")
    args = ap.parse_args()

    want = sorted(args.scales) if args.scales else DEFAULT_SCALES
    scales = available_scales(args.out_root, want)
    if not scales:
        raise FileNotFoundError(
            f"No outputs/scaling/<S>/results.csv found under {args.out_root}"
        )
    print(f"Combining scales: {scales}")

    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    per_scale_frames = []
    pct_frames = []
    for s in scales:
        rows = per_scale_stats(args.out_root / str(s) / "results.csv")
        rows["scale"] = s
        per_scale_frames.append(rows)
        pct = pct_change_vs_control(rows)
        if not pct.empty:
            pct["scale"] = s
            pct_frames.append(pct)
        print(f"  scale {s}: {len(rows)} arms across sessions")

    all_stats = pd.concat(per_scale_frames, ignore_index=True)
    all_pct = pd.concat(pct_frames, ignore_index=True)

    all_stats.to_csv(out_dir / "combined_results.csv", index=False)
    all_pct.to_csv(out_dir / "combined_pct_change.csv", index=False)
    print(f"\nCSV summaries written to {out_dir}")

    plot_trends(all_stats, scales, out_dir)
    plot_pct_change(all_pct, scales, out_dir)
    plot_accuracy(all_stats, scales, out_dir)

    print(f"\nCombined scaling figures written to {out_dir}:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()