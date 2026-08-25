#!/usr/bin/env python3
"""
src/viz/plots.py

Generate the manuscript-quality figures from the experiment results.

Figures saved to outputs/figures/:
  1. consort_flow.png           - benchmark case flow (CONSORT-style)
  2. time_distribution.png      - time-to-correct-information density
  3. intent_trajectory.png      - 2D schematic of SPD intent trajectory + GSI gate
  4. tlx_subscales.png          - NASA-TLX mean subscales
  5. latency.png                - system latency box plot
  6. subgroup_forest.png        - forest plot of time-to-info improvement
  7. gdt_vs_benchmarks.png      - GDT vs TF-IDF/BM25/CMA (time, load, latency, queries)
  8. gdt_vs_benchmarks_accuracy.png - GDT vs TF-IDF/BM25/CMA (accuracy delta)
  9. cma_components.png         - GDT system architecture diagram
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from datetime import datetime

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIG_DPI = 300

ARMS = [
    ("control", "Control (TF-IDF)", "#E74C3C"),
    ("bm25", "BM25", "#F39C12"),
    ("cma", "CMA", "#27AE60"),
    ("gdt", "GDT", "#8E44AD"),
]


def add_timestamp(ax):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.99, 0.01, timestamp,
            ha="right", va="bottom", fontsize=7, color="gray",
            transform=ax.transAxes)


def ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_consort_flow(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    n_vignettes = df["vignette_id"].nunique()
    n_sessions = len(df)
    n_per_cond = n_sessions // len(ARMS)
    # For the representative CONSORT diagram we show the vignette sizes used
    # in the scaling experiments explicitly (3, 30, and 300) rather than a
    # single numeric value.
    # Do not display numeric vignette/session counts in the CONSORT boxes;
    # keep the diagram generic across stages.
    display_vignettes = ""

    boxes = [
        ("Assessed for eligibility", 5, 9, 3.2, 1.0),
        ("Randomized, four-arm crossover", 5, 7.3, 3.2, 0.9),
        ("Control (TF-IDF)", 1.5, 5.3, 2.3, 0.9),
        ("BM25 baseline", 4.0, 5.3, 2.3, 0.9),
        ("CMA condition", 6.5, 5.3, 2.3, 0.9),
        ("GDT condition", 9.0, 5.3, 2.3, 0.9),
        ("Analysed", 5, 2.5, 4.2, 1.0),
    ]
    for text, x, y, w, h in boxes:
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor="#E8F4FD", edgecolor="#2C3E50", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold")

    # Arrows
    for x1, y1, x2, y2 in [(5, 8.5, 5, 7.8),
                           (5, 6.8, 1.5, 5.8),
                           (5, 6.8, 4.0, 5.8),
                           (5, 6.8, 6.5, 5.8),
                           (5, 6.8, 9.0, 5.8),
                           (1.5, 4.8, 4.2, 3.1),
                           (4.0, 4.8, 4.8, 3.1),
                           (6.5, 4.8, 5.2, 3.1),
                           (9.0, 4.8, 5.8, 3.1)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax.set_title("Benchmark flow and condition randomization", fontsize=13, pad=20)
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "consort_flow.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_time_distribution(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = ARMS
    for i, (cond, label, color) in enumerate(series):
        data = df[df["condition"] == cond]["time_to_info"].dropna()
        if data.nunique() >= 2:
            sns.kdeplot(data, ax=ax, fill=True, label=label, color=color, alpha=0.35, linewidth=2)
            ax.axvline(data.median(), color=color, linestyle="--", linewidth=1.5)
            ax.text(data.median() + 1, ax.get_ylim()[1] * (0.75 - i * 0.10),
                    f"median\n{data.median():.1f}s", color=color, fontsize=9)
        elif len(data) >= 1:
            # Too few sessions for a density estimate: show the observed value.
            ax.scatter([data.iloc[0]], [0.02 + i * 0.02], color=color, s=80,
                       label=f"{label} (n={len(data)})", zorder=5)
            ax.text(data.median() + 1, 0.03 + i * 0.03,
                    f"n={len(data)}\n{data.median():.1f}s", color=color, fontsize=9)

    ax.set_xlabel("Time to correct information (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Time-to-correct-information distribution")
    ax.legend()
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "time_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(df: pd.DataFrame, out_dir: Path):
    means = []
    labels = []
    colors = []
    for cond, label, color in ARMS:
        cond_acc = df[df["condition"] == cond]["accuracy"]
        if len(cond_acc) == 0:
            continue
        means.append(cond_acc.mean())
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=labels, y=means, palette=colors, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean accuracy")
    ax.set_xlabel("Condition")
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_intent_trajectory(out_dir: Path):
    """Schematic 2D trajectory on an abstract SPD manifold."""
    fig, ax = plt.subplots(figsize=(7, 6))
    np.random.seed(0)
    # Simulate a latent trajectory with a sharp pivot.
    t = np.linspace(0, 1, 60)
    x = 5 * t
    y = 2 * np.sin(2 * np.pi * t) + 0.5 * t
    y[35:] += 1.8 * np.linspace(0, 1, len(t) - 35) ** 2  # abrupt upward turn after pivot

    ax.plot(x[:35], y[:35], color="#E74C3C", linewidth=3, label="Control: context accumulates")
    ax.plot(x[34:], y[34:], color="#27AE60", linewidth=3, label="GDT: gate resets stale context")
    ax.scatter(x[34], y[34], color="#F39C12", s=120, zorder=5, label="GSI gate (κ_t)")

    # Prefetch arrow
    ax.annotate("", xy=(x[45] + 0.4, y[45] + 0.6), xytext=(x[44], y[44]),
                arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2.5, ls="--"))
    ax.text(x[46], y[46] + 0.8, "confidence-gated prefetch", color="#9B59B6", fontsize=10)

    ax.set_xlabel("Manifold coordinate 1")
    ax.set_ylabel("Manifold coordinate 2")
    ax.set_title("Session intent trajectory on an abstract SPD manifold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "intent_trajectory.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tlx(df: pd.DataFrame, out_dir: Path):
    subscales = ["tlx_mental", "tlx_physical", "tlx_temporal",
                 "tlx_performance", "tlx_effort", "tlx_frustration"]
    labels = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
    arms = ARMS
    means = {cond: [df[df["condition"] == cond][s].mean() for s in subscales] for cond, _, _ in arms}

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (cond, label, color) in enumerate(arms):
        ax.bar(x + (i - 1.5) * width, means[cond], width, label=label, color=color, alpha=0.8)
    ax.set_ylabel("NASA-TLX score (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("NASA-TLX cognitive load subscales")
    ax.legend()
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "tlx_subscales.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latency(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = [c for c, _, _ in ARMS]
    palette = {c: color for c, _, color in ARMS}
    sns.boxplot(data=df, x="condition", y="latency_ms", hue="condition",
                palette=palette, order=order, ax=ax, legend=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for _, label, _ in ARMS])
    ax.set_ylabel("Query latency (ms)")
    ax.set_xlabel("Condition")
    ax.set_title("System latency per query")
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "latency.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_subgroup_forest(stats: dict, out_dir: Path):
    """Forest plot of median % time-to-info improvement by subgroup."""
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = 0
    ticks = []
    tick_labels = []
    colors = {"specialty": "#3498DB", "experience_group": "#9B59B6", "complexity": "#E67E22"}

    for category, groups in stats.get("subgroups", {}).items():
        for name, vals in groups.items():
            med = vals["median_pct_change"]
            ci = [vals["ci_lower"], vals["ci_upper"]]
            ax.barh(y_pos, med, color=colors.get(category, "gray"), alpha=0.8, height=0.6)
            ax.plot(ci, [y_pos, y_pos], color="black", linewidth=1.5)
            ticks.append(y_pos)
            tick_labels.append(f"{category}: {name} (n={vals['n']})")
            y_pos += 1
        y_pos += 0.5

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Median % reduction in time-to-correct-information (Control – GDT)")
    ax.set_title("Subgroup forest plot (GDT vs Control)")
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "subgroup_forest.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_gdt_vs_benchmarks(stats: dict, out_dir: Path):
    """GDT (primary) vs each benchmark: grouped % change across outcomes.

    Renders one grouped bar per outcome with three bars (vs TF-IDF, BM25, CMA).
    positive values mean GDT is faster (time/latency) or lighter (cognitive
    load); accuracy bars show the accuracy delta (GDT - benchmark).
    """
    gvs = stats.get("gdt_vs_benchmarks", {})
    if not gvs:
        raise ValueError("statistics.json is missing 'gdt_vs_benchmarks'; "
                         "re-run analyze.py")

    bench_colors = {"control": "#E74C3C", "bm25": "#F39C12", "cma": "#27AE60"}
    bench_labels = {"control": "TF-IDF", "bm25": "BM25", "cma": "CMA"}
    benchmark_order = ["control", "bm25", "cma"]

    # Continuous outcomes: median % change of GDT vs each benchmark.
    metrics = [
        ("time_to_info", "Time to info", "%"),
        ("cognitive_load", "Cognitive load", "%"),
        ("latency_ms", "Latency", "%"),
        ("n_queries_issued", "Queries", "%"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (outcome, label, unit) in zip(axes.flat, metrics):
        benches = gvs.get(outcome, {})
        names = []
        vals = []
        bench_codes = []
        for bench in benchmark_order:
            pv = benches.get(bench) or {}
            v = pv.get("median_pct_change_vs_bench")
            if v is None:
                continue
            names.append(bench_labels[bench])
            bench_codes.append(bench)
            vals.append(v)
        bars = ax.bar(names, vals,
                      color=[bench_colors[b] for b in bench_codes], alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(f"% change vs benchmark")
        ax.set_title(f"GDT vs benchmarks — {label}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, weight="bold")
        ax.set_ylim(0 if all(v >= 0 for v in vals) else None)

    # Accuracy: delta (percentage points) of GDT - benchmark.
    fig.tight_layout()
    add_timestamp(axes.flat[-1])
    fig.savefig(out_dir / "gdt_vs_benchmarks.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    acc_benches = gvs.get("accuracy", {})
    names = []
    deltas = []
    bench_codes = []
    for bench in benchmark_order:
        pv = acc_benches.get(bench) or {}
        d = pv.get("accuracy_delta")
        if d is None:
            continue
        names.append(bench_labels[bench])
        bench_codes.append(bench)
        deltas.append(d)
    bars = ax2.bar(names, deltas, color=[bench_colors[b] for b in bench_codes], alpha=0.85)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Accuracy delta (pp), GDT - benchmark")
    ax2.set_title("GDT vs benchmarks — task accuracy")
    for bar, v in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:+.2f}pp", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9, weight="bold")
    fig2.tight_layout()
    add_timestamp(ax2)
    fig2.savefig(out_dir / "gdt_vs_benchmarks_accuracy.png", dpi=FIG_DPI,
                 bbox_inches="tight")
    plt.close(fig2)


def plot_ablation(report: dict, out_dir: Path):
    """Plot the ablation study with all four arms (TF-IDF, BM25, CMA, GDT).

    Reads an ablation_report.json (with 'summary' rows of variant x condition)
    and renders grouped bars per outcome. GDT uses the same arm colors as the
    main comparison figures.
    """
    bench_colors = {"control": "#E74C3C", "bm25": "#F39C12", "cma": "#27AE60",
                    "gdt": "#8E44AD"}
    bench_labels = {"control": "TF-IDF", "bm25": "BM25", "cma": "CMA", "gdt": "GDT"}
    arm_order = ["control", "bm25", "cma", "gdt"]

    summary = report.get("summary", [])
    variants = []
    for row in summary:
        label = row["variant"]
        if label not in variants:
            variants.append(label)

    metrics = [
        ("mean_time_to_info", "Mean time to info (s)"),
        ("accuracy", "Accuracy"),
        ("mean_cognitive_load", "Mean cognitive load"),
        ("mean_latency_ms", "Mean latency (ms)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        width = 0.2
        x = np.arange(len(variants))
        for i, cond in enumerate(arm_order):
            vals = []
            for label in variants:
                row = next((r for r in summary
                            if r["variant"] == label and r["condition"] == cond), None)
                vals.append(row.get(metric, np.nan) if row else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width,
                   label=bench_labels[cond], color=bench_colors[cond], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by ablation variant")
        ax.legend(fontsize=8)
    fig.tight_layout()
    add_timestamp(axes.flat[-1])
    fig.savefig(out_dir / "ablation_four_arm.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cma_components(out_dir: Path):
    """GDT system architecture diagram (retains filename for manuscript reference)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    components = [
        ("User query history", 1, 3),
        ("Session encoder\n(SPD manifold)", 3, 4.2),
        ("GSI gate κ_t\n(geodesic shift ratio)", 3, 1.8),
        ("Latent predictor\n(JEPA-style forecast)", 5.5, 3),
        ("Rank & merge\n(+ prefetch)", 8.2, 3),
    ]
    for text, x, y in components:
        w, h = 1.8, 0.9
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor="#D5F5E3", edgecolor="#1E8449", linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, weight="bold")

    arrows = [
        ((1.9, 3), (2.1, 4.2)),
        ((1.9, 3), (2.1, 1.8)),
        ((3.9, 4.2), (4.5, 3)),
        ((3.9, 1.8), (4.5, 3)),
        ((6.4, 3), (7.3, 3)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5,
                                    connectionstyle="arc3,rad=0.1"))
    ax.set_title("GDT retrieval architecture", fontsize=13, pad=20)
    fig.tight_layout()
    add_timestamp(ax)
    fig.savefig(out_dir / "cma_components.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate CMA manuscript figures")
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--stats", default="outputs/statistics.json")
    parser.add_argument("--ablation-report", default=None,
                        help="Optional ablation_report.json; renders the "
                             "four-arm ablation figure (TF-IDF vs BM25 vs CMA vs GDT).")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    plot_consort_flow(df, out_dir)
    plot_time_distribution(df, out_dir)
    plot_accuracy_comparison(df, out_dir)
    plot_intent_trajectory(out_dir)
    plot_tlx(df, out_dir)
    plot_latency(df, out_dir)
    plot_subgroup_forest(stats, out_dir)
    plot_gdt_vs_benchmarks(stats, out_dir)
    if args.ablation_report:
        report_path = Path(args.ablation_report)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            plot_ablation(report, out_dir)
            print(f"  ablation_four_arm.png (from {report_path.name})")
    plot_cma_components(out_dir)

    print(f"Figures saved to {out_dir}:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
