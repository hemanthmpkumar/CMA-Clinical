#!/usr/bin/env python3
"""
src/viz/plots.py

Generate the seven manuscript-quality figures from the experiment results.

Figures saved to outputs/figures/:
  1. consort_flow.png           - benchmark case flow (CONSORT-style)
  2. time_distribution.png      - time-to-correct-information density
   3. intent_trajectory.png      - 2D schematic of SPD intent trajectory + GSI gate
  4. tlx_subscales.png          - NASA-TLX mean subscales
  5. latency.png                - system latency box plot
  6. subgroup_forest.png        - forest plot of time-to-info improvement
  7. cma_components.png          - CMA system architecture diagram
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIG_DPI = 300


def ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_consort_flow(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    n_vignettes = df["vignette_id"].nunique()
    n_sessions = len(df)
    n_per_cond = n_sessions // 3

    boxes = [
        (f"Assessed for eligibility\n{n_vignettes} clinical EHR vignettes", 5, 9, 3.2, 1.0),
        (f"Randomized\n{n_vignettes} vignettes, three-arm crossover", 5, 7.3, 3.2, 0.9),
        (f"Control (TF-IDF)\nn = {n_per_cond} sessions", 1.8, 5.3, 2.6, 0.9),
        (f"BM25 baseline\nn = {n_per_cond} sessions", 5, 5.3, 2.6, 0.9),
        (f"CMA condition\nn = {n_per_cond} sessions", 8.2, 5.3, 2.6, 0.9),
        (f"Analysed\nN = {n_sessions} sessions ({n_per_cond} per condition)", 5, 2.5, 4.2, 1.0),
    ]
    for text, x, y, w, h in boxes:
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor="#E8F4FD", edgecolor="#2C3E50", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold")

    # Arrows
    for x1, y1, x2, y2 in [(5, 8.5, 5, 7.8),
                           (5, 6.8, 1.8, 5.8),
                           (5, 6.8, 5, 5.8),
                           (5, 6.8, 8.2, 5.8),
                           (1.8, 4.8, 4.2, 3.1),
                           (5, 4.8, 5, 3.1),
                           (8.2, 4.8, 5.8, 3.1)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax.set_title("Benchmark flow and condition randomization", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "consort_flow.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_time_distribution(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("control", "Control (TF-IDF)", "#E74C3C"),
        ("bm25", "BM25", "#F39C12"),
        ("cma", "CMA", "#27AE60"),
    ]
    for i, (cond, label, color) in enumerate(series):
        data = df[df["condition"] == cond]["time_to_info"]
        sns.kdeplot(data, ax=ax, fill=True, label=label, color=color, alpha=0.35, linewidth=2)
        ax.axvline(data.median(), color=color, linestyle="--", linewidth=1.5)
        ax.text(data.median() + 1, ax.get_ylim()[1] * (0.75 - i * 0.10),
                f"median\n{data.median():.1f}s", color=color, fontsize=9)

    ax.set_xlabel("Time to correct information (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Time-to-correct-information distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
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
    ax.plot(x[34:], y[34:], color="#27AE60", linewidth=3, label="CMA: gate resets stale context")
    ax.scatter(x[34], y[34], color="#F39C12", s=120, zorder=5, label="GSI gate")

    # Prefetch arrow
    ax.annotate("", xy=(x[45] + 0.4, y[45] + 0.6), xytext=(x[44], y[44]),
                arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2.5, ls="--"))
    ax.text(x[46], y[46] + 0.8, "JEPA prefetch", color="#9B59B6", fontsize=10)

    ax.set_xlabel("Manifold coordinate 1")
    ax.set_ylabel("Manifold coordinate 2")
    ax.set_title("Session intent trajectory on an abstract SPD manifold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "intent_trajectory.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tlx(df: pd.DataFrame, out_dir: Path):
    subscales = ["tlx_mental", "tlx_physical", "tlx_temporal",
                 "tlx_performance", "tlx_effort", "tlx_frustration"]
    labels = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
    arms = [
        ("control", "Control (TF-IDF)", "#E74C3C"),
        ("bm25", "BM25", "#F39C12"),
        ("cma", "CMA", "#27AE60"),
    ]
    means = {cond: [df[df["condition"] == cond][s].mean() for s in subscales] for cond, _, _ in arms}

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cond, label, color) in enumerate(arms):
        ax.bar(x + (i - 1) * width, means[cond], width, label=label, color=color, alpha=0.8)
    ax.set_ylabel("NASA-TLX score (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("NASA-TLX cognitive load subscales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tlx_subscales.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latency(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["control", "bm25", "cma"]
    palette = {"control": "#E74C3C", "bm25": "#F39C12", "cma": "#27AE60"}
    sns.boxplot(data=df, x="condition", y="latency_ms", hue="condition",
                palette=palette, order=order, ax=ax, legend=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Control (TF-IDF)", "BM25", "CMA"])
    ax.set_ylabel("Query latency (ms)")
    ax.set_xlabel("Condition")
    ax.set_title("System latency per query")
    fig.tight_layout()
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
    ax.set_xlabel("Median % reduction in time-to-correct-information (Control – CMA)")
    ax.set_title("Subgroup forest plot")
    fig.tight_layout()
    fig.savefig(out_dir / "subgroup_forest.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cma_components(out_dir: Path):
    """System architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    components = [
        ("User query history", 1, 3),
        ("Session encoder\n(SPD manifold)", 3, 4.2),
        ("GSI gate\n(geodesic shift ratio)", 3, 1.8),
        ("JEPA predictor\n(latent forecast)", 5.5, 3),
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
    ax.set_title("CMA retrieval architecture", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "cma_components.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate CMA manuscript figures")
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--stats", default="outputs/statistics.json")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    plot_consort_flow(df, out_dir)
    plot_time_distribution(df, out_dir)
    plot_intent_trajectory(out_dir)
    plot_tlx(df, out_dir)
    plot_latency(df, out_dir)
    plot_subgroup_forest(stats, out_dir)
    plot_cma_components(out_dir)

    print(f"Figures saved to {out_dir}:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
