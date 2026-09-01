#!/usr/bin/env python3
"""
src/analysis/analyze.py

Statistical analysis of the four-arm crossover experiment.

Conditions:
  - control : session-based TF-IDF baseline (classic sparse retrieval)
  - bm25    : session-based BM25 baseline (stronger classical sparse retrieval)
  - cma     : Continuum Memory Architecture (comparison intervention)
  - gdt     : Geodesic Diagnostic Trajectories (primary intervention)

Primary analyses:
  1. Time-to-correct-information: paired Wilcoxon signed-rank + mixed-effects
     linear model on log-transformed times.
  2. Accuracy: McNemar-like paired proportion test + generalized estimating
     equations (GEE) with logit link clustered by vignette.
  3. Secondary outcomes: NASA-TLX, latency, perceived usefulness via paired
     tests and effect sizes (Cohen's d).

Pairwise contrasts: GDT vs Control (primary), CMA vs Control, GDT vs CMA,
GDT vs BM25, CMA vs BM25, BM25 vs Control.

Outputs:
  outputs/primary_results.csv    - key summary statistics
  outputs/statistics.json      - full numerical results for figures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
except ImportError:  # pragma: no cover
    GEE = None

CONDITIONS = ["control", "bm25", "cma", "gdt"]

# GDT is the primary intervention. CMA is a benchmark (alongside TF-IDF and
# BM25), not a co-intervention: it is compared against GDT but never treated
# as a competing treatment in the primary analysis.
PRIMARY_ARM = "gdt"
BENCHMARKS = ["control", "bm25", "cma"]

# Intervention arms used for the primary (vs control) contrasts.
INTERVENTION_ARMS = ["gdt"]

# All pairwise contrasts evaluated for each outcome.
CONTRASTS = [
    ("control", "gdt"),
    ("control", "cma"),
    ("gdt", "cma"),
    ("gdt", "bm25"),
    ("cma", "bm25"),
    ("control", "bm25"),
]


def cohens_d(x, y):
    """Paired Cohen's d."""
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


import numpy as np
import pandas as pd

def safe_pct_change(target: float, control: float) -> float:
    """Calculates percentage change, safely returning np.nan on zero or missing values."""
    if pd.isna(target) or pd.isna(control) or control == 0:
        return np.nan
    return round(float((target - control) / control * 100), 2)

def summarize(
    df: pd.DataFrame, 
    outcome: str, 
    group_col: str = "condition", 
    conditions: list = None
) -> dict:
    if outcome not in df.columns or group_col not in df.columns:
        return {}

    grp = df.groupby(group_col)[outcome]
    
    # Pre-compute group aggregations ONCE for speed
    means = grp.mean()
    medians = grp.median()
    stds = grp.std()
    
    result = {}
    
    # Default to all present groups if conditions are not explicitly passed
    target_conditions = conditions if conditions is not None else list(means.index)
    
    for cond in target_conditions:
        if cond in means.index:
            m_val = means[cond]
            med_val = medians[cond]
            s_val = stds[cond]
            
            result[f"{cond}_mean"] = round(float(m_val), 3) if pd.notna(m_val) else np.nan
            result[f"{cond}_median"] = round(float(med_val), 3) if pd.notna(med_val) else np.nan
            result[f"{cond}_std"] = round(float(s_val), 3) if pd.notna(s_val) else np.nan

    # Compute percentage changes safely relative to control for all benchmarks
    # and the primary arm (CMA is now a benchmark, not an intervention).
    ctrl_mean = means.get("control", np.nan)

    for cond in list(BENCHMARKS) + [PRIMARY_ARM]:
        if cond != "control" and cond in means.index:
            result[f"{cond}_pct_change_mean"] = safe_pct_change(means.get(cond), ctrl_mean)

    return result

def paired_stats(df: pd.DataFrame, outcome: str, arm_a: str, arm_b: str) -> Optional[dict]:
    """Paired Wilcoxon signed-rank + Cohen's d between two arms."""
    wide = df.pivot(index="vignette_id", columns="condition", values=outcome).dropna()
    if arm_a not in wide.columns or arm_b not in wide.columns:
        return None
    a = wide[arm_a].values
    b = wide[arm_b].values
    if len(a) < 3:
        return None
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
    return {
        "n_pairs": int(len(a)),
        "median_difference": round(float(np.median(a - b)), 3),
        "mean_difference": round(float(np.mean(a - b)), 3),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_pvalue": float(p),
        "cohens_d": round(cohens_d(b, a), 3),
    }


def mixed_model_contrast(df: pd.DataFrame, arm: str) -> dict:
    """Mixed-effects model on log time for <arm> vs control."""
    sub = df[df["condition"].isin(["control", arm])].copy()
    sub["condition_code"] = (sub["condition"] == arm).astype(int)
    sub["log_time"] = np.log1p(sub["time_to_info"])
    X = sm.add_constant(sub[["condition_code", "period"]])
    vignette_dummies = pd.get_dummies(sub["specialty"], prefix="spec", drop_first=True)
    if not vignette_dummies.empty:
        X = pd.concat([X, vignette_dummies], axis=1)
    X = X.astype(float)

    try:
        model = MixedLM(sub["log_time"], X, groups=sub["vignette_id"])
        fit = model.fit(reml=False)
        beta = float(fit.params["condition_code"])
        ci_low, ci_high = fit.conf_int().loc["condition_code"].values
        return {
            f"{arm}_beta_log_time": round(beta, 4),
            f"{arm}_beta_95ci": [round(float(ci_low), 4), round(float(ci_high), 4)],
            "pvalue": float(fit.pvalues["condition_code"]),
            "interpretation_pct": round((np.exp(beta) - 1) * 100, 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def primary_time_analysis(df: pd.DataFrame) -> dict:
    """Paired analysis of time-to-correct-information across the four arms."""
    summary = summarize(df, "time_to_info")

    contrasts = {}
    for arm_a, arm_b in CONTRASTS:
        ps = paired_stats(df, "time_to_info", arm_a, arm_b)
        if ps is not None:
            denom = float(np.median(df[df["condition"] == arm_a]["time_to_info"]))
            if denom > 0:
                ps["median_pct_change"] = round(
                    float(np.median((ps["median_difference"] / denom) * 100)), 2
                )
            else:
                ps["median_pct_change"] = np.nan
        contrasts[f"{arm_a}_vs_{arm_b}"] = ps

    mixed_effects = {arm: mixed_model_contrast(df, arm) for arm in INTERVENTION_ARMS}

    return {
        "summary": summary,
        "paired_test": "wilcoxon_signed_rank",
        "contrasts": contrasts,
        "mixed_effects_log_time": mixed_effects,
    }


def accuracy_analysis(df: pd.DataFrame) -> dict:
    """Paired binary accuracy across the four arms."""
    wide = df.pivot(index="vignette_id", columns="condition", values="accuracy").dropna()
    acc = {}
    for cond in CONDITIONS:
        if cond in wide.columns:
            acc[f"{cond}_accuracy"] = round(float(wide[cond].astype(int).mean()), 3)

    contrasts = {}
    for arm_a, arm_b in CONTRASTS:
        if arm_a not in wide.columns or arm_b not in wide.columns:
            contrasts[f"{arm_a}_vs_{arm_b}"] = None
            continue
        a = wide[arm_a].astype(int)
        b = wide[arm_b].astype(int)
        b_like = int(((a == 0) & (b == 1)).sum())  # arm_b better
        c_like = int(((a == 1) & (b == 0)).sum())  # arm_a better
        if b_like + c_like > 0:
            m_stat = (abs(b_like - c_like) - 1) ** 2 / (b_like + c_like)
            m_p = float(stats.chi2.sf(m_stat, 1))
        else:
            m_stat = 0.0
            m_p = 1.0
        contrasts[f"{arm_a}_vs_{arm_b}"] = {
            "n_discordant_better": b_like,
            "n_discordant_worse": c_like,
            "mcnemar_statistic": round(float(m_stat), 3),
            "mcnemar_pvalue": float(m_p),
        }

    # Cluster-robust GEE for each intervention vs control.
    gee_results = {}
    if GEE is not None:
        for arm in INTERVENTION_ARMS:
            gee_results[arm] = None
            try:
                sub = df[df["condition"].isin(["control", arm])].copy()
                # Only fit the model if there is variance in accuracy
                if sub["accuracy"].nunique() > 1:
                    sub["condition_code"] = (sub["condition"] == arm).astype(int)
                    X = sm.add_constant(sub[["condition_code", "period"]])
                    spec_dummies = pd.get_dummies(sub["specialty"], prefix="spec", drop_first=True)
                    if not spec_dummies.empty:
                        X = pd.concat([X, spec_dummies], axis=1)
                    X = X.astype(float)
                    model = GEE(sub["accuracy"], X, groups=sub["vignette_id"],
                                family=Binomial(), cov_struct=Exchangeable())
                    fit = model.fit()
                    or_arm = float(np.exp(fit.params["condition_code"]))
                    gee_results[arm] = {
                        f"{arm}_odds_ratio": round(or_arm, 3),
                        "pvalue": float(fit.pvalues["condition_code"]),
                        "converged": bool(fit.converged),
                    }
                else:
                    gee_results[arm] = {"error": "Perfect separation: no variance in accuracy"}
            except Exception as exc:
                gee_results[arm] = {"error": str(exc)}

    return {
        **acc,
        "contrasts": contrasts,
        "gee_logistic": gee_results,
    }


def secondary_analyses(df: pd.DataFrame) -> dict:
    out = {}
    for outcome in ["cognitive_load", "latency_ms", "n_queries_issued"]:
        contrasts = {}
        for arm_a, arm_b in CONTRASTS:
            contrasts[f"{arm_a}_vs_{arm_b}"] = paired_stats(df, outcome, arm_a, arm_b)
        out[outcome] = {
            "summary": summarize(df, outcome),
            "contrasts": contrasts,
        }
    return out


def subgroup_analyses(df: pd.DataFrame, arm: str = "gdt") -> dict:
    """Compute <arm> effect within subgroups for the forest plot (vs control)."""
    subgroups = {}
    for col in ["specialty", "experience_group", "complexity"]:
        groups = {}
        for name, sub in df.groupby(col):
            if len(sub) < 10:
                continue
            wide = sub.pivot(index="vignette_id", columns="condition", values="time_to_info").dropna()
            if len(wide) < 3:
                continue

            # Prevent division by zero
            denom = wide["control"].replace(0, np.nan)
            pct_change = ((wide["control"] - wide[arm]) / denom * 100)
            groups[str(name)] = {
                "n": int(len(wide)),
                "median_pct_change": round(float(pct_change.median()), 2),
                "mean_pct_change": round(float(pct_change.mean()), 2),
                "ci_lower": round(float(pct_change.quantile(0.025)), 2),
                "ci_upper": round(float(pct_change.quantile(0.975)), 2),
            }
        subgroups[col] = groups
    return subgroups


def human_annotation_analysis(df: pd.DataFrame) -> dict:
    out = {}
    for outcome in ["trust"]:
        if outcome not in df.columns:
            continue
        wide = df.pivot(index="vignette_id", columns="condition", values=outcome).dropna()
        if len(wide) < 3:
            continue
        control = wide["control"].values
        for arm in INTERVENTION_ARMS:
            if arm not in wide.columns:
                continue
            arm_vals = wide[arm].values
            stat, p = wilcoxon(control, arm_vals, zero_method="zsplit")
            out[f"{arm}_{outcome}"] = {
                "summary": summarize(df, outcome),
                "wilcoxon_pvalue": float(p),
                "cohens_d": round(cohens_d(arm_vals, control), 3),
            }
    return out


def gdt_vs_benchmark_analysis(df: pd.DataFrame) -> dict:
    """Paired GDT vs each benchmark (control, BM25, CMA) across outcomes.

    GDT is the primary intervention; control/BM25/CMA are benchmarks. For each
    benchmark we report the paired comparison so manuscript figures can show
    all three benchmark-vs-GDT contrasts side by side.
    """
    result = {}
    metrics = ["time_to_info", "cognitive_load", "latency_ms", "n_queries_issued", "accuracy"]
    for outcome in metrics:
        if outcome not in df.columns:
            continue
        bench_outcomes = {}
        for bench in BENCHMARKS:
            ps = paired_stats(df, outcome, bench, "gdt")
            if ps is None:
                bench_outcomes[bench] = None
                continue
            if outcome == "accuracy":
                wide = df.pivot(index="vignette_id", columns="condition",
                                values="accuracy").dropna()
                if {"gdt", bench}.issubset(wide.columns) and len(wide) > 2:
                    g = wide["gdt"].astype(int)
                    b = wide[bench].astype(int)
                    ps["accuracy_delta"] = round(float((g - b).mean() * 100), 2)
                    discord = int(((g != b).sum()))
                    ps["n_discordant"] = discord
                    better = int(((b == 0) & (g == 1)).sum())
                    worse = int(((b == 1) & (g == 0)).sum())
                    ps["n_gdt_better"] = better
                    ps["n_gdt_worse"] = worse
                    if better + worse > 0:
                        stat = (abs(better - worse) - 1) ** 2 / (better + worse)
                        ps["mcnemar_pvalue"] = float(stats.chi2.sf(stat, 1))
                    else:
                        ps["mcnemar_pvalue"] = 1.0
                else:
                    ps["accuracy_delta"] = None
                bench_outcomes[bench] = ps
                continue
            denom = float(np.median(df[df["condition"] == bench][outcome]))
            if denom and denom > 0:
                ps["median_pct_change_vs_bench"] = round(
                    float(np.median((ps["median_difference"] / denom) * 100)), 2
                ) if ps.get("median_difference") is not None else None
                ps["mean_pct_change_vs_bench"] = round(
                    float((ps.get("mean_difference", 0.0) / denom) * 100), 2
                )
            else:
                ps["median_pct_change_vs_bench"] = None
                ps["mean_pct_change_vs_bench"] = None
            bench_outcomes[bench] = ps
        result[outcome] = bench_outcomes
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze CMA experiment results")
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--annotations-dir", default=None,
                        help="Directory with human annotation CSVs (optional)")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if "condition" not in df.columns:
        raise ValueError("results.csv must contain a 'condition' column")

    # Merge human annotations if available.
    if args.annotations_dir:
        try:
            from src.annotation.export import load_annotations, merge_with_results, apply_adjudications
            ann = load_annotations(Path(args.annotations_dir))
            df = merge_with_results(df, ann)
            df = apply_adjudications(df, ann.get("adjudications", pd.DataFrame()))
            print(f"  Merged human annotations from {args.annotations_dir}")
        except Exception as e:
            print(f"  Warning: could not load annotations: {e}")

    primary = primary_time_analysis(df)
    accuracy = accuracy_analysis(df)
    secondary = secondary_analyses(df)
    subgroups = subgroup_analyses(df, arm="gdt")
    subgroups_cma = subgroup_analyses(df, arm="cma")
    human = human_annotation_analysis(df)
    gdt_vs_bench = gdt_vs_benchmark_analysis(df)

    all_stats = {
        "n_sessions": int(len(df)),
        "n_vignettes": int(df["vignette_id"].nunique()),
        "n_per_condition": {c: int((df["condition"] == c).sum()) for c in CONDITIONS},
        "primary_time": primary,
        "accuracy": accuracy,
        "secondary": secondary,
        "human_annotations": human,
        "subgroups": subgroups,
        "subgroups_cma": subgroups_cma,
        "gdt_vs_benchmarks": gdt_vs_bench,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "statistics.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    # Human-readable primary table: GDT vs each benchmark (control, BM25, CMA).
    rows = []
    for outcome in ["time_to_info", "accuracy", "cognitive_load", "latency_ms", "n_queries_issued"]:
        if outcome not in df.columns:
            continue
        row = {"outcome": outcome, **summarize(df, outcome)}
        for bench in BENCHMARKS:
            if bench == "control":
                continue
            ctrl = gdt_vs_bench.get(outcome, {}).get(bench, {}) or {}
            row[f"gdt_vs_{bench}_p"] = round(float(ctrl.get("wilcoxon_pvalue", np.nan)), 4)
            row[f"gdt_vs_{bench}_d"] = round(float(ctrl.get("cohens_d", np.nan)), 3)
        # Primary (GDT vs Control) contrast on the row as well.
        if outcome == "time_to_info":
            ctrl = primary["contrasts"].get("control_vs_gdt", {}) or {}
            row["control_vs_gdt_p"] = round(float(ctrl.get("wilcoxon_pvalue", np.nan)), 4)
            row["control_vs_gdt_d"] = round(float(ctrl.get("cohens_d", np.nan)), 3)
        elif outcome == "accuracy":
            ctrl = accuracy["contrasts"].get("control_vs_gdt", {}) or {}
            row["control_vs_gdt_p"] = round(float(ctrl.get("mcnemar_pvalue", np.nan)), 4)
            row["control_vs_gdt_d"] = np.nan
        else:
            ctrl = secondary.get(outcome, {}).get("contrasts", {}).get("control_vs_gdt", {}) or {}
            row["control_vs_gdt_p"] = round(float(ctrl.get("wilcoxon_pvalue", np.nan)), 4)
            row["control_vs_gdt_d"] = round(float(ctrl.get("cohens_d", np.nan)), 3)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "primary_results.csv", index=False)

    print("Analysis complete.")
    print(f"  Sessions: {all_stats['n_sessions']} ({all_stats['n_per_condition']})")
    print(f"  Vignettes: {all_stats['n_vignettes']}")
    p = primary["contrasts"].get("control_vs_gdt", {}) or {}
    mpc = p.get("median_pct_change")
    wp = p.get("wilcoxon_pvalue")
    mpc_s = f"{mpc}%" if mpc is not None else "n/a"
    wp_s = f"{wp:.4f}" if wp is not None else "n/a"
    print(f"  Time-to-info GDT vs Control median pct change: {mpc_s}, p={wp_s}")
    print(f"  Accuracy: {accuracy.get('control_accuracy')} control / "
          f"{accuracy.get('bm25_accuracy')} bm25 / {accuracy.get('cma_accuracy')} cma / "
          f"{accuracy.get('gdt_accuracy')} gdt")
    for arm in INTERVENTION_ARMS:
        pct = secondary["cognitive_load"]["summary"].get(f"{arm}_pct_change_mean")
        print(f"  Cognitive load {arm} vs Control pct change: {pct}%")
        lat = secondary["latency_ms"]["summary"].get(f"{arm}_pct_change_mean")
        print(f"  Latency {arm} vs Control pct change: {lat}%")
    if human:
        for k, v in human.items():
            arm_key = k.split("_")[0]
            arm_mean = v["summary"].get(f"{arm_key}_mean")
            print(f"  {k}: control={v['summary'].get('control_mean')} {arm_key}={arm_mean} d={v['cohens_d']}")
    print(f"Outputs: {out_dir / 'statistics.json'}, {out_dir / 'primary_results.csv'}")
    print("\nNext step: python src/viz/plots.py")


if __name__ == "__main__":
    main()
