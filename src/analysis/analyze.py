#!/usr/bin/env python3
"""
src/analysis/analyze.py

Statistical analysis of the CMA crossover experiment.

Primary analyses:
  1. Time-to-correct-information: paired Wilcoxon signed-rank + mixed-effects
     linear model on log-transformed times.
  2. Accuracy: McNemar-like paired proportion test + generalized estimating
     equations (GEE) with logit link clustered by vignette.
  3. Secondary outcomes: NASA-TLX, latency, perceived usefulness via paired
     tests and effect sizes (Cohen's d / Cliff's delta).

Outputs:
  outputs/primary_results.csv    - key summary statistics
  outputs/statistics.json      - full numerical results for figures
"""

import argparse
import json
import sys
from pathlib import Path

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


def cohens_d(x, y):
    """Paired Cohen's d."""
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def hedge_correction(n):
    return 1 - (3 / (4 * n - 9))


def summarize(df: pd.DataFrame, outcome: str, group_col: str = "condition") -> dict:
    grp = df.groupby(group_col)[outcome]
    return {
        "control_mean": round(float(grp.mean().get("control", np.nan)), 3),
        "control_median": round(float(grp.median().get("control", np.nan)), 3),
        "control_std": round(float(grp.std().get("control", np.nan)), 3),
        "cma_mean": round(float(grp.mean().get("cma", np.nan)), 3),
        "cma_median": round(float(grp.median().get("cma", np.nan)), 3),
        "cma_std": round(float(grp.std().get("cma", np.nan)), 3),
        "pct_change_mean": round(float((grp.mean().get("cma", np.nan) - grp.mean().get("control", np.nan))
                                   / grp.mean().get("control", np.nan) * 100), 2),
    }


def primary_time_analysis(df: pd.DataFrame) -> dict:
    """Paired analysis of time-to-correct-information."""
    wide = df.pivot(index="vignette_id", columns="condition", values="time_to_info")
    wide = wide.dropna()
    control = wide["control"].values
    cma = wide["cma"].values
    diff = control - cma
    pct_diff = diff / control * 100

    stat, pvalue = wilcoxon(control, cma, alternative="two-sided", zero_method="zsplit")

    # Mixed-effects model on log time.
    df["log_time"] = np.log1p(df["time_to_info"])
    df = df.copy()
    # Avoid singular design with explicit intercept.
    X = sm.add_constant(df[["condition_code", "period"]])
    # Add vignette-type fixed effects safely.
    vignette_dummies = pd.get_dummies(df["specialty"], prefix="spec", drop_first=True)
    if not vignette_dummies.empty:
        X = pd.concat([X, vignette_dummies], axis=1)

    try:
        model = MixedLM(df["log_time"], X, groups=df["vignette_id"])
        fit = model.fit(reml=False)
        beta_cma = float(fit.params["condition_code"])
        ci_low, ci_high = fit.conf_int().loc["condition_code"].values
        me_result = {
            "cma_beta_log_time": round(beta_cma, 4),
            "cma_beta_95ci": [round(float(ci_low), 4), round(float(ci_high), 4)],
            "pvalue": float(fit.pvalues["condition_code"]),
            "interpretation_pct": round((np.exp(beta_cma) - 1) * 100, 2),
        }
    except Exception as exc:
        me_result = {"error": str(exc)}

    return {
        "summary": summarize(df, "time_to_info"),
        "paired_test": "wilcoxon_signed_rank",
        "n_pairs": int(len(wide)),
        "median_difference_s": round(float(np.median(diff)), 2),
        "mean_difference_s": round(float(diff.mean()), 2),
        "median_pct_change": round(float(np.median(pct_diff)), 2),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_pvalue": float(pvalue),
        "cohens_d": round(cohens_d(cma, control), 3),
        "mixed_effects_log_time": me_result,
    }


def accuracy_analysis(df: pd.DataFrame) -> dict:
    """Paired binary accuracy."""
    wide = df.pivot(index="vignette_id", columns="condition", values="accuracy")
    wide = wide.dropna()
    control_acc = wide["control"].astype(int)
    cma_acc = wide["cma"].astype(int)

    table = pd.crosstab(control_acc, cma_acc)
    # McNemar exact-like: treat discordant pairs.
    discordant = wide[control_acc != cma_acc]
    b = int(((control_acc == 0) & (cma_acc == 1)).sum())  # cma better
    c = int(((control_acc == 1) & (cma_acc == 0)).sum())  # control better

    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        mcnemar_p = float(stats.chi2.sf(mcnemar_stat, 1))
    else:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0

    # Cluster-robust GEE for non-inferiority inference.
    gee_result = None
    if GEE is not None:
        try:
            df_acc = df.copy()
            X = sm.add_constant(df_acc[["condition_code", "period"]])
            spec_dummies = pd.get_dummies(df_acc["specialty"], prefix="spec", drop_first=True)
            if not spec_dummies.empty:
                X = pd.concat([X, spec_dummies], axis=1)
            model = GEE(df_acc["accuracy"], X, groups=df_acc["vignette_id"],
                        family=Binomial(), cov_struct=Exchangeable())
            fit = model.fit()
            or_cma = float(np.exp(fit.params["condition_code"]))
            gee_result = {
                "cma_odds_ratio": round(or_cma, 3),
                "pvalue": float(fit.pvalues["condition_code"]),
                "converged": bool(fit.converged),
            }
        except Exception as exc:
            gee_result = {"error": str(exc)}

    return {
        "control_accuracy": round(float(control_acc.mean()), 3),
        "cma_accuracy": round(float(cma_acc.mean()), 3),
        "n_discordant_cma_better": b,
        "n_discordant_control_better": c,
        "mcnemar_statistic": round(float(mcnemar_stat), 3),
        "mcnemar_pvalue": float(mcnemar_p),
        "confusion_table": table.to_dict(),
        "gee_logistic": gee_result,
    }


def secondary_analyses(df: pd.DataFrame) -> dict:
    out = {}
    for outcome in ["cognitive_load", "latency_ms", "n_queries_issued"]:
        wide = df.pivot(index="vignette_id", columns="condition", values=outcome).dropna()
        control = wide["control"].values
        cma = wide["cma"].values
        stat, p = wilcoxon(control, cma, zero_method="zsplit")
        out[outcome] = {
            "summary": summarize(df, outcome),
            "wilcoxon_pvalue": float(p),
            "cohens_d": round(cohens_d(cma, control), 3),
        }
    return out


def subgroup_analyses(df: pd.DataFrame) -> dict:
    """Compute CMA effect within subgroups for forest plot."""
    subgroups = {}
    for col in ["specialty", "experience_group", "complexity"]:
        groups = {}
        for name, sub in df.groupby(col):
            if len(sub) < 10:
                continue
            wide = sub.pivot(index="vignette_id", columns="condition", values="time_to_info").dropna()
            if len(wide) < 3:
                continue
            pct_change = ((wide["control"] - wide["cma"]) / wide["control"] * 100)
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
        cma = wide["cma"].values
        stat, p = wilcoxon(control, cma, zero_method="zsplit")
        out[outcome] = {
            "summary": summarize(df, outcome),
            "wilcoxon_pvalue": float(p),
            "cohens_d": round(cohens_d(cma, control), 3),
        }
    return out


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
    subgroups = subgroup_analyses(df)
    human = human_annotation_analysis(df)

    all_stats = {
        "n_sessions": int(len(df)),
        "n_vignettes": int(df["vignette_id"].nunique()),
        "primary_time": primary,
        "accuracy": accuracy,
        "secondary": secondary,
        "human_annotations": human,
        "subgroups": subgroups,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "statistics.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    # Human-readable primary table.
    rows = []
    for outcome in ["time_to_info", "accuracy", "cognitive_load", "latency_ms", "n_queries_issued", "trust"]:
        if outcome not in df.columns:
            continue
        row = {
            "outcome": outcome,
            **summarize(df, outcome),
        }
        if outcome == "time_to_info":
            row["pvalue_wilcoxon"] = round(float(primary["wilcoxon_pvalue"]), 4)
            row["cohens_d"] = round(float(primary["cohens_d"]), 3)
        elif outcome == "accuracy":
            row["pvalue_wilcoxon"] = round(float(accuracy["mcnemar_pvalue"]), 4)
            row["cohens_d"] = np.nan
        elif outcome == "trust":
            h = human.get(outcome, {})
            row["pvalue_wilcoxon"] = round(float(h.get("wilcoxon_pvalue", np.nan)), 4)
            row["cohens_d"] = round(float(h.get("cohens_d", np.nan)), 3)
        else:
            row["pvalue_wilcoxon"] = round(float(secondary.get(outcome, {}).get("wilcoxon_pvalue", np.nan)), 4)
            row["cohens_d"] = round(float(secondary.get(outcome, {}).get("cohens_d", np.nan)), 3)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "primary_results.csv", index=False)

    print("Analysis complete.")
    print(f"  Sessions: {all_stats['n_sessions']}")
    print(f"  Vignettes: {all_stats['n_vignettes']}")
    print(f"  Time-to-info median reduction: {primary['median_pct_change']}%")
    print(f"  Accuracy control={accuracy['control_accuracy']} cma={accuracy['cma_accuracy']}")
    print(f"  Cognitive load mean reduction: {secondary['cognitive_load']['summary']['pct_change_mean']}%")
    print(f"  Latency mean reduction: {secondary['latency_ms']['summary']['pct_change_mean']}%")
    if human:
        for k, v in human.items():
            print(f"  {k}: control={v['summary']['control_mean']} cma={v['summary']['cma_mean']} d={v['cohens_d']}")
    print(f"Outputs: {out_dir / 'statistics.json'}, {out_dir / 'primary_results.csv'}")
    print("\nNext step: python src/viz/plots.py")


if __name__ == "__main__":
    main()
