"""
Event study estimation and plotting for staggered / single-treatment designs.

Generates relative-time coefficients and tests parallel pre-trends.

Usage:
    python scripts/run_event_study.py --data data/merged/panel.dta \\
                                      --outcome log_fertility \\
                                      --entity city_id --time year \\
                                      --first-treated first_treated \\
                                      --n-pre 5 --n-post 5 \\
                                      --plot event_study.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    HAS_STATS = True
except ImportError:
    HAS_STATS = False


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    raise ValueError(f"Unsupported format: {p.suffix}")


def build_relative_time(df: pd.DataFrame, entity_col: str, time_col: str,
                        first_treated_col: str, n_pre: int, n_post: int,
                        reference: int = -1) -> pd.DataFrame:
    """Add relative-time dummies to the panel.

    reference: the omitted relative period (default: -1, the period before treatment).
    """
    df = df.copy()
    df["_rel_time"] = df[time_col].astype(int) - df[first_treated_col].astype(float)

    # Cap at edges to avoid small-sample bins
    df["_rel_time_w"] = df["_rel_time"].clip(-n_pre, n_post)

    # Create dummy variables
    for t in range(-n_pre, n_post + 1):
        if t == reference:
            continue
        df[f"_rel_{t}"] = (df["_rel_time_w"] == t).astype(int)

    return df


def run_event_study(df: pd.DataFrame, outcome: str, entity_col: str, time_col: str,
                    n_pre: int, n_post: int, reference: int, controls: list[str]) -> dict:
    """Estimate event-study coefficients and test pre-trends."""

    # Build dummies
    rel_dummies = [f"_rel_{t}" for t in range(-n_pre, n_post + 1) if t != reference]
    df = build_relative_time(df, entity_col, time_col, "first_treated",
                             n_pre, n_post, reference)

    control_part = " + ".join(controls) if controls else ""
    rhs = " + ".join(rel_dummies)
    if control_part:
        rhs += f" + {control_part}"
    formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

    model = smf.ols(formula, data=df)
    results = model.fit()

    # Extract coefficients
    coefs = {}
    for t in range(-n_pre, n_post + 1):
        if t == reference:
            coefs[t] = {"coefficient": 0.0, "std_error": 0.0, "p_value": 1.0}
        else:
            key = f"_rel_{t}"
            if key in results.params:
                coefs[t] = {
                    "coefficient": float(results.params[key]),
                    "std_error": float(results.bse[key]),
                    "p_value": float(results.pvalues[key]),
                }

    # Test joint significance of pre-treatment coefficients
    pre_dummies = [f"_rel_{t}" for t in range(-n_pre, 0) if t != reference]
    pre_dummies_found = [d for d in pre_dummies if d in results.params.index]

    pre_trends_result = {"jointly_zero": None, "f_stat": None, "p_value": None}
    if pre_dummies_found:
        try:
            hypotheses = " = ".join(pre_dummies_found) + " = 0"
            f_test = results.f_test(hypotheses)
            pval = float(f_test.pvalue)
            pre_trends_result = {
                "jointly_zero": pval > 0.05,
                "f_stat": float(f_test.statistic[0][0]) if hasattr(f_test.statistic, 'shape') else float(f_test.statistic),
                "p_value": pval,
            }
        except Exception as e:
            pre_trends_result = {"error": str(e)}

    return {
        "coefficients": coefs,
        "reference_period": reference,
        "pre_trends_test": pre_trends_result,
        "n_obs": int(results.nobs),
    }


def plot_event_study(result: dict, output_path: str):
    """Generate an event-study coefficient plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    coefs = result["coefficients"]
    times = sorted(coefs.keys())
    estimates = [coefs[t]["coefficient"] for t in times]
    ses = [coefs[t]["std_error"] for t in times]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(-0.5, color="red", linewidth=0.8, linestyle="--", label="Treatment")

    ci_lower = [e - 1.96 * s for e, s in zip(estimates, ses)]
    ci_upper = [e + 1.96 * s for e, s in zip(estimates, ses)]

    ax.plot(times, estimates, marker="o", color="steelblue", linewidth=1.5)
    ax.fill_between(times, ci_lower, ci_upper, alpha=0.2, color="steelblue")

    ax.set_xlabel("Periods relative to treatment")
    ax.set_ylabel("Coefficient")
    ax.set_title("Event Study")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Event study plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Event study estimation")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--first-treated", default="first_treated", help="First-treated year column")
    parser.add_argument("--n-pre", type=int, default=5, help="Pre-treatment periods to include")
    parser.add_argument("--n-post", type=int, default=5, help="Post-treatment periods to include")
    parser.add_argument("--reference", type=int, default=-1, help="Reference period (default: -1)")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variables")
    parser.add_argument("--plot", default=None, help="Output path for event study plot (.png)")
    parser.add_argument("--output", default=None, help="Output path for coefficients (.json)")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required. Install with: pip install statsmodels")
        sys.exit(1)

    df = load_data(args.data)
    result = run_event_study(df, args.outcome, args.entity, args.time,
                             args.n_pre, args.n_post, args.reference, args.controls)

    print("\n──── Event Study ────")
    print(f"N obs: {result['n_obs']}")
    pt = result["pre_trends_test"]
    if "error" not in pt:
        label = "✓ Parallel trends hold" if pt["jointly_zero"] else "✗ Parallel trends violated"
        print(f"Pre-trends F-test: {label} (F={pt['f_stat']:.3f}, p={pt['p_value']:.4f})")
    else:
        print(f"Pre-trends test error: {pt['error']}")

    print("\nCoefficients by relative period:")
    for t in sorted(result["coefficients"].keys()):
        c = result["coefficients"][t]
        stars = "***" if c["p_value"] < 0.01 else ("**" if c["p_value"] < 0.05 else ("*" if c["p_value"] < 0.1 else ""))
        print(f"  t={t:3d}:  {c['coefficient']:8.4f}  ({c['std_error']:.4f})  p={c['p_value']:.3f} {stars}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.plot:
        plot_event_study(result, args.plot)


if __name__ == "__main__":
    main()
