"""
Goodman-Bacon decomposition for staggered DID designs.

Decomposes the TWFE estimate into all possible 2×2 DID comparisons,
flagging negative weights that indicate bias from heterogeneous treatment effects.

Usage:
    python scripts/bacon_decomp.py --data data/merged/panel.dta \\
                                   --outcome log_fertility \\
                                   --entity city_id --time year \\
                                   --treated treated --first-treated first_treated
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


def bacon_decomposition(df: pd.DataFrame, outcome: str, entity_col: str,
                        time_col: str, treated_col: str,
                        first_treated_col: str) -> dict:
    """
    Implement Goodman-Bacon decomposition manually.

    The TWFE estimator in staggered DID is a weighted average of:
    1. Early-treated vs. never-treated (or latest-treated) — valid comparisons
    2. Late-treated vs. early-treated (using already-treated as control) — problematic
       when treatment effects are heterogeneous. These get negative weights.
    """
    df = df.copy()
    df["_treated"] = df[treated_col].astype(float)
    df["_t"] = df[time_col].astype(int)
    df["_ft"] = df[first_treated_col].astype(float)

    # Assign never-treated units (NaN first_treated) a marker
    never_treated_mask = df["_ft"].isna()
    df.loc[never_treated_mask, "_ft"] = 9999

    # Identify treated entities and their treatment times
    treated = df[df["_treated"] == 1][[entity_col, "_ft"]].drop_duplicates()

    # Build list of unique treatment times
    ttimes = sorted(treated["_ft"].unique())

    comparisons = []
    total_weight = 0.0
    total_est = 0.0
    negative_weight_sum = 0.0

    for i, t1 in enumerate(ttimes):
        group1 = treated[treated["_ft"] == t1][entity_col].unique()

        # Comparison with never-treated
        never = df[df["_ft"] == 9999][entity_col].unique()
        if len(never) > 0 and len(group1) > 0:
            est, weight, n1, n2 = _compute_2x2(
                df, outcome, entity_col, time_col, "_treated",
                group1, never, t1, t1
            )
            if est is not None:
                comparisons.append({
                    "type": "treated vs never-treated",
                    "treatment_time": int(t1),
                    "control_group": "never-treated",
                    "estimate": float(est),
                    "weight": float(weight),
                })
                total_est += float(est) * float(weight)
                total_weight += float(weight)
                if float(weight) < 0:
                    negative_weight_sum += float(weight)

        # Pairwise comparisons between groups treated at different times
        for j, t2 in enumerate(ttimes):
            if t2 <= t1:
                continue
            group2 = treated[treated["_ft"] == t2][entity_col].unique()
            if len(group2) == 0:
                continue

            # Early group vs later group, using periods BEFORE t2
            # (when later group is still untreated)
            est, weight, n1, n2 = _compute_2x2(
                df, outcome, entity_col, time_col, "_treated",
                group1, group2, t1, t2, pre_period_end=t2
            )
            if est is not None:
                comparisons.append({
                    "type": "early vs late (timing comparison)",
                    "treatment_time": int(t1),
                    "control_time": int(t2),
                    "estimate": float(est),
                    "weight": float(weight),
                })
                total_est += float(est) * float(weight)
                total_weight += float(weight)
                if float(weight) < 0:
                    negative_weight_sum += float(weight)

    if total_weight > 0:
        total_est /= total_weight

    pct_negative = (negative_weight_sum / total_weight * 100) if total_weight > 0 else 0

    return {
        "comparisons": comparisons,
        "n_comparisons": len(comparisons),
        "total_estimate": float(total_est),
        "total_weight": float(total_weight),
        "negative_weight_sum": float(negative_weight_sum),
        "negative_weight_pct": float(pct_negative),
        "warning": (
            f"Negative weights account for {pct_negative:.1f}% of total. "
            "TWFE is unreliable." if pct_negative > 10
            else f"Negative weights are modest ({pct_negative:.1f}%). TWFE may be acceptable."
        ),
    }


def _compute_2x2(df: pd.DataFrame, outcome: str, entity_col: str,
                 time_col: str, treated_col: str,
                 group1_ids: np.ndarray, group2_ids: np.ndarray,
                 t1: float, t2: float,
                 pre_period_end: float = None) -> tuple:
    """
    Compute a single 2×2 DID comparison.

    group1: treated earlier (at t1)
    group2: treated later (at t2), or never-treated
    pre_period_end: if set, only use periods before this time for both groups
    """
    sub = df[df[entity_col].isin(list(group1_ids) + list(group2_ids))].copy()

    # Restrict to relevant periods
    if pre_period_end is not None:
        # Pre: before t1; Post: t1 ≤ t < pre_period_end
        sub["_period"] = sub[time_col].apply(
            lambda x: "pre" if x < t1 else ("post" if x < pre_period_end else None)
        )
    else:
        sub["_period"] = sub[time_col].apply(lambda x: "pre" if x < t1 else "post")

    sub = sub[sub["_period"].notna()]

    if len(sub) < 4:
        return None, 0, 0, 0

    n1 = len(group1_ids)
    n2 = len(group2_ids)

    # Mean outcome by group and period
    means = sub.groupby(["_period", lambda x: x[entity_col].isin(group1_ids).map({True: "treat", False: "ctrl"})])[outcome].mean()

    if len(means) < 4:
        return None, 0, n1, n2

    try:
        pre_treat = means.loc[("pre", "treat")]
        post_treat = means.loc[("post", "treat")]
        pre_ctrl = means.loc[("pre", "ctrl")]
        post_ctrl = means.loc[("post", "ctrl")]
    except KeyError:
        return None, 0, n1, n2

    did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

    # Weight: proportional to group size × variance of treatment
    weight = (n1 * n2) / (n1 + n2) * (sub["_period"] == "post").mean()

    return did, weight, n1, n2


def main():
    parser = argparse.ArgumentParser(description="Goodman-Bacon decomposition")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--treated", default="treated", help="Treatment dummy column")
    parser.add_argument("--first-treated", default="first_treated", help="First-treated year column")
    parser.add_argument("--output", default=None, help="Output path for results (.json)")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required.")
        sys.exit(1)

    df = load_data(args.data)

    result = bacon_decomposition(df, args.outcome, args.entity, args.time,
                                 args.treated, args.first_treated)

    print("\n──── Goodman-Bacon Decomposition ────")
    print(f"Number of 2×2 comparisons: {result['n_comparisons']}")
    print(f"Overall TWFE estimate:      {result['total_estimate']:.6f}")
    print(f"Negative weight share:      {result['negative_weight_pct']:.1f}%")
    print(f"\n{result['warning']}")

    print("\nComparison details:")
    for c in result["comparisons"]:
        type_label = c["type"]
        ids = f"t={c.get('treatment_time','')} ctrl={c.get('control_group','') or c.get('control_time','')}"
        print(f"  [{type_label}] {ids}:  est={c['estimate']:.4f}  weight={c['weight']:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
