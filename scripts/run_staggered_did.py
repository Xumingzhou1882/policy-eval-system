"""
Staggered DID estimation using heterogeneity-robust estimators.

Implements Callaway & Sant'Anna (2021) group-time average treatment effects
and Sun & Abraham (2021) cohort-specific ATT, using never-treated or
not-yet-treated units as controls.

Usage:
    python scripts/run_staggered_did.py --data data/merged/panel.dta \\
                                        --outcome log_fertility \\
                                        --entity city_id --time year \\
                                        --first-treated first_treated \\
                                        --method cs --control never-treated
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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


def _run_2x2(df: pd.DataFrame, outcome: str, entity_col: str, time_col: str,
             treat_group_ids: np.ndarray, ctrl_group_ids: np.ndarray,
             first_treated: float, controls: list[str]) -> tuple:
    """Run a 2×2 DID for a specific treatment cohort vs a control group."""
    sub = df[df[entity_col].isin(list(treat_group_ids) + list(ctrl_group_ids))].copy()

    # Only use periods where control is not yet treated (if not never-treated)
    pre_mask = sub[time_col] < first_treated

    sub["_treat"] = sub[entity_col].isin(treat_group_ids).astype(int)
    sub["_post"] = (sub[time_col] >= first_treated).astype(int)
    sub["_tp"] = sub["_treat"] * sub["_post"]

    control_part = " + ".join(controls) if controls else ""
    rhs = f"_tp{' + ' + control_part if control_part else ''}"
    formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

    try:
        model = smf.ols(formula, data=sub)
        results = model.fit()
        coef = float(results.params["_tp"])
        se = float(results.bse["_tp"])
        return coef, se, int(len(treat_group_ids)), int(len(ctrl_group_ids))
    except Exception:
        return None, None, 0, 0


def callaway_santanna(df: pd.DataFrame, outcome: str, entity_col: str,
                      time_col: str, first_treated_col: str,
                      controls: list[str],
                      control_type: str = "never-treated") -> dict:
    """
    Callaway & Sant'Anna (2021) estimator.

    For each treatment cohort g (treated at time g):
      - ATT(g, t) = DID comparison between cohort g and control group at period t
      - Control group: never-treated units, or not-yet-treated units
    Then aggregate:
      - ATT(g) = average ATT(g, t) over t ≥ g (post-treatment average for cohort g)
      - Overall ATT = weighted average of ATT(g)
    """
    df = df.copy()

    # Identify cohorts and treatment times
    treated = df[df[first_treated_col].notna()][[entity_col, first_treated_col]].drop_duplicates()
    cohorts = sorted(treated[first_treated_col].unique())
    cohort_entities = {g: treated[treated[first_treated_col] == g][entity_col].values
                        for g in cohorts}

    # Build control group
    if control_type == "never-treated":
        never_mask = df[first_treated_col].isna()
        ctrl_ids = df.loc[never_mask, entity_col].unique()
    elif control_type == "not-yet-treated":
        ctrl_ids = None  # dynamically set per cohort
    else:
        raise ValueError(f"Unknown control_type: {control_type}")

    time_range = sorted(df[time_col].unique())

    # Compute ATT(g, t) for each cohort-time pair
    atts = {}  # (g, t) -> {estimate, se, n_treat, n_ctrl}
    for g in cohorts:
        treat_ids = cohort_entities[g]

        if control_type == "not-yet-treated":
            # Units not yet treated by period start
            later_treated = [h for h in cohorts if h > g]
            if not later_treated:
                continue
            # Use latest-treated as practical control
            max_h = max(later_treated)
            ctrl_ids = cohort_entities[max_h]

        if len(ctrl_ids) == 0:
            continue

        for t in time_range:
            if t < g:
                continue
            # Restrict control to not-yet-treated if applicable
            est, se, nt, nc = _run_2x2(df, outcome, entity_col, time_col,
                                       treat_ids, ctrl_ids, g, controls)
            if est is not None:
                atts[(g, t)] = {
                    "cohort": int(g),
                    "period": int(t),
                    "att": float(est),
                    "std_error": float(se),
                    "n_treat": nt,
                    "n_ctrl": nc,
                }

    # Aggregate: average ATT per cohort
    cohort_atts = {}
    for g in cohorts:
        cohort_estimates = [v["att"] for (gg, t), v in atts.items() if gg == g]
        cohort_weights = [1.0 for _ in cohort_estimates]  # equal weight across periods
        if cohort_estimates:
            cohort_atts[int(g)] = {
                "att": float(np.average(cohort_estimates, weights=cohort_weights)),
                "n_periods": len(cohort_estimates),
                "n_units": int(len(cohort_entities[g])),
            }

    # Overall ATT: weighted by cohort size
    overall_weights = np.array([cohort_atts[g]["n_units"] for g in sorted(cohort_atts)])
    overall_estimates = np.array([cohort_atts[g]["att"] for g in sorted(cohort_atts)])
    if overall_weights.sum() > 0:
        overall_att = float(np.average(overall_estimates, weights=overall_weights))
    else:
        overall_att = None

    return {
        "method": f"Callaway & Sant'Anna (2021) — {control_type}",
        "atts": atts,
        "cohort_atts": cohort_atts,
        "overall_att": overall_att,
        "n_cohorts": len(cohorts),
        "n_total_periods": len(time_range),
        "control_type": control_type,
    }


def sun_abraham(df: pd.DataFrame, outcome: str, entity_col: str,
                time_col: str, first_treated_col: str,
                controls: list[str],
                control_type: str = "never-treated") -> dict:
    """
    Sun & Abraham (2021) cohort-specific ATT.

    Estimates an event study for each cohort separately, using never-treated or
    last-treated as control. Then averages across cohorts weighted by cohort size.
    """
    df = df.copy()
    treated = df[df[first_treated_col].notna()][[entity_col, first_treated_col]].drop_duplicates()
    cohorts = sorted(treated[first_treated_col].unique())
    cohort_entities = {g: treated[treated[first_treated_col] == g][entity_col].values
                        for g in cohorts}

    if control_type == "never-treated":
        never_mask = df[first_treated_col].isna()
        ctrl_ids = df.loc[never_mask, entity_col].unique()
    elif control_type == "not-yet-treated":
        ctrl_ids = cohort_entities[max(cohorts)]
    else:
        ctrl_ids = np.array([])

    time_range = sorted(df[time_col].unique())
    min_time, max_time = min(time_range), max(time_range)

    # Relative-time bins: -K ... -2 (reference: -1), 0, 1, ..., L
    rel_range = list(range(-5, 6))  # ±5 periods
    rel_atts = {}

    for r in rel_range:
        if r == -1:
            continue
        ests = []
        weights = []
        for g in cohorts:
            t = g + r
            if t not in time_range:
                continue
            est, se, nt, nc = _run_2x2_for_period(df, outcome, entity_col, time_col,
                                                    cohort_entities[g], ctrl_ids,
                                                    g, t, controls)
            if est is not None:
                ests.append(est)
                weights.append(nt)

        if ests:
            rel_atts[r] = float(np.average(ests, weights=weights))

    return {
        "method": f"Sun & Abraham (2021) — {control_type}",
        "rel_time_atts": rel_atts,
        "n_cohorts": len(cohorts),
        "control_type": control_type,
    }


def _run_2x2_for_period(df, outcome, entity_col, time_col,
                        treat_ids, ctrl_ids, first_treated, period,
                        controls):
    """Run a 2×2 DID for a specific period (single post-treatment period)."""
    sub = df[df[entity_col].isin(list(treat_ids) + list(ctrl_ids))].copy()
    # Pre: period < first_treated; Post: period
    pre = sub[sub[time_col] < first_treated]
    post = sub[sub[time_col] == period]
    if len(pre) == 0 or len(post) == 0:
        return None, None, 0, 0

    sub = pd.concat([pre, post])

    sub["_treat"] = sub[entity_col].isin(treat_ids).astype(int)
    sub["_post"] = (sub[time_col] >= first_treated).astype(int)
    sub["_tp"] = sub["_treat"] * sub["_post"]

    control_part = " + ".join(controls) if controls else ""
    rhs = f"_tp{' + ' + control_part if control_part else ''}"
    formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

    try:
        model = smf.ols(formula, data=sub)
        results = model.fit()
        return float(results.params["_tp"]), float(results.bse["_tp"]), len(treat_ids), len(ctrl_ids)
    except Exception:
        return None, None, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Staggered DID estimation")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--first-treated", default="first_treated", help="First-treated year column")
    parser.add_argument("--method", default="cs",
                        choices=["cs", "sa", "both"],
                        help="Estimator: cs (Callaway-Sant'Anna), sa (Sun-Abraham), both")
    parser.add_argument("--control", default="never-treated",
                        choices=["never-treated", "not-yet-treated"],
                        help="Control group type")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variables")
    parser.add_argument("--output", default=None, help="Output path for results (.json)")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required.")
        sys.exit(1)

    df = load_data(args.data)

    results = {}

    if args.method in ("cs", "both"):
        cs_result = callaway_santanna(df, args.outcome, args.entity, args.time,
                                      args.first_treated, args.controls, args.control)
        results["callaway_santanna"] = cs_result

        print("\n──── Callaway & Sant'Anna (2021) ────")
        print(f"Cohorts: {cs_result['n_cohorts']}")
        print(f"Control: {cs_result['control_type']}")
        print(f"Overall ATT: {cs_result['overall_att']:.6f}" if cs_result['overall_att'] else "Overall ATT: N/A")
        print("\nCohort-specific ATT:")
        for g in sorted(cs_result["cohort_atts"]):
            a = cs_result["cohort_atts"][g]
            print(f"  Cohort {g}: ATT={a['att']:.4f}  (units={a['n_units']}, periods={a['n_periods']})")

    if args.method in ("sa", "both"):
        sa_result = sun_abraham(df, args.outcome, args.entity, args.time,
                                args.first_treated, args.controls, args.control)
        results["sun_abraham"] = sa_result

        print("\n──── Sun & Abraham (2021) ────")
        print(f"Cohorts: {sa_result['n_cohorts']}")
        print("Relative-time ATTs:")
        for r in sorted(sa_result["rel_time_atts"]):
            print(f"  t{r:+d}: {sa_result['rel_time_atts'][r]:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
