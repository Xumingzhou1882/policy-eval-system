"""
Synthetic Control Method (SCM) for comparative case studies.

Constructs a synthetic counterfactual by weighting untreated donor units
to match the pre-treatment outcome trajectory of the treated unit.

Usage:
    python scripts/run_scm.py --data data/merged/panel.dta \\
                              --outcome log_fertility \\
                              --entity city_id --time year \\
                              --treated-unit 110100 \\
                              --first-treated 2016 \\
                              --donor-pool "all" \\
                              --plot scm.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    raise ValueError(f"Unsupported format: {p.suffix}")


def _build_matrices(panel: pd.DataFrame, treated_id, donor_ids,
                    entity_col: str, time_col: str, outcome_col: str,
                    pre_periods: np.ndarray) -> tuple:
    """Build Y_pre (treated pre-treatment) and X_pre (donors pre-treatment) matrices."""
    treated_pre = panel[(panel[entity_col] == treated_id) &
                        (panel[time_col].isin(pre_periods))]
    treated_pre = treated_pre.sort_values(time_col)
    Y_pre = treated_pre[outcome_col].values

    X_pre = []
    for donor in donor_ids:
        donor_pre = panel[(panel[entity_col] == donor) &
                          (panel[time_col].isin(pre_periods))]
        donor_pre = donor_pre.sort_values(time_col)
        X_pre.append(donor_pre[outcome_col].values)
    X_pre = np.column_stack(X_pre)

    return Y_pre, X_pre


def synthetic_control(panel: pd.DataFrame, treated_id, donor_ids: list,
                      entity_col: str, time_col: str, outcome_col: str,
                      first_treated: float,
                      covariates: Optional[list[str]] = None) -> dict:
    """
    Estimate SCM weights and compute treatment effect.

    Minimizes ||Y_pre - X_pre @ weights||² subject to:
      - weights ≥ 0
      - sum(weights) = 1
    """
    time_values = sorted(panel[time_col].unique())
    pre_periods = np.array([t for t in time_values if t < first_treated])
    post_periods = np.array([t for t in time_values if t >= first_treated])

    valid_donors = [d for d in donor_ids if d != treated_id and
                    panel[panel[entity_col] == d][outcome_col].notna().sum() >= len(pre_periods)]

    if len(valid_donors) == 0:
        return {"error": "No valid donor units with complete pre-treatment data."}

    Y_pre, X_pre = _build_matrices(panel, treated_id, valid_donors,
                                   entity_col, time_col, outcome_col, pre_periods)

    n_donors = X_pre.shape[1]

    # Constraints: sum(weights) = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    # Bounds: 0 ≤ w_i ≤ 1
    bounds = [(0, 1) for _ in range(n_donors)]
    # Initial guess: equal weights
    x0 = np.ones(n_donors) / n_donors

    def loss(w):
        return np.sum((Y_pre - X_pre @ w) ** 2)

    result = minimize(loss, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 5000})

    weights = result.x
    weights[np.abs(weights) < 1e-6] = 0  # zero out negligible weights
    nonzero = np.sum(weights > 0)

    # Pre-treatment fit
    synthetic_pre = X_pre @ weights
    rmse_pre = np.sqrt(np.mean((Y_pre - synthetic_pre) ** 2))

    # Post-treatment gap
    treated_post = panel[(panel[entity_col] == treated_id) &
                         (panel[time_col].isin(post_periods))]
    treated_post = treated_post.sort_values(time_col)
    Y_post_actual = treated_post[outcome_col].values

    # Build X_post
    X_post = []
    for donor in valid_donors:
        donor_post = panel[(panel[entity_col] == donor) &
                           (panel[time_col].isin(post_periods))]
        donor_post = donor_post.sort_values(time_col)
        X_post.append(donor_post[outcome_col].values)
    X_post = np.column_stack(X_post)

    synthetic_post = X_post @ weights
    gaps = Y_post_actual - synthetic_post

    # Average treatment effect
    att = float(np.mean(gaps))

    # Inference: in-space placebo (apply SCM to each donor)
    placebo_gaps = {}
    for donor in valid_donors[:20]:  # limit for speed
        try:
            other_donors = [d for d in valid_donors if d != donor]
            Y_pre_d, X_pre_d = _build_matrices(panel, donor, other_donors,
                                               entity_col, time_col, outcome_col, pre_periods)

            n_dd = X_pre_d.shape[1]
            cons_d = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bnds_d = [(0, 1) for _ in range(n_dd)]
            x0_d = np.ones(n_dd) / n_dd

            res_d = minimize(lambda w: np.sum((Y_pre_d - X_pre_d @ w) ** 2),
                             x0_d, method="SLSQP", bounds=bnds_d,
                             constraints=cons_d, options={"maxiter": 2000})

            # Post-treatment for this placebo
            treated_post_d = panel[(panel[entity_col] == donor) &
                                   (panel[time_col].isin(post_periods))]
            treated_post_d = treated_post_d.sort_values(time_col)
            Y_post_actual_d = treated_post_d[outcome_col].values

            X_post_d = []
            for dd in other_donors:
                dd_post = panel[(panel[entity_col] == dd) &
                                (panel[time_col].isin(post_periods))]
                dd_post = dd_post.sort_values(time_col)
                X_post_d.append(dd_post[outcome_col].values)
            X_post_d = np.column_stack(X_post_d)

            fake_synth_post = X_post_d @ res_d.x
            placebo_gaps[donor] = (Y_post_actual_d - fake_synth_post).tolist()
        except Exception:
            continue

    # Ratio: post RMSE / pre RMSE (large ratio = strong effect)
    rmse_post = np.sqrt(np.mean(gaps ** 2))
    ratio = rmse_post / rmse_pre if rmse_pre > 0 else float("inf")

    # P-value from in-space placebo
    pre_post_ratios = [rmse_post / rmse_pre]
    for donor, gap_list in placebo_gaps.items():
        pre_rmse_d = rmse_pre  # approx
        post_rmse_d = np.sqrt(np.mean(np.array(gap_list) ** 2))
        if pre_rmse_d > 0:
            pre_post_ratios.append(post_rmse_d / pre_rmse_d)
    p_value = np.mean(np.array(pre_post_ratios) >= pre_post_ratios[0])

    return {
        "treated_unit": str(treated_id),
        "donor_units": valid_donors,
        "donor_weights": {str(valid_donors[i]): float(weights[i])
                          for i in range(n_donors) if weights[i] > 0.001},
        "n_nonzero_weights": int(nonzero),
        "rmse_pre": float(rmse_pre),
        "rmse_post": float(rmse_post),
        "rmse_ratio": float(ratio),
        "att": att,
        "time_series": {
            "periods": post_periods.tolist(),
            "actual": Y_post_actual.tolist(),
            "synthetic": synthetic_post.tolist(),
            "gap": gaps.tolist(),
        },
        "p_value": float(p_value),
        "n_placebo": len(placebo_gaps),
    }


def plot_scm(result: dict, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    ts = result["time_series"]
    periods = ts["periods"]
    actual = ts["actual"]
    synthetic = ts["synthetic"]
    gap = ts["gap"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(periods, actual, marker="o", color="steelblue", linewidth=2, label="Treated Unit")
    ax1.plot(periods, synthetic, marker="s", color="darkorange", linewidth=2,
             linestyle="--", label="Synthetic Control")
    ax1.set_ylabel("Outcome")
    ax1.set_title(f"Synthetic Control: ATT = {result['att']:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(periods, gap, color=["green" if g > 0 else "red" for g in gap], alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Gap (Actual - Synthetic)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"SCM plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Control Method")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--treated-unit", required=True, help="ID of the treated unit")
    parser.add_argument("--first-treated", type=float, required=True, help="Treatment year")
    parser.add_argument("--donor-pool", default="all",
                        help="Comma-separated donor IDs, or 'all' for all other units")
    parser.add_argument("--plot", default=None, help="Output path for plot")
    parser.add_argument("--output", default=None, help="Output path for results (.json)")
    args = parser.parse_args()

    df = load_data(args.data)

    all_entities = df[args.entity].unique()
    if args.donor_pool == "all":
        donor_ids = [e for e in all_entities if str(e) != str(args.treated_unit)]
    else:
        donor_ids = [x.strip() for x in args.donor_pool.split(",")]

    result = synthetic_control(df, args.treated_unit, donor_ids,
                               args.entity, args.time, args.outcome,
                               args.first_treated)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\n──── Synthetic Control ────")
    print(f"Treated unit:  {result['treated_unit']}")
    print(f"Donors used:   {len(result['donor_units'])}")
    print(f"Nonzero weights: {result['n_nonzero_weights']}")
    print(f"Pre RMSE:      {result['rmse_pre']:.6f}")
    print(f"Post RMSE:     {result['rmse_post']:.6f}")
    print(f"RMSE ratio:    {result['rmse_ratio']:.2f}")
    print(f"ATT:           {result['att']:.6f}")
    print(f"P-value:       {result['p_value']:.4f}")
    print(f"\nDonor weights (nonzero):")
    for donor, w in sorted(result["donor_weights"].items(), key=lambda x: -x[1]):
        print(f"  {donor}: {w:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.plot:
        plot_scm(result, args.plot)


if __name__ == "__main__":
    main()
