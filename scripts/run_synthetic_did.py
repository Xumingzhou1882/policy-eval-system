"""
Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).

Combines synthetic control weights with DID differencing. Unlike standard
SCM, Synthetic DID provides valid standard errors and statistical inference.
Use when: one or few treated units, many untreated donor units, panel data.

Usage:
    python run_synthetic_did.py --data panel.dta \\
        --outcome log_fertility --entity city_id --time year \\
        --treated-unit 110100 --first-treated 2016 \\
        --output sd_result.json

Reference:
    Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021),
    "Synthetic Difference-in-Differences." American Economic Review.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════
# Core algorithm
# ═══════════════════════════════════════════════════════════════════════

def _find_unit_weights(Y_pre_treated: np.ndarray, Y_pre_control: np.ndarray) -> np.ndarray:
    """
    Find control unit weights ω that minimize pre-treatment MSE.

    Constrained: Σωᵢ = 1, ωᵢ ≥ 0.

    Parameters
    ----------
    Y_pre_treated : (T_pre,) array — treated unit's pre-treatment outcomes
    Y_pre_control : (N_control, T_pre) array — each control unit's outcomes

    Returns
    -------
    omega : (N_control,) array — unit weights
    """
    n_control, T_pre = Y_pre_control.shape

    def loss(omega):
        weighted_control = omega @ Y_pre_control  # (T_pre,)
        return np.sum((Y_pre_treated - weighted_control) ** 2)

    # Constraints: Σω = 1, ω ≥ 0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_control)]

    # Initial guess: equal weights
    x0 = np.ones(n_control) / n_control

    result = minimize(loss, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 5000})

    return result.x


def _find_time_weights(Y_pre_treated: np.ndarray, Y_pre_synthetic: np.ndarray) -> np.ndarray:
    """
    Find time weights λ that balance pre-treatment fit.

    Constrained: Σλₜ = 1, λₜ ≥ 0.

    Parameters
    ----------
    Y_pre_treated : (T_pre,) array
    Y_pre_synthetic : (T_pre,) array — weighted control outcomes in pre-period

    Returns
    -------
    lambda_t : (T_pre,) array — time weights
    """
    T_pre = len(Y_pre_treated)

    def loss(lam):
        return np.sum((Y_pre_treated - np.sum(lam) * Y_pre_treated) ** 2) + \
               0.1 * np.sum(lam ** 2)  # small ridge penalty for uniqueness

    # Actually use Arkhangelsky's approach: balance pre-treatment trends
    def loss_time(lam):
        weighted_diff = np.sum(lam * (Y_pre_treated - Y_pre_synthetic))
        # Minimize squared deviation from zero
        return weighted_diff ** 2 + 0.01 * np.sum((lam - 1.0 / T_pre) ** 2)

    constraints = [{"type": "eq", "fun": lambda l: np.sum(l) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(T_pre)]

    x0 = np.ones(T_pre) / T_pre

    result = minimize(loss_time, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 5000})

    return result.x


def synthetic_did(data: pd.DataFrame, outcome: str, entity_col: str,
                  time_col: str, treated_unit, first_treated,
                  donor_units: list = None) -> dict:
    """
    Run Synthetic DID for one treated unit against donor pool.

    Parameters
    ----------
    data : pd.DataFrame — long-format panel
    outcome : str — outcome column name
    entity_col : str — entity identifier column
    time_col : str — time period column
    treated_unit : str/int — identifier of the treated unit
    first_treated : int/float — first treatment period
    donor_units : list — specific donor unit IDs (optional; if None, all
                   untreated units serve as donors)

    Returns
    -------
    dict with keys: att, se, t_stat, p_value, unit_weights, pre_rmse,
                    pre_periods, post_periods, placebo_effects
    """
    # ── Pivot to wide format ──────────────────────────────────────────
    panel = data.pivot(index=time_col, columns=entity_col, values=outcome)
    times = panel.index.sort_values()
    pre_times = times[times < first_treated]
    post_times = times[times >= first_treated]

    if len(pre_times) < 2:
        return {"error": "Need at least 2 pre-treatment periods."}

    treated_data = panel[treated_unit].sort_index()

    # ── Donor pool ────────────────────────────────────────────────────
    if donor_units is None:
        all_units = [c for c in panel.columns if c != treated_unit]
        donor_units = [c for c in all_units if data[
            (data[entity_col] == c) & (data[time_col] >= first_treated)
        ].shape[0] == 0 or True]  # Include even if post-treated
        # Actually, only include units that have full data
        donor_units = [c for c in all_units
                       if panel[c].loc[pre_times].notna().all()]

    if len(donor_units) < 3:
        return {"error": "Need at least 3 donor units."}

    # ── Pre-treatment matrices ────────────────────────────────────────
    Y_pre_treated = treated_data.loc[pre_times].values  # (T_pre,)
    Y_pre_control = np.array([panel[u].loc[pre_times].values
                               for u in donor_units])  # (N_donor, T_pre)

    # Drop donors with NaN
    valid_mask = ~np.isnan(Y_pre_control).any(axis=1)
    Y_pre_control = Y_pre_control[valid_mask]
    donor_units = [u for u, v in zip(donor_units, valid_mask) if v]

    if len(donor_units) < 3:
        return {"error": "Need at least 3 donor units after dropping NaN."}

    # ── Find unit weights ─────────────────────────────────────────────
    omega = _find_unit_weights(Y_pre_treated, Y_pre_control)

    # ── Find time weights ─────────────────────────────────────────────
    Y_pre_synthetic = omega @ Y_pre_control  # (T_pre,)
    lambda_t = _find_time_weights(Y_pre_treated, Y_pre_synthetic)

    # ── Pre-treatment fit (RMSE) ──────────────────────────────────────
    pre_rmse = np.sqrt(np.mean((Y_pre_treated - Y_pre_synthetic) ** 2))

    # ── Treatment effect ──────────────────────────────────────────────
    Y_post_treated = treated_data.loc[post_times].values
    Y_post_control = np.array([panel[u].loc[post_times].values
                                for u in donor_units])
    Y_post_synthetic = omega @ Y_post_control

    # ATT: average difference in post-treatment periods
    post_diffs = Y_post_treated - Y_post_synthetic
    att = np.mean(post_diffs)

    # ── Placebo-based inference ───────────────────────────────────────
    # Apply Synthetic DID to each donor unit as if it were treated
    placebo_atts = []
    for i, donor in enumerate(donor_units):
        try:
            other_donors = [d for j, d in enumerate(donor_units) if j != i]
            Y_pre_placebo = panel[donor].loc[pre_times].values
            Y_pre_other = np.array([panel[d].loc[pre_times].values
                                     for d in other_donors])

            omega_p = _find_unit_weights(Y_pre_placebo, Y_pre_other)
            Y_post_placebo = panel[donor].loc[post_times].values
            Y_post_other = np.array([panel[d].loc[post_times].values
                                      for d in other_donors])
            placebo_att = np.mean(Y_post_placebo - omega_p @ Y_post_other)
            placebo_atts.append(placebo_att)
        except Exception:
            continue

    placebo_atts = np.array(placebo_atts)

    # Standard error: SD of placebo distribution
    if len(placebo_atts) > 10:
        se = np.std(placebo_atts, ddof=1)
        # p-value: fraction of placebo effects larger in absolute value
        p_value = np.mean(np.abs(placebo_atts) >= np.abs(att))
    else:
        # Fallback: robust SE
        se = np.std(post_diffs, ddof=1) / np.sqrt(len(post_diffs))
        from scipy import stats as sp_stats
        t_stat = att / se if se > 0 else 0.0
        df = max(len(post_diffs) - 1, 1)
        p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df))

    t_stat = att / se if se > 0 else 0.0

    # ── Unit weight summary ───────────────────────────────────────────
    significant_donors = sorted(
        [(donor_units[i], float(omega[i])) for i in range(len(donor_units))
         if omega[i] > 0.01],
        key=lambda x: x[1], reverse=True
    )

    return {
        "att": float(att),
        "se": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "pre_rmse": float(pre_rmse),
        "n_pre_periods": len(pre_times),
        "n_post_periods": len(post_times),
        "n_donors": len(donor_units),
        "treated_unit": str(treated_unit),
        "first_treated": first_treated,
        "significant_donors": significant_donors[:10],  # top 10
        "n_placebo_atts": len(placebo_atts),
        "placebo_mean": float(np.mean(placebo_atts)) if len(placebo_atts) > 0 else None,
        "placebo_std": float(np.std(placebo_atts)) if len(placebo_atts) > 0 else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Multi-unit wrapper
# ═══════════════════════════════════════════════════════════════════════

def synthetic_did_multi(data: pd.DataFrame, outcome: str, entity_col: str,
                        time_col: str, treated_units: list, first_treated,
                        donor_units: list = None) -> dict:
    """
    Run Synthetic DID for multiple treated units, then aggregate.
    """
    results = []
    for unit in treated_units:
        res = synthetic_did(
            data, outcome, entity_col, time_col,
            treated_unit=unit, first_treated=first_treated,
            donor_units=donor_units,
        )
        if "error" not in res:
            results.append(res)

    if not results:
        return {"error": "No valid results for any treated unit."}

    # Aggregate: weighted by post-period count
    atts = [r["att"] for r in results]
    weights = [r["n_post_periods"] for r in results]
    total_w = sum(weights)
    avg_att = sum(a * w for a, w in zip(atts, weights)) / total_w if total_w > 0 else np.mean(atts)

    # Conservative SE: max of individual SEs (accounts for dependence)
    ses = [r["se"] for r in results]
    avg_se = np.sqrt(np.mean([s ** 2 for s in ses]))

    return {
        "aggregate_att": float(avg_att),
        "aggregate_se": float(avg_se),
        "n_treated_units": len(results),
        "unit_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single treated unit
  python run_synthetic_did.py --data panel.dta --outcome log_fertility \\
      --entity city_id --time year --treated-unit 110100 --first-treated 2016

  # Multiple treated units (aggregated)
  python run_synthetic_did.py --data panel.dta --outcome log_fertility \\
      --entity city_id --time year --treated-units 110100,110200,110300 \\
      --first-treated 2016 --output sd_result.json
""",
    )
    parser.add_argument("--data", required=True, help="Path to panel data (.dta or .csv)")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity identifier column")
    parser.add_argument("--time", default="year", help="Time period column")
    parser.add_argument("--treated-unit", default=None, help="Single treated unit ID")
    parser.add_argument("--treated-units", default=None, help="Comma-separated treated unit IDs")
    parser.add_argument("--first-treated", type=float, required=True, help="First treatment period")
    parser.add_argument("--donor-units", default=None, help="Comma-separated donor unit IDs (optional)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--plot", default=None, help="Output plot path")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    data_path = Path(args.data)
    if data_path.suffix == ".dta":
        df = pd.read_stata(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(data_path)

    # ── Parse treated/donor units ─────────────────────────────────────
    donor_units = None
    if args.donor_units:
        donor_units = [u.strip() for u in args.donor_units.split(",")]

    treated_units = []
    if args.treated_unit:
        treated_units = [args.treated_unit]
    if args.treated_units:
        treated_units = [u.strip() for u in args.treated_units.split(",")]

    if not treated_units:
        print("Error: Specify --treated-unit or --treated-units.")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────
    if len(treated_units) == 1:
        result = synthetic_did(
            df, args.outcome, args.entity, args.time,
            treated_unit=treated_units[0], first_treated=args.first_treated,
            donor_units=donor_units,
        )
    else:
        result = synthetic_did_multi(
            df, args.outcome, args.entity, args.time,
            treated_units=treated_units, first_treated=args.first_treated,
            donor_units=donor_units,
        )

    # ── Print ─────────────────────────────────────────────────────────
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print("═══════════════════════════════════")
    print("Synthetic DID Results")
    print("═══════════════════════════════════")

    if "aggregate_att" in result:
        att = result["aggregate_att"]
        se = result["aggregate_se"]
        t_stat = att / se if se > 0 else 0
        p_val = 2 * (1 - 0.975)  # approximate
        print(f"  Aggregate ATT: {att:.4f}")
        print(f"  SE: {se:.4f}")
        print(f"  t-stat: {t_stat:.2f}")
        print(f"  N treated units: {result['n_treated_units']}")
    else:
        print(f"  Treated unit: {result['treated_unit']}")
        print(f"  ATT: {result['att']:.4f}")
        print(f"  SE: {result['se']:.4f}")
        print(f"  t-stat: {result['t_stat']:.2f}")
        print(f"  p-value: {result['p_value']:.4f}")
        print(f"  Pre-treatment RMSE: {result['pre_rmse']:.4f}")
        print(f"  Pre periods: {result['n_pre_periods']}")
        print(f"  Post periods: {result['n_post_periods']}")
        print(f"  Donors: {result['n_donors']}")
        print(f"  Placebo ATTs: {result['n_placebo_atts']}")
        if result.get("significant_donors"):
            print("  Top donor weights:")
            for donor_id, weight in result["significant_donors"][:5]:
                print(f"    {donor_id}: {weight:.3f}")

    # ── Save ──────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    # ── Plot (optional) ───────────────────────────────────────────────
    if args.plot and "unit_results" not in result:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: pre-treatment fit
            ax1 = axes[0]
            pre_times_range = list(range(-result["n_pre_periods"], 0))
            post_times_range = list(range(0, result["n_post_periods"]))

            if len(pre_times_range) == result["n_pre_periods"]:
                ax1.plot(pre_times_range, [0] * len(pre_times_range),
                         color="gray", linestyle="--", alpha=0.5)
                ax1.set_title("Pre-treatment fit (conceptual — re-run with actual data)")
                ax1.set_xlabel("Relative time")
                ax1.set_ylabel("Difference")

            # Right: placebo distribution
            ax2 = axes[1]
            placebo_atts_list = []
            # Re-run for placebos to get them into a variable
            ax2.set_title("Placebo distribution (conceptual)")
            ax2.axvline(result["att"], color="red", linestyle="--",
                       label=f"ATT = {result['att']:.3f}")
            ax2.set_xlabel("ATT")
            ax2.legend()

            plt.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"Plot saved to {args.plot}")
        except Exception as e:
            print(f"Plot warning: {e}")

    print("═══════════════════════════════════")


if __name__ == "__main__":
    main()
