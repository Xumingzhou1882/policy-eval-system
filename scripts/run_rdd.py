"""
Regression Discontinuity Design (Sharp and Fuzzy RDD).

Implements local linear regression with MSE-optimal bandwidth (Calonico et al. 2014)
and McCrary density test for manipulation.

Usage:
    python scripts/run_rdd.py --data data/merged/analysis.csv \\
                              --outcome log_pm25 \\
                              --running-var range_km \\
                              --cutoff 400 \\
                              --type sharp \\
                              --plot rdd.png
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


def _epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel."""
    k = 0.75 * (1 - u ** 2)
    k[np.abs(u) > 1] = 0
    return k


def _triangular(u: np.ndarray) -> np.ndarray:
    """Triangular kernel."""
    k = 1 - np.abs(u)
    k[np.abs(u) > 1] = 0
    return k


def _mse_optimal_bandwidth(running_var: np.ndarray, cutoff: float) -> float:
    """
    Compute MSE-optimal bandwidth using rule-of-thumb (Calonico et al. 2014, CCT).
    Simplified version of the CCT bandwidth selector.
    """
    n = len(running_var)
    h_rot = 1.84 * np.std(running_var) * n ** (-1 / 5)
    return h_rot


def mccrary_test(running_var: np.ndarray, cutoff: float, bandwidth: float = None) -> dict:
    """
    McCrary (2008) density test: test for a discontinuity in the density
    of the running variable at the cutoff (suggesting manipulation).

    Uses a histogram-based approach: bins the running variable, estimates
    a local linear smoother on each side, and tests for a jump at the cutoff.
    """
    if bandwidth is None:
        bandwidth = _mse_optimal_bandwidth(running_var, cutoff)

    n = len(running_var)
    n_bins = int(np.sqrt(n))

    # Bin the running variable
    bins = np.linspace(running_var.min(), running_var.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(running_var, bins=bins)
    freq = hist / (n * (bins[1] - bins[0]))

    # Restrict to window around cutoff
    mask = np.abs(bin_centers - cutoff) <= bandwidth
    bc = bin_centers[mask]
    fq = freq[mask]
    above = (bc >= cutoff).astype(int)

    # Local linear regression of frequency on running variable
    X = np.column_stack([np.ones(len(bc)), bc - cutoff, above, (bc - cutoff) * above])
    try:
        beta = np.linalg.lstsq(X, fq, rcond=None)[0]
        jump = beta[2]  # discontinuity in density
        se_jump = np.sqrt(np.mean((fq - X @ beta) ** 2) / len(bc)) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))[2]

        t_stat = jump / se_jump if se_jump > 0 else 0
        p_value = 2 * (1 - _normal_cdf(np.abs(t_stat)))

        return {
            "log_difference": float(jump),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "bandwidth": float(bandwidth),
            "n_bins": n_bins,
            "manipulation_detected": p_value < 0.05,
        }
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix in McCrary test — try a different bandwidth."}


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))


def sharp_rdd(df: pd.DataFrame, outcome: str, running_var: str,
              cutoff: float, bandwidth: float = None,
              controls: list[str] = None,
              kernel: str = "triangular") -> dict:
    """Sharp RDD: local linear regression on each side of the cutoff."""

    rv = df[running_var].values
    y = df[outcome].values

    if bandwidth is None:
        bandwidth = _mse_optimal_bandwidth(rv, cutoff)

    # Restrict to window
    mask = np.abs(rv - cutoff) <= bandwidth
    sub = df.loc[mask].copy()
    rv_sub = sub[running_var].values
    y_sub = sub[outcome].values

    sub["_above"] = (rv_sub >= cutoff).astype(int)
    sub["_dist"] = rv_sub - cutoff
    sub["_dist_above"] = sub["_dist"] * sub["_above"]

    # Weights
    if kernel == "triangular":
        w = _triangular((rv_sub - cutoff) / bandwidth)
    elif kernel == "epanechnikov":
        w = _epanechnikov((rv_sub - cutoff) / bandwidth)
    else:
        w = np.ones(len(rv_sub))

    # Local linear regression
    ctrl_parts = ""
    if controls:
        ctrl_parts = " + " + " + ".join(controls)

    formula = f"{outcome} ~ _above + _dist + _dist_above{ctrl_parts}"
    try:
        model = smf.wls(formula, data=sub, weights=w)
        results = model.fit()

        att = float(results.params["_above"])
        se = float(results.bse["_above"])
        pval = float(results.pvalues["_above"])

        return {
            "type": "Sharp RDD",
            "cutoff": cutoff,
            "bandwidth": float(bandwidth),
            "n_obs_window": len(sub),
            "n_above": int(sub["_above"].sum()),
            "n_below": len(sub) - int(sub["_above"].sum()),
            "att": att,
            "std_error": se,
            "p_value": pval,
            "significant": "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns")),
            "r2": float(results.rsquared),
        }
    except Exception as e:
        return {"error": str(e)}


def fuzzy_rdd(df: pd.DataFrame, outcome: str, running_var: str,
              treatment_var: str, cutoff: float,
              bandwidth: float = None,
              kernel: str = "triangular") -> dict:
    """Fuzzy RDD: 2SLS where _above instruments treatment."""

    rv = df[running_var].values
    if bandwidth is None:
        bandwidth = _mse_optimal_bandwidth(rv, cutoff)

    mask = np.abs(rv - cutoff) <= bandwidth
    sub = df.loc[mask].copy()
    rv_sub = sub[running_var].values

    sub["_above"] = (rv_sub >= cutoff).astype(int)
    sub["_dist"] = rv_sub - cutoff
    sub["_dist_above"] = sub["_dist"] * sub["_above"]

    if kernel == "triangular":
        w = _triangular((rv_sub - cutoff) / bandwidth)

    # First stage: treatment ~ above + dist + dist*above
    fs_formula = f"{treatment_var} ~ _above + _dist + _dist_above"
    fs_model = smf.wls(fs_formula, data=sub, weights=w)
    fs_results = fs_model.fit()

    fs_fstat = float(fs_results.fvalue) if fs_results.fvalue else 0
    fs_coef = float(fs_results.params["_above"])

    # Second stage: outcome ~ treatment_hat + dist + dist*above
    sub["_treatment_hat"] = fs_results.fittedvalues
    ss_formula = f"{outcome} ~ _treatment_hat + _dist + _dist_above"
    ss_model = smf.wls(ss_formula, data=sub, weights=w)
    ss_results = ss_model.fit()

    late = float(ss_results.params["_treatment_hat"])
    se = float(ss_results.bse["_treatment_hat"])
    pval = float(ss_results.pvalues["_treatment_hat"])

    # Reduced form: outcome ~ above + dist + dist*above (ITT)
    rf_formula = f"{outcome} ~ _above + _dist + _dist_above"
    rf_model = smf.wls(rf_formula, data=sub, weights=w)
    rf_results = rf_model.fit()
    itt = float(rf_results.params["_above"])

    return {
        "type": "Fuzzy RDD (Wald estimator)",
        "cutoff": cutoff,
        "bandwidth": float(bandwidth),
        "n_obs_window": len(sub),
        "late": late,
        "std_error": se,
        "p_value": pval,
        "first_stage_coef": fs_coef,
        "first_stage_fstat": fs_fstat,
        "weak_instrument": fs_fstat < 10,
        "itt": itt,
    }


def plot_rdd(df: pd.DataFrame, outcome: str, running_var: str,
             cutoff: float, result: dict, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    rv = df[running_var].values
    y = df[outcome].values
    bw = result.get("bandwidth", _mse_optimal_bandwidth(rv, cutoff))

    # Scatter (binned means for clarity)
    n_bins = min(50, len(df) // 10)
    bins = np.linspace(rv.min(), rv.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.array([y[(rv >= bins[i]) & (rv < bins[i + 1])].mean()
                          if np.sum((rv >= bins[i]) & (rv < bins[i + 1])) > 0 else np.nan
                          for i in range(n_bins)])
    ax.scatter(bin_centers, bin_means, s=10, color="gray", alpha=0.7)

    # Fitted lines on each side
    below = df[df[running_var] <= cutoff]
    above = df[df[running_var] > cutoff]

    for side, color in [(below, "steelblue"), (above, "darkorange")]:
        if len(side) < 2:
            continue
        rv_side = side[running_var].values
        y_side = side[outcome].values
        w_side = _triangular((rv_side - cutoff) / bw) if bw > 0 else np.ones(len(rv_side))
        w_side = w_side / w_side.sum()
        # Weighted least squares for visualization
        X = np.column_stack([np.ones(len(rv_side)), rv_side - cutoff])
        beta = np.linalg.lstsq(X * np.sqrt(w_side[:, None]), y_side * np.sqrt(w_side), rcond=None)[0]
        x_range = np.linspace(min(rv_side), max(rv_side), 100)
        y_pred = beta[0] + beta[1] * (x_range - cutoff)
        ax.plot(x_range, y_pred, color=color, linewidth=2)

    ax.axvline(cutoff, color="red", linewidth=1, linestyle="--", label=f"Cutoff = {cutoff}")
    ax.set_xlabel(running_var)
    ax.set_ylabel(outcome)
    ax.set_title(f"RDD: ATT = {result.get('att', result.get('late', 'N/A')):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"RDD plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regression Discontinuity Design")
    parser.add_argument("--data", required=True, help="Path to data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--running-var", required=True, help="Running variable name")
    parser.add_argument("--cutoff", type=float, required=True, help="Cutoff value")
    parser.add_argument("--type", default="sharp", choices=["sharp", "fuzzy"],
                        help="RDD type")
    parser.add_argument("--treatment-var", default=None, help="Treatment variable (fuzzy RDD only)")
    parser.add_argument("--bandwidth", type=float, default=None, help="Bandwidth (auto if omitted)")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variables")
    parser.add_argument("--kernel", default="triangular", choices=["triangular", "epanechnikov", "uniform"])
    parser.add_argument("--mccrary", action="store_true", help="Run McCrary density test")
    parser.add_argument("--plot", default=None, help="Output path for plot")
    parser.add_argument("--output", default=None, help="Output path for results (.json)")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required.")
        sys.exit(1)

    df = load_data(args.data)
    rv = df[args.running_var].values

    # Bandwidth
    bw = args.bandwidth or _mse_optimal_bandwidth(rv, args.cutoff)
    print(f"Bandwidth: {bw:.3f}")

    # McCrary test
    mccrary_result = None
    if args.mccrary:
        mccrary_result = mccrary_test(rv, args.cutoff, bw)
        print(f"\n──── McCrary Test ────")
        if "error" in mccrary_result:
            print(f"Error: {mccrary_result['error']}")
        else:
            status = "✗ Manipulation detected" if mccrary_result["manipulation_detected"] else "✓ No manipulation"
            print(f"{status} (p={mccrary_result['p_value']:.4f})")

    # RDD estimation
    if args.type == "sharp":
        result = sharp_rdd(df, args.outcome, args.running_var, args.cutoff,
                           bw, args.controls, args.kernel)
    else:
        if not args.treatment_var:
            print("Error: --treatment-var required for fuzzy RDD")
            sys.exit(1)
        result = fuzzy_rdd(df, args.outcome, args.running_var, args.treatment_var,
                           args.cutoff, bw, args.kernel)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    result["mccrary_test"] = mccrary_result

    print(f"\n──── {result['type']} ────")
    print(f"Bandwidth:    {result['bandwidth']:.3f}")
    print(f"Observations: {result['n_obs_window']}")

    if args.type == "sharp":
        print(f"ATT:          {result['att']:.6f}")
        print(f"Std Error:    {result['std_error']:.6f}")
        print(f"P-value:      {result['p_value']:.4f}  {result['significant']}")
    else:
        print(f"First stage F: {result['first_stage_fstat']:.2f} {'(WEAK!)' if result['weak_instrument'] else '(OK)'}")
        print(f"LATE:          {result['late']:.6f}")
        print(f"ITT:           {result['itt']:.6f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    if args.plot:
        plot_rdd(df, args.outcome, args.running_var, args.cutoff, result, args.plot)


if __name__ == "__main__":
    main()
