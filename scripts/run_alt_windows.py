"""
Alternative time window robustness check.

Re-estimates the main model with ±1, ±2 year windows around treatment
to verify that the result is not an artifact of the chosen time window.

Usage:
    python scripts/run_alt_windows.py --data data/merged/panel.dta \\
        --outcome log_fertility --entity city_id --time year \\
        --treated treated --post post \\
        --controls gdp population \\
        --output data/auto/stage8_alt_windows.json
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


def _run_did(df, outcome, entity_col, time_col, treat_col, post_col, controls):
    """Run a single DID regression and return coefficient, SE, p-value, N."""
    df = df.copy()
    df["_tp"] = df[treat_col].astype(float) * df[post_col].astype(float)

    control_part = " + ".join(controls) if controls else ""
    rhs = f"_tp{' + ' + control_part if control_part else ''}"
    formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

    model = smf.ols(formula, data=df)
    results = model.fit()
    return {
        "coefficient": float(results.params["_tp"]),
        "std_error": float(results.bse["_tp"]),
        "p_value": float(results.pvalues["_tp"]),
        "n_obs": int(results.nobs),
    }


def check_alt_windows(df, outcome, entity_col, time_col, treated_col,
                      post_col, first_treated_col, controls, windows):
    """Re-estimate with different time windows around treatment.

    windows: list of (label, pre_years, post_years) tuples.
    E.g., [("±1年", 1, 1), ("±2年", 2, 2), ("±3年", 3, 3)]
    """
    results = []
    df = df.copy()

    # Get treatment timing
    if first_treated_col and first_treated_col in df.columns:
        # Staggered: use the first treatment year as reference
        first_treat_year = int(df[first_treated_col].dropna().min())
    else:
        # Single period: find when post switches to 1
        post_years = df[df[post_col] == 1][time_col].unique()
        first_treat_year = int(min(post_years)) if len(post_years) > 0 else None

    if first_treat_year is None:
        return {"error": "Cannot determine treatment year", "windows": []}

    for label, pre_years, post_years in windows:
        start_year = first_treat_year - pre_years
        end_year = first_treat_year + post_years - 1  # inclusive

        sub = df[(df[time_col] >= start_year) & (df[time_col] <= end_year)].copy()

        if len(sub) < 20:
            results.append({
                "window": label,
                "years": f"{start_year}-{end_year}",
                "status": "too few observations",
                "coefficient": None, "std_error": None, "p_value": None,
            })
            continue

        try:
            est = _run_did(sub, outcome, entity_col, time_col,
                          treated_col, post_col, controls)
            est["window"] = label
            est["years"] = f"{start_year}-{end_year}"
            results.append(est)
        except Exception as e:
            results.append({
                "window": label,
                "years": f"{start_year}-{end_year}",
                "status": f"error: {str(e)[:100]}",
                "coefficient": None, "std_error": None, "p_value": None,
            })

    return results


def evaluate_stability(window_results, baseline_coef, baseline_pval):
    """Determine if results are stable across windows.

    Returns: {stable: bool, interpretation: str, details: ...}
    """
    valid = [w for w in window_results if w.get("coefficient") is not None]
    if len(valid) < 2:
        return {"stable": True, "interpretation": "有效窗口不足，无法判断"}

    coefs = [w["coefficient"] for w in valid]
    pvals = [w.get("p_value", 1) for w in valid]

    # Check 1: coefficient sign consistency
    base_sign = np.sign(baseline_coef)
    signs_consistent = all(np.sign(c) == base_sign for c in coefs)

    # Check 2: p-value consistency (same significance level)
    base_sig = baseline_pval < 0.05
    sigs_consistent = all((pv < 0.05) == base_sig for pv in pvals)

    # Check 3: coefficient magnitude stability (max/min ratio)
    abs_coefs = [abs(c) for c in coefs]
    max_ratio = max(abs_coefs) / min(abs_coefs) if min(abs_coefs) > 1e-10 else 999

    stable = signs_consistent and sigs_consistent and max_ratio < 3

    interpretation_parts = []
    if signs_consistent:
        interpretation_parts.append("系数方向在所有窗口保持一致")
    else:
        interpretation_parts.append("系数方向在不同窗口间出现变化")
    if sigs_consistent:
        interpretation_parts.append("显著性水平在各窗口中一致")
    else:
        interpretation_parts.append("显著性在不同窗口间不稳定")
    if max_ratio < 3:
        interpretation_parts.append(f"系数大小变化在合理范围内（最大/最小 = {max_ratio:.1f}）")
    else:
        interpretation_parts.append(f"系数大小波动较大（最大/最小 = {max_ratio:.1f}）")

    return {
        "stable": stable,
        "sign_consistent": signs_consistent,
        "significance_consistent": sigs_consistent,
        "max_ratio": round(max_ratio, 2),
        "interpretation": "；".join(interpretation_parts) + "。",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Alternative time window robustness check")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--treated", default="treated", help="Treatment column")
    parser.add_argument("--post", default="post", help="Post-treatment column")
    parser.add_argument("--first-treated", default="first_treated",
                        help="First-treated year column (staggered)")
    parser.add_argument("--controls", nargs="*", default=[],
                        help="Control variables")
    parser.add_argument("--windows", nargs="*", default=["1", "2", "3"],
                        help="Window sizes to test (in years each side)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels required. Install: pip install statsmodels")
        sys.exit(1)

    df = load_data(args.data)

    windows = [(f"±{w}年", int(w), int(w)) for w in args.windows]

    # Baseline: full sample
    baseline = _run_did(df, args.outcome, args.entity, args.time,
                        args.treated, args.post, args.controls)

    # Alternative windows
    window_results = check_alt_windows(
        df, args.outcome, args.entity, args.time,
        args.treated, args.post, args.first_treated,
        args.controls, windows,
    )

    stability = evaluate_stability(
        window_results, baseline["coefficient"], baseline["p_value"])

    output = {
        "check": "alternative_time_windows",
        "baseline": baseline,
        "windows": window_results,
        "stability": stability,
        "passed": stability["stable"],
    }

    print("\n──── Alternative Time Windows ────")
    print(f"Baseline (full sample): β = {baseline['coefficient']:.4f}, "
          f"SE = {baseline['std_error']:.4f}, p = {baseline['p_value']:.4f}")
    print()
    for w in window_results:
        if w.get("coefficient") is not None:
            print(f"  {w['window']} ({w['years']}): "
                  f"β = {w['coefficient']:.4f}, "
                  f"SE = {w['std_error']:.4f}, "
                  f"p = {w['p_value']:.4f}")
        else:
            print(f"  {w['window']} ({w['years']}): {w.get('status', 'N/A')}")
    print(f"\nStability: {'✓ 通过' if stability['stable'] else '✗ 未通过'}")
    print(f"  {stability['interpretation']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
