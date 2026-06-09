"""
Placebo permutation test for causal inference.

Randomly reassigns treatment assignment (or treatment timing) N times and
re-estimates the model each time. Compares the actual coefficient to the
distribution of placebo coefficients.

Usage:
    python scripts/placebo_test.py --data data/merged/panel.dta \\
                                   --outcome outcome_var \\
                                   --entity entity_id --time year \\
                                   --treated treated --post post \\
                                   --n-sim 1000 --plot placebo.png
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


def _run_single_did(df: pd.DataFrame, outcome: str, entity_col: str,
                    time_col: str, treat_col: str, post_col: str,
                    controls: list[str]) -> float:
    """Run a single DID and return the interaction coefficient."""
    df = df.copy()
    df["_tp"] = df[treat_col].astype(float) * df[post_col].astype(float)

    control_part = " + ".join(controls) if controls else ""
    rhs = f"_tp{' + ' + control_part if control_part else ''}"
    formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

    model = smf.ols(formula, data=df)
    results = model.fit()
    return float(results.params["_tp"])


def run_staggered_placebo(df: pd.DataFrame, outcome: str, entity_col: str,
                          time_col: str, treated_col: str,
                          first_treated_col: str, controls: list[str],
                          n_sim: int) -> dict:
    """
    For staggered designs: randomly reassign first-treatment years among
    treated units, keeping the set of treated units fixed.
    """
    df = df.copy()
    entities = df[entity_col].unique()
    actual_treated = df[df[treated_col] == 1][[entity_col, first_treated_col]].drop_duplicates()

    # Compute actual treatment times
    ttimes = actual_treated[first_treated_col].dropna().values
    n_treated = len(actual_treated)

    # Actual estimate
    df["_post"] = (df[time_col] >= df[first_treated_col]).astype(int)
    actual_est = _run_single_did(df, outcome, entity_col, time_col,
                                 treated_col, "_post", controls)

    # Placebo simulations
    placebo_ests = []
    rng = np.random.default_rng(42)

    for _ in range(n_sim):
        df_sim = df.copy()
        # Shuffle treatment times among treated units
        shuffled_times = rng.choice(ttimes, size=n_treated, replace=False)
        time_map = dict(zip(actual_treated[entity_col].values, shuffled_times))
        df_sim["_ft_sim"] = df_sim[entity_col].map(time_map)
        df_sim["_post_sim"] = (df_sim[time_col] >= df_sim["_ft_sim"]).astype(int)

        try:
            est = _run_single_did(df_sim, outcome, entity_col, time_col,
                                  treated_col, "_post_sim", controls)
            placebo_ests.append(float(est))
        except Exception:
            placebo_ests.append(np.nan)

    placebo_ests = np.array(placebo_ests)
    placebo_ests = placebo_ests[~np.isnan(placebo_ests)]

    p_value = np.mean(np.abs(placebo_ests) >= np.abs(actual_est))

    return {
        "actual_estimate": float(actual_est),
        "placebo_estimates": placebo_ests.tolist(),
        "placebo_mean": float(np.mean(placebo_ests)),
        "placebo_std": float(np.std(placebo_ests)),
        "p_value": float(p_value),
        "n_sim_valid": int(len(placebo_ests)),
        "passed": bool(p_value < 0.05),
    }


def run_standard_placebo(df: pd.DataFrame, outcome: str, entity_col: str,
                         time_col: str, treated_col: str, post_col: str,
                         controls: list[str], n_sim: int) -> dict:
    """
    For single-treatment-time DID: randomly reassign which units are treated,
    keeping the total number of treated units fixed.
    """
    df = df.copy()

    # Actual estimate
    actual_est = _run_single_did(df, outcome, entity_col, time_col,
                                 treated_col, post_col, controls)

    # Placebo simulations
    entities = df[entity_col].unique()
    n_treated = int(df[df[treated_col] == 1][entity_col].nunique())

    placebo_ests = []
    rng = np.random.default_rng(42)

    for _ in range(n_sim):
        df_sim = df.copy()
        fake_treated = rng.choice(entities, size=n_treated, replace=False)
        df_sim["_t_sim"] = df_sim[entity_col].isin(fake_treated).astype(int)

        try:
            est = _run_single_did(df_sim, outcome, entity_col, time_col,
                                  "_t_sim", post_col, controls)
            placebo_ests.append(float(est))
        except Exception:
            placebo_ests.append(np.nan)

    placebo_ests = np.array(placebo_ests)
    placebo_ests = placebo_ests[~np.isnan(placebo_ests)]

    p_value = np.mean(np.abs(placebo_ests) >= np.abs(actual_est))

    return {
        "actual_estimate": float(actual_est),
        "placebo_estimates": placebo_ests.tolist(),
        "placebo_mean": float(np.mean(placebo_ests)),
        "placebo_std": float(np.std(placebo_ests)),
        "p_value": float(p_value),
        "n_sim_valid": int(len(placebo_ests)),
        "passed": bool(p_value < 0.05),
    }


def plot_placebo(result: dict, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result["placebo_estimates"], bins=50, color="lightgray", edgecolor="gray", density=True)
    ax.axvline(result["actual_estimate"], color="red", linewidth=2,
               label=f"Actual ({result['actual_estimate']:.4f})")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Density")
    ax.set_title(f"Placebo Test (p={result['p_value']:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Placebo plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Placebo permutation test")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--treated", default="treated", help="Treatment dummy column")
    parser.add_argument("--post", default="post", help="Post-treatment dummy column")
    parser.add_argument("--first-treated", default=None, help="First-treated year (for staggered design)")
    parser.add_argument("--n-sim", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variables")
    parser.add_argument("--plot", default=None, help="Output path for histogram plot")
    parser.add_argument("--output", default=None, help="Output path for results (.json)")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required.")
        sys.exit(1)

    df = load_data(args.data)

    if args.first_treated:
        result = run_staggered_placebo(df, args.outcome, args.entity, args.time,
                                       args.treated, args.first_treated,
                                       args.controls, args.n_sim)
    else:
        result = run_standard_placebo(df, args.outcome, args.entity, args.time,
                                      args.treated, args.post,
                                      args.controls, args.n_sim)

    print("\n──── Placebo Test ────")
    print(f"Actual estimate:   {result['actual_estimate']:.6f}")
    print(f"Placebo mean:      {result['placebo_mean']:.6f}")
    print(f"Placebo std:       {result['placebo_std']:.6f}")
    print(f"P-value:           {result['p_value']:.4f}")
    status = "✓ Passed" if result["passed"] else "✗ Failed"
    print(f"Result:            {status} — Actual effect {'stands out from' if result['passed'] else 'is within'} noise")

    if args.output:
        out = result.copy()
        out["placebo_estimates"] = out["placebo_estimates"][:100]  # Truncate for file size
        out["_truncated"] = len(result["placebo_estimates"]) > 100
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if args.plot:
        plot_placebo(result, args.plot)


if __name__ == "__main__":
    main()
