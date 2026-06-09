"""
Standard Difference-in-Differences (single treatment time).

Usage:
    python scripts/run_did.py --data data/merged/panel.dta \\
                              --outcome log_fertility \\
                              --entity city_id --time year \\
                              --treated treated --post post
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    raise ValueError(f"Unsupported format: {p.suffix}")


def run_twfe(df: pd.DataFrame, outcome: str, entity_col: str, time_col: str,
             treat_col: str, post_col: str, controls: list[str],
             cluster: str = None) -> dict:
    """Run two-way fixed effects DID using linearmodels PanelOLS."""

    df = df.copy()
    df["_treated_post"] = df[treat_col].astype(float) * df[post_col].astype(float)

    control_part = ""
    if controls:
        control_part = " + " + " + ".join(controls)

    formula = f"{outcome} ~ _treated_post{control_part} + EntityEffects + TimeEffects"

    panel = df.set_index([entity_col, time_col])
    model = PanelOLS.from_formula(formula, data=panel)
    results = model.fit(cov_type="clustered", cluster_entity=True)

    coef = results.params["_treated_post"]
    se = results.std_errors["_treated_post"]
    pval = results.pvalues["_treated_post"]
    n_obs = int(results.nobs)

    # Extract control variable coefficients
    control_coefs = {}
    for ctrl in controls:
        if ctrl in results.params:
            control_coefs[ctrl] = {
                "coefficient": float(results.params[ctrl]),
                "std_error": float(results.std_errors[ctrl]),
                "p_value": float(results.pvalues[ctrl]),
            }

    return {
        "method": "Standard DID (TWFE)",
        "outcome": outcome,
        "coefficient": float(coef),
        "std_error": float(se),
        "p_value": float(pval),
        "significant": _significance_label(pval),
        "r2_within": float(results.rsquared_within) if results.rsquared_within else None,
        "n_obs": n_obs,
        "n_entities": int(df[entity_col].nunique()),
        "n_periods": int(df[time_col].nunique()),
        "controls": controls,
        "control_coefs": control_coefs,
    }


def run_ols(df: pd.DataFrame, outcome: str, entity_col: str, time_col: str,
            treat_col: str, post_col: str, controls: list[str],
            cluster: str = None) -> dict:
    """Fallback: OLS with entity and time dummies (statsmodels)."""

    df = df.copy()
    df["_treated_post"] = df[treat_col].astype(float) * df[post_col].astype(float)

    control_part = " + ".join(controls) if controls else ""
    main = f"{outcome} ~ _treated_post"
    if control_part:
        main += f" + {control_part}"
    main += f" + C({entity_col}) + C({time_col})"

    model = smf.ols(main, data=df)
    results = model.fit()

    coef = results.params["_treated_post"]
    se = results.bse["_treated_post"]
    pval = results.pvalues["_treated_post"]

    # Extract control variable coefficients
    control_coefs = {}
    for ctrl in controls:
        if ctrl in results.params:
            control_coefs[ctrl] = {
                "coefficient": float(results.params[ctrl]),
                "std_error": float(results.bse[ctrl]),
                "p_value": float(results.pvalues[ctrl]),
            }

    return {
        "method": "Standard DID (OLS with dummies)",
        "outcome": outcome,
        "coefficient": float(coef),
        "std_error": float(se),
        "p_value": float(pval),
        "significant": _significance_label(pval),
        "r2": float(results.rsquared),
        "n_obs": int(results.nobs),
        "n_entities": int(df[entity_col].nunique()),
        "n_periods": int(df[time_col].nunique()),
        "controls": controls,
        "control_coefs": control_coefs,
    }


def _significance_label(p: float) -> str:
    if p < 0.01:
        return "*** (1%)"
    elif p < 0.05:
        return "** (5%)"
    elif p < 0.1:
        return "* (10%)"
    return "not significant"


def main():
    parser = argparse.ArgumentParser(description="Standard DID estimation")
    parser.add_argument("--data", required=True, help="Path to panel data (.dta or .csv)")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--treated", default="treated", help="Treatment dummy column")
    parser.add_argument("--post", default="post", help="Post-treatment dummy column")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variable names")
    parser.add_argument("--cluster", default=None, help="Clustering variable (default: entity)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (no controls) plus full spec, output multi-column format")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    df = load_data(args.data)
    cluster = args.cluster or args.entity

    def _estimate(ctrls):
        if HAS_LINEARMODELS:
            return run_twfe(df, args.outcome, args.entity, args.time,
                            args.treated, args.post, ctrls, cluster)
        elif HAS_STATSMODELS:
            return run_ols(df, args.outcome, args.entity, args.time,
                           args.treated, args.post, ctrls, cluster)
        else:
            print("Error: install linearmodels or statsmodels")
            sys.exit(1)

    if args.baseline and args.controls:
        spec0 = _estimate([])
        spec1 = _estimate(args.controls)
        result = {
            "method": spec1["method"],
            "outcome": args.outcome,
            "specifications": [
                {**spec0, "label": "Baseline"},
                {**spec1, "label": "With controls"},
            ],
        }
        for spec in result["specifications"]:
            spec["r2"] = spec.pop("r2_within", spec.pop("r2", None))
            spec.pop("method", None)
            spec.pop("outcome", None)
    else:
        result = _estimate(args.controls)

    # Print summary
    if "specifications" in result:
        for i, spec in enumerate(result["specifications"], 1):
            stars = _significance_label(spec["p_value"])
            print(f"\n({i}) {spec['label']}")
            print(f"  Coefficient: {spec['coefficient']:.6f} (SE: {spec['std_error']:.6f})  {stars}")
            if spec.get("r2") is not None:
                print(f"  R²: {spec['r2']:.4f}")
            if spec["controls"]:
                print(f"  Controls: {', '.join(spec['controls'])}")
                for c, v in spec.get("control_coefs", {}).items():
                    print(f"    {c}: {v['coefficient']:.4f} ({v['std_error']:.4f})")
    else:
        print(f"\n{'='*50}")
        print(f"Method:     {result['method']}")
        print(f"Coefficient: {result['coefficient']:.6f}")
        print(f"Std Error:  {result['std_error']:.6f}")
        print(f"p-value:    {result['p_value']:.4f}  {result['significant']}")
        print(f"N:          {result['n_obs']} ({result['n_entities']} entities × {result['n_periods']} periods)")
        r2 = result.get("r2_within", result.get("r2"))
        if r2 is not None:
            print(f"R²:          {r2:.4f}")
        if result["controls"]:
            print(f"Controls:   {', '.join(result['controls'])}")
        print(f"{'='*50}")

    if args.output:
        import json
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
