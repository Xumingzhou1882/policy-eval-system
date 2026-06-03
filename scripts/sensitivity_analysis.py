"""
Sensitivity analysis for causal inference estimates.

Implements:
  1. Oster (2019) bounds — how strong would unobservables need to be
     relative to observables to explain away the treatment effect?
  2. Coefficient stability — compare controlled vs. uncontrolled estimates
  3. Rosenbaum bounds (simplified) — for matching/weighting estimators
  4. Leave-one-out influence — is the result driven by one unit?

Usage:
    python scripts/sensitivity_analysis.py --data data/merged/panel.dta \\
                                           --outcome log_fertility \\
                                           --entity city_id --time year \\
                                           --treated treated --post post \\
                                           --controls gdp population \\
                                           --output sensitivity.json
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


# ═══════════════════════════════════════════════════════════════════════
# 1. Oster (2019) bounds: selection on unobservables
# ═══════════════════════════════════════════════════════════════════════

def oster_bounds(df: pd.DataFrame, outcome: str, treatment_interaction: str,
                 entity_col: str, time_col: str,
                 controls: list[str],
                 rmax: float = 1.3,
                 delta: float = 1.0) -> dict:
    """
    Oster (2019) test for selection on unobservables.

    Compares coefficient movement and R² change between:
      - Uncontrolled regression (only treatment)
      - Controlled regression (treatment + observables)

    Parameters
    ----------
    rmax : float
        Maximum R² from a hypothetical regression including all observable
        and unobservable controls. Default 1.3×R²_controlled, or 1.0 if the
        controlled R² is high. Oster suggests min(1.3*R²_c, 1.0).
    delta : float
        How strongly unobservables affect treatment relative to observables.
        delta=1 means unobservables are as important as observables.
        The identified set is [β*, β(rmax, delta)] where β* is the bias-adjusted
        coefficient under proportional selection.

    Returns
    -------
    dict with:
      - beta_uncontrolled: coefficient without controls
      - beta_controlled: coefficient with controls
      - r2_uncontrolled, r2_controlled: R² values
      - beta_bias_adjusted: β under delta=1, Rmax assumption
      - identified_set: [beta_controlled, beta_bias_adjusted]
      - delta_for_zero: δ needed to make β=0
      - interpretation: plain-language summary
    """
    df = df.copy()

    # Uncontrolled regression: outcome ~ treatment + FE
    formula_u = f"{outcome} ~ {treatment_interaction} + C({entity_col}) + C({time_col})"
    model_u = smf.ols(formula_u, data=df)
    res_u = model_u.fit()
    beta_u = float(res_u.params.get(treatment_interaction, 0))
    r2_u = float(res_u.rsquared)

    # Controlled regression: outcome ~ treatment + controls + FE
    if controls:
        ctrl_str = " + ".join(controls)
        formula_c = f"{outcome} ~ {treatment_interaction} + {ctrl_str} + C({entity_col}) + C({time_col})"
    else:
        formula_c = formula_u
    model_c = smf.ols(formula_c, data=df)
    res_c = model_c.fit()
    beta_c = float(res_c.params.get(treatment_interaction, 0))
    r2_c = float(res_c.rsquared)

    # Oster's delta calculation
    # β* ≈ β_c - δ * (β_u - β_c) * (Rmax - R_c) / (R_c - R_u)
    Rmax = min(rmax * r2_c, 0.99) if r2_c > 0 else 0.99

    if r2_c > r2_u and (r2_c - r2_u) > 1e-10:
        # Proportional selection assumption
        beta_star = beta_c - delta * (beta_u - beta_c) * (Rmax - r2_c) / (r2_c - r2_u)
        delta_for_zero = delta * (r2_c - r2_u) / ((beta_u - beta_c) * (Rmax - r2_c) + 1e-10) if abs(beta_c) > 1e-10 else float("inf")
    else:
        beta_star = beta_c
        delta_for_zero = None

    # Identified set
    lo = min(beta_c, beta_star) if beta_star else beta_c
    hi = max(beta_c, beta_star) if beta_star else beta_c

    # Interpretation
    if abs(beta_star) < 1e-10:
        interpretation = (
            "The bias-adjusted coefficient is near zero — unobservables "
            "as strong as observables could explain away the entire effect."
        )
    elif beta_c * beta_star > 0:
        interpretation = (
            f"The bias-adjusted coefficient ({beta_star:.4f}) has the same sign "
            f"as the controlled estimate ({beta_c:.4f}). The result is robust to "
            f"selection on unobservables of equal strength to observables (δ={delta})."
        )
    else:
        interpretation = (
            f"The coefficient switches sign under bias adjustment "
            f"({beta_c:.4f} → {beta_star:.4f}). The result is NOT robust "
            f"to unobservables as strong as observables (δ={delta})."
        )

    return {
        "method": "Oster (2019) selection on unobservables",
        "beta_uncontrolled": float(beta_u),
        "beta_controlled": float(beta_c),
        "r2_uncontrolled": float(r2_u),
        "r2_controlled": float(r2_c),
        "rmax": float(Rmax),
        "delta_assumed": delta,
        "beta_bias_adjusted": float(beta_star) if beta_star is not None else None,
        "identified_set": [float(lo), float(hi)],
        "delta_for_zero": float(delta_for_zero) if delta_for_zero else None,
        "interpretation": interpretation,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Coefficient stability
# ═══════════════════════════════════════════════════════════════════════

def coefficient_stability(df: pd.DataFrame, outcome: str,
                          treatment_interaction: str,
                          entity_col: str, time_col: str,
                          control_groups: list[list[str]],
                          group_labels: list[str]) -> dict:
    """
    Add controls sequentially and track coefficient stability.

    Each control_group is a set of controls added incrementally.
    A stable coefficient across specifications suggests robustness.
    """
    df = df.copy()
    results = []

    # Baseline: no controls
    formula_base = f"{outcome} ~ {treatment_interaction} + C({entity_col}) + C({time_col})"
    res_base = smf.ols(formula_base, data=df).fit()
    beta_base = float(res_base.params.get(treatment_interaction, 0))
    se_base = float(res_base.bse.get(treatment_interaction, 0))

    results.append({
        "label": "No controls",
        "coefficient": beta_base,
        "std_error": se_base,
        "p_value": float(res_base.pvalues.get(treatment_interaction, 1)),
        "controls": [],
    })

    cumulative = []
    for i, group in enumerate(control_groups):
        cumulative.extend(group)
        ctrl_str = " + ".join(cumulative)
        formula = f"{outcome} ~ {treatment_interaction} + {ctrl_str} + C({entity_col}) + C({time_col})"
        res = smf.ols(formula, data=df).fit()
        beta = float(res.params.get(treatment_interaction, 0))
        se = float(res.bse.get(treatment_interaction, 0))

        results.append({
            "label": group_labels[i] if i < len(group_labels) else f"Model {i+1}",
            "coefficient": beta,
            "std_error": se,
            "p_value": float(res.pvalues.get(treatment_interaction, 1)),
            "controls": cumulative.copy(),
        })

    # Stability ratio: max/min absolute coefficient
    betas = [r["coefficient"] for r in results]
    abs_betas = [abs(b) for b in betas if abs(b) > 1e-10]
    stability_ratio = max(abs_betas) / min(abs_betas) if len(abs_betas) >= 2 else 1.0

    stable = stability_ratio < 2.0  # coefficient doesn't change by more than 2x

    return {
        "method": "Coefficient stability across specifications",
        "specifications": results,
        "stability_ratio": float(stability_ratio),
        "stable": stable,
        "interpretation": (
            f"The coefficient varies by {stability_ratio:.1f}× across specifications. "
            + ("The estimate is stable." if stable else "The estimate is sensitive to control choice.")
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. Rosenbaum bounds (simplified) for matched/weighted designs
# ═══════════════════════════════════════════════════════════════════════

def rosenbaum_bounds(df: pd.DataFrame, outcome: str, treatment: str,
                     covariates: list[str],
                     gamma_range: list[float] = None) -> dict:
    """
    Simplified Rosenbaum (2002) sensitivity bounds for observational studies.

    For each gamma (odds ratio of treatment assignment due to unobserved
    confounders between two matched units), computes the range of possible
    p-values for the treatment effect under the null.

    gamma=1 means no hidden bias; gamma=2 means one unit could be twice as
    likely to be treated due to unobservables.

    Uses a simplified signed-rank test approach.
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    df = df.dropna(subset=[outcome, treatment] + covariates)
    y = df[outcome].values
    d = df[treatment].values

    # Estimate propensity scores
    if HAS_STATS:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(df[covariates].values)
        ps_model = LogisticRegression(penalty=None, max_iter=2000)
        ps_model.fit(X, d)
        p_scores = ps_model.predict_proba(X)[:, 1]
    else:
        p_scores = np.full(len(y), d.mean())

    # Match on propensity scores (nearest-neighbor)
    treated_idx = np.where(d == 1)[0]
    control_idx = np.where(d == 0)[0]

    matched_diffs = []
    for ti in treated_idx:
        distances = np.abs(p_scores[control_idx] - p_scores[ti])
        ci = control_idx[np.argmin(distances)]
        matched_diffs.append(y[ti] - y[ci])

    matched_diffs = np.array(matched_diffs)
    n_pairs = len(matched_diffs)

    if n_pairs < 10:
        return {"error": "Too few matched pairs for Rosenbaum bounds."}

    # Wilcoxon signed-rank statistic
    ranks = np.argsort(np.abs(matched_diffs)) + 1
    T_pos = np.sum(ranks[matched_diffs > 0])

    # Expected T and variance under gamma (simplified)
    bounds = []
    for gamma in gamma_range:
        # Probability that the treated unit has the higher outcome under gamma
        p_plus = gamma / (1 + gamma)

        E_T = p_plus * n_pairs * (n_pairs + 1) / 2
        Var_T = p_plus * (1 - p_plus) * n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 6

        if Var_T > 0:
            z_upper = (T_pos - E_T) / np.sqrt(Var_T)
            p_upper = 2 * (1 - _normal_cdf(abs(z_upper)))
        else:
            p_upper = 1.0

        bounds.append({
            "gamma": gamma,
            "p_value_upper_bound": float(p_upper),
            "significant": p_upper < 0.05,
        })

    # Critical gamma: where significance is lost
    critical_gamma = None
    for b in bounds:
        if not b["significant"]:
            critical_gamma = b["gamma"]
            break
    if critical_gamma is None and bounds:
        critical_gamma = gamma_range[-1]

    return {
        "method": "Rosenbaum bounds — sensitivity to hidden bias",
        "n_pairs": n_pairs,
        "bounds": bounds,
        "critical_gamma": critical_gamma,
        "interpretation": (
            f"Hidden bias of Γ={critical_gamma:.1f} would be needed to overturn the "
            f"significance of the result. "
            + ("The result is highly sensitive to hidden bias." if critical_gamma < 1.5
               else ("The result is moderately sensitive." if critical_gamma < 2.5
                     else "The result is robust to substantial hidden bias."))
            if critical_gamma is not None
            else "All tested gamma values maintain significance — result is highly robust."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Leave-one-out influence analysis
# ═══════════════════════════════════════════════════════════════════════

def leave_one_out(df: pd.DataFrame, outcome: str, treatment_interaction: str,
                  entity_col: str, time_col: str,
                  controls: list[str] = None) -> dict:
    """
    Re-estimate the model dropping one entity at a time.

    Reports min/max coefficient and identifies influential units.
    """
    df = df.copy()
    controls = controls or []

    ctrl_str = " + ".join(controls) if controls else ""
    if ctrl_str:
        formula = f"{outcome} ~ {treatment_interaction} + {ctrl_str} + C({entity_col}) + C({time_col})"
    else:
        formula = f"{outcome} ~ {treatment_interaction} + C({entity_col}) + C({time_col})"

    # Full sample estimate
    res_full = smf.ols(formula, data=df).fit()
    beta_full = float(res_full.params.get(treatment_interaction, 0))

    entities = df[entity_col].unique()
    estimates = {}

    for entity in entities:
        sub = df[df[entity_col] != entity]
        try:
            res = smf.ols(formula, data=sub).fit()
            beta = float(res.params.get(treatment_interaction, 0))
            estimates[str(entity)] = float(beta)
        except Exception:
            continue

    if not estimates:
        return {"error": "Could not estimate leave-one-out for any entity."}

    betas = list(estimates.values())
    min_entity = min(estimates, key=estimates.get)
    max_entity = max(estimates, key=estimates.get)
    mean_beta = np.mean(betas)
    std_beta = np.std(betas)

    # Influential: entities whose removal changes coefficient by >2 SD
    influential = []
    for entity, beta in estimates.items():
        if abs(beta - beta_full) > 2 * std_beta:
            influential.append({
                "entity": entity,
                "coefficient_without": beta,
                "change": float(beta - beta_full),
            })

    # Sort by absolute change
    influential.sort(key=lambda x: abs(x["change"]), reverse=True)

    return {
        "method": "Leave-one-out influence analysis",
        "full_sample_coefficient": float(beta_full),
        "n_entities": len(estimates),
        "min_coefficient": float(min(betas)),
        "min_entity": str(min_entity),
        "max_coefficient": float(max(betas)),
        "max_entity": str(max_entity),
        "mean_coefficient": float(mean_beta),
        "std_coefficient": float(std_beta),
        "influential_entities": influential[:5],  # top 5
        "n_influential": len(influential),
        "interpretation": (
            f"Removing any single unit changes the coefficient from "
            f"{beta_full:.4f} to a range of [{min(betas):.4f}, {max(betas):.4f}]. "
            + (f"{len(influential)} unit(s) are influential (>2 SD change)."
               if influential else "No single unit drives the result.")
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Placebo-in-time (falsification test)
# ═══════════════════════════════════════════════════════════════════════

def placebo_in_time(df: pd.DataFrame, outcome: str, treatment_interaction: str,
                    entity_col: str, time_col: str,
                    true_treatment_time: float,
                    pre_shifts: list[int] = None,
                    controls: list[str] = None) -> dict:
    """
    Falsification: pretend treatment occurred earlier than it actually did.
    If we find a significant "effect" before the real treatment, the
    identifying assumptions are suspect.
    """
    df = df.copy()
    controls = controls or []
    if pre_shifts is None:
        pre_shifts = [-4, -3, -2]

    ctrl_str = " + ".join(controls) if controls else ""

    results = []
    for shift in pre_shifts:
        fake_time = true_treatment_time + shift  # shift is negative → earlier
        fake_post = (df[time_col] >= fake_time).astype(int)
        fake_tp = df[time_col].notna().astype(int) * fake_post  # placeholder

        # For DID: create fake treated_post
        if treatment_interaction in df.columns:
            df["_fake_tp"] = df[treatment_interaction] * fake_post
            rhs = f"_fake_tp"
        else:
            df["_fake_tp"] = fake_tp
            rhs = "_fake_tp"

        if ctrl_str:
            rhs += f" + {ctrl_str}"
        formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

        # Only use pre-treatment data
        pre_data = df[df[time_col] <= true_treatment_time]
        if len(pre_data) < 20:
            continue

        try:
            res = smf.ols(formula, data=pre_data).fit()
            beta = float(res.params.get("_fake_tp", 0))
            se = float(res.bse.get("_fake_tp", 0))
            pval = float(res.pvalues.get("_fake_tp", 1))
            results.append({
                "fake_time": fake_time,
                "shift": shift,
                "coefficient": beta,
                "std_error": se,
                "p_value": pval,
                "significant": pval < 0.05,
            })
        except Exception:
            continue

    # Test: any significant placebo effect?
    significant_placebos = [r for r in results if r["significant"]]
    passed = len(significant_placebos) == 0

    return {
        "method": "Placebo-in-time (falsification test)",
        "true_treatment_time": true_treatment_time,
        "results": results,
        "n_significant_placebos": len(significant_placebos),
        "passed": passed,
        "interpretation": (
            "No significant placebo effects in pre-treatment periods — "
            "treatment timing is credible." if passed
            else f"{len(significant_placebos)} placebo time(s) produced significant "
                 f"'effects' — anticipation or confounding may be present."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. Comprehensive sensitivity report
# ═══════════════════════════════════════════════════════════════════════

def run_all_sensitivity(df: pd.DataFrame, outcome: str,
                        treatment_interaction: str,
                        entity_col: str, time_col: str,
                        controls: list[str],
                        treatment_col: str = None,
                        true_treatment_time: float = None,
                        control_groups: list[list[str]] = None,
                        group_labels: list[str] = None) -> dict:
    """Run all sensitivity analyses and produce a combined report."""

    report = {}

    # 1. Oster bounds
    try:
        report["oster_bounds"] = oster_bounds(
            df, outcome, treatment_interaction, entity_col, time_col, controls
        )
    except Exception as e:
        report["oster_bounds"] = {"error": str(e)}

    # 2. Coefficient stability
    if control_groups:
        try:
            report["coefficient_stability"] = coefficient_stability(
                df, outcome, treatment_interaction, entity_col, time_col,
                control_groups, group_labels or []
            )
        except Exception as e:
            report["coefficient_stability"] = {"error": str(e)}

    # 3. Rosenbaum bounds
    if treatment_col and controls:
        try:
            report["rosenbaum_bounds"] = rosenbaum_bounds(
                df, outcome, treatment_col, controls
            )
        except Exception as e:
            report["rosenbaum_bounds"] = {"error": str(e)}

    # 4. Leave-one-out
    try:
        report["leave_one_out"] = leave_one_out(
            df, outcome, treatment_interaction, entity_col, time_col, controls
        )
    except Exception as e:
        report["leave_one_out"] = {"error": str(e)}

    # 5. Placebo-in-time
    if true_treatment_time is not None:
        try:
            report["placebo_in_time"] = placebo_in_time(
                df, outcome, treatment_interaction, entity_col, time_col,
                true_treatment_time, controls=controls
            )
        except Exception as e:
            report["placebo_in_time"] = {"error": str(e)}

    # Summarize
    report["summary"] = _summarize_sensitivity(report)

    return report


def _summarize_sensitivity(report: dict) -> dict:
    """Summarize all sensitivity checks into pass/fail/flag."""
    checks = []

    # Oster
    ob = report.get("oster_bounds", {})
    if "error" not in ob:
        delta_zero = ob.get("delta_for_zero")
        if delta_zero is not None and delta_zero > 1.0:
            checks.append({"name": "Oster bounds", "passed": True,
                           "interpretation": f"Unobservables would need to be {delta_zero:.1f}× as important as observables to nullify the effect (δ=1 benchmark)."})
        elif delta_zero is not None:
            checks.append({"name": "Oster bounds", "passed": False,
                           "interpretation": f"Weak: unobservables only {delta_zero:.1f}× as important as observables could explain away the effect."})
        else:
            checks.append({"name": "Oster bounds", "passed": True,
                           "interpretation": "R² did not increase with controls — bounds not informative."})

    # Coefficient stability
    cs = report.get("coefficient_stability", {})
    if "error" not in cs:
        checks.append({"name": "Coefficient stability", "passed": cs.get("stable", True),
                       "interpretation": cs.get("interpretation", "")})

    # Rosenbaum
    rb = report.get("rosenbaum_bounds", {})
    if "error" not in rb:
        cg = rb.get("critical_gamma")
        if cg is not None:
            checks.append({"name": "Rosenbaum bounds", "passed": cg >= 1.5,
                           "interpretation": rb.get("interpretation", "")})

    # Leave-one-out
    loo = report.get("leave_one_out", {})
    if "error" not in loo:
        checks.append({"name": "Leave-one-out", "passed": loo.get("n_influential", 0) == 0,
                       "interpretation": loo.get("interpretation", "")})

    # Placebo-in-time
    pit = report.get("placebo_in_time", {})
    if "error" not in pit:
        checks.append({"name": "Placebo-in-time", "passed": pit.get("passed", True),
                       "interpretation": pit.get("interpretation", "")})

    n_pass = sum(1 for c in checks if c["passed"])
    n_total = len(checks)

    return {
        "checks": checks,
        "n_pass": n_pass,
        "n_total": n_total,
        "overall": "robust" if n_pass == n_total else ("moderate" if n_pass >= n_total / 2 else "sensitive"),
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for causal inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sensitivity suite
  python sensitivity_analysis.py --data panel.dta --outcome log_y \\
      --entity id --time year --treated treated --post post \\
      --controls gdp pop --treatment-col treated --first-treated 2016

  # Oster bounds only
  python sensitivity_analysis.py --data panel.dta --outcome log_y \\
      --entity id --time year --treated treated --post post \\
      --controls gdp pop --check oster
""")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--treated", default="treated", help="Treatment dummy column")
    parser.add_argument("--post", default="post", help="Post-treatment dummy column")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variables")
    parser.add_argument("--treatment-col", default=None, help="Treatment column for Rosenbaum")
    parser.add_argument("--first-treated", type=float, default=None,
                        help="First-treated year (for placebo-in-time)")
    parser.add_argument("--check", default="all",
                        choices=["all", "oster", "stability", "rosenbaum", "leave-one-out", "placebo-time"],
                        help="Which sensitivity check to run")
    parser.add_argument("--control-groups", nargs="*", default=[],
                        help="Control variable groupings for stability test (semicolon-separated: 'gdp;pop;fiscal')")
    parser.add_argument("--output", default=None, help="Output path (.json)")
    parser.add_argument("--rmax", type=float, default=1.3, help="Rmax for Oster bounds")
    parser.add_argument("--delta", type=float, default=1.0, help="Delta for Oster bounds")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required. Install with: pip install statsmodels")
        sys.exit(1)

    df = load_data(args.data)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Build treatment interaction column
    if "treated_post" not in df.columns:
        df["treated_post"] = df[args.treated].astype(float) * df[args.post].astype(float)
    treatment_interaction = "treated_post"

    # Parse control groups
    control_groups = None
    group_labels = None
    if args.control_groups:
        control_groups = [g.split(",") for g in args.control_groups]
        group_labels = [f"Group {i+1}: {', '.join(g)}" for i, g in enumerate(control_groups)]

    if args.check == "all":
        report = run_all_sensitivity(
            df, args.outcome, treatment_interaction,
            args.entity, args.time, args.controls,
            treatment_col=args.treatment_col or args.treated,
            true_treatment_time=args.first_treated,
            control_groups=control_groups,
            group_labels=group_labels,
        )
    elif args.check == "oster":
        report = oster_bounds(df, args.outcome, treatment_interaction,
                              args.entity, args.time, args.controls,
                              rmax=args.rmax, delta=args.delta)
    elif args.check == "stability":
        report = coefficient_stability(df, args.outcome, treatment_interaction,
                                       args.entity, args.time,
                                       control_groups or [[c] for c in args.controls],
                                       group_labels or args.controls)
    elif args.check == "rosenbaum":
        report = rosenbaum_bounds(df, args.outcome,
                                  args.treatment_col or args.treated,
                                  args.controls)
    elif args.check == "leave-one-out":
        report = leave_one_out(df, args.outcome, treatment_interaction,
                               args.entity, args.time, args.controls)
    elif args.check == "placebo-time":
        if args.first_treated is None:
            print("Error: --first-treated required for placebo-in-time test")
            sys.exit(1)
        report = placebo_in_time(df, args.outcome, treatment_interaction,
                                 args.entity, args.time, args.first_treated,
                                 controls=args.controls)

    # Print results
    print("\n═══════════════════════════════════")
    print("Sensitivity Analysis")
    print("═══════════════════════════════════")

    if "summary" in report:
        s = report["summary"]
        print(f"\nOverall: {s['overall'].upper()}")
        print(f"  {s['n_pass']}/{s['n_total']} checks passed.\n")
        for c in s["checks"]:
            status = "✓" if c["passed"] else "✗"
            print(f"  {status} {c['name']}")
            print(f"    {c['interpretation']}")

    # Print individual results
    for key in ["oster_bounds", "coefficient_stability", "rosenbaum_bounds",
                "leave_one_out", "placebo_in_time"]:
        if key in report and "interpretation" in report[key]:
            print(f"\n─── {report[key].get('method', key)} ───")
            print(report[key]["interpretation"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
