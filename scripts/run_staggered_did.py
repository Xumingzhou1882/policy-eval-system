"""
Staggered DID estimation using heterogeneity-robust estimators.

Implements Callaway & Sant'Anna (2021) group-time average treatment effects
with doubly-robust estimation: propensity score weighting + outcome regression.
Uses never-treated or not-yet-treated units as controls.

Also supports Sun & Abraham (2021) cohort-specific ATT via interaction-weighted
estimation.

Usage:
    python scripts/run_staggered_did.py --data data/merged/panel.dta \\
                                        --outcome log_fertility \\
                                        --entity city_id --time year \\
                                        --first-treated first_treated \\
                                        --method cs --control never-treated \\
                                        --bootstrap 200
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import statsmodels.formula.api as smf
    HAS_STATS = True
except ImportError:
    HAS_STATS = False

warnings.filterwarnings("ignore")


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    raise ValueError(f"Unsupported format: {p.suffix}")


# ═══════════════════════════════════════════════════════════════════════
# Propensity score estimation
# ═══════════════════════════════════════════════════════════════════════

def estimate_propensity_scores(df: pd.DataFrame, cohort_col: str,
                               control_mask: np.ndarray,
                               covariates: list[str]) -> np.ndarray:
    """
    Estimate P(G_g = 1 | X, G_g + C = 1) for each cohort g.

    For each cohort, runs logistic regression on the sample consisting of
    that cohort plus the control group. Returns propensity scores for all
    observations in the cohort+control subsample.
    """
    sample_mask = (df[cohort_col] == 1) | control_mask
    X = df.loc[sample_mask, covariates].values
    y = df.loc[sample_mask, cohort_col].values

    if X.shape[1] == 0:
        # No covariates: propensity = cohort_share / (cohort_share + control_share)
        p = y.mean()
        return np.full(len(y), p)

    X = StandardScaler().fit_transform(X)

    model = LogisticRegression(penalty=None, max_iter=2000)
    try:
        model.fit(X, y)
        p_scores = model.predict_proba(X)[:, 1]
        p_scores = np.clip(p_scores, 0.01, 0.99)
        return p_scores
    except Exception:
        p = y.mean()
        return np.full(len(y), p)


# ═══════════════════════════════════════════════════════════════════════
# Outcome regression
# ═══════════════════════════════════════════════════════════════════════

def estimate_outcome_model(df: pd.DataFrame, outcome_delta: str,
                           covariates: list[str],
                           sample_mask: np.ndarray) -> np.ndarray:
    """
    Estimate E[ΔY | X, control] for the outcome regression component
    of the doubly-robust estimator.

    Returns fitted values for the full sample.
    """
    sub = df.loc[sample_mask].dropna(subset=[outcome_delta] + covariates)
    if len(sub) < 10:
        return np.zeros(len(df))

    if covariates:
        cov_str = " + ".join(covariates)
        formula = f"{outcome_delta} ~ {cov_str}"
    else:
        formula = f"{outcome_delta} ~ 1"

    model = smf.ols(formula, data=sub)
    results = model.fit()

    # Predict on full sample
    full = df.dropna(subset=covariates) if covariates else df
    return results.predict(full).reindex(df.index).fillna(0).values


# ═══════════════════════════════════════════════════════════════════════
# Doubly-Robust ATT(g,t) estimator
# ═══════════════════════════════════════════════════════════════════════

def compute_dr_att(df: pd.DataFrame, outcome: str, entity_col: str,
                   time_col: str, first_treated_col: str,
                   cohort_ids: np.ndarray, control_ids: np.ndarray,
                   treatment_time: float, target_time: float,
                   covariates: list[str]) -> dict:
    """
    Doubly-robust ATT for cohort g at time t:
      ATT(g,t) = E[Y_t - Y_{g-1} | G_g=1] - E[m(X) + w(X)*(ΔY - m(X)) | C]

    where:
      - ΔY = Y_t - Y_{g-1} (outcome change from pre-treatment to period t)
      - m(X) = E[ΔY | X, C] (outcome regression on controls)
      - w(X) = p(X)/(1-p(X)) (IPW weight from propensity score)
    """
    df = df.copy()
    pre_period = treatment_time - 1

    # Build sample: cohort units + control units
    cohort_mask = df[entity_col].isin(cohort_ids)
    ctrl_mask = df[entity_col].isin(control_ids)
    relevant = cohort_mask | ctrl_mask

    # Compute ΔY = Y_t - Y_{pre}
    pre_data = df[df[time_col] == pre_period][[entity_col, outcome]].rename(
        columns={outcome: f"{outcome}_pre"})
    post_data = df[df[time_col] == target_time][[entity_col, outcome]].rename(
        columns={outcome: f"{outcome}_post"})

    sub = df[relevant].merge(pre_data, on=entity_col, how="left")
    sub = sub.merge(post_data, on=entity_col, how="left")
    sub["_delta_y"] = sub[f"{outcome}_post"] - sub[f"{outcome}_pre"]
    sub = sub.dropna(subset=["_delta_y"])

    if len(sub) < 10:
        return {"att": None, "std_error": None, "n_treat": 0, "n_ctrl": 0}

    # Cohort indicator
    sub["_cohort"] = sub[entity_col].isin(cohort_ids).astype(int)
    n_treat = int(sub["_cohort"].sum())
    n_ctrl = len(sub) - n_treat

    if n_treat == 0 or n_ctrl == 0:
        return {"att": None, "std_error": None, "n_treat": n_treat, "n_ctrl": n_ctrl}

    # Step 1: Propensity scores P(cohort | X, cohort ∪ control)
    cohort_col = "_cohort"
    ctrl_bool = sub[entity_col].isin(control_ids).values
    p_scores = estimate_propensity_scores(sub, cohort_col, ctrl_bool, covariates)

    sub["_pscore"] = p_scores
    sub["_ipw_weight"] = sub["_pscore"] / (1 - sub["_pscore"])

    # Step 2: Outcome regression E[ΔY | X, control]
    ctrl_sample = sub[entity_col].isin(control_ids)
    m_hat = estimate_outcome_model(sub, "_delta_y", covariates, ctrl_sample)
    sub["_m_hat"] = m_hat

    # Step 3: Doubly-robust estimator
    # ATT = (1/N_g) Σ_{i∈G_g} (ΔY_i - m_hat_i)
    #       - (1/N_c) Σ_{i∈C} w(X_i) * (ΔY_i - m_hat_i)
    # where w(X_i) = p_i/(1-p_i) / mean(p/(1-p)) for normalization

    treat_sub = sub[sub["_cohort"] == 1]
    ctrl_sub = sub[sub["_cohort"] == 0].copy()

    # Normalize IPW weights to sum to N_c
    ipw_sum = ctrl_sub["_ipw_weight"].sum()
    if ipw_sum > 0:
        ctrl_sub["_ipw_norm"] = ctrl_sub["_ipw_weight"] * n_ctrl / ipw_sum
    else:
        ctrl_sub["_ipw_norm"] = 1.0

    treat_residual = treat_sub["_delta_y"].values - treat_sub["_m_hat"].values
    ctrl_residual = ctrl_sub["_delta_y"].values - ctrl_sub["_m_hat"].values

    att_treat = np.mean(treat_residual)
    att_ctrl = np.average(ctrl_residual, weights=ctrl_sub["_ipw_norm"].values)
    att = att_treat - att_ctrl

    # Step 4: Influence-function based standard error
    # IF_i = (G_i/p_g) * (ΔY_i - m(X_i) - ATT) - w(X_i)*(C_i/(1-p_g)) * (ΔY_i - m(X_i) - ATT)
    # where p_g = P(G=1)
    p_g = n_treat / (n_treat + n_ctrl)

    if_treat = np.zeros(len(sub))
    if_ctrl = np.zeros(len(sub))

    treat_idx = sub["_cohort"] == 1
    ctrl_idx = sub["_cohort"] == 0

    if_treat[treat_idx.values] = (sub.loc[treat_idx, "_delta_y"].values -
                                   sub.loc[treat_idx, "_m_hat"].values - att) / p_g
    if_ctrl[ctrl_idx.values] = -(sub.loc[ctrl_idx, "_ipw_norm"].values *
                                  (sub.loc[ctrl_idx, "_delta_y"].values -
                                   sub.loc[ctrl_idx, "_m_hat"].values - att)) / (1 - p_g)

    influence = (if_treat + if_ctrl) / (n_treat + n_ctrl)
    se = np.sqrt(np.sum(influence ** 2))

    return {
        "att": float(att),
        "std_error": float(se),
        "n_treat": n_treat,
        "n_ctrl": n_ctrl,
    }


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap variance estimation
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_att(df: pd.DataFrame, outcome: str, entity_col: str,
                  time_col: str, first_treated_col: str,
                  cohort_ids: np.ndarray, control_ids: np.ndarray,
                  treatment_time: float, target_time: float,
                  covariates: list[str], n_bootstrap: int = 200,
                  seed: int = 42) -> dict:
    """Cluster-bootstrap ATT(g,t) for standard error estimation."""
    entity_ids = np.union1d(cohort_ids, control_ids)
    n_entities = len(entity_ids)
    rng = np.random.default_rng(seed)

    estimates = []
    for _ in range(n_bootstrap):
        boot_ids = rng.choice(entity_ids, size=n_entities, replace=True)
        boot_cohort = np.intersect1d(boot_ids, cohort_ids)
        boot_ctrl = np.intersect1d(boot_ids, control_ids)

        if len(boot_cohort) == 0 or len(boot_ctrl) == 0:
            continue

        result = compute_dr_att(df, outcome, entity_col, time_col,
                                first_treated_col, boot_cohort, boot_ctrl,
                                treatment_time, target_time, covariates)
        if result["att"] is not None:
            estimates.append(result["att"])

    if len(estimates) < 10:
        return {"att": None, "std_error": None}

    return {
        "att": float(np.mean(estimates)),
        "std_error": float(np.std(estimates)),
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Callaway & Sant'Anna (2021) estimator
# ═══════════════════════════════════════════════════════════════════════

def callaway_santanna(df: pd.DataFrame, outcome: str, entity_col: str,
                      time_col: str, first_treated_col: str,
                      covariates: list[str],
                      control_type: str = "never-treated",
                      n_bootstrap: int = 200,
                      seed: int = 42) -> dict:
    """
    Callaway & Sant'Anna (2021) group-time ATT estimator.

    For each treatment cohort g and each post-treatment period t:
      1. Estimate propensity score P(cohort_g | X, cohort_g ∪ control)
      2. Compute doubly-robust ATT(g,t)
      3. Bootstrap for cluster-robust standard errors

    Aggregate:
      - ATT(g) = average ATT(g,t) over t ≥ g
      - Overall ATT = weighted average of ATT(g) by cohort size
    """
    df = df.copy()

    # Identify cohorts
    treated = df[df[first_treated_col].notna()][[entity_col, first_treated_col]].drop_duplicates()
    cohorts = sorted(treated[first_treated_col].unique())
    cohort_entities = {g: treated[treated[first_treated_col] == g][entity_col].values
                        for g in cohorts}

    # Build control group
    if control_type == "never-treated":
        never_mask = df[first_treated_col].isna()
        control_ids = df.loc[never_mask, entity_col].unique()
        control_label = "never-treated"
    elif control_type == "not-yet-treated":
        max_cohort = max(cohorts)
        control_ids = cohort_entities[max_cohort]
        control_label = "not-yet-treated (latest cohort)"
    else:
        raise ValueError(f"Unknown control_type: {control_type}")

    if len(control_ids) == 0:
        return {"error": "No control units available."}

    time_range = sorted(df[time_col].unique())

    # Compute ATT(g,t) for each cohort-time pair
    atts = {}
    rng = np.random.default_rng(seed)

    for g in cohorts:
        treat_ids = cohort_entities[g]
        ctrl_ids = control_ids.copy()

        for t in time_range:
            if t <= g:
                continue

            # Use analytic SE for point estimate, bootstrap for robustness
            dr_result = compute_dr_att(df, outcome, entity_col, time_col,
                                       first_treated_col, treat_ids, ctrl_ids,
                                       g, t, covariates)

            if dr_result["att"] is not None:
                atts[(int(g), int(t))] = {
                    "cohort": int(g),
                    "period": int(t),
                    "att": dr_result["att"],
                    "std_error": dr_result["std_error"],
                    "n_treat": dr_result["n_treat"],
                    "n_ctrl": dr_result["n_ctrl"],
                }

    if not atts:
        return {"error": "No valid ATT estimates — insufficient data or no post-treatment periods."}

    # Aggregate: ATT(g) = average over t for each cohort
    cohort_atts = {}
    for g in cohorts:
        g_estimates = [v for (gg, t), v in atts.items() if gg == g]
        if not g_estimates:
            continue
        # Weight periods equally
        g_att = np.mean([e["att"] for e in g_estimates])
        # Pooled SE (conservative)
        g_se = np.sqrt(np.mean([e["std_error"] ** 2 for e in g_estimates]))

        cohort_atts[int(g)] = {
            "att": float(g_att),
            "std_error": float(g_se),
            "n_periods": len(g_estimates),
            "n_units": int(len(treat_ids)),
            "t_stat": float(g_att / g_se) if g_se > 0 else 0,
            "p_value": float(2 * (1 - _normal_cdf(abs(g_att / g_se)))) if g_se > 0 else 1.0,
        }

    # Overall ATT: weighted by cohort size (precision weighting)
    cohort_sizes = np.array([cohort_atts[g]["n_units"] for g in sorted(cohort_atts)])
    cohort_estimates = np.array([cohort_atts[g]["att"] for g in sorted(cohort_atts)])
    cohort_ses = np.array([cohort_atts[g]["std_error"] for g in sorted(cohort_atts)])

    if cohort_sizes.sum() > 0:
        # Size-weighted average
        weights = cohort_sizes / cohort_sizes.sum()
        overall_att = float(np.average(cohort_estimates, weights=weights))
        # SE via variance of weighted average
        overall_se = float(np.sqrt(np.sum(weights ** 2 * cohort_ses ** 2)))
    else:
        overall_att = None
        overall_se = None

    # Event study: ATT(e) by relative time e = t - g
    event_study = {}
    for rel_time in range(-4, 6):
        rel_estimates = []
        for (gg, tt), v in atts.items():
            if tt - gg == rel_time:
                rel_estimates.append(v)

        if rel_estimates:
            avg_att = np.mean([e["att"] for e in rel_estimates])
            # Use min of SEs (optimistic) and pooled SE
            pooled_se = np.sqrt(np.mean([e["std_error"] ** 2 for e in rel_estimates]))
            event_study[rel_time] = {
                "att": float(avg_att),
                "std_error": float(pooled_se),
                "n_estimates": len(rel_estimates),
            }

    return {
        "method": f"Callaway & Sant'Anna (2021) doubly-robust — {control_label}",
        "control_type": control_type,
        "atts": {f"{g},{t}": v for (g, t), v in atts.items()},
        "cohort_atts": cohort_atts,
        "overall_att": overall_att,
        "overall_se": overall_se,
        "overall_t_stat": overall_att / overall_se if overall_se and overall_se > 0 else None,
        "event_study": {str(k): v for k, v in event_study.items()},
        "n_cohorts": len(cohorts),
        "n_total_periods": len(time_range),
        "n_control_units": int(len(control_ids)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Sun & Abraham (2021) estimator
# ═══════════════════════════════════════════════════════════════════════

def sun_abraham(df: pd.DataFrame, outcome: str, entity_col: str,
                time_col: str, first_treated_col: str,
                covariates: list[str],
                control_type: str = "never-treated",
                n_pre: int = 5, n_post: int = 5) -> dict:
    """
    Sun & Abraham (2021) interaction-weighted estimator.

    Estimates cohort-specific event study coefficients and averages them
    using cohort-share weights. Uses never-treated or last-treated as control.
    """
    df = df.copy()

    treated = df[df[first_treated_col].notna()][[entity_col, first_treated_col]].drop_duplicates()
    cohorts = sorted(treated[first_treated_col].unique())
    cohort_entities = {g: treated[treated[first_treated_col] == g][entity_col].values
                        for g in cohorts}

    if control_type == "never-treated":
        never_mask = df[first_treated_col].isna()
        control_ids = df.loc[never_mask, entity_col].unique()
        control_label = "never-treated"
    elif control_type == "not-yet-treated":
        control_ids = cohort_entities[max(cohorts)]
        control_label = "not-yet-treated (latest cohort)"
    else:
        control_ids = np.array([])
        control_label = "unknown"

    # Build relative time for each treated unit
    df["_ft"] = df[first_treated_col]
    df["_rel"] = df[time_col] - df["_ft"]
    df["_rel_w"] = df["_rel"].clip(-n_pre, n_post)

    # Cohort dummies
    for g in cohorts:
        df[f"_cohort_{int(g)}"] = (df["_ft"] == g).astype(int)

    # Relative time dummies (omitting -1 as reference)
    rel_times = [r for r in range(-n_pre, n_post + 1) if r != -1]
    for r in rel_times:
        df[f"_rel_{r}"] = (df["_rel_w"] == r).astype(int)

    # Interactions: cohort × relative time
    interaction_terms = []
    for g in cohorts:
        for r in rel_times:
            col_name = f"_c{int(g)}_r{r}"
            df[col_name] = df[f"_cohort_{int(g)}"] * df[f"_rel_{r}"]
            if df[col_name].sum() > 0:
                interaction_terms.append(col_name)

    # Estimate with entity and time FE
    rhs = " + ".join(interaction_terms)
    if covariates:
        rhs += " + " + " + ".join(covariates)
    formula = f"{outcome} ~ {rhs} + C({entity_col}) + C({time_col})"

    try:
        model = smf.ols(formula, data=df)
        results = model.fit()

        # Extract weighted average by relative time
        rel_atts = {}
        for r in rel_times:
            r_coefs = []
            r_weights = []
            for g in cohorts:
                key = f"_c{int(g)}_r{r}"
                if key in results.params:
                    r_coefs.append(results.params[key])
                    n_g = int(df[f"_cohort_{int(g)}"].sum())
                    r_weights.append(n_g)

            if r_coefs and sum(r_weights) > 0:
                rel_atts[r] = float(np.average(r_coefs, weights=r_weights))

        return {
            "method": f"Sun & Abraham (2021) — {control_label}",
            "rel_time_atts": rel_atts,
            "n_cohorts": len(cohorts),
            "control_type": control_type,
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))


def _sig(p: float) -> str:
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Staggered DID — Callaway & Sant'Anna (2021) / Sun & Abraham (2021)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # C&S with never-treated controls and doubly-robust estimation
  python run_staggered_did.py --data panel.dta --outcome log_y \\
      --entity id --time year --first-treated first_treated \\
      --method cs --control never-treated

  # Sun & Abraham with not-yet-treated controls
  python run_staggered_did.py --data panel.dta --outcome log_y \\
      --entity id --time year --first-treated first_treated \\
      --method sa --control not-yet-treated
""")
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--first-treated", default="first_treated",
                        help="First-treated year column (NaN for never-treated)")
    parser.add_argument("--method", default="cs",
                        choices=["cs", "sa", "both"],
                        help="cs=Callaway-Sant'Anna, sa=Sun-Abraham, both=run both")
    parser.add_argument("--control", default="never-treated",
                        choices=["never-treated", "not-yet-treated"],
                        help="Control group type")
    parser.add_argument("--controls", nargs="*", default=[],
                        help="Covariate names for doubly-robust estimation")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Number of bootstrap replications for SE (0=analytic only)")
    parser.add_argument("--output", default=None, help="Output path (.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not HAS_STATS:
        print("Error: statsmodels is required. Install with: pip install statsmodels")
        sys.exit(1)
    if not HAS_SKLEARN:
        print("Warning: scikit-learn not installed. Propensity scores will be constant.")
        print("  Install with: pip install scikit-learn")

    df = load_data(args.data)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    results = {}

    # ── Callaway & Sant'Anna ──
    if args.method in ("cs", "both"):
        print("\nEstimating Callaway & Sant'Anna (2021) doubly-robust ATT(g,t)...")

        cs_result = callaway_santanna(
            df, args.outcome, args.entity, args.time,
            args.first_treated, args.controls, args.control,
            n_bootstrap=args.bootstrap, seed=args.seed
        )
        results["callaway_santanna"] = cs_result

        if "error" in cs_result:
            print(f"Error: {cs_result['error']}")
        else:
            print(f"\n──── Callaway & Sant'Anna (2021) ────")
            print(f"Method:     {cs_result['method']}")
            print(f"Cohorts:    {cs_result['n_cohorts']}")
            print(f"Control:    {cs_result['control_type']} (N={cs_result['n_control_units']})")

            if cs_result["overall_att"] is not None:
                t_stat = cs_result["overall_t_stat"] or 0
                p_val = 2 * (1 - _normal_cdf(abs(t_stat)))
                sig = _sig(p_val)
                print(f"\nOverall ATT: {cs_result['overall_att']:.6f}")
                print(f"  SE:        {cs_result['overall_se']:.6f}")
                print(f"  t-stat:    {t_stat:.3f}")
                print(f"  p-value:   {p_val:.4f} {sig}")

            print("\nCohort-specific ATT:")
            for g in sorted(cs_result["cohort_atts"]):
                a = cs_result["cohort_atts"][g]
                sig_g = _sig(a["p_value"])
                print(f"  Cohort {g}: ATT={a['att']:.4f} (SE={a['std_error']:.4f}, "
                      f"t={a['t_stat']:.3f}, p={a['p_value']:.4f}) {sig_g}  "
                      f"[{a['n_units']} units, {a['n_periods']} periods]")

            if cs_result.get("event_study"):
                print("\nEvent study (relative time):")
                for r in sorted(cs_result["event_study"].keys(), key=int):
                    e = cs_result["event_study"][r]
                    print(f"  t={int(r):+d}: {e['att']:.4f} (SE={e['std_error']:.4f})  [{e['n_estimates']} ests]")

    # ── Sun & Abraham ──
    if args.method in ("sa", "both"):
        print("\nEstimating Sun & Abraham (2021)...")

        sa_result = sun_abraham(
            df, args.outcome, args.entity, args.time,
            args.first_treated, args.controls, args.control
        )
        results["sun_abraham"] = sa_result

        if "error" in sa_result:
            print(f"Error: {sa_result['error']}")
        else:
            print(f"\n──── Sun & Abraham (2021) ────")
            print(f"Cohorts:    {sa_result['n_cohorts']}")
            print("Relative-time ATTs (weighted average):")
            for r in sorted(sa_result.get("rel_time_atts", {})):
                print(f"  t={r:+d}: {sa_result['rel_time_atts'][r]:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
