"""
Stage 3: Theoretical method analysis — deterministic decision engine.

Takes structured inputs about the policy's assignment mechanism (filled in by
the LLM after Stage 2 policy research) and outputs a theoretically appropriate
identification strategy. Does NOT consider data availability.

The decision tree is encoded deterministically: same inputs → same output, every time.

Usage:
    python scripts/stage3_analyze.py --mechanism staggered_policy_shock \\
                                     --staggered true \\
                                     --has-control-group true \\
                                     --panel-available true \\
                                     --output stage3_result.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Decision tree data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Assumption:
    name: str
    description: str
    testable: bool
    test_method: str = ""


@dataclass
class Fallback:
    method: str
    condition: str
    assumption_relaxed: str


@dataclass
class MethodRecommendation:
    primary_method: str
    source_of_variation: str
    why: list[str]
    assumptions: list[Assumption]
    fallbacks: list[Fallback]
    required_vars: list[str]
    optional_vars: list[str]
    literature_key: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Assignment mechanism definitions
# ═══════════════════════════════════════════════════════════════════════

MECHANISM_DEFINITIONS = {
    "random_assignment": {
        "label": "Random assignment (lottery, RCT, natural randomization)",
        "description": "Treatment is assigned by a random process independent of potential outcomes.",
    },
    "threshold_rule": {
        "label": "Threshold-based eligibility",
        "description": "A continuous score determines eligibility with a sharp or fuzzy cutoff.",
    },
    "selection_on_observables": {
        "label": "Selection on observable characteristics",
        "description": "Treatment depends on measured characteristics; no time dimension or cross-section only.",
    },
    "staggered_policy_shock": {
        "label": "Staggered policy shock (different units treated at different times)",
        "description": "A policy is rolled out in batches; treatment timing varies across units.",
    },
    "single_policy_shock": {
        "label": "Single-time policy shock",
        "description": "A policy takes effect at one known time for all treated units simultaneously.",
    },
    "time_varying_unobservables": {
        "label": "Selection on time-varying unobservables",
        "description": "Confounders that change over time and differ across units drive treatment assignment.",
    },
    "continuous_intensity": {
        "label": "Continuous treatment intensity",
        "description": "All units are treated, but at different doses/intensities.",
    },
    "multiple_overlapping_policies": {
        "label": "Multiple overlapping policies",
        "description": "Several policies affect the same units concurrently.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# The decision tree (deterministic)
# ═══════════════════════════════════════════════════════════════════════

def decide_method(
    mechanism: str,
    staggered: bool = False,
    has_control_group: bool = True,
    threshold_type: Optional[str] = None,  # "sharp" | "fuzzy"
    panel_available: bool = True,
    has_instrument: bool = False,
    multiple_policies: bool = False,
    treatment_cross_section: bool = False,
    everyone_treated_eventually: bool = False,
) -> MethodRecommendation:
    """
    Deterministic decision tree for causal identification strategy.

    Parameters
    ----------
    mechanism : str
        One of the assignment mechanism keys from MECHANISM_DEFINITIONS.
    staggered : bool
        Whether treatment is rolled out at different times for different units.
    has_control_group : bool
        Whether there are units that never receive treatment.
    threshold_type : str or None
        "sharp" if treatment jumps 0→1 at cutoff; "fuzzy" if jump < 1.
    panel_available : bool
        Whether panel data structure exists (same units observed over time).
    has_instrument : bool
        Whether a plausibly valid instrument exists.
    multiple_policies : bool
        Whether multiple policies affect the same outcome concurrently.
    treatment_cross_section : bool
        Whether treatment varies only in the cross-section (no time dimension).
    everyone_treated_eventually : bool
        Whether all units are eventually treated (no never-treated group).

    Returns
    -------
    MethodRecommendation
    """
    why = []
    assumptions = []
    fallbacks = []
    required_vars = []
    optional_vars = []

    # ── Mechanism: Random assignment ─────────────────────────────────
    if mechanism == "random_assignment":
        return MethodRecommendation(
            primary_method="Randomization inference / Fisher exact test",
            source_of_variation="Random assignment breaks the statistical link between treatment and all confounders (observed and unobserved).",
            why=[
                "Treatment is randomly assigned — the gold standard for causal identification.",
                "No need for control variables; simple difference-in-means is unbiased.",
                "Randomization inference provides exact p-values without distributional assumptions.",
            ],
            assumptions=[
                Assumption("SUTVA (Stable Unit Treatment Value Assumption)", "No interference between units; treatment of one unit does not affect outcomes of others.", False, "Cannot be formally tested — requires institutional argument."),
                Assumption("No selective attrition", "Units do not drop out of the study in a way correlated with treatment.", True, "Compare attrition rates between treatment and control; test if baseline covariates predict attrition."),
                Assumption("Compliance", "Units assigned to treatment actually receive it; units assigned to control do not.", True, "Check actual treatment receipt rates by assignment group."),
            ],
            fallbacks=[
                Fallback("IV (treatment assignment as instrument)", "If compliance is imperfect", "Relaxes perfect compliance; identifies LATE for compliers."),
            ],
            required_vars=["treatment_assignment", "outcome"],
            optional_vars=["baseline_covariates (for balance check)", "compliance_indicator"],
            literature_key="Imbens & Rubin (2015), Causal Inference for Statistics, Social, and Biomedical Sciences.",
        )

    # ── Mechanism: Threshold rule ────────────────────────────────────
    if mechanism == "threshold_rule":
        is_sharp = threshold_type == "sharp"
        method_name = "Sharp RDD" if is_sharp else "Fuzzy RDD"
        variation_desc = (
            "Units just above and just below the threshold are as-good-as-randomly assigned. "
            "Comparing them identifies the local average treatment effect at the cutoff."
        )

        why = [
            f"A known continuous threshold ({'sharp' if is_sharp else 'fuzzy'}) determines treatment eligibility.",
            "In a narrow window around the cutoff, treatment is approximately random.",
            f"{'Treatment probability jumps from 0 to 1 at the cutoff → Sharp RDD.' if is_sharp else 'Treatment probability jumps by less than 1 at the cutoff (imperfect compliance) → Fuzzy RDD (IV).'}",
        ]

        assumptions = [
            Assumption("No precise manipulation of the running variable", "Units cannot precisely control their position relative to the cutoff.", True, "McCrary (2008) density test: check for discontinuity in the density of the running variable at the cutoff."),
            Assumption("Continuity of potential outcomes", "The expected potential outcomes are smooth functions of the running variable at the cutoff.", False, "Cannot be directly tested; supported by covariate balance checks at the cutoff."),
            Assumption("Local linearity", "The relationship between running variable and outcome is approximately linear near the cutoff.", True, "Compare linear vs. quadratic specifications; test sensitivity to bandwidth choice (CCT optimal bandwidth)."),
        ]

        if not is_sharp:
            assumptions.append(
                Assumption("First-stage strength (fuzzy RDD)", "The jump in treatment probability at the cutoff is statistically significant.", True, "First-stage F-statistic at the cutoff; F < 10 indicates weak instrument — use robust inference.")
            )

        fallbacks = [
            Fallback("Donut-hole RDD", "If McCrary test detects manipulation near the cutoff", "Relaxes no-manipulation; drops observations in a window around the cutoff."),
            Fallback("DID", "If RDD assumptions fail and a pre-treatment period exists", "Uses time variation instead of threshold variation; requires parallel trends."),
        ]

        required_vars = ["running_variable", "outcome", "cutoff_value"]
        optional_vars = ["baseline_covariates (for balance check)", "treatment_indicator (fuzzy RDD)"]

        return MethodRecommendation(
            primary_method=method_name,
            source_of_variation=variation_desc,
            why=why,
            assumptions=assumptions,
            fallbacks=fallbacks,
            required_vars=required_vars,
            optional_vars=optional_vars,
            literature_key="Calonico, Cattaneo & Titiunik (2014), 'Robust Nonparametric Confidence Intervals for RDD.'; Lee & Lemieux (2010), 'Regression Discontinuity Designs in Economics.'",
        )

    # ── Mechanism: Selection on observables ──────────────────────────
    if mechanism == "selection_on_observables":
        return MethodRecommendation(
            primary_method="Propensity Score Matching (PSM) / Inverse Probability Weighting (IPW) / Doubly-Robust (AIPW)",
            source_of_variation="Conditional on observed covariates, treatment assignment is as-good-as-random.",
            why=[
                "Treatment depends on characteristics that are measured in the data.",
                "By conditioning on these characteristics (via matching, weighting, or regression), we can compare comparable units.",
                "Doubly-robust methods (AIPW) combine outcome modeling and propensity score modeling — only one needs to be correct.",
            ],
            assumptions=[
                Assumption("Unconfoundedness (CIA / Selection on Observables)", "All variables that affect both treatment and outcome are measured and included.", False, "Cannot be formally tested. Must be argued from institutional knowledge. Sensitivity analysis: Rosenbaum bounds, Oster (2019) delta."),
                Assumption("Common support / overlap", "For every treated unit, there exists a comparable untreated unit (and vice versa).", True, "Check propensity score distributions; trim off-support observations."),
                Assumption("No interference (SUTVA)", "Treatment of one unit does not affect outcomes of other units.", False, "Argue from study design."),
            ],
            fallbacks=[
                Fallback("CEM (Coarsened Exact Matching)", "If PSM balance is poor", "Non-parametric; balances on coarsened covariates. Preferred when balance is more important than sample size."),
                Fallback("Entropy Balancing", "If PSM fails to achieve balance in higher moments", "Balances on first, second, and higher moments of covariate distributions."),
                Fallback("DID", "If panel data becomes available", "Controls for time-invariant unobserved confounders — relaxes CIA."),
            ],
            required_vars=["outcome", "treatment_indicator", "covariates (all variables affecting selection)"],
            optional_vars=["instrument (for sensitivity analysis)"],
            literature_key="Rosenbaum & Rubin (1983), 'The Central Role of the Propensity Score.'; Imbens (2004), 'Nonparametric Estimation of Average Treatment Effects under Exogeneity.'",
        )

    # ── Mechanism: Single policy shock ───────────────────────────────
    if mechanism == "single_policy_shock":
        return MethodRecommendation(
            primary_method="Standard Difference-in-Differences (TWFE)",
            source_of_variation="Within-unit change over time, differenced between treated and untreated units.",
            why=[
                "A single policy event at a known time affects some units (treated) but not others (control).",
                "The DID estimator removes time-invariant unit-specific confounders and period-specific shocks.",
                "Standard TWFE is unbiased when treatment effects are homogeneous and there is a single treatment time.",
            ],
            assumptions=[
                Assumption("Parallel trends", "In the absence of treatment, treated and control units would have followed the same outcome trajectory.", True, "Event study: test whether pre-treatment coefficients are jointly zero. Visual inspection of pre-treatment trends."),
                Assumption("No anticipation", "Units did not change behavior before the policy took effect.", True, "Check if the t-1 coefficient in the event study is zero."),
                Assumption("No spillover (SUTVA)", "Treatment of one unit does not affect outcomes of control units.", False, "Argue from institutional context; consider spatial placebo tests."),
                Assumption("Stable unit composition", "The composition of treated and control groups does not change over time.", True, "Check for entry/exit of units; use balanced panel or document attrition."),
            ],
            fallbacks=[
                Fallback("PSM-DID", "If treated and control differ systematically in observables", "Re-weights control units to match treated units on pre-treatment characteristics, then applies DID."),
                Fallback("SCM", "If the number of treated units is small and no clear control group exists", "Constructs a synthetic counterfactual from a weighted combination of untreated units."),
            ],
            required_vars=["outcome", "treatment_indicator", "post_period_indicator", "entity_id", "time"],
            optional_vars=["control_variables", "pre_treatment_outcome_trends"],
            literature_key="Angrist & Pischke (2009), Mostly Harmless Econometrics, Ch. 5.; Bertrand, Duflo & Mullainathan (2004), 'How Much Should We Trust DID Estimates?'",
        )

    # ── Mechanism: Staggered policy shock ────────────────────────────
    if mechanism == "staggered_policy_shock":
        if everyone_treated_eventually and not has_control_group:
            return MethodRecommendation(
                primary_method="Callaway & Sant'Anna (2021) — not-yet-treated as control",
                source_of_variation="Differential treatment timing: earlier-treated units serve as control for later-treated units (before they are treated).",
                why=[
                    "All units are eventually treated — there is no never-treated group.",
                    "C&S (2021) uses not-yet-treated units as the control group for each cohort.",
                    "This identifies ATT for each cohort over the window before the control group itself gets treated.",
                    "Important caveat: only identifies effects for periods before the control group's own treatment date.",
                ],
                assumptions=[
                    Assumption("Parallel trends (conditional on covariates)", "Treated and not-yet-treated units would have followed parallel paths absent treatment.", True, "Pre-treatment event study coefficients jointly zero. Can condition on covariates to strengthen."),
                    Assumption("No anticipation", "Units do not adjust behavior before their treatment date.", True, "Check t-1 coefficient; if significant, re-code treatment date or use S&A 'anticipation-aware' estimator."),
                    Assumption("Limited treatment effect heterogeneity", "C&S estimator aggregates cohort-specific ATTs. Strong heterogeneity across cohorts weakens the aggregate interpretation.", True, "Report cohort-specific ATTs; check if they are similar."),
                ],
                fallbacks=[
                    Fallback("Sun & Abraham (2021)", "If anticipation effects are suspected", "Allows specification of anticipation periods via 'pre-periods' argument."),
                    Fallback("Borusyak, Jaravel & Spiess (2024)", "If the factor structure is more appropriate", "Imputation-based: estimates treatment effects by imputing counterfactuals from untreated observations using a factor model."),
                    Fallback("de Chaisemartin & D'Haultfoeuille (2020)", "If interested in instantaneous switchers", "Focuses on the effect in the first period after treatment switch; uses switchers vs. stable units."),
                ],
                required_vars=["outcome", "entity_id", "time", "first_treatment_year"],
                optional_vars=["covariates", "treatment_indicator"],
                literature_key="Callaway & Sant'Anna (2021), 'Difference-in-Differences with Multiple Time Periods.'; Goodman-Bacon (2021), 'Difference-in-Differences with Variation in Treatment Timing.'",
            )
        else:
            return MethodRecommendation(
                primary_method="Callaway & Sant'Anna (2021) — never-treated as control",
                source_of_variation="Differential treatment timing across cohorts, compared against never-treated units.",
                why=[
                    "Policy rollout is staggered: different units are treated at different times.",
                    "Traditional TWFE in staggered designs produces a weighted average of all 2×2 DIDs, and can assign negative weights when treatment effects are heterogeneous (Goodman-Bacon 2021).",
                    "C&S (2021) estimates group-time ATTs using never-treated units as the control group, avoiding the negative-weight problem.",
                    "Cohort-specific effects can be aggregated to an overall ATT with clear weighting.",
                ],
                assumptions=[
                    Assumption("Parallel trends (conditional on covariates)", "Treated and never-treated units would have followed parallel paths absent treatment.", True, "Pre-treatment event study coefficients jointly zero. Can condition on covariates to strengthen."),
                    Assumption("No anticipation", "Units do not adjust behavior before their treatment date.", True, "Check t-1 coefficient; if significant, re-code treatment date."),
                    Assumption("Never-treated units are a valid counterfactual", "Units that are never treated are comparable to treated units (after conditioning on covariates).", False, "Argue from institutional knowledge: why were some units never treated? If selection is non-random, this assumption may fail."),
                ],
                fallbacks=[
                    Fallback("Sun & Abraham (2021)", "If anticipation effects are suspected", "Allows specification of anticipation periods."),
                    Fallback("Borusyak, Jaravel & Spiess (2024)", "If factor structure is more appropriate", "Imputation-based estimator, robust to some forms of non-parallel trends."),
                    Fallback("SCM", "If only a few units are treated", "Better suited for small-N treated groups; constructs a synthetic counterfactual from donor pool."),
                ],
                required_vars=["outcome", "entity_id", "time", "first_treatment_year", "never_treated_indicator"],
                optional_vars=["covariates"],
                literature_key="Callaway & Sant'Anna (2021); Sun & Abraham (2021); Goodman-Bacon (2021); de Chaisemartin & D'Haultfoeuille (2020).",
            )

    # ── Mechanism: Time-varying unobservables ────────────────────────
    if mechanism == "time_varying_unobservables":
        if has_instrument:
            return MethodRecommendation(
                primary_method="Instrumental Variables (2SLS / LIML)",
                source_of_variation="Exogenous variation in the instrument induces variation in treatment that is independent of unobserved confounders.",
                why=[
                    "Unobserved confounders that change over time make DID and matching invalid.",
                    "A valid instrument provides exogenous variation in treatment — it affects the outcome only through treatment.",
                    "2SLS identifies the LATE: the average treatment effect for compliers (units whose treatment status is affected by the instrument).",
                ],
                assumptions=[
                    Assumption("Relevance", "The instrument significantly predicts treatment.", True, "First-stage F-statistic (Montiel Olea & Pflueger 2013 effective F). Critical value depends on number of instruments and desired maximal bias."),
                    Assumption("Exclusion restriction", "The instrument affects the outcome ONLY through its effect on treatment.", False, "Cannot be formally tested. Must be argued from institutional/economic theory. Partial test: overidentification test (Hansen J) if multiple instruments exist."),
                    Assumption("Monotonicity", "The instrument affects treatment in the same direction for all units (no defiers).", False, "Plausible if the instrument is a policy eligibility rule that can only increase treatment probability."),
                ],
                fallbacks=[
                    Fallback("LIML", "If instruments are weak (effective F < critical value)", "Limited Information Maximum Likelihood is more robust to weak instruments than 2SLS."),
                    Fallback("SCM", "If no valid instrument exists", "Constructs a synthetic counterfactual without requiring instruments."),
                ],
                required_vars=["outcome", "treatment", "instrument(s)"],
                optional_vars=["covariates", "multiple_instruments (for overid test)"],
                literature_key="Angrist, Imbens & Rubin (1996), 'Identification of Causal Effects Using Instrumental Variables.'; Montiel Olea & Pflueger (2013), 'A Robust Test for Weak Instruments.'",
            )
        else:
            return MethodRecommendation(
                primary_method="Synthetic Control Method (SCM) or Interactive Fixed Effects",
                source_of_variation="A weighted combination of untreated units (donors) constructs a counterfactual that matches the treated unit's pre-treatment trajectory.",
                why=[
                    "Unobserved time-varying confounders make standard panel methods invalid.",
                    "No valid instrument is available.",
                    "SCM constructs a data-driven counterfactual by weighting untreated units to match pre-treatment outcomes.",
                    "Interactive fixed effects (Bai 2009, Xu 2017) model unobserved confounders as a factor structure — more flexible than additive fixed effects.",
                ],
                assumptions=[
                    Assumption("Good pre-treatment fit", "The synthetic control closely tracks the treated unit before treatment.", True, "Pre-treatment RMSE; visual inspection of the pre-treatment fit."),
                    Assumption("No unobserved time-varying confounders in the post-treatment period", "After conditioning on the factor structure, no residual time-varying confounding.", False, "Cannot be directly tested. Supported by in-space placebo: applying SCM to untreated units should show null effects."),
                    Assumption("Donor pool adequacy", "The donor pool contains units similar enough to the treated unit to construct a meaningful counterfactual.", True, "Check pre-treatment RMSE; trim donors that are too dissimilar."),
                ],
                fallbacks=[
                    Fallback("Generalized SCM (Xu 2017)", "If multiple treated units exist", "Extends SCM to multiple treated units using interactive fixed effects."),
                    Fallback("Matrix Completion (Athey et al. 2021)", "If the panel has missing data or complex patterns", "Treats causal inference as a matrix completion problem with nuclear norm regularization."),
                    Fallback("Honest reporting: not identifiable", "If no method's assumptions are plausible", "Report that the causal effect cannot be credibly identified without stronger assumptions."),
                ],
                required_vars=["outcome", "entity_id", "time", "treatment_timing", "donor_pool"],
                optional_vars=["covariates", "multiple_outcome_measures"],
                literature_key="Abadie, Diamond & Hainmueller (2010), 'Synthetic Control Methods for Comparative Case Studies.'; Xu (2017), 'Generalized Synthetic Control Method.'; Athey et al. (2021), 'Matrix Completion Methods for Causal Panel Data Models.'",
            )

    # ── Mechanism: Continuous intensity ──────────────────────────────
    if mechanism == "continuous_intensity":
        return MethodRecommendation(
            primary_method="Intensity DID / Continuous-treatment DiD",
            source_of_variation="Cross-sectional variation in treatment intensity, differenced over time.",
            why=[
                "All units receive some treatment, but at varying intensities (doses).",
                "Intensity DID replaces the binary treatment indicator with a continuous dose variable.",
                "Key challenge: intensity is often endogenous — units that choose higher doses may differ systematically.",
            ],
            assumptions=[
                Assumption("Parallel trends in the dose-response relationship", "Units with different doses would have followed parallel trends absent treatment.", True, "Event study interacted with dose: pre-trend coefficients should be zero for all dose levels."),
                Assumption("Exogenous intensity variation", "The cross-sectional variation in intensity is not driven by unobserved confounders.", False, "Argue from policy design: why did some units get higher doses? If random/administrative, this is plausible."),
            ],
            fallbacks=[
                Fallback("IV with intensity", "If an instrument for intensity exists", "Use an exogenous shifter of intensity as an instrument — identifies LATE for units whose intensity is affected by the instrument."),
                Fallback("Dichotomize at median intensity", "If intensity IV is not available", "Split into high/low intensity groups and use staggered DID. Loses information but relaxes linearity."),
            ],
            required_vars=["outcome", "treatment_intensity (continuous)", "entity_id", "time"],
            optional_vars=["instrument_for_intensity", "covariates"],
            literature_key="Callaway, Goodman-Bacon & Sant'Anna (2024), 'Difference-in-Differences with a Continuous Treatment.'; de Chaisemartin & D'Haultfoeuille (2023), 'Two-way Fixed Effects and Differences-in-Differences with Continuous Treatments.'",
        )

    # ── Mechanism: Multiple overlapping policies ─────────────────────
    if mechanism == "multiple_overlapping_policies":
        return MethodRecommendation(
            primary_method="Triple Difference (DDD) or controlled DID",
            source_of_variation="Differential exposure across dimensions: a group exposed to policy A but not B vs. a group exposed to both.",
            why=[
                "Multiple policies affect the same outcome concurrently, making it impossible to isolate the effect of a single policy using standard DID.",
                "DDD adds a third difference: compare the DID estimate in a group exposed to policy A only vs. the DID estimate in a group exposed to both policies.",
                "If policies are separable by group, time, or region, DDD can net out the confounding policy.",
            ],
            assumptions=[
                Assumption("Parallel trends in the triple-differenced sense", "The difference-in-differences between groups follows parallel trends.", True, "Triple-difference event study; pre-trend coefficients jointly zero."),
                Assumption("No policy interaction effects", "The effects of policy A and policy B are additive (no synergy or offsetting effects).", False, "Argue from policy design: do the policies interact mechanically? If A changes eligibility for B, DDD fails."),
                Assumption("At least one group is exposed to only one policy", "There exists a group that receives policy A but not B (or vice versa).", True, "Check directly from policy coverage data."),
            ],
            fallbacks=[
                Fallback("Control function approach", "If policies are not separable by group", "Model the other policy explicitly; include it as a control variable. Assumes additive separability."),
                Fallback("Flag as fundamentally unidentified", "If policies are inseparable and affect all units equally", "Honest reporting: the effect of one policy cannot be separately identified. Report the combined effect and note the limitation."),
            ],
            required_vars=["outcome", "treatment_A", "treatment_B", "entity_id", "time", "group_dimension (for DDD)"],
            optional_vars=["covariates", "policy_intensity_measures"],
            literature_key="Gruber (1994), 'The Incidence of Mandated Maternity Benefits.'; Olden & Møen (2022), 'The Triple Difference Estimator.'",
        )

    # ── Fallback: unknown mechanism ──────────────────────────────────
    return MethodRecommendation(
        primary_method="Cannot determine — mechanism not recognized",
        source_of_variation="N/A",
        why=[f"Unknown mechanism: '{mechanism}'. Must be one of: {list(MECHANISM_DEFINITIONS.keys())}"],
        assumptions=[],
        fallbacks=[],
        required_vars=[],
        optional_vars=[],
    )


# ═══════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════

def format_method_report(
    policy: str,
    outcome: str,
    mechanism: str,
    rec: MethodRecommendation,
) -> str:
    """Format the method recommendation into a readable report."""
    mech_info = MECHANISM_DEFINITIONS.get(mechanism, {})

    lines = []
    lines.append("═══════════════════════════════════")
    lines.append("Theoretical Method Analysis (Stage 3)")
    lines.append("═══════════════════════════════════")
    lines.append("")
    lines.append(f"Policy:  {policy}")
    lines.append(f"Outcome: {outcome}")
    lines.append("")
    lines.append(f"Assignment mechanism: {mech_info.get('label', mechanism)}")
    lines.append("")
    lines.append(f"Why this mechanism:")
    lines.append(f"  {mech_info.get('description', 'N/A')}")
    lines.append("")
    lines.append(f"Primary theoretical recommendation: {rec.primary_method}")
    lines.append("")
    lines.append("Why:")
    for i, reason in enumerate(rec.why, 1):
        lines.append(f"  {i}. {reason}")
    lines.append("")
    lines.append(f"Source of variation: {rec.source_of_variation}")
    lines.append("")
    lines.append("Key identifying assumptions (theoretical — not yet checked against data):")
    for a in rec.assumptions:
        testable = "Testable" if a.testable else "Argument required"
        method = f" → {a.test_method}" if a.test_method else ""
        lines.append(f"  ├── {a.name}")
        lines.append(f"  │   {a.description}")
        lines.append(f"  │   [{testable}{method}]")
    lines.append("")
    lines.append("Fallback strategies (if primary assumptions fail):")
    for i, fb in enumerate(rec.fallbacks, 1):
        lines.append(f"  {i}. {fb.method}")
        lines.append(f"     When: {fb.condition}")
        lines.append(f"     Relaxes: {fb.assumption_relaxed}")
    lines.append("")
    lines.append("Required data (preview for Stage 4):")
    for v in rec.required_vars:
        lines.append(f"  ├── [Essential] {v}")
    for v in rec.optional_vars:
        lines.append(f"  └── [Optional]  {v}")
    lines.append("")
    lines.append(f"Key literature: {rec.literature_key}")
    lines.append("")
    lines.append("═══════════════════════════════════")

    return "\n".join(lines)


def to_dict(rec: MethodRecommendation) -> dict:
    """Serialize to a JSON-compatible dict."""
    return {
        "primary_method": rec.primary_method,
        "source_of_variation": rec.source_of_variation,
        "why": rec.why,
        "assumptions": [asdict(a) for a in rec.assumptions],
        "fallbacks": [asdict(f) for f in rec.fallbacks],
        "required_variables": rec.required_vars,
        "optional_variables": rec.optional_vars,
        "literature_key": rec.literature_key,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Theoretical method analysis (deterministic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Staggered DID with never-treated controls
  python stage3_analyze.py --mechanism staggered_policy_shock --staggered --has-control-group

  # Fuzzy RDD
  python stage3_analyze.py --mechanism threshold_rule --threshold-type fuzzy

  # IV with available instrument
  python stage3_analyze.py --mechanism time_varying_unobservables --has-instrument
""",
    )
    parser.add_argument("--mechanism", required=True,
                        choices=list(MECHANISM_DEFINITIONS.keys()),
                        help="Assignment mechanism type")
    parser.add_argument("--staggered", action="store_true", default=False,
                        help="Treatment is rolled out at different times")
    parser.add_argument("--has-control-group", action="store_true", default=True,
                        help="There exists a never-treated control group")
    parser.add_argument("--no-control-group", action="store_true", default=False,
                        help="No never-treated control group exists")
    parser.add_argument("--threshold-type", default=None,
                        choices=["sharp", "fuzzy"],
                        help="For threshold_rule mechanism: sharp or fuzzy")
    parser.add_argument("--panel-available", action="store_true", default=True,
                        help="Panel data structure is available")
    parser.add_argument("--no-panel", action="store_true", default=False,
                        help="Panel data is NOT available")
    parser.add_argument("--has-instrument", action="store_true", default=False,
                        help="A plausibly valid instrument exists")
    parser.add_argument("--multiple-policies", action="store_true", default=False,
                        help="Multiple policies affect the same outcome")
    parser.add_argument("--cross-section-only", action="store_true", default=False,
                        help="Treatment varies only cross-sectionally (no time dimension)")
    parser.add_argument("--everyone-treated-eventually", action="store_true", default=False,
                        help="All units are eventually treated")
    parser.add_argument("--policy", default="(unspecified)", help="Policy name for report header")
    parser.add_argument("--outcome", default="(unspecified)", help="Outcome name for report header")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--report", default=None, help="Output text report path")
    args = parser.parse_args()

    # Resolve flags
    has_control = args.has_control_group and not args.no_control_group
    panel_ok = args.panel_available and not args.no_panel

    rec = decide_method(
        mechanism=args.mechanism,
        staggered=args.staggered,
        has_control_group=has_control,
        threshold_type=args.threshold_type,
        panel_available=panel_ok,
        has_instrument=args.has_instrument,
        multiple_policies=args.multiple_policies,
        treatment_cross_section=args.cross_section_only,
        everyone_treated_eventually=args.everyone_treated_eventually,
    )

    # Print report
    report = format_method_report(args.policy, args.outcome, args.mechanism, rec)
    print(report)

    # Save JSON
    if args.output:
        output = {
            "mechanism": args.mechanism,
            "mechanism_label": MECHANISM_DEFINITIONS.get(args.mechanism, {}).get("label", ""),
            "inputs": {
                "staggered": args.staggered,
                "has_control_group": has_control,
                "threshold_type": args.threshold_type,
                "panel_available": panel_ok,
                "has_instrument": args.has_instrument,
                "multiple_policies": args.multiple_policies,
                "everyone_treated_eventually": args.everyone_treated_eventually,
            },
            "recommendation": to_dict(rec),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nJSON saved to {args.output}")

    # Save text report
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
