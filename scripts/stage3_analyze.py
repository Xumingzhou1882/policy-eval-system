"""
Stage 3: Theoretical method analysis — deterministic decision engine.

Takes structured inputs about the policy's assignment mechanism (filled in by
the LLM after Stage 2 policy research) and outputs a theoretically appropriate
identification strategy. Does NOT consider data availability.

The decision tree has TWO levels:
  Level 1 — "What family of methods?"   → routes to the correct branch function
  Level 2 — "Which estimator in that family?" → selects based on data structure

This two-level structure is expressed in code by splitting decide_method()
into routing + dedicated branch functions. Each branch function only receives
parameters relevant to that family.

Usage:
    python scripts/stage3_analyze.py --mechanism staggered_policy_shock \\
                                     --staggered true \\
                                     --has-control-group true \\
                                     --output stage3_result.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Data structures
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
    warnings: list[str] = None
    heterogeneity_note: str = ""

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ═══════════════════════════════════════════════════════════════════════
# Assignment mechanism definitions (mapping to branch functions)
# ═══════════════════════════════════════════════════════════════════════

MECHANISM_DEFINITIONS = {
    "random_assignment": {
        "label": "Random assignment (lottery, RCT, natural randomization)",
        "description": "Treatment is assigned by a random process independent of potential outcomes.",
        "family": "Experimental",
    },
    "threshold_rule": {
        "label": "Threshold-based eligibility",
        "description": "A continuous score determines eligibility with a sharp or fuzzy cutoff.",
        "family": "RDD",
    },
    "selection_on_observables": {
        "label": "Selection on observable characteristics",
        "description": "Treatment depends on measured characteristics; no time dimension or the relevant variation is cross-sectional only.",
        "family": "Matching / ML",
    },
    "single_policy_shock": {
        "label": "Single-time policy shock",
        "description": "A policy takes effect at one known time for all treated units simultaneously. Panel data required.",
        "family": "DID",
    },
    "staggered_policy_shock": {
        "label": "Staggered policy shock (different units treated at different times)",
        "description": "A policy is rolled out in batches; treatment timing varies across units. Panel data required.",
        "family": "DID",
    },
    "time_varying_unobservables": {
        "label": "Selection on time-varying unobservables",
        "description": "Confounders that change over time and differ across units drive treatment assignment.",
        "family": "IV / SCM",
    },
    "continuous_intensity": {
        "label": "Continuous treatment intensity",
        "description": "All units are treated, but at different doses/intensities.",
        "family": "Intensity",
    },
    "multiple_overlapping_policies": {
        "label": "Multiple overlapping policies",
        "description": "Several policies affect the same units concurrently.",
        "family": "DDD",
    },
}

# Cross-cutting: appended to every branch's recommendation
CROSS_CUTTING_HETEROGENEITY = (
    "Consider Causal Forest (run_causal_forest.py) to estimate treatment effect "
    "heterogeneity: which units benefit most? Who is unaffected? This is recommended "
    "as a Stage 7+ supplement regardless of the primary method chosen above."
)

CROSS_CUTTING_SENSITIVITY = (
    "After estimation, run sensitivity_analysis.py to check robustness: "
    "Oster bounds (unobservables), coefficient stability, and leave-one-out influence."
)


# ═══════════════════════════════════════════════════════════════════════
# Helper: shared assumptions across DID-family methods
# ═══════════════════════════════════════════════════════════════════════

def _assumption_parallel_trends(control_desc: str = "control") -> Assumption:
    return Assumption(
        "Parallel trends",
        f"In the absence of treatment, treated and {control_desc} units would have "
        f"followed the same outcome trajectory.",
        True,
        "Event study: pre-treatment coefficients jointly zero (F-test). "
        "Visual inspection of pre-treatment trends."
    )


def _assumption_no_anticipation() -> Assumption:
    return Assumption(
        "No anticipation",
        "Units did not change behavior before the policy took effect.",
        True,
        "Check if the t-1 coefficient in the event study is zero."
    )


def _assumption_sutva() -> Assumption:
    return Assumption(
        "No spillover (SUTVA)",
        "Treatment of one unit does not affect outcomes of other units.",
        False,
        "Argue from institutional context; spatial placebo tests provide partial evidence."
    )


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 1: Facts → Mechanism classification
# ═══════════════════════════════════════════════════════════════════════

def classify_mechanism(facts: dict) -> dict:
    """
    Level 1 of the decision tree: classify the assignment mechanism from
    observable policy facts (produced by Stage 2 research).

    This function does NOT select an estimator. It only determines:
      - which mechanism type best describes the policy
      - what Level 2 flags to pass to the branch function

    The classification follows a priority order. Early rules dominate
    later ones because they represent stronger sources of identifying variation.

    Parameters
    ----------
    facts : dict
        The Stage 2 structured facts. Expected keys match the template
        in SKILL.md: q1_assignment, q2_threshold, q3_timing, q4_control_group,
        q5_treatment_type, q6_concurrent_policies, q7_instrument.

    Returns
    -------
    dict with keys: mechanism, threshold_type, panel_available, has_control_group,
    staggered, everyone_treated_eventually, has_instrument, high_dimensional_controls,
    multiple_policies, reasoning (list of classification steps taken)
    """
    q1 = facts.get("q1_assignment", {})
    q2 = facts.get("q2_threshold", {})
    q3 = facts.get("q3_timing", {})
    q4 = facts.get("q4_control_group", {})
    q5 = facts.get("q5_treatment_type", {})
    q6 = facts.get("q6_concurrent_policies", {})
    q7 = facts.get("q7_instrument", {})

    reasoning = []

    # Default flags
    flags = {
        "threshold_type": None,
        "panel_available": True,
        "has_control_group": True,
        "staggered": False,
        "everyone_treated_eventually": False,
        "has_instrument": q7.get("has_plausible_instrument", False),
        "high_dimensional_controls": False,
        "multiple_policies": False,
        "unobservables_risk": False,
    }

    # ── Rule 1: Random assignment ─────────────────────────────────────
    how_assigned = q1.get("how_is_treatment_assigned", "")
    if how_assigned == "by_lottery":
        reasoning.append("Q1: Treatment assigned by lottery → 'random_assignment'.")
        flags["has_control_group"] = True
        return _finalize_classification("random_assignment", flags, reasoning, facts)

    # ── Rule 2: Threshold-based eligibility ───────────────────────────
    if q2.get("has_eligibility_threshold", False):
        reasoning.append(
            f"Q2: Eligibility determined by threshold on '{q2.get('threshold_variable', 'unknown')}' "
            f"→ 'threshold_rule'."
        )
        flags["threshold_type"] = "sharp" if q2.get("compliance_is_perfect", True) else "fuzzy"
        return _finalize_classification("threshold_rule", flags, reasoning, facts)

    # ── Rule 3: Plausible instrument → time-varying unobservables ─────
    # A credible instrument signals that the researcher is operating in a
    # setting with unobserved time-varying confounders. IV/SCM family
    # provides stronger identification than DID in this case.
    if q7.get("has_plausible_instrument", False):
        reasoning.append(
            f"Q7: Plausible instrument identified: '{q7.get('instrument_description', 'unspecified')}' "
            f"→ 'time_varying_unobservables'. IV/SCM family uses exogenous variation "
            f"in the instrument to identify the causal effect."
        )
        flags["has_instrument"] = True
        return _finalize_classification("time_varying_unobservables", flags, reasoning, facts)

    # ── Rule 4: No time dimension → selection on observables ──────────
    has_time = q3.get("has_known_start_time", False)
    if not has_time:
        reasoning.append(
            "Q3: No known treatment start time — no before/after comparison possible. "
            "Treatment variation is cross-sectional → 'selection_on_observables'."
        )
        # If many covariates mentioned, flag for DML
        n_controls = facts.get("data_environment", {}).get("control_variable_count", "moderate")
        if n_controls in ("many", "high", "very_high"):
            flags["high_dimensional_controls"] = True
            reasoning.append(f"  Control count = '{n_controls}' → DML preferred.")
        return _finalize_classification("selection_on_observables", flags, reasoning, facts)

    # ── Rule 5: Multiple overlapping policies ─────────────────────────
    if q6.get("has_overlapping_policies", False):
        reasoning.append(
            f"Q6: Multiple policies affect the same outcome concurrently: "
            f"{q6.get('overlapping_list', [])} → 'multiple_overlapping_policies'."
        )
        flags["multiple_policies"] = True
        return _finalize_classification("multiple_overlapping_policies", flags, reasoning, facts)

    # ── Rule 6: Continuous treatment intensity ────────────────────────
    if q5.get("has_intensity_variation", False):
        reasoning.append(
            f"Q5: Treatment has intensity variation: '{q5.get('intensity_detail', '')}' "
            f"→ 'continuous_intensity'."
        )
        return _finalize_classification("continuous_intensity", flags, reasoning, facts)

    # ── Rule 7: Policy shock with known timing (DID family) ───────────
    # At this point, has_time=True, binary treatment, not multiple policies,
    # not continuous intensity, no instrument.
    is_staggered = q3.get("treatment_starts_at_different_times", False)
    has_never_treated = q4.get("has_never_treated_units", False)
    all_treated = q4.get("all_units_eventually_treated", False)

    flags["has_control_group"] = has_never_treated
    flags["everyone_treated_eventually"] = all_treated

    # DID-family methods rely on parallel trends to control for unobservables.
    # If the researcher initially considered an instrument (q7 was explored but
    # none found), flag the risk that DID assumptions may not hold.
    if not q7.get("has_plausible_instrument", False):
        flags["unobservables_risk"] = True

    if is_staggered:
        reasoning.append(
            f"Q3: Treatment starts at different times (staggered rollout). "
            f"First: {q3.get('first_treatment_time')}, Last: {q3.get('last_treatment_time')}."
        )
        if has_never_treated:
            reasoning.append(
                "Q4: Never-treated units exist → C&S with never-treated control."
            )
        else:
            reasoning.append(
                "Q4: All units eventually treated → C&S with not-yet-treated control."
            )
        flags["staggered"] = True
        return _finalize_classification("staggered_policy_shock", flags, reasoning, facts)
    else:
        reasoning.append(
            f"Q3: Single treatment time: {q3.get('first_treatment_time')}."
        )
        if has_never_treated:
            reasoning.append("Q4: Never-treated control group exists → standard DID.")
        else:
            reasoning.append(
                "Q4: No control group → SCM (synthetic control as counterfactual)."
            )
        return _finalize_classification("single_policy_shock", flags, reasoning, facts)


# ═══════════════════════════════════════════════════════════════════════
# Helper: secondary feature detection (mixed mechanisms)
# ═══════════════════════════════════════════════════════════════════════

def _detect_secondary_features(facts: dict, primary_mechanism: str) -> list[dict]:
    """
    After primary classification, scan the remaining q2-q7 facts for features
    that exist but were NOT captured by the primary mechanism.

    These serve as alternative identification strategies if the primary method's
    assumptions fail in Stage 6. Each feature includes a deterministic suggestion
    for how to exploit it.
    """
    q2 = facts.get("q2_threshold", {})
    q3 = facts.get("q3_timing", {})
    q4 = facts.get("q4_control_group", {})
    q5 = facts.get("q5_treatment_type", {})
    q6 = facts.get("q6_concurrent_policies", {})
    q7 = facts.get("q7_instrument", {})

    secondary = []

    # ── Threshold (not captured by threshold_rule) ─────────────────────
    if primary_mechanism != "threshold_rule" and q2.get("has_eligibility_threshold"):
        tv = q2.get("threshold_variable", "unknown")
        cv = q2.get("cutoff_value", "unknown")
        is_sharp = q2.get("compliance_is_perfect", True)
        secondary.append({
            "feature": "eligibility_threshold",
            "detail": f"Eligibility determined by {tv} ≥ {cv}",
            "alternative_method": "Sharp RDD" if is_sharp else "Fuzzy RDD",
            "alternative_condition": (
                f"If a continuous measure of {tv} is available, the {cv} cutoff "
                f"provides local identification that does not require the primary "
                f"method's assumptions."
            ),
        })

    # ── Instrument (not captured by time_varying_unobservables) ────────
    if primary_mechanism != "time_varying_unobservables" and q7.get("has_plausible_instrument"):
        secondary.append({
            "feature": "plausible_instrument",
            "detail": q7.get("instrument_description", "unspecified"),
            "alternative_method": "IV / 2SLS",
            "alternative_condition": (
                "A plausible instrument was identified. IV provides identification "
                "from exogenous variation and does not rely on the primary method's "
                "assumptions — useful as a robustness check."
            ),
        })

    # ── Staggered rollout (not captured by staggered_policy_shock) ─────
    is_staggered = q3.get("treatment_starts_at_different_times", False)
    if primary_mechanism != "staggered_policy_shock" and is_staggered:
        secondary.append({
            "feature": "staggered_rollout",
            "detail": (
                f"Treatment rolled out in batches: "
                f"first={q3.get('first_treatment_time')}, "
                f"last={q3.get('last_treatment_time')}"
            ),
            "alternative_method": "Callaway & Sant'Anna (2021) staggered DID",
            "alternative_condition": (
                "Staggered rollout timing provides an alternative source of "
                "identifying variation. If the primary method's assumptions fail "
                "and panel data is available, C&S (2021) can estimate group-time ATTs."
            ),
        })

    # ── Single time shock (not captured by single_policy_shock) ────────
    has_time = q3.get("has_known_start_time", False)
    if primary_mechanism != "single_policy_shock" and has_time and not is_staggered:
        secondary.append({
            "feature": "single_time_shock",
            "detail": f"Treatment took effect at a single time: {q3.get('first_treatment_time')}",
            "alternative_method": "Standard TWFE DID",
            "alternative_condition": (
                "A clear before/after timing exists. Standard DID can difference out "
                "time-invariant confounders if the primary method's assumptions fail."
            ),
        })

    # ── Continuous intensity (not captured by continuous_intensity) ────
    if primary_mechanism != "continuous_intensity" and q5.get("has_intensity_variation"):
        secondary.append({
            "feature": "continuous_intensity",
            "detail": q5.get("intensity_detail", "unspecified"),
            "alternative_method": "Intensity DID / dose-response analysis",
            "alternative_condition": (
                "Treatment intensity varies across units. A dose-response design "
                "can test whether larger treatment doses produce larger effects, "
                "providing a different dimension of evidence."
            ),
        })

    # ── Overlapping policies (not captured by multiple_overlapping_policies) ──
    if primary_mechanism != "multiple_overlapping_policies" and q6.get("has_overlapping_policies"):
        overlapping = q6.get("overlapping_list", [])
        secondary.append({
            "feature": "overlapping_policies",
            "detail": str(overlapping),
            "alternative_method": "Triple Difference (DDD) / controlled DID",
            "alternative_condition": (
                f"Other policies ({overlapping}) affect the same outcome. If the "
                "other policy can be separated by group, region, or time, DDD can "
                "isolate the target policy's effect."
            ),
        })

    # ── Never-treated group (informational) ────────────────────────────
    if primary_mechanism not in ("single_policy_shock", "staggered_policy_shock") \
            and q4.get("has_never_treated_units"):
        secondary.append({
            "feature": "never_treated_control",
            "detail": "Some units were never treated",
            "alternative_method": "DID-family methods with never-treated control",
            "alternative_condition": (
                "Never-treated units provide a natural comparison group. "
                "If the primary method fails, consider a DID design using "
                "these units as the control group."
            ),
        })

    return secondary


def _secondary_features_to_fallbacks(secondary_features: list[dict]) -> list[Fallback]:
    """Convert detected secondary features into Fallback strategy entries."""
    result = []
    for sf in secondary_features:
        result.append(Fallback(
            method=sf["alternative_method"],
            condition=sf["alternative_condition"],
            assumption_relaxed=(
                f"Exploits {sf['feature']} instead of the primary method's "
                f"identifying assumption"
            ),
        ))
    return result


def _finalize_classification(
    mechanism: str, flags: dict, reasoning: list, facts: dict
) -> dict:
    """Attach secondary features and return the classification dict."""
    flags["secondary_features"] = _detect_secondary_features(facts, mechanism)
    return {**flags, "mechanism": mechanism, "reasoning": reasoning}


def decide_from_facts(facts: dict) -> MethodRecommendation:
    """
    Full two-level decision tree: Level 1 (facts → mechanism) + Level 2 (mechanism → method).

    This is the primary entry point when Stage 2 has produced a structured facts file.
    """
    # Level 1
    classification = classify_mechanism(facts)
    mechanism = classification.pop("mechanism")
    reasoning = classification.pop("reasoning", [])

    # Level 2
    rec = decide_method(mechanism=mechanism, **classification)

    # Attach the Level 1 reasoning to the recommendation
    rec.literature_key = (
        f"[Classification: {' | '.join(reasoning)}] "
        + rec.literature_key
    )

    return rec


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 2: Router — mechanism → branch function
# ═══════════════════════════════════════════════════════════════════════

def decide_method(
    mechanism: str,
    # ── Flags for Level 2 branching ──
    threshold_type: Optional[str] = None,     # RDD family
    panel_available: bool = True,              # DID / IV / SCM families
    has_control_group: bool = True,            # DID family
    staggered: bool = False,                   # DID family
    everyone_treated_eventually: bool = False, # DID family (staggered)
    has_instrument: bool = False,              # Unobservables / cross-family
    high_dimensional_controls: bool = False,   # Matching/ML family
    multiple_policies: bool = False,           # DDD family
    unobservables_risk: bool = False,          # DID family: no instrument found
    secondary_features: list[dict] = None,     # Mixed mechanism: other features
) -> MethodRecommendation:
    """
    Level 2 of the decision tree: given an assignment mechanism and data
    structure flags, select the specific identification strategy.

    This function routes to the appropriate branch function based on mechanism.
    Each branch function selects the specific estimator within that family.
    """
    # ── Level 1: Route ──────────────────────────────────────────────────

    sf = secondary_features or []

    if mechanism == "random_assignment":
        return _branch_random_assignment(secondary_features=sf)

    if mechanism == "threshold_rule":
        return _branch_threshold(threshold_type, secondary_features=sf)

    if mechanism == "selection_on_observables":
        return _branch_selection_on_observables(
            high_dimensional_controls=high_dimensional_controls,
            panel_available=panel_available,
            has_instrument=has_instrument,
            secondary_features=sf,
        )

    if mechanism == "single_policy_shock":
        return _branch_single_policy_shock(
            panel_available=panel_available,
            has_control_group=has_control_group,
            has_instrument=has_instrument,
            unobservables_risk=unobservables_risk,
            secondary_features=sf,
        )

    if mechanism == "staggered_policy_shock":
        return _branch_staggered_policy_shock(
            panel_available=panel_available,
            has_control_group=has_control_group,
            everyone_treated_eventually=everyone_treated_eventually,
            has_instrument=has_instrument,
            unobservables_risk=unobservables_risk,
            secondary_features=sf,
        )

    if mechanism == "time_varying_unobservables":
        return _branch_time_varying_unobservables(
            has_instrument=has_instrument,
            panel_available=panel_available,
            secondary_features=sf,
        )

    if mechanism == "continuous_intensity":
        return _branch_continuous_intensity(
            panel_available=panel_available,
            secondary_features=sf,
        )

    if mechanism == "multiple_overlapping_policies":
        return _branch_multiple_policies(secondary_features=sf)

    # ── Unknown mechanism ───────────────────────────────────────────────
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
# LEVEL 2: Branch functions — select the specific estimator
# ═══════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# Branch A: Random assignment
# ───────────────────────────────────────────────────────────────────────

def _branch_random_assignment(
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the Experimental family:
      - Only one estimator (randomization inference), but different diagnostics
        depending on perfect vs. imperfect compliance.
    """
    fallbacks = [
        Fallback("IV (assignment as instrument)", "If compliance is imperfect",
                 "Relaxes perfect compliance; identifies LATE for compliers."),
    ]
    if secondary_features:
        fallbacks = fallbacks + _secondary_features_to_fallbacks(secondary_features)

    rec = MethodRecommendation(
        primary_method="Randomization inference / Fisher exact test",
        source_of_variation="Random assignment breaks the statistical link between treatment and all confounders (observed and unobserved).",
        why=[
            "Treatment is randomly assigned — the gold standard for causal identification.",
            "No need for control variables; simple difference-in-means is unbiased.",
            "Randomization inference provides exact p-values without distributional assumptions.",
        ],
        assumptions=[
            Assumption("SUTVA", "No interference between units.", False,
                       "Cannot be formally tested — requires institutional argument."),
            Assumption("No selective attrition", "Units do not drop out in a way correlated with treatment.", True,
                       "Compare attrition rates between treatment and control."),
            Assumption("Compliance", "Units assigned to treatment actually receive it.", True,
                       "Check actual treatment receipt rates by assignment group."),
        ],
        fallbacks=fallbacks,
        required_vars=["treatment_assignment", "outcome"],
        optional_vars=["baseline_covariates (balance check)", "compliance_indicator"],
        literature_key="Imbens & Rubin (2015), Causal Inference for Statistics, Social, and Biomedical Sciences.",
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch B: Threshold (RDD family)
# ───────────────────────────────────────────────────────────────────────

def _branch_threshold(
    threshold_type: Optional[str],
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the RDD family:
      sharp → Sharp RDD (local linear regression at cutoff)
      fuzzy → Fuzzy RDD (IV at cutoff, for imperfect compliance)
    """
    if threshold_type not in ("sharp", "fuzzy"):
        return MethodRecommendation(
            primary_method="RDD — threshold type not specified",
            source_of_variation="Discontinuity in treatment probability at a known cutoff.",
            why=["A threshold rule determines eligibility."],
            assumptions=[],
            fallbacks=(_secondary_features_to_fallbacks(secondary_features)
                       if secondary_features else []),
            required_vars=["running_variable", "cutoff_value", "outcome"],
            optional_vars=[],
            warnings=["threshold_type must be 'sharp' or 'fuzzy' — re-run with --threshold-type"],
        )

    is_sharp = threshold_type == "sharp"
    method_name = "Sharp RDD" if is_sharp else "Fuzzy RDD (IV at cutoff)"

    assumptions = [
        Assumption("No manipulation of running variable",
                   "Units cannot precisely control their position relative to the cutoff.",
                   True, "McCrary (2008) density test."),
        Assumption("Continuity of potential outcomes",
                   "Expected potential outcomes are smooth across the cutoff.",
                   False, "Cannot be directly tested; supported by covariate balance at cutoff."),
        Assumption("Local linearity",
                   "Relationship between running variable and outcome is approximately linear near the cutoff.",
                   True, "Compare linear vs. quadratic; CCT optimal bandwidth sensitivity."),
    ]

    if not is_sharp:
        assumptions.append(
            Assumption("First-stage strength (fuzzy RDD)",
                       "The jump in treatment probability at the cutoff is statistically significant.",
                       True, "F-statistic at cutoff; F < 10 indicates weak instrument.")
        )

    fallbacks = [
        Fallback("Donut-hole RDD", "If McCrary test detects manipulation near cutoff",
                 "Drops observations in a window around the cutoff."),
        Fallback("DID", "If both RDD assumptions fail and pre-treatment data exists",
                 "Uses time variation instead of threshold variation."),
    ]
    if secondary_features:
        fallbacks = fallbacks + _secondary_features_to_fallbacks(secondary_features)

    rec = MethodRecommendation(
        primary_method=method_name,
        source_of_variation=(
            "Units just above and just below the threshold are as-good-as-randomly assigned. "
            "Comparing them identifies the local average treatment effect at the cutoff."
        ),
        why=[
            f"A known continuous threshold ({'sharp' if is_sharp else 'fuzzy'}) determines treatment eligibility.",
            "In a narrow window around the cutoff, treatment is approximately random.",
            f"{'Treatment jumps from 0 to 1 at cutoff → Sharp RDD.' if is_sharp else 'Treatment jumps by less than 1 (imperfect compliance) → Fuzzy RDD (IV).'}",
        ],
        assumptions=assumptions,
        fallbacks=fallbacks,
        required_vars=["running_variable", "outcome", "cutoff_value"],
        optional_vars=["baseline_covariates", "treatment_indicator (fuzzy RDD)"],
        literature_key="Calonico, Cattaneo & Titiunik (2014); Lee & Lemieux (2010).",
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch C: Selection on observables (Matching / ML family)
# ───────────────────────────────────────────────────────────────────────

def _branch_selection_on_observables(
    high_dimensional_controls: bool = False,
    panel_available: bool = True,
    has_instrument: bool = False,
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the Matching/ML family:
      few controls      → PSM / IPW / AIPW (traditional, interpretable)
      many controls     → DML (flexible ML, valid inference)
      panel available   → DID is a superior fallback (relaxes CIA)
    """
    if high_dimensional_controls:
        rec = MethodRecommendation(
            primary_method="Double/Debiased Machine Learning (DML)",
            source_of_variation="Conditional on high-dimensional covariates (flexibly modeled via ML), treatment assignment is as-good-as-random.",
            why=[
                "The number of confounders is large relative to sample size.",
                "Traditional PSM/IPW may perform poorly due to curse of dimensionality.",
                "DML uses ML (RandomForest, GradientBoosting, Lasso) to flexibly model nuisance functions.",
                "Neyman-orthogonal scores + cross-fitting debias ML estimates for valid inference.",
            ],
            assumptions=[
                Assumption("Unconfoundedness (CIA)", "All confounders are measured.", False,
                           "Cannot be tested. Use Oster (2019) bounds: sensitivity_analysis.py."),
                Assumption("Overlap / positivity", "Every unit has positive probability of either treatment status.", True,
                           "Check propensity score histograms; DML handles limited overlap better than PSM."),
                Assumption("ML model quality", "Nuisance models converge at n^(-1/4) rate or faster.", True,
                           "CV R² of nuisance models from run_dml.py; both should exceed 0.1."),
            ],
            fallbacks=[
                Fallback("Causal Forest", "If heterogeneity is the primary interest",
                         "Estimates CATE directly; see run_causal_forest.py."),
                Fallback("AIPW", "If DML fails (poor overlap or small sample)",
                         "Doubly-robust with parametric models — more stable in small samples."),
                Fallback("IV / 2SLS", "If a plausible instrument is available and CIA is suspect",
                         "Uses exogenous variation in instrument to identify LATE. See run_iv.py."),
                Fallback("DID", "If panel data becomes available",
                         "Controls for time-invariant unobserved confounders — relaxes CIA."),
            ],
            required_vars=["outcome", "treatment_indicator", "all_confounders"],
            optional_vars=["instrument (for IV as supplementary check)"],
            literature_key="Chernozhukov et al. (2018), 'Double/Debiased Machine Learning.'; "
                           "Chernozhukov et al. (2024), 'Applied Causal Inference Powered by ML and AI.'",
            heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
        )
        if secondary_features:
            rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
        if has_instrument:
            rec.why.append(
                "A plausible instrument was identified in Stage 2. IV/2SLS "
                "can serve as a supplementary identification strategy that does "
                "not rely on the CIA — useful if unmeasured confounding is a concern."
            )
        return rec

    # Standard (few controls)
    warnings_list = []
    if panel_available:
        warnings_list.append(
            "Panel data is available — consider whether a DID design (single_policy_shock or "
            "staggered_policy_shock) could relax the unconfoundedness assumption by differencing "
            "out time-invariant unobservables. Selection-on-observables methods cannot handle "
            "unmeasured confounders."
        )
    if has_instrument:
        warnings_list.append(
            "A plausible instrument was identified in Stage 2. Consider IV/2SLS "
            "(run_iv.py) as a supplementary check — it does not rely on the CIA "
            "and provides a different source of identifying variation."
        )

    rec = MethodRecommendation(
        primary_method="PSM / IPW / Doubly-Robust (AIPW) — with DML as modern alternative",
        source_of_variation="Conditional on observed covariates, treatment assignment is as-good-as-random.",
        why=[
            "Treatment depends on measured characteristics.",
            "Conditioning on these characteristics (matching, weighting, or regression) makes units comparable.",
            "DML with Gradient Boosting provides more flexible confounding adjustment than parametric PSM.",
            "For moderate numbers of controls (<20), traditional methods are well-understood and interpretable.",
        ],
        assumptions=[
            Assumption("Unconfoundedness (CIA)", "All confounders are measured and included.", False,
                       "Cannot be tested. Use Oster (2019) delta: sensitivity_analysis.py."),
            Assumption("Common support / overlap", "For each treated unit, a comparable untreated unit exists.", True,
                       "Check propensity score distributions; trim off-support observations."),
            _assumption_sutva(),
        ],
        fallbacks=[
            Fallback("DML (Gradient Boosting)", "If PSM balance is poor or many controls exist",
                     "Flexible ML nuisance estimation with valid inference. Use run_dml.py."),
            Fallback("Causal Forest", "If heterogeneity is of interest",
                     "Estimates CATE; see run_causal_forest.py."),
            Fallback("CEM", "If PSM balance is poor and interpretability is key",
                     "Non-parametric; balances on coarsened covariates."),
            Fallback("IV / 2SLS", "If a plausible instrument exists and CIA is suspect",
                     "Uses exogenous variation in instrument — does not require CIA. See run_iv.py."),
            Fallback("DID", "If panel data is available",
                     "Controls for time-invariant unobservables — relaxes CIA."),
        ],
        required_vars=["outcome", "treatment_indicator", "covariates (all variables affecting selection)"],
        optional_vars=["instrument (for IV / sensitivity analysis)"],
        literature_key="Rosenbaum & Rubin (1983); Chernozhukov et al. (2018); Imbens (2004).",
        warnings=warnings_list if warnings_list else None,
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    if secondary_features:
        rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch D: Single policy shock (DID family, standard)
# ───────────────────────────────────────────────────────────────────────

def _branch_single_policy_shock(
    panel_available: bool = True,
    has_control_group: bool = True,
    has_instrument: bool = False,
    unobservables_risk: bool = False,
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the DID family (single time):
      has control group → Standard TWFE DID
      no control group  → SCM (construct synthetic counterfactual)
    """
    warnings_list = []
    if not panel_available:
        warnings_list.append(
            "DID requires panel data (same units observed over time). "
            "If only repeated cross-sections are available, consider repeated cross-section DID "
            "or synthetic cohort methods."
        )

    if has_control_group:
        if unobservables_risk:
            warnings_list.append(
                "No instrument was identified in Stage 2. DID relies entirely on parallel trends "
                "to control for unobserved confounders. If pre-treatment trends diverge in Stage 6, "
                "the fallback options (PSM-DID, SCM) also cannot handle time-varying unobservables. "
                "Consider whether additional data could support an IV strategy."
            )
        if has_instrument:
            warnings_list.append(
                "A plausible instrument was identified in Stage 2 but was not the primary "
                "classification path. Consider IV/2SLS (run_iv.py) as a supplementary check — "
                "it identifies the causal effect from a different source of variation than DID."
            )

        rec = MethodRecommendation(
            primary_method="Standard Difference-in-Differences (TWFE)",
            source_of_variation="Within-unit change over time, differenced between treated and untreated units.",
            why=[
                "A single policy event at a known time affects some units (treated) but not others (control).",
                "DID removes time-invariant unit-specific confounders and period-specific shocks.",
                "TWFE is unbiased when treatment effects are homogeneous and timing is uniform.",
            ],
            assumptions=[
                _assumption_parallel_trends("control"),
                _assumption_no_anticipation(),
                _assumption_sutva(),
                Assumption("Stable unit composition", "Treated/control group composition does not change over time.",
                           True, "Check for unit entry/exit; use balanced panel or document attrition."),
            ],
            fallbacks=[
                Fallback("PSM-DID", "If treated and control differ in observables",
                         "Re-weights control units to match treated on pre-treatment characteristics."),
                Fallback("SCM", "If few treated units and no clear control group",
                         "Constructs a synthetic counterfactual from donor pool."),
                Fallback("IV / 2SLS", "If a valid instrument is available and parallel trends fail",
                         "Uses exogenous variation — does not require parallel trends. See run_iv.py."),
            ],
            required_vars=["outcome", "treatment_indicator", "post_period_indicator", "entity_id", "time"],
            optional_vars=["control_variables", "pre_treatment_outcome_trends"],
            literature_key="Angrist & Pischke (2009), Ch. 5; Bertrand, Duflo & Mullainathan (2004).",
            warnings=warnings_list if warnings_list else None,
            heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
        )
        if secondary_features:
            rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
        return rec

    # No control group
    warnings_list.append(
        "No untreated control group — standard DID is not applicable. "
        "Synthetic DID constructs a weighted counterfactual from donor units "
        "and applies DID differencing, providing valid standard errors."
    )
    if has_instrument:
        warnings_list.append(
            "A plausible instrument was identified in Stage 2. IV/2SLS (run_iv.py) "
            "may be a viable alternative — it does not require a control group, "
            "only a valid instrument. Consider as a supplementary strategy."
        )
    rec = MethodRecommendation(
        primary_method="Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)",
        source_of_variation="A weighted combination of untreated donor units creates a synthetic counterfactual; DID differencing removes time-invariant confounders and provides valid inference.",
        why=[
            "A single policy event but no untreated control group.",
            "Synthetic DID combines SCM-style unit weighting with DID-style time differencing.",
            "Unlike traditional SCM, Synthetic DID provides valid standard errors and p-values via placebo-based inference.",
            "Pre-treatment fit quality is directly verifiable via RMSE.",
            "The method is designed for the 'one treated unit, many controls' setting.",
        ],
        assumptions=[
            Assumption("Good pre-treatment fit", "Synthetic control closely tracks the treated unit before treatment.",
                       True, "Pre-treatment RMSE; visual inspection of fitted trajectory."),
            Assumption("No unobserved time-varying confounders post-treatment",
                       "After weighting and differencing, no residual confounding.",
                       False, "In-space placebo: apply Synthetic DID to each donor unit → null distribution."),
            Assumption("Donor pool adequacy", "Donor units are similar enough to construct meaningful weights.",
                       True, "Pre-treatment RMSE; trim dissimilar donors; check weight distribution."),
        ],
        fallbacks=[
            Fallback("Interactive Fixed Effects (Xu 2017)", "If pre-treatment fit is poor or multiple treated units exist",
                     "Models unobserved confounders via factor structure."),
            Fallback("Traditional SCM (Abadie et al. 2010)", "If placebo-based inference is unstable",
                     "Classic SCM with permutation-based inference. See run_scm.py."),
            Fallback("IV / 2SLS", "If a valid instrument is available",
                     "Does not require a donor pool — uses exogenous variation. See run_iv.py."),
        ],
        required_vars=["outcome", "entity_id", "time", "first_treated", "donor_pool"],
        optional_vars=["covariates", "multiple_outcome_measures"],
        literature_key="Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), 'Synthetic Difference-in-Differences.' AER.",
        warnings=warnings_list,
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    if secondary_features:
        rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch D': Staggered policy shock (DID family, staggered)
# ───────────────────────────────────────────────────────────────────────

def _branch_staggered_policy_shock(
    panel_available: bool = True,
    has_control_group: bool = True,
    everyone_treated_eventually: bool = False,
    has_instrument: bool = False,
    unobservables_risk: bool = False,
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the DID family (staggered):
      has never-treated       → C&S (2021) with never-treated control
      all eventually treated  → C&S (2021) with not-yet-treated control
    """
    warnings_list = []
    if not panel_available:
        warnings_list.append(
            "Staggered DID requires panel data. Without panel data, "
            "staggered timing cannot be exploited for identification."
        )

    if unobservables_risk:
        warnings_list.append(
            "No instrument was identified in Stage 2. Staggered DID relies entirely "
            "on parallel trends to control for unobserved confounders. If pre-treatment "
            "trends diverge in Stage 6, the fallback options also require parallel trends "
            "or factor structure — neither provides the clean identification of an instrument. "
            "Consider whether additional data could support an IV strategy."
        )
    if has_instrument:
        warnings_list.append(
            "A plausible instrument was identified in Stage 2 but was not the primary "
            "classification path. Consider IV/2SLS (run_iv.py) as a robustness check — "
            "it identifies the causal effect from a different source of variation than "
            "staggered DID, providing a useful complement."
        )

    if not has_control_group and everyone_treated_eventually:
        rec = MethodRecommendation(
            primary_method="Callaway & Sant'Anna (2021) — not-yet-treated as control",
            source_of_variation="Differential treatment timing: earlier-treated serve as control for later-treated (before they are treated).",
            why=[
                "Policy rollout is staggered: different units treated at different times.",
                "All units are eventually treated — no never-treated group exists.",
                "C&S (2021) uses not-yet-treated units as control for each cohort.",
                "Traditional TWFE produces negative weights with heterogeneous effects (Goodman-Bacon 2021).",
                "Important: only identifies effects over the window before the control group gets treated.",
            ],
            assumptions=[
                _assumption_parallel_trends("not-yet-treated"),
                _assumption_no_anticipation(),
                Assumption("Limited heterogeneity across cohorts",
                           "Cohort-specific ATTs should be similar for meaningful aggregation.",
                           True, "Report and compare cohort-specific ATTs."),
            ],
            fallbacks=[
                Fallback("Sun & Abraham (2021)", "If anticipation effects are suspected",
                         "Allows specification of anticipation periods."),
                Fallback("Borusyak, Jaravel & Spiess (2024)", "If factor structure is more appropriate",
                         "Imputation-based estimator."),
                Fallback("de Chaisemartin & D'Haultfoeuille (2020)", "If interested in instantaneous effects",
                         "Focuses on switchers vs. stable units."),
                Fallback("IV / 2SLS", "If a valid instrument is available and staggered DID assumptions fail",
                         "Uses exogenous variation — does not require parallel trends. See run_iv.py."),
            ],
            required_vars=["outcome", "entity_id", "time", "first_treatment_year"],
            optional_vars=["covariates", "treatment_indicator"],
            literature_key="Callaway & Sant'Anna (2021); Goodman-Bacon (2021); Sun & Abraham (2021).",
            warnings=warnings_list if warnings_list else None,
            heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
        )
        if secondary_features:
            rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
        return rec

    # Has never-treated control group
    rec = MethodRecommendation(
        primary_method="Callaway & Sant'Anna (2021) — never-treated as control",
        source_of_variation="Differential treatment timing across cohorts, compared against never-treated units.",
        why=[
            "Policy rollout is staggered: different units treated at different times.",
            "Traditional TWFE in staggered designs can assign negative weights with heterogeneous effects.",
            "C&S (2021) estimates group-time ATTs using never-treated units as control.",
            "Cohort-specific ATTs are aggregated to an overall ATT with clear weighting.",
        ],
        assumptions=[
            _assumption_parallel_trends("never-treated"),
            _assumption_no_anticipation(),
            Assumption("Never-treated are valid counterfactual",
                       "Never-treated units are comparable to treated units (conditional on covariates).",
                       False, "Argue from institutional knowledge: why were some units never treated?"),
        ],
        fallbacks=[
            Fallback("Sun & Abraham (2021)", "If anticipation effects are suspected",
                     "Allows anticipation-period specification."),
            Fallback("Borusyak, Jaravel & Spiess (2024)", "If factor structure is more appropriate",
                     "Imputation-based estimator."),
            Fallback("SCM", "If only a few units are treated",
                     "Better suited for small-N treated groups."),
            Fallback("IV / 2SLS", "If a valid instrument is available and staggered DID assumptions fail",
                     "Uses exogenous variation — does not require parallel trends. See run_iv.py."),
        ],
        required_vars=["outcome", "entity_id", "time", "first_treatment_year", "never_treated_indicator"],
        optional_vars=["covariates"],
        literature_key="Callaway & Sant'Anna (2021); Sun & Abraham (2021); Goodman-Bacon (2021).",
        warnings=warnings_list if warnings_list else None,
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    if secondary_features:
        rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch E: Time-varying unobservables (IV / SCM family)
# ───────────────────────────────────────────────────────────────────────

def _branch_time_varying_unobservables(
    has_instrument: bool = False,
    panel_available: bool = True,
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the IV/SCM family:
      has instrument  → IV / 2SLS (LATE for compliers)
      no instrument   → SCM / Interactive FE
    """
    if has_instrument:
        rec = MethodRecommendation(
            primary_method="Instrumental Variables (2SLS / LIML)",
            source_of_variation="Exogenous variation in the instrument induces variation in treatment that is independent of unobserved confounders.",
            why=[
                "Unobserved time-varying confounders make DID and matching invalid.",
                "A valid instrument provides exogenous variation in treatment.",
                "2SLS identifies LATE: average treatment effect for compliers.",
            ],
            assumptions=[
                Assumption("Relevance", "The instrument significantly predicts treatment.", True,
                           "First-stage F (Montiel Olea & Pflueger 2013 effective F). Below critical → weak."),
                Assumption("Exclusion restriction", "Instrument affects outcome ONLY through treatment.", False,
                           "Cannot be formally tested. Partial: overidentification Hansen J test."),
                Assumption("Monotonicity", "Instrument affects treatment in same direction for all units.", False,
                           "Plausible if instrument is a policy eligibility rule."),
            ],
            fallbacks=[
                Fallback("LIML", "If instruments are weak (effective F < critical)",
                         "More robust to weak instruments than 2SLS."),
                Fallback("SCM", "If no valid instrument exists",
                         "Constructs synthetic counterfactual without instruments."),
            ],
            required_vars=["outcome", "treatment", "instrument(s)"],
            optional_vars=["covariates", "multiple_instruments (overid test)"],
            literature_key="Angrist, Imbens & Rubin (1996); Montiel Olea & Pflueger (2013).",
            heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
        )
        if secondary_features:
            rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
        return rec

    # No instrument
    warnings_list = []
    if not panel_available:
        warnings_list.append(
            "SCM requires panel data. Without panel or instrument, "
            "the causal effect may not be credibly identifiable."
        )

    rec = MethodRecommendation(
        primary_method="Synthetic Control Method (SCM) or Interactive Fixed Effects",
        source_of_variation="Weighted combination of untreated donors constructs a counterfactual matching the treated unit's pre-treatment trajectory.",
        why=[
            "Unobserved time-varying confounders make standard panel methods invalid.",
            "No valid instrument is available.",
            "SCM constructs data-driven counterfactual by weighting untreated units.",
            "Interactive fixed effects (Xu 2017) models confounders as factor structure.",
        ],
        assumptions=[
            Assumption("Good pre-treatment fit", "Synthetic control closely tracks treated unit before treatment.",
                       True, "Pre-treatment RMSE; visual inspection."),
            Assumption("No unobserved time-varying confounders post-treatment", "Factor structure captures confounding.",
                       False, "In-space placebo: apply SCM to untreated units."),
            Assumption("Donor pool adequacy", "Donors are similar enough to construct meaningful counterfactual.",
                       True, "Pre-treatment RMSE; trim dissimilar donors."),
        ],
        fallbacks=[
            Fallback("Generalized SCM (Xu 2017)", "If multiple treated units exist",
                     "Extends SCM to multiple treated units."),
            Fallback("Matrix Completion (Athey et al. 2021)", "If panel has missing data or complex patterns",
                     "Nuclear norm regularization for counterfactual imputation."),
            Fallback("Honest reporting: not identifiable", "If no method's assumptions are plausible",
                     "Report that the causal effect cannot be credibly identified."),
        ],
        required_vars=["outcome", "entity_id", "time", "treatment_timing", "donor_pool"],
        optional_vars=["covariates", "multiple_outcome_measures"],
        literature_key="Abadie, Diamond & Hainmueller (2010); Xu (2017); Athey et al. (2021).",
        warnings=warnings_list if warnings_list else None,
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    if secondary_features:
        rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch F: Continuous treatment intensity
# ───────────────────────────────────────────────────────────────────────

def _branch_continuous_intensity(
    panel_available: bool = True,
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the Intensity family:
      panel available → Intensity DID
      cross-section   → IV with exogenous intensity shifter needed
    """
    warnings_list = []
    if not panel_available:
        warnings_list.append(
            "Intensity DID requires panel data to difference out time-invariant confounders. "
            "Cross-sectional intensity variation is almost always endogenous — "
            "an exogenous instrument for intensity is needed."
        )

    rec = MethodRecommendation(
        primary_method="Intensity DID / Continuous-treatment DiD",
        source_of_variation="Cross-sectional variation in treatment intensity, differenced over time.",
        why=[
            "All units receive some treatment, but at varying doses.",
            "Replaces binary treatment indicator with continuous dose variable.",
            "Critical challenge: intensity is often endogenous — high-dose units may differ systematically.",
        ],
        assumptions=[
            Assumption("Parallel trends in dose-response", "Units with different doses would follow parallel trends absent treatment.",
                       True, "Event study interacted with dose; pre-trend coefficients should be zero."),
            Assumption("Exogenous intensity variation", "Cross-sectional intensity variation is not driven by unobservables.",
                       False, "Argue from policy design: why did some units get higher doses?"),
        ],
        fallbacks=[
            Fallback("IV with intensity", "If an instrument for intensity exists",
                     "Identifies LATE for units whose intensity is shifted by the instrument."),
            Fallback("Dichotomize at median", "If no intensity IV",
                     "Split into high/low groups and use staggered DID. Loses information."),
        ],
        required_vars=["outcome", "treatment_intensity (continuous)", "entity_id", "time"],
        optional_vars=["instrument_for_intensity", "covariates"],
        literature_key="Callaway, Goodman-Bacon & Sant'Anna (2024); de Chaisemartin & D'Haultfoeuille (2023).",
        warnings=warnings_list if warnings_list else None,
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    if secondary_features:
        rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
    return rec


# ───────────────────────────────────────────────────────────────────────
# Branch G: Multiple overlapping policies
# ───────────────────────────────────────────────────────────────────────

def _branch_multiple_policies(
    secondary_features: list[dict] = None,
) -> MethodRecommendation:
    """
    Level 2 within the DDD family:
      separable by group/region/time → DDD
      not separable                   → Controlled DID or flag as unidentified
    """
    rec = MethodRecommendation(
        primary_method="Triple Difference (DDD) or controlled DID",
        source_of_variation="Differential exposure across dimensions: a group exposed to policy A but not B vs. a group exposed to both.",
        why=[
            "Multiple policies affect the same outcome concurrently.",
            "DDD adds a third difference: compare DID estimates across groups differentially exposed to each policy.",
            "If policies are separable by group, time, or region, DDD can isolate the effect of one policy.",
        ],
        assumptions=[
            Assumption("Parallel trends in triple-differenced sense",
                       "DID estimates between groups follow parallel trends.",
                       True, "Triple-difference event study; pre-trend coefficients jointly zero."),
            Assumption("No policy interaction effects",
                       "Effects of policy A and B are additive (no synergy or offsetting).",
                       False, "Argue from policy design: do policies interact mechanically?"),
            Assumption("At least one group exposed to only one policy",
                       "Exists a group that receives A but not B (or vice versa).",
                       True, "Check directly from policy coverage data."),
        ],
        fallbacks=[
            Fallback("Control function approach", "If policies are not separable by group",
                     "Model other policy explicitly as control variable."),
            Fallback("Flag as fundamentally unidentified", "If policies are inseparable",
                     "Report combined effect; note that single-policy effect cannot be isolated."),
        ],
        required_vars=["outcome", "treatment_A", "treatment_B", "entity_id", "time", "group_dimension"],
        optional_vars=["covariates", "policy_intensity_measures"],
        literature_key="Gruber (1994); Olden & Møen (2022).",
        heterogeneity_note=CROSS_CUTTING_HETEROGENEITY,
    )
    if secondary_features:
        rec.fallbacks = rec.fallbacks + _secondary_features_to_fallbacks(secondary_features)
    return rec


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
    family = mech_info.get("family", "Unknown")

    lines = []
    lines.append("═══════════════════════════════════")
    lines.append("Theoretical Method Analysis (Stage 3)")
    lines.append("═══════════════════════════════════")
    lines.append("")
    lines.append(f"Policy:  {policy}")
    lines.append(f"Outcome: {outcome}")
    lines.append("")
    lines.append(f"Method family:       {family}")
    lines.append(f"Assignment mechanism: {mech_info.get('label', mechanism)}")
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
    lines.append("Key identifying assumptions:")
    for a in rec.assumptions:
        testable = "Testable" if a.testable else "Argument required"
        method = f" → {a.test_method}" if a.test_method else ""
        lines.append(f"  ├── {a.name}")
        lines.append(f"  │   {a.description}")
        lines.append(f"  │   [{testable}{method}]")
    lines.append("")
    lines.append("Fallback strategies:")
    for i, fb in enumerate(rec.fallbacks, 1):
        lines.append(f"  {i}. {fb.method}")
        lines.append(f"     When: {fb.condition}")
        lines.append(f"     Relaxes: {fb.assumption_relaxed}")
    lines.append("")

    if rec.warnings:
        lines.append("Data compatibility warnings:")
        for w in rec.warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    lines.append("Required data (preview for Stage 4):")
    for v in rec.required_vars:
        lines.append(f"  ├── [Essential] {v}")
    for v in rec.optional_vars:
        lines.append(f"  └── [Optional]  {v}")
    lines.append("")

    if rec.heterogeneity_note:
        lines.append("Heterogeneity analysis (Stage 7+):")
        lines.append(f"  {rec.heterogeneity_note}")
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
        "warnings": rec.warnings,
        "heterogeneity_note": rec.heterogeneity_note,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Theoretical method analysis (two-level decision tree)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full two-level: facts → mechanism → method
  python stage3_analyze.py --from-facts data/auto/stage2_facts.json

  # Level 2 only (mechanism already known):
  python stage3_analyze.py --mechanism staggered_policy_shock --has-control-group

  # Fuzzy RDD
  python stage3_analyze.py --mechanism threshold_rule --threshold-type fuzzy

  # DML for high-dimensional selection on observables
  python stage3_analyze.py --mechanism selection_on_observables --high-dimensional-controls

  # Save output
  python stage3_analyze.py --from-facts stage2_facts.json --output stage3_result.json
""",
    )
    parser.add_argument("--from-facts", default=None,
                        help="Path to Stage 2 facts JSON (runs Level 1 + Level 2)")
    parser.add_argument("--mechanism", default=None,
                        choices=list(MECHANISM_DEFINITIONS.keys()),
                        help="Assignment mechanism type (Level 2 only; skip Level 1)")
    # Level 2 flags (only used by relevant branches)
    parser.add_argument("--threshold-type", default=None,
                        choices=["sharp", "fuzzy"],
                        help="RDD family: sharp or fuzzy")
    parser.add_argument("--panel-available", action="store_true", default=True,
                        help="DID/IV/SCM families: panel data is available")
    parser.add_argument("--no-panel", action="store_true", default=False,
                        help="DID/IV/SCM families: panel data is NOT available")
    parser.add_argument("--has-control-group", action="store_true", default=True,
                        help="DID family: never-treated control group exists")
    parser.add_argument("--no-control-group", action="store_true", default=False,
                        help="DID family: no never-treated control group")
    parser.add_argument("--staggered", action="store_true", default=False,
                        help="DID family: treatment is rolled out at different times")
    parser.add_argument("--everyone-treated-eventually", action="store_true", default=False,
                        help="DID family (staggered): all units are eventually treated")
    parser.add_argument("--has-instrument", action="store_true", default=False,
                        help="IV/SCM family: a valid instrument exists")
    parser.add_argument("--high-dimensional-controls", action="store_true", default=False,
                        help="Matching/ML family: many confounders → prefer DML")
    parser.add_argument("--multiple-policies", action="store_true", default=False,
                        help="DDD family: multiple policies affect same outcome")
    # Report options
    parser.add_argument("--policy", default="(unspecified)", help="Policy name")
    parser.add_argument("--outcome", default="(unspecified)", help="Outcome name")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--report", default=None, help="Output text report path")
    args = parser.parse_args()

    # Must specify either --from-facts or --mechanism
    if not args.from_facts and not args.mechanism:
        parser.error("Either --from-facts (Level 1+2) or --mechanism (Level 2 only) is required.")

    # Resolve conflicting flags
    panel_ok = args.panel_available and not args.no_panel
    has_control = args.has_control_group and not args.no_control_group

    # ── Mode A: Full two-level (facts → mechanism → method) ────────────
    if args.from_facts:
        with open(args.from_facts, encoding="utf-8") as f:
            facts = json.load(f)

        # Run Level 1 classification
        classification = classify_mechanism(facts)
        mechanism = classification.pop("mechanism")
        reasoning = classification.pop("reasoning", [])

        print("═══════════════════════════════════")
        print("Level 1: Mechanism Classification")
        print("═══════════════════════════════════")
        for step in reasoning:
            print(f"  {step}")
        print(f"  → Mechanism: {mechanism}")
        print(f"  → Family: {MECHANISM_DEFINITIONS.get(mechanism, {}).get('family', 'Unknown')}")
        print()

        # Run Level 2
        rec = decide_method(mechanism=mechanism, **classification)

        # Attach reasoning
        rec.literature_key = (
            f"[Classification: {' | '.join(reasoning)}]\n"
            + rec.literature_key
        )

        # Print report
        report = format_method_report(
            facts.get("policy_name", args.policy),
            facts.get("outcome", args.outcome),
            mechanism, rec
        )
        print(report)

        # Save JSON
        if args.output:
            output = {
                "mechanism": mechanism,
                "mechanism_label": MECHANISM_DEFINITIONS.get(mechanism, {}).get("label", ""),
                "method_family": MECHANISM_DEFINITIONS.get(mechanism, {}).get("family", ""),
                "classification_reasoning": reasoning,
                "inputs": classification,
                "recommendation": to_dict(rec),
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"\nJSON saved to {args.output}")

        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(report, encoding="utf-8")
            print(f"Report saved to {args.report}")

        return

    # ── Mode B: Level 2 only (mechanism already known) ─────────────────
    rec = decide_method(
        mechanism=args.mechanism,
        threshold_type=args.threshold_type,
        panel_available=panel_ok,
        has_control_group=has_control,
        staggered=args.staggered,
        everyone_treated_eventually=args.everyone_treated_eventually,
        has_instrument=args.has_instrument,
        high_dimensional_controls=args.high_dimensional_controls,
        multiple_policies=args.multiple_policies,
    )

    # Print report
    report = format_method_report(args.policy, args.outcome, args.mechanism, rec)
    print(report)

    # Save JSON
    if args.output:
        output = {
            "mechanism": args.mechanism,
            "mechanism_label": MECHANISM_DEFINITIONS.get(args.mechanism, {}).get("label", ""),
            "method_family": MECHANISM_DEFINITIONS.get(args.mechanism, {}).get("family", ""),
            "inputs": {
                "threshold_type": args.threshold_type,
                "panel_available": panel_ok,
                "has_control_group": has_control,
                "staggered": args.staggered,
                "everyone_treated_eventually": args.everyone_treated_eventually,
                "has_instrument": args.has_instrument,
                "high_dimensional_controls": args.high_dimensional_controls,
                "multiple_policies": args.multiple_policies,
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
