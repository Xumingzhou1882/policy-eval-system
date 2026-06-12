"""
Stage 6: Final Method Confirmation — deterministic assumption verification.

Reads the Stage 3 theoretical method recommendation and validates its
identifying assumptions against actual data. If the primary method's testable
assumptions fail, tries fallback strategies in ranked order from Stage 3.

Produces stage6_confirmation.json with assumption verdicts, final method
decision, and specification details for Stage 7 consumption.

Usage:
    # From a pipeline state file
    python stage6_confirm.py --stage3 data/auto/stage3_result.json \\
        --data data/merged/panel.dta --output data/auto/stage6_confirmation.json

    # With validation report and data status
    python stage6_confirm.py --stage3 data/auto/stage3_result.json \\
        --data data/merged/panel.dta --validate-report data/auto/validation_report.json \\
        --data-status '{"gdp": {"tier": "A", "status": "fetched"}}' \\
        --output data/auto/stage6_confirmation.json
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
AUTO_DIR = DATA_DIR / "auto"
MERGED_DIR = DATA_DIR / "merged"


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticResult:
    diagnostic_name: str
    test_description: str
    status: str          # "PASS" | "FAIL" | "UNCERTAIN" | "NOT_RUN"
    values: dict = field(default_factory=dict)
    threshold: str = ""
    interpretation: str = ""


@dataclass
class AssumptionVerdict:
    assumption_name: str
    description: str
    testable: bool
    test_method: str
    diagnostic: Optional[DiagnosticResult] = None
    verdict: str = "UNCERTAIN"   # "PASS" | "FAIL" | "UNCERTAIN"
    reasoning: str = ""


@dataclass
class FallbackAttempt:
    method_name: str
    condition: str
    assumption_relaxed: str
    diagnoses: list = field(default_factory=list)
    outcome: str = "skipped"     # "selected" | "rejected" | "skipped"
    rejection_reason: str = ""


@dataclass
class MethodConfirmation:
    mechanism: str
    theoretical_method: str
    final_method: str
    method_changed: bool
    chain: list = field(default_factory=list)
    gap_explanation: str = ""
    assumption_verdicts: list = field(default_factory=list)
    fallback_attempts: list = field(default_factory=list)
    data_quality_summary: dict = field(default_factory=dict)
    specification: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    causal_claim_strength: str = "moderate"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_data(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    elif p.suffix == ".json":
        return pd.read_json(str(p))
    raise ValueError(f"Unsupported data format: {p.suffix}")


def _resolve_data(data_path: str) -> str:
    """Resolve data path, trying merged dir if not found."""
    p = Path(data_path)
    if p.exists():
        return str(p.resolve())
    alt = MERGED_DIR / p.name
    if alt.exists():
        return str(alt)
    return str(p.resolve())


def _run_py(script_name: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run a peer script in the same directory."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# Data quality integration
# ═══════════════════════════════════════════════════════════════════════════

def _load_validation_report(data_path: str, output_col: str,
                            entity_col: str, time_col: str,
                            validate_report_path: str = None,
                            output_dir: str = None) -> dict:
    """Run validate_data.py or load a cached report."""
    if validate_report_path and Path(validate_report_path).exists():
        with open(validate_report_path, encoding="utf-8") as f:
            report = json.load(f)
    else:
        resolved = _resolve_data(data_path)
        out_d = Path(output_dir) if output_dir else AUTO_DIR
        tmp_out = str(out_d / "_stage6_validation.json")
        proc = _run_py("validate_data.py", [
            "--data", resolved, "--outcome", output_col,
            "--entity", entity_col, "--time", time_col,
            "--output", tmp_out,
        ])
        if proc.returncode != 0:
            return {"overall": "UNKNOWN", "error": proc.stderr}
        with open(tmp_out, encoding="utf-8") as f:
            report = json.load(f)

    summary = report.get("summary", {})
    return {
        "overall": summary.get("overall", "UNKNOWN"),
        "total_issues": summary.get("total_issues", 0),
        "critical_issues": summary.get("critical_details", []),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Inline diagnostic functions (no subprocess)
# ═══════════════════════════════════════════════════════════════════════════

def _check_overlap(df: pd.DataFrame, treatment_col: str,
                   covariates: list[str]) -> DiagnosticResult:
    """Check propensity score overlap between treated and control."""
    if not covariates or treatment_col not in df.columns:
        return DiagnosticResult(
            diagnostic_name="propensity_score_overlap",
            test_description="Propensity score support overlap between treated and control",
            status="NOT_RUN",
            values={"error": "Missing treatment column or covariates"},
            interpretation="Cannot check overlap without treatment column and covariates.",
        )

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return DiagnosticResult(
            diagnostic_name="propensity_score_overlap",
            test_description="Propensity score support overlap",
            status="NOT_RUN",
            values={"error": "scikit-learn not installed"},
            interpretation="Install scikit-learn for overlap diagnostics.",
        )

    valid = df[[treatment_col] + covariates].dropna()
    if len(valid) < 10:
        return DiagnosticResult(
            diagnostic_name="propensity_score_overlap",
            test_description="Propensity score support overlap",
            status="UNCERTAIN",
            values={"n_valid": len(valid)},
            interpretation="Too few observations for overlap check.",
        )

    X = valid[covariates].values
    y = valid[treatment_col].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]

    ps_treated = ps[y == 1]
    ps_control = ps[y == 0]

    if len(ps_treated) == 0 or len(ps_control) == 0:
        return DiagnosticResult(
            diagnostic_name="propensity_score_overlap",
            test_description="Propensity score support overlap",
            status="FAIL",
            values={"n_treated": int(y.sum()), "n_control": int((1 - y).sum())},
            threshold="Both groups must have observations",
            interpretation="One treatment group is empty — overlap cannot be assessed.",
        )

    min_ps = max(ps_treated.min(), ps_control.min())
    max_ps = min(ps_treated.max(), ps_control.max())

    if min_ps >= max_ps:
        return DiagnosticResult(
            diagnostic_name="propensity_score_overlap",
            test_description="Propensity score support overlap",
            status="FAIL",
            values={"ps_treated_range": [float(ps_treated.min()), float(ps_treated.max())],
                    "ps_control_range": [float(ps_control.min()), float(ps_control.max())]},
            threshold="Overlapping propensity score ranges",
            interpretation="No overlap in propensity scores. Treatment and control are not comparable.",
        )

    return DiagnosticResult(
        diagnostic_name="propensity_score_overlap",
        test_description="Propensity score support overlap",
        status="PASS",
        values={"ps_overlap_range": [float(min_ps), float(max_ps)]},
        threshold="Overlapping propensity score range",
        interpretation="Propensity scores overlap between treated and control groups.",
    )


def _check_treatment_variation(df: pd.DataFrame, treated_col: str,
                               entity_col: str) -> DiagnosticResult:
    """Check if treatment has sufficient within-entity variation."""
    if treated_col not in df.columns:
        return DiagnosticResult(
            diagnostic_name="treatment_variation",
            test_description="Within-entity treatment status variation",
            status="NOT_RUN",
            values={"error": "Treatment column not found"},
        )

    grp = df.groupby(entity_col)[treated_col]
    n_switchers = int((grp.nunique() > 1).sum())
    n_total = int(df[entity_col].nunique())

    pct_switch = n_switchers / n_total * 100 if n_total else 0

    status = "PASS" if pct_switch >= 5 else ("FAIL" if pct_switch == 0 else "UNCERTAIN")
    return DiagnosticResult(
        diagnostic_name="treatment_variation",
        test_description="Fraction of entities that change treatment status",
        status=status,
        values={"n_switchers": n_switchers, "n_entities": n_total, "pct_switchers": pct_switch},
        threshold=">= 5% of entities switch treatment status",
        interpretation=(
            f"{pct_switch:.1f}% of entities change treatment status. "
            + ("Sufficient variation for panel identification."
               if status == "PASS" else "Insufficient treatment variation for fixed-effects estimation.")
        ),
    )


def _check_covariate_balance(df: pd.DataFrame, treatment_col: str,
                             covariates: list[str]) -> DiagnosticResult:
    """Standardized mean difference for covariates between treated and control."""
    if not covariates or treatment_col not in df.columns:
        return DiagnosticResult(
            diagnostic_name="covariate_balance",
            test_description="Standardized mean differences between treated and control",
            status="NOT_RUN",
        )

    valid = df[[treatment_col] + covariates].dropna()
    treated = valid[valid[treatment_col] == 1]
    control = valid[valid[treatment_col] == 0]

    if len(treated) == 0 or len(control) == 0:
        return DiagnosticResult(
            diagnostic_name="covariate_balance",
            test_description="Standardized mean differences",
            status="FAIL",
            values={"error": "One group is empty"},
        )

    diffs = {}
    for c in covariates:
        mt, mc = treated[c].mean(), control[c].mean()
        sd_pooled = np.sqrt((treated[c].var() + control[c].var()) / 2)
        if sd_pooled > 1e-10:
            diffs[c] = float((mt - mc) / sd_pooled)

    if not diffs:
        return DiagnosticResult(
            diagnostic_name="covariate_balance",
            test_description="Standardized mean differences",
            status="UNCERTAIN",
            values={"error": "No valid covariates with variance"},
        )

    max_diff = max(abs(v) for v in diffs.values())
    status = "PASS" if max_diff < 0.25 else "FAIL"
    return DiagnosticResult(
        diagnostic_name="covariate_balance",
        test_description="Standardized mean differences (treated vs control)",
        status=status,
        values={"std_diffs": diffs, "max_abs_std_diff": max_diff},
        threshold="max |std_diff| < 0.25",
        interpretation=(
            f"Max standardized difference is {max_diff:.3f}. "
            + ("Covariates are balanced across groups." if status == "PASS"
               else "Some covariates are imbalanced — consider adding controls or matching.")
        ),
    )


def _check_compliance(df: pd.DataFrame, assignment_col: str,
                      treatment_col: str) -> DiagnosticResult:
    """Check compliance rate: P(treatment=1 | assignment=1)."""
    if assignment_col not in df.columns or treatment_col not in df.columns:
        return DiagnosticResult(
            diagnostic_name="compliance_check",
            test_description="Treatment compliance rate",
            status="NOT_RUN",
            values={"error": "Required columns not found"},
        )

    assigned = df[df[assignment_col] == 1]
    if len(assigned) == 0:
        return DiagnosticResult(
            diagnostic_name="compliance_check",
            test_description="Treatment compliance rate",
            status="UNCERTAIN",
            values={"error": "No assigned-to-treatment observations"},
        )

    rate = float(assigned[treatment_col].mean())
    status = "PASS" if rate >= 0.80 else ("FAIL" if rate < 0.50 else "UNCERTAIN")
    return DiagnosticResult(
        diagnostic_name="compliance_check",
        test_description="Fraction of assigned units that received treatment",
        status=status,
        values={"compliance_rate": rate, "n_assigned": int(len(assigned))},
        threshold="compliance_rate >= 0.80",
        interpretation=f"Compliance rate is {rate:.1%}.",
    )


def _check_attrition(df: pd.DataFrame, entity_col: str, time_col: str,
                     treated_col: str) -> DiagnosticResult:
    """Compare attrition rates between treated and control groups."""
    if treated_col not in df.columns:
        return DiagnosticResult(
            diagnostic_name="attrition_rate_comparison",
            test_description="Attrition rate difference between groups",
            status="NOT_RUN",
            values={"error": "Treatment column not found"},
        )

    grp = df.groupby(entity_col)
    max_t = df[time_col].max()
    min_t = df[time_col].min()

    present_start = set(df[df[time_col] == min_t][entity_col])
    present_end = set(df[df[time_col] == max_t][entity_col])

    # Determine treatment status from the data
    treated_entities = set(df[df[treated_col] == 1][entity_col].unique())
    control_entities = set(df[df[treated_col] == 0][entity_col].unique())

    def attrition_rate(entities):
        ents = entities & present_start
        if not ents:
            return None
        dropped = ents - present_end
        return len(dropped) / len(ents)

    rate_treat = attrition_rate(treated_entities)
    rate_ctrl = attrition_rate(control_entities)

    if rate_treat is None or rate_ctrl is None:
        return DiagnosticResult(
            diagnostic_name="attrition_rate_comparison",
            test_description="Attrition rate difference",
            status="UNCERTAIN",
            values={"error": "Cannot compute attrition for one or both groups"},
        )

    diff = abs(rate_treat - rate_ctrl)
    status = "PASS" if diff < 0.05 else "FAIL"
    return DiagnosticResult(
        diagnostic_name="attrition_rate_comparison",
        test_description="Difference in attrition rates (treated vs control)",
        status=status,
        values={"attrition_treated": rate_treat, "attrition_control": rate_ctrl, "diff": diff},
        threshold="|diff| < 0.05",
        interpretation=(
            f"Attrition difference is {diff:.3f}. "
            + ("No evidence of differential attrition." if status == "PASS"
               else "Differential attrition detected — results may be biased.")
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess diagnostic runners
# ═══════════════════════════════════════════════════════════════════════════

def _run_event_study_diagnostic(data_path: str, outcome: str, entity: str,
                                time: str, first_treated: str,
                                n_pre: int = 5, controls: list[str] = None,
                                output_dir: str = None) -> dict:
    """Call run_event_study.py and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_event_study.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--entity", entity,
        "--time", time,
        "--first-treated", first_treated,
        "--n-pre", str(n_pre),
        "--n-post", "0",
        "--output", tmp_out,
    ]
    # Only pass controls that are actual columns in the data
    if controls:
        df_cols = set(_load_data(data_path).columns)
        valid_controls = [c for c in controls if c in df_cols]
        if valid_controls:
            args.extend(["--controls"] + valid_controls)

    proc = _run_py("run_event_study.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr, "pre_trends_test": {}, "coefficients": {}}

    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


def _run_bacon_diagnostic(data_path: str, outcome: str, entity: str,
                          time: str, treated: str, first_treated: str,
                          output_dir: str = None) -> dict:
    """Call bacon_decomp.py and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_bacon.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--entity", entity,
        "--time", time,
        "--treated", treated,
        "--first-treated", first_treated,
        "--output", tmp_out,
    ]
    proc = _run_py("bacon_decomp.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr, "negative_weight_pct": 100, "n_comparisons": 0}
    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


def _run_mccrary_diagnostic(data_path: str, outcome: str, running_var: str,
                            cutoff: float, output_dir: str = None) -> dict:
    """Call run_rdd.py with --mccrary and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_mccrary.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--running-var", running_var,
        "--cutoff", str(cutoff),
        "--mccrary",
        "--output", tmp_out,
    ]
    proc = _run_py("run_rdd.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr, "mccrary_test": {}}
    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


def _run_iv_diagnostic(data_path: str, outcome: str, treatment: str,
                       instruments: list[str], controls: list[str] = None,
                       output_dir: str = None) -> dict:
    """Call run_iv.py and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_iv.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--treatment", treatment,
        "--instruments"] + instruments + [
        "--output", tmp_out,
    ]
    if controls:
        args.extend(["--controls"] + controls)

    proc = _run_py("run_iv.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr}
    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


def _run_scm_diagnostic(data_path: str, outcome: str, entity: str,
                        time: str, treated_unit: str, first_treated: float,
                        output_dir: str = None) -> dict:
    """Call run_scm.py and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_scm.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--entity", entity,
        "--time", time,
        "--treated-unit", str(treated_unit),
        "--first-treated", str(first_treated),
        "--output", tmp_out,
    ]
    proc = _run_py("run_scm.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr, "rmse_pre": 999, "p_value": 1}
    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


def _run_synthetic_did_diagnostic(data_path: str, outcome: str, entity: str,
                                  time: str, treated_unit: str, first_treated: float,
                                  output_dir: str = None) -> dict:
    """Call run_synthetic_did.py and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_sdid.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--entity", entity,
        "--time", time,
        "--treated-unit", str(treated_unit),
        "--first-treated", str(first_treated),
        "--output", tmp_out,
    ]
    proc = _run_py("run_synthetic_did.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr, "pre_rmse": 999, "p_value": 1}
    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


def _run_dml_diagnostic(data_path: str, outcome: str, treatment: str,
                        controls: list[str], ml_model: str = "gradient_boosting",
                        output_dir: str = None) -> dict:
    """Call run_dml.py and return parsed JSON."""
    out_dir = Path(output_dir) if output_dir else AUTO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out_dir / "_stage6_dml.json")

    args = [
        "--data", _resolve_data(data_path),
        "--outcome", outcome,
        "--treatment", treatment,
        "--controls"] + controls + [
        "--ml-model", ml_model,
        "--cv", "5",
        "--output", tmp_out,
    ]
    proc = _run_py("run_dml.py", args)
    if proc.returncode != 0:
        return {"error": proc.stderr, "nuisance_scores": {}}
    with open(tmp_out, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Assumption → Diagnostic dispatch
# ═══════════════════════════════════════════════════════════════════════════

def _diagnose_assumption(
    assumption: dict,
    mechanism: str,
    data_path: str,
    df: pd.DataFrame,
    output_dir: str,
    entity_col: str = "city_id",
    time_col: str = "year",
    outcome_col: str = "outcome",
) -> DiagnosticResult:
    """
    Dispatch a single Stage 3 assumption to its diagnostic test.
    Returns DiagnosticResult, or None for non-testable assumptions (no dispatch needed).
    """
    name = assumption.get("name", "")
    testable = assumption.get("testable", False)

    if not testable:
        return None  # Will be handled as UNCERTAIN by the caller

    # ── DID-family assumptions ──
    if "parallel trends" in name.lower():
        diag = _run_event_study_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            assumption.get("first_treated_col", "first_treated"),
            controls=assumption.get("controls", []),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="event_study_parallel_trends",
                test_description="F-test of joint significance of pre-treatment event study coefficients",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        pt = diag.get("pre_trends_test", {})
        f_stat = pt.get("f_stat")
        p_value = pt.get("p_value")
        if f_stat is None or p_value is None:
            return DiagnosticResult(
                diagnostic_name="event_study_parallel_trends",
                test_description="F-test of pre-treatment coefficients",
                status="UNCERTAIN",
                values=pt,
                threshold="p_value > 0.05",
                interpretation="Could not compute parallel trends test.",
            )
        passed = p_value > 0.05
        return DiagnosticResult(
            diagnostic_name="event_study_parallel_trends",
            test_description="F-test of joint significance of pre-treatment event study coefficients",
            status="PASS" if passed else "FAIL",
            values={"f_stat": f_stat, "p_value": p_value},
            threshold="p_value > 0.05",
            interpretation=(
                f"Pre-trends F-test p={p_value:.4f}. "
                + ("Parallel trends assumption is supported." if passed
                   else "Pre-treatment trends diverge — parallel trends is violated.")
            ),
        )

    if "no anticipation" in name.lower():
        diag = _run_event_study_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            assumption.get("first_treated_col", "first_treated"),
            controls=assumption.get("controls", []),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="event_study_t_minus_1",
                test_description="Test whether the coefficient at t=-1 is zero",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        coefs = diag.get("coefficients", {})
        c_minus_1 = coefs.get("-1", {})
        if not c_minus_1:
            return DiagnosticResult(
                diagnostic_name="event_study_t_minus_1",
                test_description="Test whether the coefficient at t=-1 is zero",
                status="UNCERTAIN",
                values={"error": "t=-1 coefficient not found in event study output"},
                threshold="p_value > 0.05",
                interpretation="Could not find t=-1 coefficient for anticipation check.",
            )
        p_val = c_minus_1.get("p_value", 1.0)
        passed = p_val > 0.05
        return DiagnosticResult(
            diagnostic_name="event_study_t_minus_1",
            test_description="Test whether the coefficient at t=-1 is zero",
            status="PASS" if passed else "FAIL",
            values={"coef_t_minus_1": c_minus_1.get("coefficient"), "p_value": p_val},
            threshold="p_value > 0.05",
            interpretation=(
                f"t=-1 coefficient p={p_val:.4f}. "
                + ("No evidence of anticipation effects." if passed
                   else "Significant effect at t=-1 suggests anticipation.")
            ),
        )

    if "limited heterogeneity" in name.lower() or "bacon" in name.lower():
        diag = _run_bacon_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            assumption.get("treated_col", "treated"),
            assumption.get("first_treated_col", "first_treated"),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="bacon_decomp",
                test_description="Goodman-Bacon decomposition negative weight share",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        neg_pct = diag.get("negative_weight_pct", 0)
        passed = neg_pct <= 10
        return DiagnosticResult(
            diagnostic_name="bacon_decomp",
            test_description="Goodman-Bacon decomposition: negative weight share",
            status="PASS" if passed else "FAIL",
            values={"negative_weight_pct": neg_pct, "n_comparisons": diag.get("n_comparisons", 0)},
            threshold="negative_weight_pct <= 10%",
            interpretation=(
                f"Negative weight share is {neg_pct:.1f}%. "
                + ("TWFE is reliable for this design." if passed
                   else "TWFE is unreliable — use heterogeneity-robust estimator (C&S / S&A).")
            ),
        )

    if "stable unit composition" in name.lower() or "panel balance" in name.lower():
        grp = df.groupby(entity_col)[time_col]
        n_total = int(df[entity_col].nunique())
        n_periods = int(df[time_col].nunique())
        n_incomplete = int((grp.count() < n_periods).sum())

        missing_pct = n_incomplete / n_total * 100 if n_total else 0
        passed = missing_pct < 5
        return DiagnosticResult(
            diagnostic_name="panel_balance",
            test_description=f"Entities present in all {n_periods} periods",
            status="PASS" if passed else "FAIL",
            values={"n_entities": n_total, "n_periods": n_periods,
                    "n_incomplete": n_incomplete, "missing_pct": missing_pct},
            threshold="< 5% of entities have incomplete panels",
            interpretation=(
                f"{missing_pct:.1f}% of entities have missing periods. "
                + ("Panel is balanced." if passed else "Panel imbalance may affect DID estimates.")
            ),
        )

    # ── RDD assumptions ──
    if "no manipulation" in name.lower() or "mccrary" in name.lower():
        diag = _run_mccrary_diagnostic(
            data_path, outcome_col,
            assumption.get("running_var", "running_var"),
            assumption.get("cutoff", 0),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="mccrary_density_test",
                test_description="McCrary (2008) density test at cutoff",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        mc = diag.get("mccrary_test", {})
        if not mc:
            return DiagnosticResult(
                diagnostic_name="mccrary_density_test",
                test_description="McCrary density test at cutoff",
                status="UNCERTAIN",
                values={"error": "McCrary test not run (use --mccrary flag on run_rdd.py)"},
                threshold="p_value > 0.05",
                interpretation="McCrary test was not included in RDD output.",
            )
        manipulation = mc.get("manipulation_detected", False)
        p_val = mc.get("p_value", 0)
        return DiagnosticResult(
            diagnostic_name="mccrary_density_test",
            test_description="McCrary (2008) density test for manipulation at cutoff",
            status="PASS" if not manipulation else "FAIL",
            values={"log_difference": mc.get("log_difference"), "p_value": p_val,
                    "manipulation_detected": manipulation},
            threshold="p_value > 0.05",
            interpretation=(
                f"McCrary test p={p_val:.4f}. "
                + ("No evidence of manipulation at cutoff." if not manipulation
                   else "Density discontinuity detected — possible manipulation. Consider donut-hole RDD.")
            ),
        )

    if "continuity of potential outcomes" in name.lower() or "covariate balance" in name.lower():
        return _check_covariate_balance(df, assumption.get("treated_col", "treated"),
                                        assumption.get("covariates", []))

    if "local linearity" in name.lower():
        # Bandwidth sensitivity — runs RDD at 0.5x and 2x bandwidth for stability check
        return DiagnosticResult(
            diagnostic_name="bandwidth_sensitivity",
            test_description="Coefficient stability across bandwidth choices",
            status="UNCERTAIN",
            values={"note": "Bandwidth sensitivity requires manual inspection or additional RDD runs."},
            threshold="ATT stable across 0.5x to 2x CCT bandwidth",
            interpretation="Bandwidth sensitivity not automatically checked. Verify by re-running RDD with different --bandwidth values.",
        )

    if "first-stage strength" in name.lower():
        diag = _run_mccrary_diagnostic(
            data_path, outcome_col,
            assumption.get("running_var", "running_var"),
            assumption.get("cutoff", 0),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="rdd_first_stage_f",
                test_description="First-stage F-statistic at cutoff (fuzzy RDD)",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        fs_f = diag.get("first_stage_fstat", 0)
        passed = fs_f >= 10
        return DiagnosticResult(
            diagnostic_name="rdd_first_stage_f",
            test_description="First-stage F-statistic at cutoff (fuzzy RDD)",
            status="PASS" if passed else "FAIL",
            values={"first_stage_fstat": fs_f},
            threshold="F >= 10",
            interpretation=(
                f"First-stage F={fs_f:.1f}. "
                + ("Strong first stage at cutoff." if passed else "Weak first stage — consider alternative methods.")
            ),
        )

    # ── IV assumptions ──
    if "relevance" in name.lower():
        diag = _run_iv_diagnostic(
            data_path, outcome_col,
            assumption.get("treatment_var", "treatment"),
            assumption.get("instruments", []),
            assumption.get("controls", []),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="first_stage_f_test",
                test_description="Montiel Olea & Pflueger (2013) effective F-statistic",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        fs = diag.get("2sls", {})
        effective_f = fs.get("effective_f", 0)
        critical_val = fs.get("mp_critical_value", 23.1)
        weak = fs.get("weak_instrument", True)
        return DiagnosticResult(
            diagnostic_name="first_stage_f_test",
            test_description="Montiel Olea & Pflueger (2013) effective F-statistic",
            status="PASS" if not weak else "FAIL",
            values={"first_stage_f": fs.get("first_stage_f"), "effective_f": effective_f,
                    "mp_critical_value": critical_val},
            threshold=f"effective F > {critical_val}",
            interpretation=(
                f"Effective F={effective_f:.1f} (critical={critical_val:.1f}). "
                + ("Instrument is strong." if not weak else "Weak instrument — use LIML or find stronger instruments.")
            ),
        )

    if "exclusion" in name.lower():
        diag = _run_iv_diagnostic(
            data_path, outcome_col,
            assumption.get("treatment_var", "treatment"),
            assumption.get("instruments", []),
            assumption.get("controls", []),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="overidentification_hansen_j",
                test_description="Hansen J overidentification test",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        oid = diag.get("2sls", {}).get("overid_test", {})
        if not oid:
            return DiagnosticResult(
                diagnostic_name="overidentification_hansen_j",
                test_description="Hansen J overidentification test",
                status="UNCERTAIN",
                values={"note": "Only one instrument — overidentification test not applicable."},
                threshold="J-test p > 0.05",
                interpretation="Overidentification test requires multiple instruments.",
            )
        p_val = oid.get("p_value", 0)
        passed = p_val > 0.05
        return DiagnosticResult(
            diagnostic_name="overidentification_hansen_j",
            test_description="Hansen J overidentification test",
            status="PASS" if passed else "FAIL",
            values={"j_statistic": oid.get("j_statistic"), "p_value": p_val},
            threshold="p_value > 0.05",
            interpretation=(
                f"J-test p={p_val:.4f}. "
                + ("No evidence against instrument validity." if passed
                   else "At least one instrument may be invalid.")
            ),
        )

    # ── SCM assumptions ──
    if "pre-treatment fit" in name.lower():
        diag = _run_scm_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            assumption.get("treated_unit", ""),
            assumption.get("first_treated", 0),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="scm_pre_rmse",
                test_description="Pre-treatment RMSE of synthetic control",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        rmse = diag.get("rmse_pre", 999)
        sd_pre = assumption.get("outcome_sd_pre", 1.0)
        threshold_val = 0.1 * sd_pre
        passed = rmse < threshold_val

        # Try to compute actual pre-treatment SD
        try:
            ft = assumption.get("first_treated", 0)
            pre_data = df[df[time_col] < ft][outcome_col].dropna()
            if len(pre_data) > 1:
                sd_pre = float(pre_data.std())
                threshold_val = 0.1 * sd_pre
                passed = rmse < threshold_val
        except Exception:
            pass

        return DiagnosticResult(
            diagnostic_name="scm_pre_rmse",
            test_description="Pre-treatment RMSE: how well synthetic control matches treated unit",
            status="PASS" if passed else "FAIL",
            values={"rmse_pre": rmse, "outcome_sd_pre": sd_pre if sd_pre else 1.0,
                    "rmse_ratio": diag.get("rmse_ratio")},
            threshold=f"rmse_pre < 0.1 * sd_pre",
            interpretation=(
                f"Pre-treatment RMSE={rmse:.4f}. "
                + ("Synthetic control fits well." if passed else "Poor pre-treatment fit — SCM may be unreliable.")
            ),
        )

    if "unobserved time-varying" in name.lower() or "in-space placebo" in name.lower():
        diag = _run_scm_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            assumption.get("treated_unit", ""),
            assumption.get("first_treated", 0),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="in_space_placebo",
                test_description="In-space placebo: SCM applied to each donor unit",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        p_val = diag.get("p_value", 1.0)
        passed = p_val < 0.05
        return DiagnosticResult(
            diagnostic_name="in_space_placebo",
            test_description="In-space placebo: treated unit effect vs. donor placebo distribution",
            status="PASS" if passed else "FAIL",
            values={"p_value": p_val, "n_placebo": diag.get("n_placebo", 0)},
            threshold="p_value < 0.05",
            interpretation=(
                f"Placebo p={p_val:.4f}. "
                + ("Treated unit effect stands out from placebo distribution." if passed
                   else "Treated unit effect is not distinguishable from placebo — SCM result is not robust.")
            ),
        )

    if "donor pool" in name.lower():
        diag = _run_scm_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            assumption.get("treated_unit", ""),
            assumption.get("first_treated", 0),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="donor_pool_adequacy",
                test_description="Donor pool size and pre-treatment fit",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        n_donors = diag.get("n_donors", 0)
        n_nonzero = diag.get("n_nonzero_weights", 0)
        passed = n_nonzero >= 3
        return DiagnosticResult(
            diagnostic_name="donor_pool_adequacy",
            test_description="Number of donors with non-zero weight in synthetic control",
            status="PASS" if passed else "FAIL",
            values={"n_donors": n_donors, "n_nonzero_weights": n_nonzero},
            threshold=">= 3 donors with non-zero weight",
            interpretation=(
                f"{n_nonzero} donors have non-zero weights. "
                + ("Donor pool is adequate." if passed
                   else "Too few donors contribute — SCM extrapolates too far.")
            ),
        )

    # ── Selection-on-observables assumptions ──
    if "overlap" in name.lower() or "common support" in name.lower():
        return _check_overlap(df, assumption.get("treatment_col", "treated"),
                              assumption.get("covariates", []))

    if "ml model quality" in name.lower() or "nuisance" in name.lower():
        diag = _run_dml_diagnostic(
            data_path, outcome_col,
            assumption.get("treatment_var", "treatment"),
            assumption.get("controls", []),
            output_dir=output_dir,
        )
        if "error" in diag:
            return DiagnosticResult(
                diagnostic_name="dml_nuisance_cv_r2",
                test_description="Cross-validated R² of DML nuisance models",
                status="NOT_RUN",
                values={"error": diag.get("error", "")},
            )
        ns = diag.get("nuisance_scores", {})
        outcome_r2 = ns.get("outcome_cv_r2", 0)
        treatment_r2 = ns.get("treatment_cv_r2", 0)
        passed = outcome_r2 > 0.1 and treatment_r2 > 0.1
        return DiagnosticResult(
            diagnostic_name="dml_nuisance_cv_r2",
            test_description="Cross-validated R² of outcome and treatment nuisance models",
            status="PASS" if passed else "FAIL",
            values={"outcome_cv_r2": outcome_r2, "treatment_cv_r2": treatment_r2},
            threshold="Both CV R² > 0.1",
            interpretation=(
                f"Outcome CV R²={outcome_r2:.3f}, Treatment CV R²={treatment_r2:.3f}. "
                + ("ML models fit reasonably well." if passed
                   else "Poor nuisance model fit — DML estimates may be unreliable.")
            ),
        )

    # ── Randomized experiment assumptions ──
    if "compliance" in name.lower():
        return _check_compliance(df, assumption.get("assignment_col", "assignment"),
                                 assumption.get("treatment_col", "treated"))

    if "attrition" in name.lower() or "selective attrition" in name.lower():
        return _check_attrition(df, entity_col, time_col,
                                assumption.get("treated_col", "treated"))

    # ── Intensity DID assumptions ──
    if "dose-response" in name.lower() or "parallel trends in dose" in name.lower():
        intensity_col = assumption.get("intensity_col", "treatment_intensity")
        if intensity_col not in df.columns:
            return DiagnosticResult(
                diagnostic_name="dose_response_event_study",
                test_description="Parallel trends in dose-response",
                status="NOT_RUN",
                values={"error": f"Intensity column '{intensity_col}' not found"},
            )

        try:
            import statsmodels.formula.api as smf
            df_test = df.copy()
            df_test["_intensity"] = df_test[intensity_col]
            df_test["_post"] = (df_test[time_col] > assumption.get("treatment_time", 0)).astype(int)
            formula = f"{outcome_col} ~ _post * _intensity + C({entity_col}) + C({time_col})"
            model = smf.ols(formula, data=df_test).fit()
            f_test = model.f_test("_post:_intensity = 0")
            p_val = float(f_test.pvalue)
            passed = p_val > 0.05
            return DiagnosticResult(
                diagnostic_name="dose_response_event_study",
                test_description="F-test for dose-response interaction in pre-treatment periods",
                status="PASS" if passed else "FAIL",
                values={"p_value": p_val},
                threshold="p_value > 0.05",
                interpretation=(
                    f"Interaction test p={p_val:.4f}. "
                    + ("No evidence against parallel dose-response trends." if passed
                       else "Dose-response trends may not be parallel.")
                ),
            )
        except Exception as e:
            return DiagnosticResult(
                diagnostic_name="dose_response_event_study",
                test_description="Parallel trends in dose-response",
                status="NOT_RUN",
                values={"error": str(e)},
            )

    # ── DDD assumptions ──
    if "group exposed" in name.lower() or "at least one group" in name.lower():
        policy_a_col = assumption.get("policy_a_col", "")
        policy_b_col = assumption.get("policy_b_col", "")
        if policy_a_col not in df.columns or policy_b_col not in df.columns:
            return DiagnosticResult(
                diagnostic_name="policy_exposure_overlap",
                test_description="Check for groups exposed to only one policy",
                status="NOT_RUN",
                values={"error": "Policy columns not found"},
            )
        a_only = int(((df[policy_a_col] == 1) & (df[policy_b_col] == 0)).any())
        b_only = int(((df[policy_a_col] == 0) & (df[policy_b_col] == 1)).any())
        passed = (a_only + b_only) >= 2
        return DiagnosticResult(
            diagnostic_name="policy_exposure_overlap",
            test_description="At least one group exposed to each policy in isolation",
            status="PASS" if passed else "FAIL",
            values={"group_a_only_exists": bool(a_only), "group_b_only_exists": bool(b_only)},
            threshold="Both policies have exclusive exposure groups",
            interpretation=(
                "Both policies have exclusive groups — DDD is feasible."
                if passed else "Policies overlap completely — cannot isolate individual effects via DDD.")
        )

    # ── Fallthrough: unknown assumption ──
    return DiagnosticResult(
        diagnostic_name="unknown_assumption",
        test_description=name,
        status="UNCERTAIN",
        values={"note": f"No diagnostic registered for: {name}"},
        interpretation=f"No automated test available for '{name}'.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fallback chain engine
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_VAR_REQUIREMENTS = {
    # Keys are method substrings; values are (required_roles, actual_columns_from_context)
    # Actual columns are resolved via _check_fallback_data using the data context.
    "callaway": ["entity_id", "time", "outcome", "first_treated"],
    "sun & abraham": ["entity_id", "time", "outcome", "first_treated"],
    "borusyak": ["entity_id", "time", "outcome", "first_treated"],
    "de chaisemartin": ["entity_id", "time", "outcome", "first_treated"],
    "scm": ["entity_id", "time", "outcome", "first_treated"],
    "synthetic did": ["entity_id", "time", "outcome", "first_treated"],
    "synthetic control": ["entity_id", "time", "outcome", "first_treated"],
    "did": ["entity_id", "time", "outcome", "treatment_indicator", "post_period"],
    "twfe": ["entity_id", "time", "outcome", "treatment_indicator", "post_period"],
    "psm": ["outcome", "treatment_indicator"],
    "ipw": ["outcome", "treatment_indicator"],
    "aipw": ["outcome", "treatment_indicator"],
    "dml": ["outcome", "treatment_indicator"],
    "causal forest": ["outcome", "treatment_indicator"],
    "cem": ["outcome", "treatment_indicator"],
    "iv": ["outcome", "treatment", "instrument"],
    "2sls": ["outcome", "treatment", "instrument"],
    "liml": ["outcome", "treatment", "instrument"],
    "rdd": ["running_variable", "outcome"],
    "donut-hole rdd": ["running_variable", "outcome"],
    "intensity did": ["outcome", "treatment_intensity", "entity_id", "time"],
    "ddd": ["outcome", "entity_id", "time"],
    "interactive fixed effects": ["entity_id", "time", "outcome", "first_treated"],
    "matrix completion": ["entity_id", "time", "outcome", "first_treated"],
    "honest reporting": [],
}


def _check_fallback_data(fallback_method: str, df: pd.DataFrame,
                         entity_col: str, time_col: str,
                         outcome_col: str, first_treated_col: str,
                         treated_col: str) -> tuple[bool, list[str]]:
    """Check if data has the columns needed for a fallback method.

    Uses actual column names from the data context (not Stage 3 abstract names).
    """
    method_lower = fallback_method.lower()

    # Map abstract role names → actual column names from data context
    role_map = {
        "entity_id": entity_col,
        "time": time_col,
        "outcome": outcome_col,
        "first_treated": first_treated_col,
        "treatment_indicator": treated_col,
        "treatment": treated_col,
    }

    for key, required_roles in FALLBACK_VAR_REQUIREMENTS.items():
        if key in method_lower:
            required_cols = []
            for role in required_roles:
                actual = role_map.get(role, role)
                required_cols.append(actual)
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return False, missing
            return True, []
    return True, []


def _resolve_fallback_chain(
    primary_failures: list[AssumptionVerdict],
    fallbacks: list[dict],
    data_path: str,
    df: pd.DataFrame,
    stage3_rec: dict,
    output_dir: str,
    entity_col: str,
    time_col: str,
    outcome_col: str,
    first_treated_col: str,
    treated_col: str,
) -> tuple[Optional[FallbackAttempt], list[FallbackAttempt]]:
    """Try fallbacks in ranked order. Returns (selected_fallback, all_attempts)."""
    all_attempts = []

    for fb in fallbacks:
        method_name = fb.get("method", "")
        condition = fb.get("condition", "")
        relaxed = fb.get("assumption_relaxed", "")

        attempt = FallbackAttempt(
            method_name=method_name,
            condition=condition,
            assumption_relaxed=relaxed,
        )

        # Step 1: Check data availability using actual column names
        has_data, missing = _check_fallback_data(
            method_name, df, entity_col, time_col, outcome_col,
            first_treated_col, treated_col,
        )
        if not has_data and missing:
            attempt.outcome = "skipped"
            attempt.rejection_reason = f"Missing required columns: {missing}"
            all_attempts.append(attempt)
            continue

        # Step 2: Run method-specific diagnostics
        diagnoses = _run_fallback_diagnostics(
            method_name, data_path, df, output_dir,
            entity_col, time_col, outcome_col, stage3_rec,
        )

        attempt.diagnoses = diagnoses
        failures = [d for d in diagnoses if d.status == "FAIL"]

        if not failures:
            attempt.outcome = "selected"
            all_attempts.append(attempt)
            return attempt, all_attempts
        else:
            attempt.outcome = "rejected"
            attempt.rejection_reason = "Failed diagnostics: " + ", ".join(
                f"{d.diagnostic_name}" for d in failures
            )
            all_attempts.append(attempt)

    return None, all_attempts


def _run_fallback_diagnostics(
    method_name: str, data_path: str, df: pd.DataFrame,
    output_dir: str, entity_col: str, time_col: str,
    outcome_col: str, stage3_rec: dict,
) -> list[DiagnosticResult]:
    """Run appropriate diagnostics for a fallback method."""
    diagnoses = []
    ml = method_name.lower()

    # DID-family fallbacks: check parallel trends via event study
    if any(k in ml for k in ("did", "callaway", "sun & abraham", "staggered",
                              "borusyak", "de chaisemartin", "twfe")):
        diag = _run_event_study_diagnostic(
            data_path, outcome_col, entity_col, time_col,
            "first_treated", output_dir=output_dir,
        )
        if "error" not in diag:
            pt = diag.get("pre_trends_test", {})
            p_val = pt.get("p_value")
            if p_val is not None:
                passed = p_val > 0.05
                diagnoses.append(DiagnosticResult(
                    diagnostic_name="fallback_parallel_trends",
                    test_description=f"Parallel trends for {method_name}",
                    status="PASS" if passed else "FAIL",
                    values={"p_value": p_val, "f_stat": pt.get("f_stat")},
                    threshold="p_value > 0.05",
                    interpretation="Pre-trends " + ("hold" if passed else "violated"),
                ))

    # SCM / Synthetic DID
    if any(k in ml for k in ("scm", "synthetic did", "synthetic control")):
        inputs = stage3_rec.get("inputs", {})
        treated_unit = inputs.get("treated_unit", "")
        first_treated = inputs.get("first_treated", 0)

        if "synthetic did" in ml:
            diag = _run_synthetic_did_diagnostic(
                data_path, outcome_col, entity_col, time_col,
                str(treated_unit), float(first_treated), output_dir=output_dir,
            )
        else:
            diag = _run_scm_diagnostic(
                data_path, outcome_col, entity_col, time_col,
                str(treated_unit), float(first_treated), output_dir=output_dir,
            )

        if "error" not in diag:
            rmse = diag.get("rmse_pre", diag.get("pre_rmse", 999))
            sd_pre = float(df[outcome_col].std())
            passed = rmse < 0.1 * sd_pre if sd_pre > 0 else False
            diagnoses.append(DiagnosticResult(
                diagnostic_name="fallback_pre_treatment_fit",
                test_description=f"Pre-treatment fit for {method_name}",
                status="PASS" if passed else "FAIL",
                values={"rmse_pre": rmse, "sd_pre": sd_pre},
                threshold="rmse_pre < 0.1 * sd_pre",
                interpretation=f"Pre-treatment RMSE={rmse:.4f}",
            ))

    # IV fallbacks: check first-stage strength
    if any(k in ml for k in ("iv", "2sls", "liml")):
        inputs = stage3_rec.get("inputs", {})
        instruments = inputs.get("instruments", [])
        treatment_var = inputs.get("treatment_var", "treatment")
        if instruments:
            diag = _run_iv_diagnostic(
                data_path, outcome_col, treatment_var, instruments,
                output_dir=output_dir,
            )
            if "error" not in diag:
                fs = diag.get("2sls", {})
                weak = fs.get("weak_instrument", True)
                diagnoses.append(DiagnosticResult(
                    diagnostic_name="fallback_first_stage_f",
                    test_description=f"First-stage strength for {method_name}",
                    status="PASS" if not weak else "FAIL",
                    values={"effective_f": fs.get("effective_f"), "first_stage_f": fs.get("first_stage_f")},
                    threshold="effective F > MP critical value",
                    interpretation="Instrument is " + ("strong" if not weak else "weak"),
                ))

    # RDD fallbacks: run McCrary
    if "rdd" in ml:
        inputs = stage3_rec.get("inputs", {})
        running_var = inputs.get("running_var", "running_var")
        cutoff = inputs.get("cutoff", 0)
        diag = _run_mccrary_diagnostic(
            data_path, outcome_col, running_var, float(cutoff), output_dir=output_dir,
        )
        if "error" not in diag:
            mc = diag.get("mccrary_test", {})
            if mc:
                manip = mc.get("manipulation_detected", True)
                diagnoses.append(DiagnosticResult(
                    diagnostic_name="fallback_mccrary",
                    test_description=f"McCrary test for {method_name}",
                    status="PASS" if not manip else "FAIL",
                    values={"p_value": mc.get("p_value")},
                    threshold="p_value > 0.05",
                ))

    # DML / matching fallbacks: check overlap
    if any(k in ml for k in ("psm", "ipw", "aipw", "dml", "cem", "causal forest")):
        diagnoses.append(_check_overlap(df, "treated",
                          stage3_rec.get("recommendation", {}).get("optional_variables", [])))

    return diagnoses


# ═══════════════════════════════════════════════════════════════════════════
# Main confirmation engine
# ═══════════════════════════════════════════════════════════════════════════

def confirm_method(
    stage3_output: dict,
    data_path: str,
    validate_report: dict = None,
    data_status: dict = None,
    entity_col: str = "city_id",
    time_col: str = "year",
    output_dir: str = None,
) -> MethodConfirmation:
    """Confirm the theoretical method against actual data."""

    mechanism = stage3_output.get("mechanism", "unknown")
    rec = stage3_output.get("recommendation", {})
    theoretical_method = rec.get("primary_method", "Unknown")
    assumptions = rec.get("assumptions", [])
    fallbacks = rec.get("fallbacks", [])
    inputs = stage3_output.get("inputs", {})

    outcome_col = stage3_output.get("outcome",
                   stage3_output.get("outcome_var", "outcome"))

    # Resolve data path
    resolved_data = _resolve_data(data_path)
    df = _load_data(resolved_data)

    # Infer column names from data and stage3 inputs
    if entity_col not in df.columns:
        for cand in ["city_id", "entity_id", "id", "entity"]:
            if cand in df.columns:
                entity_col = cand
                break
    if time_col not in df.columns:
        for cand in ["year", "time", "period", "date"]:
            if cand in df.columns:
                time_col = cand
                break

    # Determine key columns from Stage 3 inputs
    first_treated_col = inputs.get("first_treated_col", "first_treated")
    if first_treated_col not in df.columns:
        for cand in ["first_treated", "first_treatment_year", "treatment_year"]:
            if cand in df.columns:
                first_treated_col = cand
                break

    treated_col = "treated"
    if treated_col not in df.columns:
        if first_treated_col in df.columns:
            df["treated"] = df[first_treated_col].notna().astype(int)
        else:
            for cand in ["treated", "treatment", "treat", "treated_unit"]:
                if cand in df.columns:
                    treated_col = cand
                    break

    outcome_col_actual = outcome_col
    if outcome_col not in df.columns:
        for cand in ["log_birth_rate", "log_fertility", "log_outcome", "outcome",
                     "birth_rate", "fertility_rate", "y", "dep_var"]:
            if cand in df.columns:
                outcome_col_actual = cand
                break
    else:
        outcome_col_actual = outcome_col

    # ── Step 1: Data quality ──
    data_quality = _load_validation_report(
        data_path, outcome_col_actual, entity_col, time_col,
        validate_report.get("path") if validate_report else None,
        output_dir,
    )

    # ── Step 2: Verify each assumption ──
    verdicts = []
    has_failure = False

    # Enrich assumptions with column info from inputs/stage
    # Filter Stage 3's abstract variable names to actual data columns
    df_cols = set(df.columns)
    abstract_optional = rec.get("optional_variables", [])
    actual_covariates = [c for c in abstract_optional if c in df_cols]

    enriched_assumptions = []
    for a in assumptions:
        enriched = dict(a)
        enriched["first_treated_col"] = first_treated_col
        enriched["treated_col"] = treated_col
        enriched["entity_col"] = entity_col
        enriched["time_col"] = time_col
        enriched["first_treated"] = inputs.get("first_treated", 0)
        enriched["treated_unit"] = inputs.get("treated_unit", "")
        enriched["cutoff"] = inputs.get("cutoff", 0)
        enriched["running_var"] = inputs.get("running_var", "running_var")
        enriched["treatment_var"] = inputs.get("treatment_var", "treatment")
        enriched["instruments"] = inputs.get("instruments", [])
        enriched["controls"] = actual_covariates
        enriched["covariates"] = actual_covariates
        enriched["intensity_col"] = inputs.get("intensity_col", "treatment_intensity")
        enriched["treatment_time"] = inputs.get("first_treated", inputs.get("treatment_time", 0))
        enriched["assignment_col"] = inputs.get("assignment_col", "assignment")
        enriched_assumptions.append(enriched)

    for a in enriched_assumptions:
        name = a.get("name", "")
        testable = a.get("testable", False)
        test_method = a.get("test_method", "")

        if not testable:
            verdicts.append(AssumptionVerdict(
                assumption_name=name,
                description=a.get("description", ""),
                testable=False,
                test_method=test_method,
                diagnostic=None,
                verdict="UNCERTAIN",
                reasoning="This assumption cannot be empirically tested. It must be argued from institutional knowledge or study design.",
            ))
            continue

        diag = _diagnose_assumption(
            a, mechanism, data_path, df, output_dir or str(AUTO_DIR),
            entity_col, time_col, outcome_col_actual,
        )

        if diag is None:
            verdicts.append(AssumptionVerdict(
                assumption_name=name,
                description=a.get("description", ""),
                testable=True,
                test_method=test_method,
                diagnostic=None,
                verdict="UNCERTAIN",
                reasoning="No diagnostic was dispatched for this testable assumption.",
            ))
            continue

        verdict = AssumptionVerdict(
            assumption_name=name,
            description=a.get("description", ""),
            testable=True,
            test_method=test_method,
            diagnostic=diag,
            verdict=diag.status if diag.status in ("PASS", "FAIL") else "UNCERTAIN",
            reasoning=diag.interpretation,
        )
        verdicts.append(verdict)
        if diag.status == "FAIL":
            has_failure = True

    # ── Step 3: Handle failures via fallback chain ──
    fallback_attempts = []
    final_method = theoretical_method
    method_changed = False
    chain = [theoretical_method]
    gap_explanation = ""

    if has_failure:
        selected, attempts = _resolve_fallback_chain(
            [v for v in verdicts if v.verdict == "FAIL"],
            fallbacks, data_path, df, stage3_output,
            output_dir or str(AUTO_DIR), entity_col, time_col, outcome_col_actual,
            first_treated_col, treated_col,
        )
        fallback_attempts = attempts

        if selected is not None:
            final_method = selected.method_name
            method_changed = True
            chain.append(final_method)
            failed_names = [v.assumption_name for v in verdicts if v.verdict == "FAIL"]
            gap_explanation = (
                f"{theoretical_method} was theoretically preferred, but data revealed: "
                + ", ".join(failed_names)
                + f". Falling back to {final_method}. "
                + f"Relaxes: {selected.assumption_relaxed}"
            )
        else:
            failed_names = [v.assumption_name for v in verdicts if v.verdict == "FAIL"]
            gap_explanation = (
                f"{theoretical_method} failed diagnostics ({', '.join(failed_names)}). "
                "No fallback method passed its diagnostics."
            )
            # Keep theoretical method but flag major limitation
            final_method = theoretical_method

    # ── Step 4: Build specification for Stage 7 ──
    spec = _build_specification(final_method, inputs, rec, entity_col, time_col,
                                outcome_col_actual, first_treated_col, treated_col)

    # ── Step 5: Assess causal claim strength ──
    n_pass = sum(1 for v in verdicts if v.verdict == "PASS")
    n_fail = sum(1 for v in verdicts if v.verdict == "FAIL")
    n_total = len(verdicts)

    if n_fail == 0 and n_pass >= n_total * 0.75 and not method_changed:
        strength = "strong"
    elif n_fail == 0 or (method_changed and n_fail == 0):
        strength = "moderate"
    elif n_fail <= 2 and method_changed:
        strength = "suggestive"
    else:
        strength = "not identifiable"

    # ── Step 6: Collect warnings and limitations ──
    warnings = list(rec.get("warnings", []))
    limitations = []

    for v in verdicts:
        if v.verdict == "UNCERTAIN" and v.testable:
            warnings.append(f"Could not verify '{v.assumption_name}'. {v.reasoning}")
        elif v.verdict == "UNCERTAIN" and not v.testable:
            limitations.append(f"'{v.assumption_name}' is not testable. {v.reasoning}")

    if method_changed:
        limitations.append(f"Method downgraded from {theoretical_method} to {final_method}. {gap_explanation}")

    # CIA-based methods need Oster bounds in Stage 8
    if any("unconfoundedness" in v.assumption_name.lower() for v in verdicts):
        warnings.append(
            "Final method relies on unconfoundedness (CIA) which is not testable. "
            "Oster bounds and coefficient stability checks (Stage 8 sensitivity_analysis.py) "
            "are essential for assessing result credibility."
        )

    return MethodConfirmation(
        mechanism=mechanism,
        theoretical_method=theoretical_method,
        final_method=final_method,
        method_changed=method_changed,
        chain=chain,
        gap_explanation=gap_explanation,
        assumption_verdicts=verdicts,
        fallback_attempts=fallback_attempts,
        data_quality_summary=data_quality,
        specification=spec,
        warnings=warnings,
        limitations=limitations,
        causal_claim_strength=strength,
    )


def _build_specification(final_method: str, inputs: dict, rec: dict,
                         entity_col: str, time_col: str, outcome_col: str,
                         first_treated_col: str, treated_col: str) -> dict:
    """Build the specification dict for Stage 7 consumption."""
    ml = final_method.lower()

    spec = {
        "entity_col": entity_col,
        "time_col": time_col,
        "outcome": outcome_col,
        "covariates": rec.get("optional_variables", []),
    }

    if any(k in ml for k in ("callaway", "staggered", "sun & abraham",
                              "borusyak", "de chaisemartin")):
        spec.update({
            "first_treated_col": first_treated_col,
            "control_type": "never-treated" if "never-treated" in final_method else "not-yet-treated",
            "method": "cs",
            "no_estimation_script": False,
        })
    elif "did" in ml or "twfe" in ml:
        spec.update({
            "treated_col": treated_col,
            "post_col": inputs.get("post_col", "post"),
            "method": "twfe",
            "no_estimation_script": False,
        })
    elif "rdd" in ml:
        spec.update({
            "running_var": inputs.get("running_var", "running_var"),
            "cutoff": inputs.get("cutoff", 0),
            "rdd_type": inputs.get("threshold_type", "sharp"),
            "method": "rdd",
            "no_estimation_script": False,
        })
    elif any(k in ml for k in ("iv", "2sls", "liml")):
        spec.update({
            "treatment_var": inputs.get("treatment_var", "treatment"),
            "instruments": inputs.get("instruments", []),
            "method": "iv",
            "no_estimation_script": False,
        })
    elif "synthetic did" in ml or "synthetic difference" in ml:
        spec.update({
            "treated_unit": inputs.get("treated_unit", ""),
            "first_treated": inputs.get("first_treated", 0),
            "method": "sdid",
            "no_estimation_script": False,
        })
    elif "scm" in ml or "synthetic control" in ml:
        spec.update({
            "treated_unit": inputs.get("treated_unit", ""),
            "first_treated": inputs.get("first_treated", 0),
            "method": "scm",
            "no_estimation_script": False,
        })
    elif any(k in ml for k in ("dml", "psm", "ipw", "aipw", "causal forest", "cem")):
        spec.update({
            "treatment_var": inputs.get("treatment_var", "treatment"),
            "method": "dml" if "dml" in ml else ("causal_forest" if "causal forest" in ml else "psm"),
            "no_estimation_script": "dml" not in ml and "causal forest" not in ml,
        })
    elif "ddd" in ml or "triple difference" in ml:
        spec.update({
            "treatment_a": inputs.get("policy_a_col", "treatment_a"),
            "treatment_b": inputs.get("policy_b_col", "treatment_b"),
            "method": "ddd",
            "no_estimation_script": True,
        })
    elif "intensity" in ml:
        spec.update({
            "intensity_col": inputs.get("intensity_col", "treatment_intensity"),
            "method": "intensity_did",
            "no_estimation_script": True,
        })
    elif "random" in ml:
        spec.update({
            "treatment_var": inputs.get("treatment_var", "treatment"),
            "method": "randomization",
            "no_estimation_script": True,
        })
    else:
        spec["method"] = "unknown"
        spec["no_estimation_script"] = True
        spec["note"] = f"No estimation script registered for: {final_method}"

    return spec


# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════

def confirmation_to_dict(conf: MethodConfirmation) -> dict:
    """Serialize MethodConfirmation to JSON-compatible dict."""
    return {
        "mechanism": conf.mechanism,
        "theoretical_method": conf.theoretical_method,
        "final_method": conf.final_method,
        "method_changed": conf.method_changed,
        "chain": conf.chain,
        "gap_explanation": conf.gap_explanation,
        "assumption_verdicts": [
            {
                "assumption_name": v.assumption_name,
                "description": v.description,
                "testable": v.testable,
                "test_method": v.test_method,
                "diagnostic": {
                    "diagnostic_name": v.diagnostic.diagnostic_name,
                    "test_description": v.diagnostic.test_description,
                    "status": v.diagnostic.status,
                    "values": v.diagnostic.values,
                    "threshold": v.diagnostic.threshold,
                    "interpretation": v.diagnostic.interpretation,
                } if v.diagnostic else None,
                "verdict": v.verdict,
                "reasoning": v.reasoning,
            }
            for v in conf.assumption_verdicts
        ],
        "fallback_attempts": [
            {
                "method_name": fa.method_name,
                "condition": fa.condition,
                "assumption_relaxed": fa.assumption_relaxed,
                "diagnoses": [
                    {
                        "diagnostic_name": d.diagnostic_name,
                        "status": d.status,
                        "values": d.values,
                        "interpretation": d.interpretation,
                    }
                    for d in fa.diagnoses
                ],
                "outcome": fa.outcome,
                "rejection_reason": fa.rejection_reason,
            }
            for fa in conf.fallback_attempts
        ],
        "data_quality_summary": conf.data_quality_summary,
        "specification": conf.specification,
        "warnings": conf.warnings,
        "limitations": conf.limitations,
        "causal_claim_strength": conf.causal_claim_strength,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Final Method Confirmation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From pipeline state
  python stage6_confirm.py --stage3 data/auto/stage3_result.json \\
      --data data/merged/panel.dta --output data/auto/stage6_confirmation.json

  # With validation report
  python stage6_confirm.py --stage3 data/auto/stage3_result.json \\
      --data data/merged/panel.dta --validate-report data/auto/validation_report.json \\
      --output data/auto/stage6_confirmation.json
""")
    parser.add_argument("--stage3", required=True, help="Path to Stage 3 output JSON")
    parser.add_argument("--data", required=True, help="Path to analysis-ready panel data")
    parser.add_argument("--output", required=True, help="Output path for confirmation JSON")
    parser.add_argument("--validate-report", default=None,
                        help="Path to cached validate_data.py report")
    parser.add_argument("--data-status", default=None,
                        help="JSON string of Stage 5 data_status dict")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    args = parser.parse_args()

    # Load Stage 3 output
    with open(args.stage3, encoding="utf-8") as f:
        stage3_output = json.load(f)

    # Load optional validation report
    validate_report = None
    if args.validate_report and Path(args.validate_report).exists():
        validate_report = {"path": args.validate_report}

    # Load optional data status
    data_status = None
    if args.data_status:
        try:
            data_status = json.loads(args.data_status)
        except json.JSONDecodeError:
            pass

    print(f"\n{'='*60}")
    print("Stage 6: Final Method Confirmation")
    print(f"{'='*60}")
    print(f"Theoretical method: {stage3_output.get('recommendation', {}).get('primary_method', 'Unknown')}")
    print(f"Mechanism: {stage3_output.get('mechanism', 'unknown')}")
    print(f"Data: {args.data}")
    print()

    output_dir = str(Path(args.output).parent)
    confirmation = confirm_method(
        stage3_output=stage3_output,
        data_path=args.data,
        validate_report=validate_report,
        data_status=data_status,
        entity_col=args.entity,
        time_col=args.time,
        output_dir=output_dir,
    )

    # Print summary
    print("\n─── Assumption Verification ───")
    for v in confirmation.assumption_verdicts:
        symbol = {"PASS": "✓", "FAIL": "✗", "UNCERTAIN": "?"}.get(v.verdict, "?")
        print(f"  {symbol} {v.assumption_name}: {v.verdict}")
        if v.reasoning:
            print(f"      {v.reasoning}")

    if confirmation.fallback_attempts:
        print("\n─── Fallback Chain ───")
        for fa in confirmation.fallback_attempts:
            outcome_sym = {"selected": "→", "rejected": "✗", "skipped": "○"}.get(fa.outcome, "?")
            print(f"  {outcome_sym} {fa.method_name}: {fa.outcome}")
            if fa.rejection_reason:
                print(f"      {fa.rejection_reason}")

    print(f"\n─── Final Decision ───")
    print(f"Method chain: {' → '.join(confirmation.chain)}")
    if confirmation.method_changed:
        print(f"Method changed: Yes — {confirmation.gap_explanation}")
    else:
        print("Method changed: No — theoretical method confirmed.")
    print(f"Causal claim strength: {confirmation.causal_claim_strength}")

    if confirmation.warnings:
        print(f"\n─── Warnings ───")
        for w in confirmation.warnings:
            print(f"  ⚠ {w}")

    if confirmation.limitations:
        print(f"\n─── Limitations ───")
        for lim in confirmation.limitations:
            print(f"  • {lim}")

    # Save
    output = confirmation_to_dict(confirmation)
    output["status"] = "completed"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nConfirmation saved to {args.output}")


if __name__ == "__main__":
    main()
