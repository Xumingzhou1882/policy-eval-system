"""
Generate a final causal inference report from estimation results.

Usage:
    python scripts/output_report.py --stage3 stage3_method.json \\
                                    --stage6 stage6_final.json \\
                                    --stage7 main_result.json \\
                                    --stage8 robustness.json \\
                                    --policy "Long-term Care Insurance Pilot" \\
                                    --outcome "Fertility Rate" \\
                                    --output report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sig(p: float) -> str:
    if p < 0.01:
        return "*** (significant at 1%)"
    elif p < 0.05:
        return "** (significant at 5%)"
    elif p < 0.1:
        return "* (significant at 10%)"
    return "not statistically significant"


def _effect_direction(coef: float) -> str:
    if coef > 0:
        return "increased"
    elif coef < 0:
        return "decreased"
    return "had no measurable effect on"


def _check_label(passed: bool) -> str:
    return "Passed" if passed else "Failed"


def generate_report(policy: str, outcome: str, stage3: dict = None,
                    stage6: dict = None, stage7: dict = None,
                    stage8: dict = None, data_source: str = "",
                    data_span: str = "", n_obs: str = "") -> str:
    """Generate a formatted causal inference report."""

    lines = []

    lines.append("═══════════════════════════════════")
    lines.append("Causal Inference Report")
    lines.append("═══════════════════════════════════")
    lines.append("")
    lines.append(f"Policy:  {policy}")
    lines.append(f"Outcome: {outcome}")

    # Method chain
    if stage3 and stage6:
        theory_method = stage3.get("primary_method", stage3.get("method", "N/A"))
        final_method = stage6.get("final_method", stage6.get("method", "N/A"))
        lines.append(f"Method chain: {theory_method} → {final_method}")
        if theory_method != final_method:
            reason = _get_method_switch_reason(stage6)
            if reason:
                lines.append(f"  Switch reason: {reason}")
    elif stage6:
        lines.append(f"Method: {stage6.get('final_method', stage6.get('method', 'N/A'))}")

    lines.append(f"Data:  {data_source}, {data_span}, N = {n_obs}")
    lines.append("")

    # Main result
    lines.append("─── Main Result ───")
    if stage7:
        coef = _extract_coef(stage7)
        se = _extract_se(stage7)
        pval = _extract_pval(stage7)

        direction = _effect_direction(coef)
        sig = _sig(pval)
        lines.append(f"The policy {direction} {outcome}.")
        effect_mag = _extract_effect_magnitude(stage7)
        if effect_mag:
            lines.append(f"Effect size: {effect_mag}")
        lines.append(f"Coefficient = {coef:.6f} (SE = {se:.6f}), p = {pval:.4f} [{sig}]")
    else:
        lines.append("[No estimation results provided.]")
    lines.append("")

    # Robustness
    lines.append("─── Robustness ───")
    if stage8 and "checks" in stage8:
        checks = stage8["checks"]
        n_pass = sum(1 for c in checks if c.get("passed", False))
        n_total = len(checks)
        lines.append(f"{n_pass}/{n_total} robustness checks passed.")
        lines.append("")
        for c in checks:
            status = "✓" if c.get("passed") else "✗"
            lines.append(f"{status} {c['name']}: {c.get('interpretation', '')}")
    elif stage8:
        lines.append(f"Robustness results: {json.dumps(stage8, indent=2)}")
    else:
        lines.append("[No robustness results provided.]")
    lines.append("")

    # Key assumption
    lines.append("─── Key Assumption ───")
    if stage6 and "assumptions" in stage6:
        for a in stage6["assumptions"]:
            status = "✓ Holds" if a.get("holds") else ("✗ Violated" if a.get("holds") is False else "? Unverified")
            lines.append(f"{a['name']}: {status} — {a.get('evidence', '')}")
    elif stage6 and "limitations" in stage6:
        lines.append(f"See limitations below.")
    else:
        lines.append("[No assumption verification provided.]")
    lines.append("")

    # Limitations
    lines.append("─── Limitations ───")
    limitations = []
    if stage6 and "limitations" in stage6:
        limitations.extend(stage6["limitations"])
    if stage8 and "limitations" in stage8:
        limitations.extend(stage8["limitations"])
    if limitations:
        for i, lim in enumerate(limitations, 1):
            lines.append(f"{i}. {lim}")
    else:
        lines.append("1. No explicit limitations documented.")
    lines.append("")

    # Causal claim strength
    lines.append("─── Causal Claim Strength ───")
    strength = _assess_strength(stage6, stage8)
    lines.append(f"[{strength['level']}]")
    lines.append(strength["justification"])
    lines.append("")

    lines.append("═══════════════════════════════════")

    return "\n".join(lines)


def _extract_coef(result: dict) -> float:
    """Extract main coefficient from various possible result structures."""
    for key in ["coefficient", "att", "overall_att", "late"]:
        if key in result and result[key] is not None:
            return float(result[key])
    return 0.0


def _extract_se(result: dict) -> float:
    for key in ["std_error", "se"]:
        if key in result:
            return float(result[key])
    return 0.0


def _extract_pval(result: dict) -> float:
    for key in ["p_value", "pval"]:
        if key in result:
            return float(result[key])
    return 1.0


def _extract_effect_magnitude(result: dict) -> str:
    coef = _extract_coef(result)
    # For log-transformed outcomes, interpret as % change
    outcome = result.get("outcome", "")
    if outcome.startswith("log_"):
        pct = (np.exp(coef) - 1) * 100
        return f"{pct:.2f}% (log-point interpretation)"
    return ""


def _get_method_switch_reason(stage6: dict) -> str:
    for key in ["switch_reason", "reason"]:
        if key in stage6:
            return stage6[key]
    return ""


def _assess_strength(stage6: dict = None, stage8: dict = None) -> dict:
    """
    Assess causal claim strength:
    - Strong: Theory = final method, all checks pass, key assumptions hold
    - Moderate: Minor method switch or some checks fail
    - Suggestive: Major method switch or critical assumptions violated
    """
    score = 0
    reasons = []

    # Method alignment
    if stage6:
        if stage6.get("method_switch", False) is False:
            score += 1
            reasons.append("Theoretical method was applied without compromise.")
        else:
            reasons.append("Method was adjusted due to data constraints (documented in method chain).")
    else:
        reasons.append("Method selection chain not documented.")

    # Assumptions
    if stage6 and "assumptions" in stage6:
        holds = sum(1 for a in stage6["assumptions"] if a.get("holds"))
        total = len(stage6["assumptions"])
        if total > 0 and holds == total:
            score += 1
            reasons.append("All testable assumptions hold.")
        else:
            reasons.append(f"Some assumptions could not be verified or were violated ({holds}/{total} hold).")

    # Robustness
    if stage8 and "checks" in stage8:
        checks = stage8["checks"]
        passed = sum(1 for c in checks if c.get("passed"))
        total = len(checks)
        if total > 0 and passed == total:
            score += 1
            reasons.append("All robustness checks passed.")
        elif total > 0:
            reasons.append(f"{passed}/{total} robustness checks passed — some sensitivity detected.")

    if score >= 3:
        return {"level": "Strong", "justification": " ".join(reasons)}
    elif score >= 2:
        return {"level": "Moderate", "justification": " ".join(reasons)}
    else:
        return {"level": "Suggestive", "justification": " ".join(reasons)}


def main():
    parser = argparse.ArgumentParser(description="Generate causal inference report")
    parser.add_argument("--policy", required=True, help="Policy name")
    parser.add_argument("--outcome", required=True, help="Outcome name")
    parser.add_argument("--stage3", default=None, help="Stage 3 theoretical method JSON")
    parser.add_argument("--stage6", default=None, help="Stage 6 final method confirmation JSON")
    parser.add_argument("--stage7", default=None, help="Stage 7 estimation results JSON")
    parser.add_argument("--stage8", default=None, help="Stage 8 robustness results JSON")
    parser.add_argument("--data-source", default="", help="Data source description")
    parser.add_argument("--data-span", default="", help="Data time span")
    parser.add_argument("--n-obs", default="", help="Number of observations")
    parser.add_argument("--output", default="report.md", help="Output path for report")
    args = parser.parse_args()

    stage3 = load_json(args.stage3) if args.stage3 else None
    stage6 = load_json(args.stage6) if args.stage6 else None
    stage7 = load_json(args.stage7) if args.stage7 else None
    stage8 = load_json(args.stage8) if args.stage8 else None

    report = generate_report(args.policy, args.outcome, stage3, stage6, stage7, stage8,
                             args.data_source, args.data_span, args.n_obs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
