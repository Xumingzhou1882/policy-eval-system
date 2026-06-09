"""
Stage 8 result consolidation.

Reads all Stage 8 check outputs (placebo, Bacon, sensitivity, alt windows)
and produces a unified pass/fail summary for Stage 9 report ingestion.

Usage:
    python scripts/stage8_combine.py \\
        --placebo data/auto/stage8_placebo.json \\
        --bacon data/auto/stage8_bacon.json \\
        --sensitivity data/auto/stage8_sensitivity.json \\
        --alt-windows data/auto/stage8_alt_windows.json \\
        --output data/auto/stage8_summary.json
"""

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _check_placebo(placebo: dict) -> dict:
    if not placebo:
        return {"name": "Placebo permutation test", "passed": None,
                "interpretation": "检验未运行。"}

    p_val = placebo.get("p_value", 1)
    actual = placebo.get("actual_estimate", 0)
    placebo_mean = placebo.get("placebo_mean", 0)
    passed = p_val < 0.05

    return {
        "name": "安慰剂置换检验",
        "passed": passed,
        "values": {
            "actual_coef": round(actual, 6),
            "placebo_mean": round(placebo_mean, 6),
            "p_value": round(p_val, 4),
        },
        "interpretation": (
            f"实际效应{'显著区别于' if passed else '未显著区别于'}安慰剂分布"
            f"（p = {p_val:.4f}），结果{'非' if passed else '可能为'}随机噪声。"
        ),
    }


def _check_bacon(bacon: dict) -> dict:
    if not bacon:
        return {"name": "Bacon分解", "passed": None,
                "interpretation": "检验未运行（非交错DID设计，无需此检验）。"}

    neg_pct = bacon.get("negative_weight_pct", 0)
    n_comps = bacon.get("n_comparisons", 0)
    passed = neg_pct <= 10

    return {
        "name": "Goodman-Bacon分解",
        "passed": passed,
        "values": {
            "n_comparisons": n_comps,
            "negative_weight_pct": round(neg_pct, 1),
        },
        "interpretation": (
            f"共{n_comps}个2×2比较，负权重{neg_pct:.1f}%。"
            f"{'通过。' if passed else '负权重>10%，TWFE不可靠。'}"
        ),
    }


def _check_sensitivity(sensitivity: dict) -> list[dict]:
    checks = []
    if not sensitivity:
        return [{"name": "敏感性分析", "passed": None,
                 "interpretation": "检验未运行。"}]

    # Oster bounds
    oster = sensitivity.get("oster", {})
    if oster:
        delta = oster.get("delta", 0)
        delta_passed = delta > 1
        checks.append({
            "name": "Oster遗漏变量检验",
            "passed": delta_passed,
            "values": {"delta": round(delta, 2)},
            "interpretation": (
                f"δ = {delta:.2f}，"
                f"{'通过（δ>1）。' if delta_passed else '未通过（δ≤1）。'}"
            ),
        })

    # Coefficient stability
    stability = sensitivity.get("coefficient_stability", {})
    if stability:
        coef_ratio = stability.get("ratio", 0)
        stab_passed = abs(1 - coef_ratio) < 0.3
        checks.append({
            "name": "系数稳定性检验",
            "passed": stab_passed,
            "values": {"ratio": round(coef_ratio, 3)},
            "interpretation": (
                f"系数比 = {coef_ratio:.3f}，"
                f"{'通过。' if stab_passed else '系数变化较大。'}"
            ),
        })

    # Placebo-in-time
    pit = sensitivity.get("placebo_in_time", {})
    if pit:
        pit_passed = pit.get("passed", True)
        checks.append({
            "name": "时间安慰剂检验",
            "passed": pit_passed,
            "values": pit.get("values", {}),
            "interpretation": (
                "前移处理时间后未发现显著虚假效应。" if pit_passed
                else "前移处理时间后出现显著效应。"
            ),
        })

    # Leave-one-out
    loo = sensitivity.get("leave_one_out", {})
    if loo:
        loo_min = loo.get("min_coef", 0)
        loo_max = loo.get("max_coef", 0)
        loo_baseline = loo.get("baseline_coef", 0)
        loo_range = abs(loo_max - loo_min)
        loo_passed = loo_range < abs(loo_baseline) if loo_baseline != 0 else True
        checks.append({
            "name": "留一法影响分析",
            "passed": loo_passed,
            "values": {
                "min_coef": round(loo_min, 6),
                "max_coef": round(loo_max, 6),
                "range": round(loo_range, 6),
            },
            "interpretation": (
                f"系数范围 [{loo_min:.4f}, {loo_max:.4f}]，"
                f"{'没有单一城市驱动结果。' if loo_passed else '个别城市影响较大。'}"
            ),
        })

    # Rosenbaum bounds
    rb = sensitivity.get("rosenbaum", {})
    if rb:
        gamma = rb.get("gamma", 1)
        rb_passed = gamma > 2
        checks.append({
            "name": "Rosenbaum边界分析",
            "passed": rb_passed,
            "values": {"gamma": round(gamma, 2)},
            "interpretation": (
                f"Γ = {gamma:.2f}，"
                f"{'通过（Γ>2）。' if rb_passed else '未通过（Γ≤2）。'}"
            ),
        })

    return checks


def _check_alt_windows(alt: dict) -> dict:
    if not alt:
        return {"name": "替代时间窗口检验", "passed": None,
                "interpretation": "检验未运行。"}

    stability = alt.get("stability", {})
    passed = stability.get("stable", False)
    baseline = alt.get("baseline", {})
    windows = alt.get("windows", [])
    n_windows = len([w for w in windows if w.get("coefficient") is not None])

    return {
        "name": "替代时间窗口检验",
        "passed": passed,
        "values": {
            "baseline_coef": round(baseline.get("coefficient", 0), 4),
            "n_valid_windows": n_windows,
        },
        "interpretation": (
            f"{n_windows}个替代窗口下"
            f"{'系数方向和显著性一致，通过。' if passed else '存在不稳定。'}"
        ),
    }


def combine(placebo, bacon, sensitivity, alt_windows) -> dict:
    checks = []

    # Placebo
    checks.append(_check_placebo(placebo))

    # Bacon (only for staggered)
    if bacon:
        checks.append(_check_bacon(bacon))

    # Alternative windows
    checks.append(_check_alt_windows(alt_windows))

    # Sensitivity sub-checks
    sens_checks = _check_sensitivity(sensitivity)
    checks.extend(sens_checks)

    # Summary
    run_checks = [c for c in checks if c["passed"] is not None]
    n_pass = sum(1 for c in run_checks if c["passed"])
    n_fail = sum(1 for c in run_checks if not c["passed"])
    n_total = len(run_checks)
    n_not_run = len(checks) - n_total

    overall_pass = n_fail == 0 and n_pass > 0

    return {
        "checks": checks,
        "summary": {
            "total_checks": len(checks),
            "checks_run": n_total,
            "passed": n_pass,
            "failed": n_fail,
            "not_run": n_not_run,
            "overall_pass": overall_pass,
            "interpretation": (
                f"共{len(checks)}项稳健性检验，其中{n_total}项已完成："
                f"{n_pass}项通过，{n_fail}项未通过。"
                f"{'总体结果稳健。' if overall_pass else '存在未通过的检验项，结果需谨慎解读。'}"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate Stage 8 robustness check results")
    parser.add_argument("--placebo", default=None, help="Placebo test JSON")
    parser.add_argument("--bacon", default=None, help="Bacon decomposition JSON")
    parser.add_argument("--sensitivity", default=None, help="Sensitivity analysis JSON")
    parser.add_argument("--alt-windows", default=None, help="Alt windows JSON")
    parser.add_argument("--output", required=True, help="Output path for summary JSON")
    args = parser.parse_args()

    placebo = load_json(args.placebo) if args.placebo else None
    bacon = load_json(args.bacon) if args.bacon else None
    sensitivity = load_json(args.sensitivity) if args.sensitivity else None
    alt_windows = load_json(args.alt_windows) if args.alt_windows else None

    result = combine(placebo, bacon, sensitivity, alt_windows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print summary
    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"Stage 8 Summary: {s['passed']}/{s['checks_run']} passed"
          + (f" ({s['not_run']} not run)" if s['not_run'] else ""))
    print(f"{'='*60}")
    for c in result["checks"]:
        if c["passed"] is True:
            icon = "✓"
        elif c["passed"] is False:
            icon = "✗"
        else:
            icon = "—"
        print(f"  {icon} {c['name']}")
        print(f"    {c['interpretation']}")
    print(f"\n{s['interpretation']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
