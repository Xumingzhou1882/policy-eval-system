"""
Data validation for policy evaluation datasets.

Post-acquisition quality checks to run before estimation:
  - Panel balance and structure
  - Missing value patterns
  - Outlier detection
  - Duplicate entity-time detection
  - Variable type and range validation
  - Treatment variable consistency
  - Pre-treatment data sufficiency

Usage:
    python scripts/validate_data.py --data data/merged/panel.dta \\
                                    --entity city_id --time year \\
                                    --outcome log_fertility \\
                                    --treated treated \\
                                    --first-treated first_treated \\
                                    --output validation_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    elif p.suffix == ".json":
        return pd.read_json(str(p))
    raise ValueError(f"Unsupported format: {p.suffix}")


# ═══════════════════════════════════════════════════════════════════════
# 1. Panel structure checks
# ═══════════════════════════════════════════════════════════════════════

def check_panel_structure(df: pd.DataFrame, entity_col: str,
                          time_col: str) -> dict:
    """Check basic panel structure: balance, dimensions, gaps."""
    n_rows = len(df)
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()

    # Expected rows for balanced panel
    expected = n_entities * n_periods
    balanced = n_rows == expected
    missing_pct = (1 - n_rows / expected) * 100 if expected > 0 else 0

    # Check for entity-time duplicates
    dup_mask = df.duplicated(subset=[entity_col, time_col], keep=False)
    n_duplicates = int(dup_mask.sum())
    duplicate_examples = []
    if n_duplicates > 0:
        dup_groups = df[dup_mask].groupby([entity_col, time_col]).size()
        duplicate_examples = [
            {"entity": str(k[0]), "time": str(k[1]), "count": int(v)}
            for k, v in dup_groups.head(10).items()
        ]

    # Time gaps: for each entity, are periods consecutive?
    time_values = sorted(df[time_col].unique())
    time_diffs = np.diff(time_values)
    expected_interval = np.median(time_diffs) if len(time_diffs) > 0 else 1
    irregular_intervals = [
        {"from": time_values[i], "to": time_values[i + 1], "gap": float(diff)}
        for i, diff in enumerate(time_diffs)
        if abs(diff - expected_interval) > expected_interval * 0.1
    ]

    # Entity presence: do all entities appear in all periods?
    entity_periods = df.groupby(entity_col)[time_col].apply(set)
    full_period_set = set(time_values)
    incomplete_entities = {
        str(e): sorted(list(full_period_set - entity_periods[e]))
        for e in df[entity_col].unique()
        if len(entity_periods[e]) < n_periods
    }

    return {
        "n_rows": n_rows,
        "n_entities": n_entities,
        "n_periods": n_periods,
        "expected_rows": expected,
        "balanced": balanced,
        "missing_pct": round(missing_pct, 1),
        "n_duplicates": n_duplicates,
        "duplicate_examples": duplicate_examples[:5],
        "time_range": [min(time_values), max(time_values)],
        "expected_interval": float(expected_interval),
        "irregular_intervals": irregular_intervals[:5],
        "n_entities_incomplete": len(incomplete_entities),
        "issues": [
            "Panel is unbalanced" if not balanced else None,
            f"{n_duplicates} duplicate entity-time rows" if n_duplicates > 0 else None,
            f"{len(irregular_intervals)} irregular time intervals" if irregular_intervals else None,
            f"{len(incomplete_entities)} entities with missing periods" if incomplete_entities else None,
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Missing value analysis
# ═══════════════════════════════════════════════════════════════════════

def check_missing_values(df: pd.DataFrame, key_vars: list[str]) -> dict:
    """Analyze missing value patterns for key variables."""
    var_stats = {}
    for var in key_vars:
        if var in df.columns:
            missing = df[var].isna().sum()
            missing_pct = missing / len(df) * 100
            var_stats[var] = {
                "missing_count": int(missing),
                "missing_pct": round(missing_pct, 1),
                "complete_count": int(len(df) - missing),
            }

    # Joint missingness: which observations are missing the most key vars?
    key_present = [v for v in key_vars if v in df.columns]
    if key_present:
        joint_missing = df[key_present].isna().sum(axis=1)
        complete_rows = int((joint_missing == 0).sum())
        all_missing = int((joint_missing == len(key_present)).sum())

        # Missing by time period
        missing_by_time = {}
        time_col_candidates = [c for c in df.columns if "year" in c.lower() or "time" in c.lower()]
        if time_col_candidates:
            tc = time_col_candidates[0]
            if tc in df.columns:
                for t in sorted(df[tc].unique()):
                    mask = df[tc] == t
                    missing_by_time[str(t)] = {
                        "total": int(mask.sum()),
                        "complete": int((~df.loc[mask, key_present].isna().any(axis=1)).sum()),
                    }
    else:
        complete_rows = len(df)
        all_missing = 0
        missing_by_time = {}

    return {
        "variable_stats": var_stats,
        "n_complete_rows": complete_rows,
        "n_all_missing": all_missing,
        "complete_pct": round(complete_rows / len(df) * 100, 1) if len(df) > 0 else 0,
        "missing_by_time": missing_by_time,
        "issues": [
            f"{v}: {s['missing_pct']:.0f}% missing" for v, s in var_stats.items()
            if s["missing_pct"] > 5
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. Outlier detection
# ═══════════════════════════════════════════════════════════════════════

def check_outliers(df: pd.DataFrame, vars_to_check: list[str],
                   method: str = "iqr", threshold: float = 3.0) -> dict:
    """Detect outliers in continuous variables using IQR or z-score."""
    outlier_summary = {}

    for var in vars_to_check:
        if var not in df.columns:
            continue
        data = df[var].dropna()
        if len(data) < 10:
            continue

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            n_low = int((data < lower).sum())
            n_high = int((data > upper).sum())
        else:  # z-score
            z = (data - data.mean()) / data.std()
            n_low = int((z < -threshold).sum())
            n_high = int((z > threshold).sum())

        outlier_summary[var] = {
            "n": len(data),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "p1": float(data.quantile(0.01)),
            "p99": float(data.quantile(0.99)),
            "n_outliers_low": n_low,
            "n_outliers_high": n_high,
            "outlier_pct": round((n_low + n_high) / len(data) * 100, 1),
        }

    flagged = [v for v, s in outlier_summary.items() if s["outlier_pct"] > 1]

    return {
        "method": f"{method} (threshold={threshold})",
        "variables": outlier_summary,
        "flagged_variables": flagged,
        "issues": [
            f"{v}: {s['outlier_pct']:.1f}% outliers (min={s['min']:.2f}, max={s['max']:.2f})"
            for v, s in outlier_summary.items() if s["outlier_pct"] > 1
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Variable type and range checks
# ═══════════════════════════════════════════════════════════════════════

def check_variable_types(df: pd.DataFrame, outcome: str,
                         treated_col: str = None,
                         entity_col: str = None,
                         time_col: str = None) -> dict:
    """Verify variable types and reasonable ranges."""
    checks = []

    # Outcome should be numeric
    if outcome in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[outcome])
        checks.append({
            "variable": outcome,
            "check": "numeric type",
            "passed": is_numeric,
            "detail": f"dtype={df[outcome].dtype}",
        })

    # Treatment should be 0/1 or boolean
    if treated_col and treated_col in df.columns:
        values = df[treated_col].dropna().unique()
        is_binary = set(values).issubset({0, 1, 0.0, 1.0, True, False})
        checks.append({
            "variable": treated_col,
            "check": "binary (0/1)",
            "passed": is_binary,
            "detail": f"unique values: {sorted(values)[:10]}",
        })

    # Entity ID should not be null
    if entity_col and entity_col in df.columns:
        has_null = df[entity_col].isna().any()
        checks.append({
            "variable": entity_col,
            "check": "no null IDs",
            "passed": not has_null,
            "detail": f"{int(df[entity_col].isna().sum())} nulls" if has_null else "0 nulls",
        })

    # Time should be numeric or datetime
    if time_col and time_col in df.columns:
        is_time = pd.api.types.is_numeric_dtype(df[time_col]) or pd.api.types.is_datetime64_dtype(df[time_col])
        checks.append({
            "variable": time_col,
            "check": "numeric or datetime",
            "passed": is_time,
            "detail": f"dtype={df[time_col].dtype}, range=[{df[time_col].min()}, {df[time_col].max()}]",
        })

    # Constant variables
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            checks.append({
                "variable": col,
                "check": "not constant",
                "passed": False,
                "detail": f"Only {df[col].nunique()} unique value(s) — column is constant.",
            })

    failed = [c for c in checks if not c["passed"]]
    return {
        "checks": checks,
        "n_failed": len(failed),
        "issues": [f"{c['variable']}: {c['check']} — {c['detail']}" for c in failed],
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. Treatment variable consistency
# ═══════════════════════════════════════════════════════════════════════

def check_treatment_consistency(df: pd.DataFrame, entity_col: str,
                                time_col: str, treated_col: str = None,
                                first_treated_col: str = None,
                                post_col: str = None) -> dict:
    """Verify logical consistency of treatment-related variables."""
    issues = []

    # treated should be time-invariant (once treated, always treated)
    if treated_col and treated_col in df.columns:
        treated_switches = df.groupby(entity_col)[treated_col].nunique()
        switching = treated_switches[treated_switches > 1]
        if len(switching) > 0:
            issues.append(f"{len(switching)} entities have non-constant treatment status.")

    # first_treated should be unique per entity
    if first_treated_col and first_treated_col in df.columns:
        ft_unique = df.groupby(entity_col)[first_treated_col].nunique()
        inconsistent = ft_unique[ft_unique > 1]
        if len(inconsistent) > 0:
            issues.append(f"{len(inconsistent)} entities have multiple first-treatment years.")

    # Post should be 0 before first_treated
    if first_treated_col and post_col and post_col in df.columns and first_treated_col in df.columns:
        pre = df[df[time_col] < df[first_treated_col]]
        post_violations = (pre[post_col] == 1).sum()
        if post_violations > 0:
            issues.append(f"{int(post_violations)} rows have post=1 before first_treated.")

    # Check never-treated entities have no first_treated
    if treated_col and first_treated_col and treated_col in df.columns and first_treated_col in df.columns:
        never_treated = df[df[treated_col] == 0]
        has_ft = never_treated[first_treated_col].notna().sum()
        if has_ft > 0:
            issues.append(f"{int(has_ft)} never-treated rows have a first_treated value.")

    # Check for entities treated at the very end of the panel
    if first_treated_col and first_treated_col in df.columns:
        max_time = df[time_col].max()
        late_treated = df[df[first_treated_col] >= max_time]
        n_late = late_treated[entity_col].nunique()
        if n_late > 0:
            issues.append(f"{n_late} entities treated at or after the last time period — no post-treatment data.")

    return {
        "issues": issues,
        "passed": len(issues) == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. Pre-treatment data sufficiency
# ═══════════════════════════════════════════════════════════════════════

def check_pretreatment_sufficiency(df: pd.DataFrame, entity_col: str,
                                   time_col: str, outcome: str,
                                   first_treated_col: str,
                                   min_pre_periods: int = 2) -> dict:
    """Check if there's enough pre-treatment data for DID/event study."""
    issues = []

    if first_treated_col not in df.columns:
        return {"issues": ["No first_treated column — cannot check pre-treatment data."], "passed": False}

    treated = df[df[first_treated_col].notna()]

    entity_pre_counts = {}
    for entity in treated[entity_col].unique():
        entity_data = df[df[entity_col] == entity]
        ft = entity_data[first_treated_col].iloc[0]
        pre_periods = entity_data[entity_data[time_col] < ft]
        n_pre = len(pre_periods)
        n_pre_outcome = pre_periods[outcome].notna().sum()
        entity_pre_counts[str(entity)] = {
            "first_treated": float(ft),
            "n_pre_periods": n_pre,
            "n_pre_outcome_nonmissing": int(n_pre_outcome),
        }

    low_pre = {e: v for e, v in entity_pre_counts.items()
               if v["n_pre_outcome_nonmissing"] < min_pre_periods}
    no_pre = {e: v for e, v in entity_pre_counts.items()
              if v["n_pre_outcome_nonmissing"] == 0}

    if no_pre:
        issues.append(f"{len(no_pre)} entities have NO pre-treatment outcome data — cannot be used in DID.")
    if low_pre:
        issues.append(f"{len(low_pre)} entities have fewer than {min_pre_periods} pre-treatment periods.")

    return {
        "entities_with_no_pre_data": len(no_pre),
        "entities_with_low_pre_data": len(low_pre),
        "min_pre_periods_required": min_pre_periods,
        "passed": len(no_pre) == 0,
        "issues": issues,
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. Comprehensive validation
# ═══════════════════════════════════════════════════════════════════════

def validate_all(df: pd.DataFrame, entity_col: str, time_col: str,
                 outcome: str, treated_col: str = None,
                 first_treated_col: str = None,
                 post_col: str = None,
                 controls: list[str] = None) -> dict:
    """Run all validation checks and produce a comprehensive report."""

    report = {}

    # 1. Panel structure
    report["panel_structure"] = check_panel_structure(df, entity_col, time_col)

    # 2. Missing values (outcome + treatment + controls)
    key_vars = [outcome]
    if treated_col:
        key_vars.append(treated_col)
    if first_treated_col:
        key_vars.append(first_treated_col)
    if controls:
        key_vars.extend(controls)
    report["missing_values"] = check_missing_values(df, key_vars)

    # 3. Outliers (outcome + controls)
    numeric_vars = [outcome] + (controls or [])
    numeric_vars = [v for v in numeric_vars
                    if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
    report["outliers"] = check_outliers(df, numeric_vars)

    # 4. Variable types
    report["variable_types"] = check_variable_types(
        df, outcome, treated_col, entity_col, time_col
    )

    # 5. Treatment consistency
    report["treatment_consistency"] = check_treatment_consistency(
        df, entity_col, time_col, treated_col, first_treated_col, post_col
    )

    # 6. Pre-treatment sufficiency
    if first_treated_col:
        report["pretreatment_sufficiency"] = check_pretreatment_sufficiency(
            df, entity_col, time_col, outcome, first_treated_col
        )

    # Summary
    all_issues = []
    for section in report.values():
        if isinstance(section, dict) and "issues" in section:
            issues = section["issues"]
            if isinstance(issues, list):
                all_issues.extend([i for i in issues if i])

    critical = [i for i in all_issues if "no pre-treatment" in str(i).lower()
                or "constant" in str(i).lower()
                or "duplicate" in str(i).lower()]

    report["summary"] = {
        "total_issues": len(all_issues),
        "critical_issues": len(critical),
        "critical_details": critical,
        "overall": "FAIL" if critical else ("WARN" if all_issues else "PASS"),
    }

    return report


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data validation for policy evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to panel data")
    parser.add_argument("--entity", default="city_id", help="Entity ID column")
    parser.add_argument("--time", default="year", help="Time column")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--treated", default=None, help="Treatment dummy column")
    parser.add_argument("--first-treated", default=None, help="First-treated year column")
    parser.add_argument("--post", default=None, help="Post-treatment dummy column")
    parser.add_argument("--controls", nargs="*", default=[], help="Control variable names")
    parser.add_argument("--output", default=None, help="Output path (.json)")
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    report = validate_all(
        df, args.entity, args.time, args.outcome,
        treated_col=args.treated,
        first_treated_col=args.first_treated,
        post_col=args.post,
        controls=args.controls,
    )

    # Print report
    print("\n═══════════════════════════════════")
    print("Data Validation Report")
    print("═══════════════════════════════════")

    summary = report["summary"]
    status_symbol = "✗ FAIL" if summary["overall"] == "FAIL" else ("⚠ WARN" if summary["overall"] == "WARN" else "✓ PASS")
    print(f"\nOverall: {status_symbol}")
    print(f"  {summary['total_issues']} issues found ({summary['critical_issues']} critical)")

    if summary["critical_details"]:
        print("\nCritical issues:")
        for i in summary["critical_details"]:
            print(f"  ✗ {i}")

    sections = [
        ("Panel Structure", "panel_structure"),
        ("Missing Values", "missing_values"),
        ("Outliers", "outliers"),
        ("Variable Types", "variable_types"),
        ("Treatment Consistency", "treatment_consistency"),
        ("Pre-treatment Sufficiency", "pretreatment_sufficiency"),
    ]

    for title, key in sections:
        if key not in report:
            continue
        section = report[key]
        print(f"\n─── {title} ───")
        for field, value in section.items():
            if field == "issues":
                continue
            if isinstance(value, (int, float, str, bool)):
                print(f"  {field}: {value}")

        if "issues" in section:
            issues = section["issues"]
            if isinstance(issues, list):
                for issue in [i for i in issues if i]:
                    print(f"  ⚠ {issue}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
