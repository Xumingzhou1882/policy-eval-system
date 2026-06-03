"""
Merge and clean panel data from multiple sources into analysis-ready format.

Handles:
  - Entity ID normalization (北京市 → 110000) via GB/T 2260 codes
  - Fuzzy matching for unmatched names (edit distance, substring)
  - Raw file format validation
  - Multi-source merging with consistent keys
  - Balance checks and missing-value reports

Usage:
    python scripts/clean_panel.py --outcome data/auto/fertility.json \\
                                  --treatment data/manual/pilot_dates.csv \\
                                  --controls data/auto/gdp.json data/auto/population.json \\
                                  --entity-col city_id --time-col year \\
                                  --output data/merged/panel.dta
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Entity ID normalization
# ═══════════════════════════════════════════════════════════════════════

def load_entity_map(map_path: str = None) -> dict[str, str]:
    """
    Load entity ID mapping table. Maps alternative names to canonical IDs.

    Default location: data/auto/entity_map.json
    Format: {"alternative_name": "canonical_id", ...}
    """
    if map_path is None:
        skill_root = Path(__file__).resolve().parent.parent
        map_path = skill_root / "data" / "auto" / "entity_map.json"

    map_path = Path(map_path)
    if map_path.exists():
        with open(map_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _clean_name(name: str) -> str:
    """Strip punctuation, whitespace, and common suffixes from a name."""
    name = str(name).strip()
    # Remove parentheses and their content: "通化市(地级)" → "通化市"
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'（[^）]*）', '', name)
    # Remove autonomous prefecture/county suffixes (longest first)
    ethnic_suffixes = [
        "土家族苗族自治州", "苗族侗族自治州", "布依族苗族自治州",
        "傣族景颇族自治州", "蒙古族藏族自治州", "藏族羌族自治州",
        "哈尼族彝族自治州", "傈僳族自治州", "柯尔克孜自治州",
        "哈萨克自治州", "蒙古自治州", "彝族自治州", "藏族自治州",
        "回族自治州", "壮族自治州", "朝鲜族自治州", "白族自治州",
        "自治州", "自治县",
    ]
    for sfx in ethnic_suffixes:
        if name.endswith(sfx) and len(name) > len(sfx) + 1:
            name = name[:-len(sfx)]
            break
    # Remove common region suffixes (longest first)
    region_suffixes = [
        "特别行政区", "经济技术开发区", "高新技术产业开发区",
        "自治区", "开发区", "新区", "地区", "市", "省", "州", "盟",
    ]
    for sfx in region_suffixes:
        if name.endswith(sfx) and len(name) > len(sfx) + 1:
            name = name[:-len(sfx)]
    return name.strip()


def _fuzzy_match(name: str, candidates: dict[str, str],
                 min_score: float = 0.75) -> Optional[str]:
    """
    Find the best matching canonical code for an unmatched name.

    Uses:
      1. Exact match on cleaned names
      2. Substring match (name contains candidate or vice versa)
      3. SequenceMatcher ratio
    """
    cleaned = _clean_name(name)
    if not cleaned:
        return None

    # 1. Exact match on cleaned name
    if cleaned in candidates:
        return candidates[cleaned]

    # 2. Substring matching
    for cand, code in candidates.items():
        cand_clean = _clean_name(cand)
        if len(cleaned) >= 2 and len(cand_clean) >= 2:
            if cleaned in cand_clean or cand_clean in cleaned:
                return code

    # 3. Character-level fuzzy match
    best_score = 0.0
    best_code = None
    for cand, code in candidates.items():
        cand_clean = _clean_name(cand)
        ratio = SequenceMatcher(None, cleaned, cand_clean).ratio()
        if ratio > best_score:
            best_score = ratio
            best_code = code

    if best_score >= min_score:
        return best_code

    return None


def normalize_entity_ids(df: pd.DataFrame, entity_col: str,
                         entity_map: dict[str, str],
                         auto_fuzzy: bool = True) -> pd.DataFrame:
    """
    Normalize entity IDs using a mapping table with fuzzy fallback.

    Steps:
      1. Direct lookup in entity_map
      2. Clean name and retry lookup
      3. Fuzzy match against known entities (if auto_fuzzy=True)
      4. Keep original as-is if no match

    Returns DataFrame with normalized entity_id column.
    Reports unmatched entities for manual review.
    """
    df = df.copy()
    original_ids = df[entity_col].astype(str).str.strip()

    # Build cleaned-name lookup from entity_map
    cleaned_candidates = {_clean_name(k): v for k, v in entity_map.items()}

    unmatched: set[str] = set()
    fuzzy_matched: dict[str, str] = {}  # original → matched_name
    normalized = []

    for eid in original_ids:
        if not eid or eid.lower() in ("nan", "none", "null", ""):
            normalized.append(eid)
            continue

        # Step 1: Direct lookup
        if eid in entity_map:
            normalized.append(entity_map[eid])
            continue

        # Step 2: Clean and retry
        cleaned = _clean_name(eid)
        if cleaned in entity_map:
            normalized.append(entity_map[cleaned])
            continue
        if cleaned in cleaned_candidates:
            normalized.append(cleaned_candidates[cleaned])
            continue

        # Step 3: Fuzzy match
        if auto_fuzzy:
            matched_code = _fuzzy_match(eid, entity_map)
            if matched_code:
                normalized.append(matched_code)
                fuzzy_matched[eid] = matched_code
                continue

        # No match
        normalized.append(eid)
        unmatched.add(eid)

    df[entity_col] = normalized

    if fuzzy_matched:
        print(f"  Fuzzy-matched {len(fuzzy_matched)} entities:")
        for orig, code in sorted(fuzzy_matched.items())[:10]:
            print(f"    {orig} → {code}")
        if len(fuzzy_matched) > 10:
            print(f"    ... and {len(fuzzy_matched) - 10} more")

    if unmatched:
        print(f"  ⚠ {len(unmatched)} unmatched entities:")
        for u in sorted(unmatched)[:15]:
            print(f"    '{u}'")
        if len(unmatched) > 15:
            print(f"    ... and {len(unmatched) - 15} more")

    return df


def generate_entity_map(dfs: list[pd.DataFrame],
                        entity_col: str) -> dict[str, str]:
    """
    Auto-generate entity map by matching non-numeric IDs to numeric codes
    across multiple DataFrames.

    Uses heuristics:
      - If one df has numeric codes and another has Chinese names,
        try to find the common entities by name pattern matching.
      - If multiple non-numeric IDs map to the same numeric code,
        they are aliases.
    """
    all_ids: set[str] = set()
    for df in dfs:
        if entity_col in df.columns:
            ids = df[entity_col].dropna().astype(str).str.strip()
            all_ids.update(ids.unique())

    numeric = {x for x in all_ids if re.match(r'^\d{4,6}(\.0)?$', x)}
    non_numeric = all_ids - numeric

    if not non_numeric or not numeric:
        return {}

    # Load canonical map for cross-reference
    canonical = load_entity_map()

    entity_map: dict[str, str] = {}

    for name in sorted(non_numeric):
        # Try canonical first
        if name in canonical:
            entity_map[name] = canonical[name]
            continue

        cleaned = _clean_name(name)
        if cleaned in canonical:
            entity_map[name] = canonical[cleaned]
            continue

        # Try to match against numeric codes in the data
        # Heuristic: check if any numeric code appears alongside this name
        # in the same data source (same DataFrame)
        for df in dfs:
            if entity_col not in df.columns:
                continue
            # Find rows where this name appears
            mask = df[entity_col].astype(str).str.strip() == name
            if not mask.any():
                continue
            # Check other columns for numeric identifiers
            for col in df.columns:
                if col == entity_col:
                    continue
                vals = df.loc[mask, col].dropna().astype(str).unique()
                for v in vals:
                    v_clean = v.replace(".0", "")
                    if re.match(r'^\d{6}$', v_clean):
                        entity_map[name] = v_clean
                        break

    return entity_map


# ═══════════════════════════════════════════════════════════════════════
# Raw file format validation
# ═══════════════════════════════════════════════════════════════════════

RAW_FILE_SPEC = """
Raw data files placed in data/raw/ must follow this format:

1. File format: CSV (UTF-8) or Excel (.xlsx)
2. Required columns: entity_id, year, value
   - entity_id: string or integer — unique identifier for each unit
   - year: integer — time period (use 2020, not "2020年")
   - value: numeric — the variable value
3. No duplicate (entity_id, year) pairs
4. Missing values: leave empty (not "N/A" or "missing")
5. Optional: a "source" column indicating data provenance
"""


def validate_raw_format(df: pd.DataFrame, variable_name: str,
                        entity_col: str = "entity_id",
                        time_col: str = "year",
                        value_col: str = "value") -> list[str]:
    """
    Validate that a raw DataFrame conforms to the required schema.
    Returns a list of issues (empty = valid).
    """
    issues = []

    # Check required columns
    for col in [entity_col, time_col, value_col]:
        if col not in df.columns:
            issues.append(f"Missing required column: '{col}'")
    if issues:
        return issues

    # Check types
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        # Try to convert
        try:
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        except Exception:
            issues.append(f"Column '{time_col}' must be numeric (year)")

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        try:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        except Exception:
            issues.append(f"Column '{value_col}' must be numeric")

    # Check for duplicates
    dupes = df.duplicated(subset=[entity_col, time_col]).sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate ({entity_col}, {time_col}) pairs")

    # Check for common bad patterns in entity_id
    sample_ids = df[entity_col].dropna().astype(str).head(5).tolist()
    bad_patterns = ["N/A", "missing", "null", "none", "nan", "暂无"]
    for bad in bad_patterns:
        if any(bad.lower() in str(x).lower() for x in sample_ids):
            issues.append(f"Entity IDs contain '{bad}' — use empty cells for missing values")
            break

    return issues


def load_and_validate(path: str, variable_name: str,
                      entity_col: str = "entity_id",
                      time_col: str = "year",
                      value_col: str = "value") -> pd.DataFrame:
    """
    Load a data file (JSON, CSV, Excel) and validate it against the raw format spec.

    Prints validation results. Returns DataFrame with standardized column names.
    Raises ValueError if critical issues found.
    """
    p = Path(path)

    # Load
    if p.suffix == ".json":
        df = pd.read_json(path)
    elif p.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif p.suffix == ".dta":
        df = pd.read_stata(path)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    # Detect column names if not standard
    if entity_col not in df.columns:
        # Try common alternatives
        for alt in ["city_id", "city", "province", "region", "code", "行政区划代码",
                    "城市代码", "地区", "编号", "id", "ID"]:
            if alt in df.columns:
                print(f"  [{variable_name}] Using '{alt}' as entity_id column")
                entity_col = alt
                break

    if time_col not in df.columns:
        for alt in ["yr", "Year", "年份", "时间", "period"]:
            if alt in df.columns:
                print(f"  [{variable_name}] Using '{alt}' as year column")
                time_col = alt
                break

    if value_col not in df.columns:
        # Try the remaining numeric column (common in 3-column files)
        used_cols = {entity_col, time_col}
        remaining = [c for c in df.columns if c not in used_cols]
        numeric_remaining = [
            c for c in remaining
            if pd.api.types.is_numeric_dtype(df[c])
            or df[c].apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit()
                          if pd.notna(x) else True).all()
        ]
        if len(numeric_remaining) == 1:
            print(f"  [{variable_name}] Using '{numeric_remaining[0]}' as value column")
            value_col = numeric_remaining[0]
        elif len(remaining) == 1:
            print(f"  [{variable_name}] Using '{remaining[0]}' as value column")
            value_col = remaining[0]

    # Validate
    issues = validate_raw_format(df, variable_name, entity_col, time_col, value_col)
    if issues:
        print(f"  ⚠ [{variable_name}] Format issues:")
        for issue in issues:
            print(f"    - {issue}")
        if any("Missing required column" in i for i in issues):
            raise ValueError(
                f"Cannot load {variable_name}: missing required columns.\n"
                f"{RAW_FILE_SPEC}"
            )

    # Standardize column names
    df = df.rename(columns={entity_col: "entity_id", time_col: "year",
                            value_col: "value"})

    # Clean types
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows with missing keys
    df = df.dropna(subset=["entity_id", "year"])
    df = df.drop_duplicates(subset=["entity_id", "year"])

    return df[["entity_id", "year", "value"]]


# ═══════════════════════════════════════════════════════════════════════
# Merge
# ═══════════════════════════════════════════════════════════════════════

def merge_panels(dataframes: dict[str, pd.DataFrame],
                 entity_col: str, time_col: str) -> pd.DataFrame:
    """
    Merge multiple DataFrames on entity and time.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Variable name → DataFrame (must have entity_id, year, value columns).
    entity_col, time_col : str
        Column names to merge on.

    Returns
    -------
    Wide panel DataFrame with one column per variable.
    """
    merged = None
    for var_name, df in dataframes.items():
        if df.empty:
            print(f"  [skip] {var_name}: empty DataFrame")
            continue

        # Rename value column to variable name
        df_renamed = df.rename(columns={"value": var_name})

        if merged is None:
            merged = df_renamed
        else:
            merged = merged.merge(df_renamed, on=[entity_col, time_col], how="outer")

    if merged is None:
        return pd.DataFrame()

    return merged


def balance_check(df: pd.DataFrame, entity_col: str, time_col: str) -> dict:
    """Report panel balance."""
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    n_rows = len(df)
    complete_rows = n_entities * n_periods

    return {
        "entities": n_entities,
        "periods": n_periods,
        "rows": n_rows,
        "expected_rows": complete_rows,
        "balanced": n_rows == complete_rows,
        "missing_pct": round(
            (1 - n_rows / complete_rows) * 100, 1
        ) if complete_rows > 0 else 100.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Missing value analysis and interpolation
# ═══════════════════════════════════════════════════════════════════════

def analyze_missing_patterns(df: pd.DataFrame, entity_col: str,
                              time_col: str,
                              variables: list[str]) -> dict:
    """
    Classify missing data patterns for each variable.

    Returns a report categorizing each missing value as:
      - sporadic: isolated single-year gap (interpolatable)
      - boundary: missing at start/end of entity's time series
      - systematic: entity always missing this variable
      - gap_run: consecutive multi-year gap

    Does NOT modify data. Returns diagnostic report only.
    """
    report = {}
    time_values = sorted(df[time_col].unique())

    for var in variables:
        if var not in df.columns:
            report[var] = {"status": "not_in_dataframe"}
            continue

        series = df[var]
        total = len(series)
        missing = int(series.isna().sum())
        complete = total - missing

        var_report = {
            "total_rows": total,
            "missing": missing,
            "missing_pct": round(missing / total * 100, 1) if total > 0 else 0,
            "complete": complete,
            "sporadic_count": 0,
            "boundary_count": 0,
            "systematic_entities": [],
            "gap_runs": [],
            "verdict": "ok",
        }

        if missing == 0:
            report[var] = var_report
            continue

        # Per-entity analysis
        entities = df[entity_col].unique()
        sporadic_total = 0
        boundary_total = 0
        systematic_entities = []
        gap_runs = []

        for entity in entities:
            entity_mask = df[entity_col] == entity
            entity_data = df.loc[entity_mask, [time_col, var]].sort_values(time_col)
            entity_missing = entity_data[var].isna()

            if entity_missing.all():
                systematic_entities.append(str(entity))
                continue
            if not entity_missing.any():
                continue

            # Classify each missing segment
            missing_indices = entity_missing.values.nonzero()[0]
            # Group consecutive missing indices
            runs = []
            run_start = missing_indices[0]
            run_len = 1
            for i in range(1, len(missing_indices)):
                if missing_indices[i] == missing_indices[i-1] + 1:
                    run_len += 1
                else:
                    runs.append((run_start, run_len))
                    run_start = missing_indices[i]
                    run_len = 1
            runs.append((run_start, run_len))

            entity_times = entity_data[time_col].values
            n_periods = len(entity_times)

            for start_idx, length in runs:
                year_val = entity_times[start_idx]
                if length == 1:
                    # Single gap — check if at boundary
                    if start_idx == 0 or start_idx == n_periods - 1:
                        boundary_total += 1
                    else:
                        sporadic_total += 1
                else:
                    # Multi-year gap
                    gap_years = entity_times[start_idx:start_idx + length]
                    gap_runs.append({
                        "entity": str(entity),
                        "years": gap_years.tolist(),
                        "length": length,
                    })

        var_report["sporadic_count"] = sporadic_total
        var_report["boundary_count"] = boundary_total
        var_report["systematic_entities"] = systematic_entities[:10]
        var_report["n_systematic"] = len(systematic_entities)
        var_report["gap_runs"] = gap_runs[:10]
        var_report["n_gap_runs"] = len(gap_runs)

        # Verdict
        if len(systematic_entities) > len(entities) * 0.3:
            var_report["verdict"] = "warn_systematic"
        elif sporadic_total > len(entities) * 0.3:
            var_report["verdict"] = "warn_sporadic"
        elif len(gap_runs) > 0:
            var_report["verdict"] = "has_gaps"
        elif missing > 0:
            var_report["verdict"] = "minor"
        else:
            var_report["verdict"] = "ok"

        report[var] = var_report

    return report


def impute_sporadic(df: pd.DataFrame, entity_col: str, time_col: str,
                     variables: list[str],
                     max_gap: int = 1) -> pd.DataFrame:
    """
    Linearly interpolate isolated single-year gaps within each entity.

    Only fills gaps of size ≤ max_gap that are NOT at the time series
    boundary. Boundary gaps and multi-year gaps are left as NaN.

    Parameters
    ----------
    df : pd.DataFrame
    entity_col, time_col : str
    variables : list[str]
        Variables to check and interpolate.
    max_gap : int
        Maximum gap length (in periods) to interpolate. Default 1.

    Returns
    -------
    pd.DataFrame with interpolated values (new column: {var}_imputed).
    """
    df = df.copy()
    entities = df[entity_col].unique()

    for var in variables:
        if var not in df.columns:
            continue

        imputed_col = f"{var}_imputed"
        df[imputed_col] = df[var].copy()
        n_filled = 0

        for entity in entities:
            mask = df[entity_col] == entity
            entity_idx = df.loc[mask].sort_values(time_col).index

            if len(entity_idx) < 3:
                continue

            values = df.loc[entity_idx, var].values.astype(float)
            n = len(values)

            for i in range(1, n - 1):
                if np.isnan(values[i]):
                    # Check gap length
                    gap_len = 1
                    while i + gap_len < n and np.isnan(values[i + gap_len]):
                        gap_len += 1
                    # Also check backwards
                    j = i - 1
                    while j >= 0 and np.isnan(values[j]):
                        gap_len += 1
                        j -= 1

                    if gap_len <= max_gap:
                        # Linear interpolation
                        left_val = values[i - 1] if i > 0 and not np.isnan(values[i - 1]) else None
                        # Find next non-NaN
                        right_val = None
                        for k in range(i + 1, n):
                            if not np.isnan(values[k]):
                                right_val = values[k]
                                break
                        if left_val is not None and right_val is not None:
                            frac = 1.0 / (n - 1) if n > 1 else 0.5
                            # Simple: average neighbors for single gap
                            df.loc[entity_idx[i], imputed_col] = (
                                left_val + right_val) / 2.0
                            n_filled += 1

        if n_filled > 0:
            print(f"  {var}: interpolated {n_filled} sporadic gaps "
                  f"(→ '{imputed_col}')")

    return df


def print_missing_report(report: dict, entity_col: str = "entity_id"):
    """Pretty-print a missing value analysis report."""
    print("\n" + "═" * 60)
    print("Missing Value Analysis Report")
    print("═" * 60)

    verdict_icons = {
        "ok": "✓", "minor": "○", "has_gaps": "⚠",
        "warn_sporadic": "⚠", "warn_systematic": "✗",
        "not_in_dataframe": "✗",
    }

    for var, r in report.items():
        icon = verdict_icons.get(r.get("verdict", "?"), "?")
        print(f"\n{icon} {var}: {r.get('verdict', '?')} "
              f"({r.get('missing', '?')}/{r.get('total_rows', '?')} missing, "
              f"{r.get('missing_pct', '?')}%)")

        if r.get("sporadic_count", 0) > 0:
            print(f"   └ Sporadic gaps (interpolatable): {r['sporadic_count']}")
        if r.get("boundary_count", 0) > 0:
            print(f"   └ Boundary gaps (start/end of series): {r['boundary_count']}")
        if r.get("n_systematic", 0) > 0:
            ents = r.get("systematic_entities", [])[:3]
            print(f"   └ Systematically missing ({r['n_systematic']} entities): "
                  f"{', '.join(ents)}" + ("..." if r['n_systematic'] > 3 else ""))
        if r.get("n_gap_runs", 0) > 0:
            gaps = r.get("gap_runs", [])[:3]
            for g in gaps:
                print(f"   └ Multi-year gap: {g['entity']} years {g['years']} "
                      f"({g['length']} periods)")

    print("\n" + "─" * 60)
    print("Recommendations:")
    for var, r in report.items():
        if r.get("verdict") == "warn_systematic":
            print(f"  {var}: Too many entities lack data — "
                  f"consider dropping this variable")
        elif r.get("sporadic_count", 0) > 0:
            print(f"  {var}: {r['sporadic_count']} sporadic gaps — "
                  f"linear interpolation recommended")
        elif r.get("n_gap_runs", 0) > 0:
            print(f"  {var}: Has multi-year gaps — "
                  f"check data source, do NOT interpolate blindly")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Merge and clean panel data with entity ID normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Raw file format specification:
{RAW_FILE_SPEC}

Examples:
  # Basic merge
  python clean_panel.py --outcome data/auto/fertility.json \\
      --treatment data/raw/pilot_dates.csv \\
      --controls data/auto/gdp.json --output data/merged/panel.dta

  # With entity ID mapping
  python clean_panel.py --outcome data/auto/fertility.json \\
      --treatment data/raw/pilot_dates.csv \\
      --entity-map data/auto/entity_map.json \\
      --output data/merged/panel.dta
""",
    )
    parser.add_argument("--outcome", required=True, help="Path to outcome variable data")
    parser.add_argument("--treatment", required=True, help="Path to treatment assignment data")
    parser.add_argument("--controls", nargs="*", default=[], help="Paths to control variable data")
    parser.add_argument("--entity-col", default="entity_id", help="Entity identifier column")
    parser.add_argument("--time-col", default="year", help="Time column")
    parser.add_argument("--entity-map", default=None, help="Path to entity_map.json")
    parser.add_argument("--output", default=None, help="Output path (.dta or .csv)")
    parser.add_argument("--missing-report", action="store_true",
                        help="Generate missing value analysis report")
    parser.add_argument("--impute", action="store_true",
                        help="Linearly interpolate sporadic single-year gaps")
    args = parser.parse_args()

    # ── Load entity map ───────────────────────────────────────────────
    entity_map = load_entity_map(args.entity_map)
    if entity_map:
        print(f"Loaded entity map: {len(entity_map)} entries")
    else:
        print("No entity map loaded — entity IDs will be kept as-is.")

    # ── Load and validate each file ───────────────────────────────────
    dataframes = {}

    # Outcome
    print(f"\nLoading outcome: {args.outcome}")
    df_outcome = load_and_validate(args.outcome, "outcome",
                                   args.entity_col, args.time_col)
    dataframes["outcome"] = df_outcome

    # Treatment
    print(f"Loading treatment: {args.treatment}")
    df_treat = load_and_validate(args.treatment, "treatment",
                                 args.entity_col, args.time_col)
    dataframes["treatment"] = df_treat

    # Controls
    for i, cp in enumerate(args.controls):
        ctrl_name = Path(cp).stem
        print(f"Loading control [{i+1}/{len(args.controls)}]: {cp}")
        try:
            df_ctrl = load_and_validate(cp, ctrl_name,
                                        args.entity_col, args.time_col)
            # Standardize with entity map
            if entity_map:
                df_ctrl = normalize_entity_ids(df_ctrl, "entity_id", entity_map)
            dataframes[ctrl_name] = df_ctrl
        except ValueError as e:
            print(f"  ✗ {e}")

    # Standardize outcome and treatment with entity map
    if entity_map:
        for key in ["outcome", "treatment"]:
            if key in dataframes:
                dataframes[key] = normalize_entity_ids(
                    dataframes[key], "entity_id", entity_map
                )

    # ── Merge ─────────────────────────────────────────────────────────
    merged = merge_panels(dataframes, "entity_id", "year")
    if merged.empty:
        print("Error: No data after merge.")
        sys.exit(1)

    print(f"\nMerged panel: {merged.shape[0]} rows × {merged.shape[1]} columns")

    # ── Balance ───────────────────────────────────────────────────────
    balance = balance_check(merged, "entity_id", "year")
    print(f"Entities: {balance['entities']}, Periods: {balance['periods']}")
    if not balance["balanced"]:
        print(
            f"Unbalanced panel: {balance['missing_pct']}% missing "
            f"({balance['rows']}/{balance['expected_rows']})"
        )
    else:
        print("Panel is balanced.")

    # ── Mark treated/post ─────────────────────────────────────────────
    treat_col = "treatment"
    first_treat_col = "first_treated"
    if treat_col in merged.columns and first_treat_col in merged.columns:
        merged["post"] = (merged["year"] >= merged[first_treat_col]).astype(int)
        merged["treated_post"] = merged[treat_col] * merged["post"]
        print("Created 'post' and 'treated_post' columns.")
    elif treat_col in merged.columns:
        print(f"Note: '{first_treat_col}' column not found — 'post' not created.")

    # ── Missing value analysis ─────────────────────────────────────────
    if args.missing_report or args.impute:
        numeric_vars = [c for c in merged.columns
                       if c not in ("entity_id", "year", "post",
                                    "treated_post")
                       and pd.api.types.is_numeric_dtype(merged[c])]
        if numeric_vars:
            report = analyze_missing_patterns(
                merged, "entity_id", "year", numeric_vars)
            print_missing_report(report, "entity_id")

    if args.impute:
        numeric_vars = [c for c in merged.columns
                       if c not in ("entity_id", "year", "post",
                                    "treated_post")
                       and pd.api.types.is_numeric_dtype(merged[c])]
        merged = impute_sporadic(merged, "entity_id", "year", numeric_vars)

    # ── Save ──────────────────────────────────────────────────────────
    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent.parent / "data" / "merged" / "panel.dta"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".dta":
        merged.to_stata(str(output_path), write_index=False)
    else:
        merged.to_csv(str(output_path), index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
