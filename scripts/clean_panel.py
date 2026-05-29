"""
Merge and clean panel data from multiple sources into analysis-ready format.

Usage:
    python scripts/clean_panel.py --outcome data/auto/fertility.json \\
                                  --treatment data/manual/pilot_dates.csv \\
                                  --controls data/auto/province_gdp.json \\
                                  --output data/merged/panel.dta
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_json(path: str) -> pd.DataFrame:
    """Load a JSON file produced by fetch_* scripts into a DataFrame."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def reshape_to_panel(df: pd.DataFrame, entity_col: str, time_col: str,
                     value_col: str) -> pd.DataFrame:
    """Ensure the DataFrame is in panel format: one row per entity-time."""
    panel = df[[entity_col, time_col, value_col]].drop_duplicates()
    panel = panel.dropna(subset=[entity_col, time_col])
    return panel.rename(columns={value_col: value_col})


def merge_panels(outcome: pd.DataFrame, treatment: pd.DataFrame,
                 controls: list[pd.DataFrame], entity_col: str,
                 time_col: str) -> pd.DataFrame:
    """Merge outcome, treatment, and control DataFrames on entity and time."""
    merged = outcome.merge(treatment, on=[entity_col, time_col], how="left")

    for ctrl in controls:
        ctrl_cols = list(ctrl.columns)
        on_cols = []
        if entity_col in ctrl_cols:
            on_cols.append(entity_col)
        if time_col in ctrl_cols:
            on_cols.append(time_col)
        if len(on_cols) == 0:
            # Cannot merge — use cross-join or skip
            print(f"  [skip] control missing entity/time columns: {ctrl_cols}")
            continue
        merged = merged.merge(ctrl, on=on_cols, how="left")

    return merged


def balance_check(df: pd.DataFrame, entity_col: str, time_col: str) -> dict:
    """Report panel balance: number of entities, time periods, missingness."""
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
        "missing_pct": (1 - n_rows / complete_rows) * 100 if complete_rows > 0 else 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge and clean panel data")
    parser.add_argument("--outcome", required=True, help="Path to outcome variable data (JSON/CSV)")
    parser.add_argument("--treatment", required=True, help="Path to treatment assignment data (CSV)")
    parser.add_argument("--controls", nargs="*", default=[], help="Paths to control variable data files")
    parser.add_argument("--entity-col", default="city_id", help="Entity identifier column name")
    parser.add_argument("--time-col", default="year", help="Time column name")
    parser.add_argument("--output", default=None, help="Output path (.dta or .csv)")
    args = parser.parse_args()

    # Load outcome
    outcome_path = Path(args.outcome)
    outcome = load_json(str(outcome_path)) if outcome_path.suffix == ".json" else load_csv(str(outcome_path))

    # Load treatment
    treatment_path = Path(args.treatment)
    treatment = load_csv(str(treatment_path))

    # Load controls
    controls = []
    for cp in args.controls:
        cp_path = Path(cp)
        ctrl = load_json(str(cp_path)) if cp_path.suffix == ".json" else load_csv(str(cp_path))
        controls.append(ctrl)

    # Merge
    merged = merge_panels(outcome, treatment, controls, args.entity_col, args.time_col)
    print(f"Merged panel: {merged.shape[0]} rows × {merged.shape[1]} columns")

    # Balance
    balance = balance_check(merged, args.entity_col, args.time_col)
    print(f"Entities: {balance['entities']}, Periods: {balance['periods']}")
    if not balance["balanced"]:
        print(f"Unbalanced panel: {balance['missing_pct']:.1f}% missing ({balance['rows']}/{balance['expected_rows']})")

    # Mark treated/post
    if "treated" in merged.columns and "first_treated" in merged.columns:
        merged["post"] = (merged[args.time_col] >= merged["first_treated"]).astype(int)
        merged["treated_post"] = merged["treated"] * merged["post"]
        print("Created 'post' and 'treated_post' columns.")

    # Save
    output_path = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / "data" / "merged" / "panel.dta"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".dta":
        merged.to_stata(str(output_path), write_index=False)
    else:
        merged.to_csv(str(output_path), index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
