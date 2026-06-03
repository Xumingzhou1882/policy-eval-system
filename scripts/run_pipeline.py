"""
Pipeline orchestrator for the policy evaluation system.

Chains the 9 stages together, passing outputs from one stage as inputs
to the next. Supports:
  - Automatic stage detection and execution
  - Restart from any stage
  - Dry-run mode to preview the pipeline
  - State persistence in JSON

Usage:
    # Run full pipeline automatically
    python run_pipeline.py --policy "LTCI Pilot" --outcome "Fertility Rate" \\
                           --state pipeline_state.json

    # Resume from Stage 5
    python run_pipeline.py --state pipeline_state.json --from-stage 5

    # Dry run (show what would execute)
    python run_pipeline.py --policy "LTCI Pilot" --outcome "Fertility Rate" --dry-run
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
MERGED_DIR = DATA_DIR / "merged"
AUTO_DIR = DATA_DIR / "auto"


# ═══════════════════════════════════════════════════════════════════════
# State management
# ═══════════════════════════════════════════════════════════════════════

def load_state(state_path: str) -> dict:
    """Load pipeline state from JSON, or create a new one."""
    p = Path(state_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {
        "pipeline": {
            "started": datetime.now().isoformat(),
            "current_stage": 0,
            "completed_stages": [],
        },
        "stages": {},
    }


def save_state(state: dict, state_path: str):
    """Persist pipeline state to JSON."""
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def mark_stage_complete(state: dict, stage: int, result: dict, state_path: str):
    """Mark a stage as completed and save state."""
    state["stages"][f"stage{stage}"] = result
    state["pipeline"]["current_stage"] = stage
    completed = state["pipeline"].get("completed_stages", [])
    if stage not in completed:
        completed.append(stage)
    state["pipeline"]["completed_stages"] = sorted(completed)
    state["pipeline"]["last_updated"] = datetime.now().isoformat()
    save_state(state, state_path)


# ═══════════════════════════════════════════════════════════════════════
# Stage runners
# ═══════════════════════════════════════════════════════════════════════

def run_stage3(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 3: Theoretical method analysis."""
    s2 = state["stages"].get("stage2", {})
    flags = s2.get("flags", {})
    mechanism = s2.get("mechanism", "staggered_policy_shock")
    policy = state["stages"].get("stage1", {}).get("policy", state.get("policy", ""))
    outcome = state["stages"].get("stage1", {}).get("outcome", state.get("outcome", ""))

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "stage3_analyze.py"),
        "--mechanism", mechanism,
        "--policy", policy,
        "--outcome", outcome,
    ]

    if flags.get("staggered"):
        cmd.append("--staggered")
    if flags.get("has_control_group", True):
        cmd.append("--has-control-group")
    if flags.get("no_control_group"):
        cmd.append("--no-control-group")
    if flags.get("threshold_type"):
        cmd.extend(["--threshold-type", flags["threshold_type"]])
    if flags.get("has_instrument"):
        cmd.append("--has-instrument")
    if flags.get("multiple_policies"):
        cmd.append("--multiple-policies")
    if flags.get("everyone_treated_eventually"):
        cmd.append("--everyone-treated-eventually")

    output_path = str(AUTO_DIR / f"stage3_{_safe_filename(policy)}.json")
    cmd.extend(["--output", output_path])

    if dry_run:
        return {"status": "dry_run", "command": " ".join(cmd), "output": output_path}

    print(f"\n{'='*60}")
    print("Stage 3: Theoretical Method Analysis")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return {"status": "error", "stderr": result.stderr}

    # Load the JSON output
    with open(output_path, encoding="utf-8") as f:
        stage3_output = json.load(f)

    stage3_output["status"] = "completed"
    stage3_output["output_file"] = output_path
    mark_stage_complete(state, 3, stage3_output, state_path)
    return stage3_output


def run_stage7(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 7: Estimation — run the chosen method."""
    s6 = state["stages"].get("stage6", {})
    s1 = state["stages"].get("stage1", {})
    final_method = s6.get("final_method", "")
    data_path = state.get("data_path") or str(MERGED_DIR / "panel.dta")

    outcome = s1.get("outcome", state.get("outcome", ""))
    entity = s6.get("entity_col", "city_id")
    time_col = s6.get("time_col", "year")

    results = {}

    # Determine which estimation script to run
    if "callaway" in final_method.lower() or "staggered" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_staggered_did.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--first-treated", s6.get("first_treated_col", "first_treated"),
            "--method", "cs",
            "--control", s6.get("control_type", "never-treated"),
        ]
        controls = s6.get("covariates", [])
        if controls:
            cmd.extend(["--controls"] + controls)

    elif "did" in final_method.lower() or "twfe" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_did.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--post", s6.get("post_col", "post"),
        ]
        controls = s6.get("covariates", [])
        if controls:
            cmd.extend(["--controls"] + controls)

    elif "rdd" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_rdd.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--running-var", s6.get("running_var", "running_var"),
            "--cutoff", str(s6.get("cutoff", 0)),
            "--type", s6.get("rdd_type", "sharp"),
        ]

    elif "iv" in final_method.lower() or "2sls" in final_method.lower():
        instruments = s6.get("instruments", [])
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_iv.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--treatment", s6.get("treatment_var", "treatment"),
            "--instruments"] + instruments
        controls = s6.get("covariates", [])
        if controls:
            cmd.extend(["--controls"] + controls)

    elif "synthetic did" in final_method.lower() or "synthetic difference" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_synthetic_did.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--first-treated", str(s6.get("first_treated", 0)),
        ]
        treated_unit = s6.get("treated_unit", "")
        if treated_unit:
            cmd.extend(["--treated-unit", str(treated_unit)])

    elif "scm" in final_method.lower() or "synthetic control" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_scm.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated-unit", str(s6.get("treated_unit", "")),
            "--first-treated", str(s6.get("first_treated", 0)),
        ]

    else:
        return {"status": "error", "message": f"Unknown method: {final_method}"}

    output_path = str(AUTO_DIR / "stage7_main_result.json")
    cmd.extend(["--output", output_path])

    if dry_run:
        return {"status": "dry_run", "command": " ".join(cmd), "output": output_path}

    print(f"\n{'='*60}")
    print("Stage 7: Estimation")
    print(f"{'='*60}")
    print(f"Method: {final_method}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return {"status": "error", "stderr": result.stderr}

    results["main_result"] = {"output_file": output_path}
    results["status"] = "completed"

    # Also run event study
    try:
        event_cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_event_study.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--first-treated", s6.get("first_treated_col", "first_treated"),
            "--plot", str(AUTO_DIR / "event_study.png"),
            "--output", str(AUTO_DIR / "stage7_event_study.json"),
        ]
        event_result = subprocess.run(event_cmd, capture_output=True, text=True)
        print(event_result.stdout)
        results["event_study"] = {"output_file": str(AUTO_DIR / "stage7_event_study.json")}
    except Exception as e:
        print(f"Event study warning: {e}")

    mark_stage_complete(state, 7, results, state_path)
    return results


def run_stage8(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 8: Robustness checks — placebo, Bacon, and sensitivity."""
    s1 = state["stages"].get("stage1", {})
    s6 = state["stages"].get("stage6", {})
    data_path = state.get("data_path") or str(MERGED_DIR / "panel.dta")

    outcome = s1.get("outcome", state.get("outcome", ""))
    entity = s6.get("entity_col", "city_id")
    time_col = s6.get("time_col", "year")

    results = {}

    # 1. Placebo test
    try:
        placebo_cmd = [
            sys.executable, str(SCRIPTS_DIR / "placebo_test.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--post", s6.get("post_col", "post"),
            "--n-sim", "500",
            "--plot", str(AUTO_DIR / "placebo.png"),
            "--output", str(AUTO_DIR / "stage8_placebo.json"),
        ]
        first_treated = s6.get("first_treated_col")
        if first_treated:
            placebo_cmd.extend(["--first-treated", first_treated])

        if not dry_run:
            p_result = subprocess.run(placebo_cmd, capture_output=True, text=True)
            print(p_result.stdout)
            results["placebo"] = {"output_file": str(AUTO_DIR / "stage8_placebo.json")}
    except Exception as e:
        print(f"Placebo test warning: {e}")

    # 2. Bacon decomposition (for staggered designs)
    try:
        bacon_cmd = [
            sys.executable, str(SCRIPTS_DIR / "bacon_decomp.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--first-treated", s6.get("first_treated_col", "first_treated"),
        ]
        if not dry_run:
            b_result = subprocess.run(bacon_cmd, capture_output=True, text=True)
            print(b_result.stdout)
    except Exception as e:
        print(f"Bacon decomp warning: {e}")

    # 3. Sensitivity analysis
    try:
        sens_cmd = [
            sys.executable, str(SCRIPTS_DIR / "sensitivity_analysis.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--post", s6.get("post_col", "post"),
            "--output", str(AUTO_DIR / "stage8_sensitivity.json"),
        ]
        controls = s6.get("covariates", [])
        if controls:
            sens_cmd.extend(["--controls"] + controls)
        treatment_col = s6.get("treatment_col")
        if treatment_col:
            sens_cmd.extend(["--treatment-col", treatment_col])
        first_treated_val = s6.get("first_treated")
        if first_treated_val:
            sens_cmd.extend(["--first-treated", str(first_treated_val)])

        if not dry_run:
            s_result = subprocess.run(sens_cmd, capture_output=True, text=True)
            print(s_result.stdout)
            results["sensitivity"] = {"output_file": str(AUTO_DIR / "stage8_sensitivity.json")}
    except Exception as e:
        print(f"Sensitivity analysis warning: {e}")

    results["status"] = "completed"
    mark_stage_complete(state, 8, results, state_path)
    return results


def run_stage9(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 9: Generate final report."""
    s1 = state["stages"].get("stage1", {})
    policy = s1.get("policy", state.get("policy", ""))
    outcome = s1.get("outcome", state.get("outcome", ""))

    report_cmd = [
        sys.executable, str(SCRIPTS_DIR / "output_report.py"),
        "--policy", policy,
        "--outcome", outcome,
        "--stage3", str(AUTO_DIR / f"stage3_{_safe_filename(policy)}.json"),
        "--output", str(AUTO_DIR / "final_report.md"),
    ]

    s7 = state["stages"].get("stage7", {})
    if s7.get("main_result", {}).get("output_file"):
        report_cmd.extend(["--stage7", s7["main_result"]["output_file"]])

    s8 = state["stages"].get("stage8", {})
    if s8.get("sensitivity", {}).get("output_file"):
        report_cmd.extend(["--stage8", s8["sensitivity"]["output_file"]])

    if dry_run:
        return {"status": "dry_run", "command": " ".join(report_cmd)}

    print(f"\n{'='*60}")
    print("Stage 9: Final Report")
    print(f"{'='*60}")

    result = subprocess.run(report_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return {"status": "error", "stderr": result.stderr}

    results = {"status": "completed", "report_path": str(AUTO_DIR / "final_report.md")}
    mark_stage_complete(state, 9, results, state_path)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    """Convert a policy name to a safe filename component."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name).lower()


def _print_pipeline_summary(state: dict):
    """Print a summary of the pipeline state."""
    print("\n═══════════════════════════════════")
    print("Pipeline Summary")
    print("═══════════════════════════════════")

    stages_config = state.get("stages", {})
    completed = state.get("pipeline", {}).get("completed_stages", [])

    stage_names = {
        1: "Problem Definition",
        2: "Policy Research",
        3: "Theoretical Method",
        4: "Data Requirements",
        5: "Data Acquisition",
        6: "Final Method Confirmation",
        7: "Estimation",
        8: "Robustness Checks",
        9: "Result Report",
    }

    for i in range(1, 10):
        status = "✓" if i in completed else ("●" if i == state["pipeline"].get("current_stage", 0) + 1 else "○")
        note = ""
        if f"stage{i}" in stages_config:
            s = stages_config[f"stage{i}"]
            if "primary_method" in s:
                note = f" → {s['primary_method']}"
            elif "final_method" in s:
                note = f" → {s['final_method']}"
            elif "status" in s:
                note = f" [{s['status']}]"
        print(f"  {status} Stage {i}: {stage_names.get(i, 'Unknown')}{note}")

    print(f"\n  Data: {state.get('data_path', 'Not set')}")
    print(f"  State: {len(completed)}/9 stages completed")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Policy Evaluation Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new pipeline
  python run_pipeline.py --policy "Long-term Care Insurance" \\
      --outcome "Fertility Rate" --state my_pipeline.json

  # Resume from Stage 5 with data ready
  python run_pipeline.py --state my_pipeline.json --from-stage 5 \\
      --data data/merged/panel.dta

  # Dry run to preview commands
  python run_pipeline.py --policy "LTCI" --outcome "Fertility" --dry-run
""")
    parser.add_argument("--policy", default=None, help="Policy name")
    parser.add_argument("--outcome", default=None, help="Outcome variable name")
    parser.add_argument("--state", default="pipeline_state.json",
                        help="Pipeline state file path")
    parser.add_argument("--from-stage", type=int, default=None,
                        help="Resume from a specific stage (1-9)")
    parser.add_argument("--to-stage", type=int, default=9,
                        help="Stop after this stage (default: 9)")
    parser.add_argument("--data", default=None, help="Path to analysis-ready data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview pipeline without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status and exit")
    args = parser.parse_args()

    state = load_state(args.state)

    # Status mode
    if args.status:
        _print_pipeline_summary(state)
        return

    # Initialize new pipeline
    if args.policy and args.outcome:
        if "stage1" not in state["stages"]:
            state["stages"]["stage1"] = {
                "policy": args.policy,
                "outcome": args.outcome,
                "status": "defined",
            }
            state["pipeline"]["completed_stages"] = [1]
            state["pipeline"]["current_stage"] = 1
        state["policy"] = args.policy
        state["outcome"] = args.outcome

    if args.data:
        state["data_path"] = args.data

    if not state.get("policy"):
        print("Error: No policy specified. Use --policy and --outcome, or provide an existing --state file.")
        sys.exit(1)

    save_state(state, args.state)

    # Determine starting stage
    start_stage = args.from_stage or (state["pipeline"].get("current_stage", 0) + 1)
    start_stage = max(start_stage, 1)

    print(f"Pipeline: {state.get('policy')} → {state.get('outcome')}")
    print(f"Starting from Stage {start_stage}, running through Stage {args.to_stage}")
    if args.dry_run:
        print("[DRY RUN — no commands will be executed]")

    # Execute stages
    stages_to_run = range(start_stage, args.to_stage + 1)

    for stage in stages_to_run:
        if stage == 1:
            print("\nStage 1: Problem Definition — already defined.")
            continue

        if stage == 2:
            print("\nStage 2: Policy Research — must be completed interactively.")
            print("  The system will search for policy details and present findings.")
            print("  Once policy research is complete, run:")
            print(f"    python {__file__} --state {args.state} --from-stage 3")
            break

        if stage == 3:
            run_stage3(state, args.state, dry_run=args.dry_run)

        if stage == 4:
            print("\nStage 4: Data Requirements — derived from Stage 3 output.")
            s3 = state["stages"].get("stage3", {})
            rec = s3.get("recommendation", {})
            required = rec.get("required_variables", [])
            optional = rec.get("optional_variables", [])
            print(f"  Required: {required}")
            print(f"  Optional: {optional}")
            print(f"  Once you've identified data sources, update {args.state} and re-run with --from-stage 5.")

        if stage == 5:
            print("\nStage 5: Data Acquisition — must be completed interactively.")
            print("  Place acquired data files in data/auto/ or data/raw/,")
            print(f"  then run: python {__file__} --state {args.state} --from-stage 6 --data <path>")

        if stage == 6:
            print("\nStage 6: Final Method Confirmation — requires user review.")
            print("  After verifying assumptions against actual data, update")
            print(f"  stage6 in {args.state} and re-run with --from-stage 7.")

        if stage == 7:
            if not state.get("data_path"):
                print("Stage 7: No data path set. Use --data to specify the panel data file.")
                break
            run_stage7(state, args.state, dry_run=args.dry_run)

        if stage == 8:
            run_stage8(state, args.state, dry_run=args.dry_run)

        if stage == 9:
            run_stage9(state, args.state, dry_run=args.dry_run)

    # Print summary
    _print_pipeline_summary(state)

    print(f"\nState saved to {args.state}")


if __name__ == "__main__":
    main()
