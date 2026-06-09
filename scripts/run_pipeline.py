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


def run_stage6(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 6: Final method confirmation — verify assumptions against data."""
    s3 = state["stages"].get("stage3", {})
    policy = state["stages"].get("stage1", {}).get("policy", state.get("policy", ""))
    data_path = state.get("data_path") or str(MERGED_DIR / "panel.dta")

    stage3_file = AUTO_DIR / f"stage3_{_safe_filename(policy)}.json"
    if not stage3_file.exists():
        # Try to find the stage3 output from stage3 data
        s3_output_file = s3.get("output_file", "")
        if s3_output_file and Path(s3_output_file).exists():
            stage3_file = Path(s3_output_file)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "stage6_confirm.py"),
        "--stage3", str(stage3_file),
        "--data", _resolve_data_path(data_path),
        "--output", str(AUTO_DIR / "stage6_confirmation.json"),
    ]

    # Pass validation report if available
    validation_report = AUTO_DIR / "validation_report.json"
    if validation_report.exists():
        cmd.extend(["--validate-report", str(validation_report)])

    if dry_run:
        return {"status": "dry_run", "command": " ".join(cmd),
                "output": str(AUTO_DIR / "stage6_confirmation.json")}

    print(f"\n{'='*60}")
    print("Stage 6: Final Method Confirmation")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return {"status": "error", "stderr": result.stderr}

    # Load the JSON output
    out_path = AUTO_DIR / "stage6_confirmation.json"
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            stage6_output = json.load(f)
    else:
        return {"status": "error", "stderr": "stage6 output file not found"}

    stage6_output["status"] = "completed"
    stage6_output["output_file"] = str(out_path)
    mark_stage_complete(state, 6, stage6_output, state_path)
    return stage6_output


def _resolve_data_path(data_path: str) -> str:
    """Resolve a data path, trying merged dir as fallback."""
    p = Path(data_path)
    if p.exists():
        return str(p.resolve())
    alt = MERGED_DIR / p.name
    if alt.exists():
        return str(alt)
    return str(p.resolve())


def run_stage7(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 7: Estimation — run the chosen method."""
    s6 = state["stages"].get("stage6", {})
    s1 = state["stages"].get("stage1", {})
    final_method = s6.get("final_method", "")
    data_path = state.get("data_path") or str(MERGED_DIR / "panel.dta")

    spec = s6.get("specification", {})
    outcome = spec.get("outcome", s1.get("outcome", state.get("outcome", "")))
    entity = spec.get("entity_col", s6.get("entity_col", "city_id"))
    time_col = spec.get("time_col", s6.get("time_col", "year"))

    results = {}

    # Determine which estimation script to run
    if "callaway" in final_method.lower() or "staggered" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_staggered_did.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--first-treated", spec.get("first_treated_col", s6.get("first_treated_col", "first_treated")),
            "--method", spec.get("method", "cs"),
            "--control", spec.get("control_type", s6.get("control_type", "never-treated")),
        ]
        controls = spec.get("covariates", s6.get("covariates", []))
        if controls:
            cmd.extend(["--controls"] + controls)

    elif "did" in final_method.lower() or "twfe" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_did.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated", spec.get("treated_col", s6.get("treated_col", "treated")),
            "--post", spec.get("post_col", s6.get("post_col", "post")),
        ]
        controls = spec.get("covariates", s6.get("covariates", []))
        if controls:
            cmd.extend(["--controls"] + controls)
            cmd.append("--baseline")

    elif "rdd" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_rdd.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--running-var", spec.get("running_var", s6.get("running_var", "running_var")),
            "--cutoff", str(spec.get("cutoff", s6.get("cutoff", 0))),
            "--type", spec.get("rdd_type", s6.get("rdd_type", "sharp")),
        ]

    elif "iv" in final_method.lower() or "2sls" in final_method.lower():
        instruments = spec.get("instruments", s6.get("instruments", []))
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_iv.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--treatment", spec.get("treatment_var", s6.get("treatment_var", "treatment")),
            "--instruments"] + instruments
        controls = spec.get("covariates", s6.get("covariates", []))
        if controls:
            cmd.extend(["--controls"] + controls)

    elif "synthetic did" in final_method.lower() or "synthetic difference" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_synthetic_did.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--first-treated", str(spec.get("first_treated", s6.get("first_treated", 0))),
        ]
        treated_unit = spec.get("treated_unit", s6.get("treated_unit", ""))
        if treated_unit:
            cmd.extend(["--treated-unit", str(treated_unit)])

    elif "scm" in final_method.lower() or "synthetic control" in final_method.lower():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_scm.py"),
            "--data", data_path,
            "--outcome", outcome,
            "--entity", entity,
            "--time", time_col,
            "--treated-unit", str(spec.get("treated_unit", s6.get("treated_unit", ""))),
            "--first-treated", str(spec.get("first_treated", s6.get("first_treated", 0))),
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
            "--first-treated", spec.get("first_treated_col", s6.get("first_treated_col", "first_treated")),
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
    """Stage 8: Robustness checks — placebo, Bacon, alt windows, sensitivity, combined."""
    s1 = state["stages"].get("stage1", {})
    s6 = state["stages"].get("stage6", {})
    data_path = state.get("data_path") or str(MERGED_DIR / "panel.dta")

    spec = s6.get("specification", {})
    outcome = spec.get("outcome", s1.get("outcome", state.get("outcome", "")))
    entity = spec.get("entity_col", s6.get("entity_col", "city_id"))
    time_col = spec.get("time_col", s6.get("time_col", "year"))

    results = {}
    placebo_file = str(AUTO_DIR / "stage8_placebo.json")
    bacon_file = str(AUTO_DIR / "stage8_bacon.json")
    sens_file = str(AUTO_DIR / "stage8_sensitivity.json")
    alt_file = str(AUTO_DIR / "stage8_alt_windows.json")
    summary_file = str(AUTO_DIR / "stage8_summary.json")

    if dry_run:
        return {"status": "dry_run",
                "checks": ["placebo", "bacon", "alt_windows", "sensitivity", "combine"]}

    print(f"\n{'='*60}")
    print("Stage 8: Robustness Checks")
    print(f"{'='*60}")

    # 1. Placebo test
    print("\n[1/5] Placebo permutation test...")
    try:
        placebo_cmd = [
            sys.executable, str(SCRIPTS_DIR / "placebo_test.py"),
            "--data", data_path, "--outcome", outcome,
            "--entity", entity, "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--post", s6.get("post_col", "post"),
            "--n-sim", "500",
            "--plot", str(AUTO_DIR / "placebo.png"),
            "--output", placebo_file,
        ]
        if s6.get("first_treated_col"):
            placebo_cmd.extend(["--first-treated", s6["first_treated_col"]])
        p_result = subprocess.run(placebo_cmd, capture_output=True, text=True)
        print(p_result.stdout)
        results["placebo"] = {"output_file": placebo_file}
    except Exception as e:
        print(f"  ✗ Placebo test error: {e}")

    # 2. Bacon decomposition (staggered designs only)
    print("\n[2/5] Goodman-Bacon decomposition...")
    try:
        bacon_cmd = [
            sys.executable, str(SCRIPTS_DIR / "bacon_decomp.py"),
            "--data", data_path, "--outcome", outcome,
            "--entity", entity, "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--first-treated", s6.get("first_treated_col", "first_treated"),
            "--output", bacon_file,
        ]
        b_result = subprocess.run(bacon_cmd, capture_output=True, text=True)
        print(b_result.stdout)
        results["bacon"] = {"output_file": bacon_file}
    except Exception as e:
        print(f"  — Bacon decomp skipped: {e}")

    # 3. Alternative time windows
    print("\n[3/5] Alternative time windows...")
    try:
        alt_cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_alt_windows.py"),
            "--data", data_path, "--outcome", outcome,
            "--entity", entity, "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--post", s6.get("post_col", "post"),
            "--first-treated", s6.get("first_treated_col", "first_treated"),
            "--output", alt_file,
        ]
        controls = s6.get("covariates", [])
        if controls:
            alt_cmd.extend(["--controls"] + controls)
        a_result = subprocess.run(alt_cmd, capture_output=True, text=True)
        print(a_result.stdout)
        results["alt_windows"] = {"output_file": alt_file}
    except Exception as e:
        print(f"  ✗ Alt windows error: {e}")

    # 4. Sensitivity analysis
    print("\n[4/5] Sensitivity analysis (Oster, stability, Rosenbaum, LOO)...")
    try:
        sens_cmd = [
            sys.executable, str(SCRIPTS_DIR / "sensitivity_analysis.py"),
            "--data", data_path, "--outcome", outcome,
            "--entity", entity, "--time", time_col,
            "--treated", s6.get("treated_col", "treated"),
            "--post", s6.get("post_col", "post"),
            "--output", sens_file,
        ]
        controls = s6.get("covariates", [])
        if controls:
            sens_cmd.extend(["--controls"] + controls)
        if s6.get("treatment_col"):
            sens_cmd.extend(["--treatment-col", s6["treatment_col"]])
        if s6.get("first_treated"):
            sens_cmd.extend(["--first-treated", str(s6["first_treated"])])
        s_result = subprocess.run(sens_cmd, capture_output=True, text=True)
        print(s_result.stdout)
        results["sensitivity"] = {"output_file": sens_file}
    except Exception as e:
        print(f"  ✗ Sensitivity analysis error: {e}")

    # 5. Consolidate
    print("\n[5/5] Consolidating Stage 8 results...")
    try:
        combine_cmd = [
            sys.executable, str(SCRIPTS_DIR / "stage8_combine.py"),
            "--placebo", placebo_file,
            "--bacon", bacon_file,
            "--sensitivity", sens_file,
            "--alt-windows", alt_file,
            "--output", summary_file,
        ]
        c_result = subprocess.run(combine_cmd, capture_output=True, text=True)
        print(c_result.stdout)
        results["summary"] = {"output_file": summary_file}
    except Exception as e:
        print(f"  ✗ Consolidation error: {e}")

    results["status"] = "completed"
    mark_stage_complete(state, 8, results, state_path)
    return results


def run_stage9(state: dict, state_path: str, dry_run: bool = False) -> dict:
    """Stage 9: Generate final report — structured JSON + Markdown + XeLaTeX."""
    s1 = state["stages"].get("stage1", {})
    s6 = state["stages"].get("stage6", {})
    s7 = state["stages"].get("stage7", {})
    s8 = state["stages"].get("stage8", {})
    policy = s1.get("policy", state.get("policy", ""))
    outcome = s1.get("outcome", state.get("outcome", ""))

    # Stage 3 output file — find it properly
    stage3_file = None
    # First try: from stage3's own output_file field
    s3 = state["stages"].get("stage3", {})
    if s3.get("output_file") and Path(s3["output_file"]).exists():
        stage3_file = s3["output_file"]
    # Second try: standard naming pattern
    if not stage3_file:
        candidate = str(AUTO_DIR / f"stage3_{_safe_filename(policy)}.json")
        if Path(candidate).exists():
            stage3_file = candidate
    # Third try: glob
    if not stage3_file:
        candidates = sorted(AUTO_DIR.glob("stage3_*.json"))
        if candidates:
            stage3_file = str(candidates[0])

    # Generate output path: Desktop/policy_eval_output/<policy_slug>/<timestamp>/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path.home() / "Desktop" / "policy_eval_output"
    subject_dir = _find_or_create_subject_dir(policy, output_base)
    run_dir = subject_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract structured data
    report_data_path = str(run_dir / "report_data.json")
    extract_cmd = [
        sys.executable, str(SCRIPTS_DIR / "output_report.py"),
        "--policy", policy,
        "--outcome", outcome,
        "--stage3", stage3_file,
        "--output", report_data_path,
    ]

    stage6_file = s6.get("output_file", str(AUTO_DIR / "stage6_confirmation.json"))
    if Path(stage6_file).exists():
        extract_cmd.extend(["--stage6", stage6_file])

    if s7.get("main_result", {}).get("output_file"):
        extract_cmd.extend(["--stage7", s7["main_result"]["output_file"]])

    if s8.get("sensitivity", {}).get("output_file"):
        extract_cmd.extend(["--stage8", s8["sensitivity"]["output_file"]])
    if s8.get("placebo", {}).get("output_file"):
        extract_cmd.extend(["--stage8-placebo", s8["placebo"]["output_file"]])
    if s8.get("summary", {}).get("output_file"):
        extract_cmd.extend(["--stage8-summary", s8["summary"]["output_file"]])

    # Pass panel data path for descriptive stats computation
    data_path = state.get("data_path", "")
    if data_path and Path(data_path).exists():
        extract_cmd.extend(["--data", data_path])
        extract_cmd.extend(["--data-source", data_path])

    # Pass data_status from Stage 5 if available
    s5 = state["stages"].get("stage5", {})
    data_status = s5.get("data_status", None)
    if data_status:
        import tempfile
        tmp_status = Path(tempfile.gettempdir()) / f"stage5_data_status_{_safe_filename(policy)}.json"
        tmp_status.write_text(json.dumps(data_status, ensure_ascii=False, indent=2), encoding="utf-8")
        extract_cmd.extend(["--data-status", str(tmp_status)])

    # Pass data span and n_obs from metadata
    data_meta = state.get("data_meta", {})
    if data_meta.get("span"):
        extract_cmd.extend(["--data-span", str(data_meta["span"])])
    if data_meta.get("n_obs"):
        extract_cmd.extend(["--n-obs", str(data_meta["n_obs"])])

    # Stage 2 narrative sections
    stage2_sections_file = AUTO_DIR / "stage2_sections.json"
    if stage2_sections_file.exists():
        extract_cmd.extend(["--stage2-sections", str(stage2_sections_file)])

    # Event study (from Stage 7) — search with glob, not hardcoded names
    event_study_file = AUTO_DIR / "stage7_event_study.json"
    if not event_study_file.exists():
        # Also look for event study from s7's own records
        if s7.get("event_study", {}).get("output_file"):
            es_candidate = Path(s7["event_study"]["output_file"])
            if es_candidate.exists():
                event_study_file = es_candidate
    if not event_study_file.exists():
        # Generic glob for any event study JSON
        es_candidates = sorted(AUTO_DIR.glob("*event_study*.json"))
        if not es_candidates:
            es_candidates = sorted(AUTO_DIR.glob("*es_*.json"))
        if es_candidates:
            event_study_file = es_candidates[0]
    if event_study_file.exists():
        extract_cmd.extend(["--event-study", str(event_study_file)])

    # Step 2: Render Markdown + XeLaTeX
    safe_policy = _safe_filename(policy)
    md_dir = run_dir / "markdown"
    tex_dir = run_dir / "latex"
    md_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    md_path = str(md_dir / f"{safe_policy}_report.md")
    tex_path = str(tex_dir / f"{safe_policy}_report.tex")

    render_cmd = [
        sys.executable, str(SCRIPTS_DIR / "render_report.py"),
        "--data", report_data_path,
        "--md", md_path,
        "--tex", tex_path,
        "--compile",
    ]

    if dry_run:
        return {"status": "dry_run",
                "commands": [" ".join(extract_cmd), " ".join(render_cmd)]}

    print(f"\n{'='*60}")
    print("Stage 9: Final Report")
    print(f"{'='*60}")

    # Run extraction
    print(f"Extracting: {' '.join(extract_cmd)}")
    result1 = subprocess.run(extract_cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    if result1.stdout:
        print(result1.stdout)
    if result1.returncode != 0:
        print(f"Extraction error: {result1.stderr}")
        return {"status": "error", "step": "extract", "stderr": result1.stderr}

    # Run rendering
    print(f"Rendering: {' '.join(render_cmd)}")
    result2 = subprocess.run(render_cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    if result2.stdout:
        print(result2.stdout)
    if result2.returncode != 0:
        print(f"Render error: {result2.stderr}")
        return {"status": "error", "step": "render", "stderr": result2.stderr}

    results = {
        "status": "completed",
        "run_dir": str(run_dir),
        "report_data": report_data_path,
        "report_md": md_path,
        "report_tex": tex_path,
    }
    mark_stage_complete(state, 9, results, state_path)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    """Convert a policy name to a safe filename component."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name).lower()


def _find_or_create_subject_dir(policy_name: str, base_dir: Path) -> Path:
    """Find an existing subject folder or create a new one."""
    base_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_filename(policy_name)

    existing = [d for d in base_dir.iterdir() if d.is_dir()]

    # 1. Exact match
    for d in existing:
        if d.name.lower() == slug:
            return d

    # 2. Substring match
    for d in existing:
        if slug in d.name.lower() or d.name.lower() in slug:
            return d

    # 3. First 5 chars match
    for d in existing:
        if len(d.name) >= 5 and len(slug) >= 5:
            if d.name[:5].lower() == slug[:5]:
                return d

    # 4. No match — create new
    new_dir = base_dir / slug
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


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
            policy = state["stages"].get("stage1", {}).get("policy", state.get("policy", ""))
            stage3_file = AUTO_DIR / f"stage3_{_safe_filename(policy)}.json"
            if not stage3_file.exists():
                s3 = state["stages"].get("stage3", {})
                if s3.get("output_file") and Path(s3["output_file"]).exists():
                    stage3_file = Path(s3["output_file"])
            if stage3_file.exists():
                cmd = [
                    sys.executable, str(SCRIPTS_DIR / "stage4_requirements.py"),
                    "--stage3", str(stage3_file),
                    "--output", str(AUTO_DIR / "stage4_requirements.json"),
                    "--text",
                ]
                if not args.dry_run:
                    print(f"\n{'='*60}")
                    print("Stage 4: Data Requirements")
                    print(f"{'='*60}")
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                           encoding="utf-8", errors="replace")
                    print(result.stdout)
                    if result.returncode != 0:
                        print(f"Error: {result.stderr}")
                    else:
                        # Load and store in state
                        req_path = AUTO_DIR / "stage4_requirements.json"
                        if req_path.exists():
                            with open(req_path, encoding="utf-8") as f:
                                stage4_output = json.load(f)
                            stage4_output["status"] = "completed"
                            stage4_output["output_file"] = str(req_path)
                            mark_stage_complete(state, 4, stage4_output, args.state)
                            # Print unmatched variables that need attention
                            unmatched = stage4_output.get("unmatched", [])
                            if unmatched:
                                print(f"\n  ⚠ {len(unmatched)} variable(s) not matched in variable_map.json:")
                                for u in unmatched:
                                    print(f"    - {u}")
                                print("  These require manual source specification or Tier B/C handling.")
                else:
                    print(f"[DRY RUN] {' '.join(cmd)}")
            else:
                print("Stage 4: Stage 3 output not found. Run Stage 3 first.")

        if stage == 5:
            print(f"\n{'='*60}")
            print("Stage 5: Data Acquisition")
            print(f"{'='*60}")

            # Read Stage 4 requirements for auto-fetch
            req_path = AUTO_DIR / "stage4_requirements.json"
            if req_path.exists():
                with open(req_path, encoding="utf-8") as f:
                    req = json.load(f)

                auto_vars = [v["matched_key"] for v in req.get("variables", [])
                            if v.get("acquisition") == "auto" and v.get("matched_key")]
                manual_vars = [v for v in req.get("variables", [])
                              if v.get("acquisition") != "auto"]

                if auto_vars and not args.dry_run:
                    print(f"\n  Auto-fetching {len(auto_vars)} Tier A variables: {auto_vars}")
                    try:
                        from fetch_data import fetch_from_variable_map, results_to_data_status
                        results = fetch_from_variable_map(auto_vars)
                        data_status = results_to_data_status(results)

                        # Count successes
                        n_ok = sum(1 for r in results.values() if r.is_ok())
                        print(f"  ✓ {n_ok}/{len(auto_vars)} variables fetched successfully")

                        # Store in state
                        stage5_output = state["stages"].get("stage5", {})
                        # Merge with existing data_status
                        existing = stage5_output.get("data_status", {})
                        existing.update(data_status)
                        stage5_output["data_status"] = existing
                        stage5_output["status"] = "in_progress"
                        mark_stage_complete(state, 5, stage5_output, args.state)
                    except Exception as e:
                        print(f"  ✗ Auto-fetch error: {e}")
                elif args.dry_run:
                    print(f"  [DRY RUN] Would auto-fetch: {auto_vars}")

                if manual_vars:
                    print(f"\n  ⚠ {len(manual_vars)} variable(s) require manual handling:")
                    for mv in manual_vars:
                        print(f"    - {mv['concept']} (Tier {mv.get('tier', '?')}): "
                              f"{mv.get('description', '')}")
                        if mv.get("alternatives"):
                            alts = [a["key"] for a in mv["alternatives"][:3]]
                            print(f"      Alternatives in variable_map: {alts}")
                    print(f"\n  Place Tier B/C data in data/raw/ or data/manual/,")
                    print(f"  then re-run with: --from-stage 5")
            else:
                print("  Stage 4 requirements not found. Run Stage 4 first.")
                if not args.dry_run:
                    print(f"  Or place data files in data/raw/ and run:")
                    print(f"    python {__file__} --state {args.state} --from-stage 6 --data <path>")

        if stage == 6:
            if not state.get("data_path"):
                print("Stage 6: No data path set. Use --data to specify the panel data file.")
                break
            run_stage6(state, args.state, dry_run=args.dry_run)

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
