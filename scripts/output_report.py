"""
Extract structured data from all stage outputs into a unified JSON format
for downstream rendering (Markdown / XeLaTeX).

Usage:
    python scripts/output_report.py --policy "Environmental Tax Reform" --outcome "SO2 Emissions" \\
        --stage6 data/auto/stage6_confirmation.json \\
        --stage7 data/auto/stage7_main_result.json \\
        --stage8 data/auto/stage8_sensitivity.json \\
        --stage8-placebo data/auto/stage8_placebo.json \\
        --output data/auto/report_data.json

    # Also print a text preview
    python scripts/output_report.py ... --text
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.stats import norm
except ImportError:
    norm = None


def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Failed to load {path}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7 extraction
# ═══════════════════════════════════════════════════════════════════════════

def _unwrap_stage7(result: dict) -> dict:
    # All known wrapper keys from estimation scripts
    wrapper_keys = [
        "callaway_santanna", "2sls", "liml", "synthetic_did",
        "scm", "synthetic_control", "dml", "causal_forest",
        "rdd", "sharp_rdd", "fuzzy_rdd",
        "did", "twfe",
    ]
    for wk in wrapper_keys:
        if wk in result and isinstance(result[wk], dict):
            return result[wk]
    return result


def _extract_coef(result: dict) -> float:
    r = _unwrap_stage7(result)
    for key in ["coefficient", "att", "overall_att", "late", "ate", "aggregate_att"]:
        if key in r and r[key] is not None:
            return float(r[key])
    return 0.0


def _extract_se(result: dict) -> float:
    r = _unwrap_stage7(result)
    for key in ["std_error", "se", "overall_se", "aggregate_se"]:
        if key in r and r[key] is not None:
            return float(r[key])
    if "placebo_std" in result and result["placebo_std"] is not None:
        return float(result["placebo_std"])
    return 0.0


def _extract_pval(result: dict) -> float:
    r = _unwrap_stage7(result)
    for key in ["p_value", "pval", "overall_p_value"]:
        if key in r and r[key] is not None:
            return float(r[key])
    t_stat = r.get("t_stat") or r.get("overall_t_stat")
    if t_stat is not None and norm is not None:
        return float(2 * (1 - norm.cdf(abs(float(t_stat)))))
    return 1.0


def _extract_method_name(result: dict) -> str:
    return _unwrap_stage7(result).get("method", "")


def _direction(coef: float) -> str:
    if coef > 0:
        return "increased"
    elif coef < 0:
        return "decreased"
    return "had no measurable effect on"


def _effect_magnitude(result: dict, coef: float) -> str:
    r = _unwrap_stage7(result)
    if r.get("outcome", "").startswith("log_"):
        pct = (np.exp(coef) - 1) * 100
        return f"{pct:.2f}% (log-point interpretation)"
    return ""


def _treatment_var_label(mechanism: str) -> str:
    """Return the correct treatment variable label for a given mechanism."""
    labels = {
        "staggered_policy_shock": "ATT(g,t) — 组别-时间特定平均处理效应",
        "single_policy_shock": "Treated × Post",
        "threshold_rule": "Above Threshold",
        "time_varying_unobservables": "Treatment (endogenous)",
        "selection_on_observables": "Treatment indicator",
        "continuous_intensity": "Treatment Intensity × Post",
        "random_assignment": "Treatment assignment",
        "multiple_overlapping_policies": "Treated × Post (per policy)",
    }
    return labels.get(mechanism, "Treated × Post")


def _has_entity_fe(mechanism: str) -> bool:
    """Whether entity fixed effects are typically used for this method."""
    no_fe_methods = {"threshold_rule", "selection_on_observables"}
    return mechanism not in no_fe_methods


def _has_time_fe(mechanism: str) -> bool:
    """Whether time fixed effects are typically used for this method."""
    no_fe_methods = {"threshold_rule", "selection_on_observables", "time_varying_unobservables"}
    return mechanism not in no_fe_methods


# ═══════════════════════════════════════════════════════════════════════════
# Stage 8 normalization
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_checks(stage8: dict, stage8_placebo: dict = None,
                       stage8_summary: dict = None,
                       stage8_alt_windows: dict = None) -> list[dict]:
    # If we have the consolidated summary, use it directly
    if stage8_summary:
        checks = stage8_summary.get("checks", [])
        # Translate names from English (raw script output) to Chinese where needed
        for c in checks:
            if c.get("name") == "Placebo permutation test":
                c["name"] = "安慰剂置换检验"
            elif c.get("name") == "Bacon decomposition":
                c["name"] = "Goodman-Bacon分解（负权重检验）"
        return checks

    # Fallback: build from individual outputs
    checks = []
    if stage8:
        summary = stage8.get("summary", {})
        raw = summary.get("checks", []) or stage8.get("checks", [])
        for c in raw:
            checks.append({
                "name": c.get("name", "Unknown"),
                "passed": c.get("passed", False),
                "interpretation": c.get("interpretation", ""),
            })
    if stage8_placebo:
        passed = stage8_placebo.get("passed", False)
        p_val = stage8_placebo.get("p_value", 1.0)
        checks.append({
            "name": "Placebo permutation test",
            "passed": passed,
            "interpretation": (
                f"Actual effect {'stands out from' if passed else 'does not differ from'} "
                f"placebo distribution (p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'})."
            ),
        })
    if stage8_alt_windows:
        for entry in stage8_alt_windows.get("windows", stage8_alt_windows.get("results", [])):
            window_name = entry.get("window", entry.get("name", "Unknown window"))
            coef = entry.get("coefficient", entry.get("coef", None))
            passed = coef is not None
            checks.append({
                "name": f"Alternative window: {window_name}",
                "passed": passed,
                "interpretation": (
                    f"Coefficient = {coef:.4f}" if coef is not None
                    else "Window estimation failed"
                ),
            })
    return checks


# ═══════════════════════════════════════════════════════════════════════════
# Data description builder (uses Stage 4+5 results)
# ═══════════════════════════════════════════════════════════════════════════

def _build_data_desc(data_path, data_meta, main_result, data_status, data_span):
    """Generate a data source description paragraph for section 3.1.

    Uses Stage 5 data_status (what was actually acquired, from where) and
    data_meta (computed from the actual panel) to produce publication-quality
    prose describing where each variable came from.
    """
    parts = []
    n_cities = data_meta.get("n_cities", 0)
    n_treated = data_meta.get("n_treated", 0)
    n_control = data_meta.get("n_control", 0)
    n_years = data_meta.get("n_years", 0)
    span = data_meta.get("time_span", data_span or f"{n_years}年")
    entity_label = "省级行政区" if (isinstance(n_cities, int) and n_cities <= 35) else "地级及以上城市"

    # Paragraph 1: Panel structure
    p1 = f"本文使用的数据集为{span}中国{n_cities}个{entity_label}的面板数据"
    if n_treated and n_control:
        p1 += f"，其中处理组{n_treated}个城市、对照组{n_control}个城市"
    n_obs = data_meta.get("n_obs", "")
    if n_obs:
        p1 += f"，共{n_obs}个观测值"
    p1 += "。"
    parts.append(p1)

    # Paragraph 2: Variable sources from data_status (Stage 5)
    if data_status:
        # Group by tier
        tier_a_vars = []
        tier_b_vars = []
        tier_c_vars = []
        for var_name, info in data_status.items():
            tier = info.get("tier", "?")
            desc = info.get("description", var_name)
            status = info.get("status", "?")
            source_info = info.get("source", "")
            entry = (var_name, desc, source_info, status)
            if tier == "A":
                tier_a_vars.append(entry)
            elif tier == "B":
                tier_b_vars.append(entry)
            elif tier == "C":
                tier_c_vars.append(entry)

        if tier_a_vars:
            p2 = "数据来自以下公开来源（Tier A）："
            for var_name, desc, src, status in tier_a_vars:
                status_note = "" if status == "fetched" else f"（{status}）"
                p2 += f"{desc}{status_note}；"
            p2 = p2.rstrip("；") + "。"
            parts.append(p2)

        if tier_b_vars:
            p3 = "以下数据来自需申请的微观调查数据库（Tier B）："
            for var_name, desc, src, status in tier_b_vars:
                p3 += f"{desc}（来源：{src or '未注明'}）；"
            p3 = p3.rstrip("；") + "。"
            parts.append(p3)

        if tier_c_vars:
            p4 = "以下数据通过手动收集获得（Tier C）："
            for var_name, desc, src, status in tier_c_vars:
                p4 += f"{desc}；"
            p4 = p4.rstrip("；") + "。"
            parts.append(p4)
    else:
        # Fallback without data_status — build a basic description
        outcome_label = main_result.get("outcome", "被解释变量")
        outcome_name_raw = outcome_label if isinstance(outcome_label, str) else "被解释变量"
        # Use CTRL_VARDEFS lookup for proper Chinese label
        outcome_name = CTRL_VARDEFS.get(str(outcome_name_raw), (str(outcome_name_raw),))[0]
        method = main_result.get("method", "双重差分")
        is_cs = "Callaway" in method or "Sant'Anna" in method
        if is_cs:
            parts.append(f"被解释变量为{outcome_name}，核心解释变量为组别-时间特定平均处理效应ATT(g,t)。估计方法为{method}。")
        else:
            parts.append(f"被解释变量为{outcome_name}，核心解释变量为处理变量与时间虚拟变量的交互项。估计方法为{method}。")
        controls = main_result.get("controls", [])
        ctrl_labels = {
            "log_gdp_pc": "人均GDP（对数）", "log_population": "人口规模（对数）",
            "gdp_per_capita": "人均GDP", "population_10k": "人口规模",
            "fiscal_revenue_pc": "人均财政收入", "n_hospitals": "医院数量",
            "elderly_ratio": "老龄化率（65岁以上占比）",
            "female_labor_participation": "女性劳动参与率",
            "urbanization_rate": "城镇化率", "industrial_rate": "工业化率",
            "energy_consumption": "能源消费总量",
        }
        if controls:
            ctrl_descs = [ctrl_labels.get(c, c) for c in controls]
            parts.append(f"控制变量包括{'、'.join(ctrl_descs)}。")
        # Add treatment info
        n_treated = data_meta.get("n_treated", 0)
        n_control = data_meta.get("n_control", 0)
        if n_treated and n_control:
            parts.append(f"处理组包含{n_treated}个单元，对照组包含{n_control}个单元。")

    # Paragraph 3: Processing notes
    if data_path:
        p = Path(data_path)
        parts.append(f"原始数据经清洗、合并后形成分析用面板数据（{p.name}，{p.suffix.upper().lstrip('.')}格式）。")

    return "\n\n".join(parts)


def _build_variable_source_map(data_status, data_path=None):
    """Build a mapping from variable column name to data source description.

    Uses Stage 5 data_status to look up where each variable came from.
    Returns a dict: variable_name → source_string
    """
    source_map = {}
    if not data_status:
        return source_map

    # Direct matches from data_status
    for var_name, info in data_status.items():
        tier = info.get("tier", "?")
        desc = info.get("description", "")
        src = info.get("source", "")

        tier_label = {"A": "公开数据库", "B": "微观调查数据", "C": "手动收集", "D": "不可获取"}
        label = tier_label.get(tier, f"Tier {tier}")

        if src:
            source_str = f"{label}：{src}"
        elif desc:
            source_str = f"{label}：{desc}"
        else:
            source_str = label

        source_map[var_name] = source_str

    return source_map


# ═══════════════════════════════════════════════════════════════════════════
# Output path helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    """Convert a name to a safe filename slug."""
    slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).lower()
    return slug.strip("_") or "report"


def _find_or_create_subject_dir(policy_name: str, base_dir: Path) -> Path:
    """Find an existing subject folder or create a new one.

    Avoids creating duplicate folders for the same policy entered with
    different spellings (e.g., 'env-tax' vs '环境保护税改革').
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_filename(policy_name)

    # List existing subject directories (not timestamp subdirs)
    existing = [d for d in base_dir.iterdir() if d.is_dir()]

    # 1. Exact match
    for d in existing:
        if d.name.lower() == slug:
            return d

    # 2. Slug is a substring of an existing dir (or vice versa)
    for d in existing:
        if slug in d.name.lower() or d.name.lower() in slug:
            return d

    # 3. First 5 chars match (covers most spelling variations)
    for d in existing:
        if len(d.name) >= 5 and len(slug) >= 5:
            if d.name[:5].lower() == slug[:5]:
                return d

    # 4. No match — create new
    new_dir = base_dir / slug
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


# ═══════════════════════════════════════════════════════════════════════════
# Reference extraction (runs before rendering; incomplete refs flagged for LLM web search)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_author_year(authors_str: str, year_str: str) -> str:
    """Normalize author names and year into a lookup key.

    "Callaway & Sant'Anna" + "2021" → "callaway|santanna|2021"
    "Zhang" + "2020" → "zhang|2020"
    """
    parts = re.split(r'\s+(?:&|and)\s+|,\s*', authors_str)
    last_names = []
    for p in parts:
        p = p.strip().lower()
        p = re.sub(r'\.', '', p)
        words = p.split()
        if words:
            last_names.append(words[-1])
    return "|".join(last_names) + "|" + year_str


def _build_reference_database():
    """Canonical reference database. Maps normalized author-year keys to full citations.

    When a citation is found in prose text but not in this database, it is
    flagged for LLM web search during the Stage 9 translation step.
    """
    db = {}

    # ── Staggered / Heterogeneous DID ──
    db["callaway|santanna|2021"] = (
        "Callaway, B. & Sant'Anna, P.H. (2021). "
        "Difference-in-differences with multiple time periods. "
        "Journal of Econometrics, 225(2), 200-230."
    )
    db["goodman-bacon|2021"] = (
        "Goodman-Bacon, A. (2021). "
        "Difference-in-differences with variation in treatment timing. "
        "Journal of Econometrics, 225(2), 254-277."
    )
    db["sun|abraham|2021"] = (
        "Sun, L. & Abraham, S. (2021). "
        "Estimating dynamic treatment effects in event studies with "
        "heterogeneous treatment effects. Journal of Econometrics, 225(2), 175-199."
    )
    db["dechaisemartin|dhaultfoeuille|2020"] = (
        "de Chaisemartin, C. & D'Haultfoeuille, X. (2020). "
        "Two-way fixed effects estimators with heterogeneous treatment effects. "
        "American Economic Review, 110(9), 2964-2996."
    )
    db["baker|larcker|wang|2022"] = (
        "Baker, A.C., Larcker, D.F. & Wang, C.C. (2022). "
        "How much should we trust staggered difference-in-differences estimates? "
        "Journal of Financial Economics, 144(2), 370-395."
    )
    db["borusyak|jaravel|spiess|2024"] = (
        "Borusyak, K., Jaravel, X. & Spiess, J. (2024). "
        "Revisiting event study designs: Robust and efficient estimation. "
        "Review of Economic Studies, 91(2), 1023-1060."
    )
    db["gardner|2022"] = (
        "Gardner, J. (2022). "
        "Two-stage differences in differences. arXiv:2207.05943."
    )

    # ── Synthetic Control / SCM ──
    db["abadie|gardeazabal|2003"] = (
        "Abadie, A. & Gardeazabal, J. (2003). "
        "The economic costs of conflict: A case study of the Basque Country. "
        "American Economic Review, 93(1), 113-132."
    )
    db["abadie|diamond|hainmueller|2010"] = (
        "Abadie, A., Diamond, A. & Hainmueller, J. (2010). "
        "Synthetic control methods for comparative case studies: Estimating the "
        "effect of California's tobacco control program. "
        "Journal of the American Statistical Association, 105(490), 493-505."
    )
    db["arkhangelsky|athey|hirshberg|imbens|wager|2021"] = (
        "Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. & Wager, S. (2021). "
        "Synthetic difference-in-differences. American Economic Review, 111(12), 4088-4118."
    )

    # ── RDD ──
    db["imbens|lemieux|2008"] = (
        "Imbens, G.W. & Lemieux, T. (2008). "
        "Regression discontinuity designs: A guide to practice. "
        "Journal of Econometrics, 142(2), 615-635."
    )
    db["mccrary|2008"] = (
        "McCrary, J. (2008). "
        "Manipulation of the running variable in the regression discontinuity design: "
        "A density test. Journal of Econometrics, 142(2), 698-714."
    )
    db["calonico|cattaneo|titiunik|2014"] = (
        "Calonico, S., Cattaneo, M.D. & Titiunik, R. (2014). "
        "Robust nonparametric confidence intervals for regression-discontinuity designs. "
        "Econometrica, 82(6), 2295-2326."
    )

    # ── Causal Forest / ML ──
    db["athey|imbens|2016"] = (
        "Athey, S. & Imbens, G. (2016). "
        "Recursive partitioning for heterogeneous causal effects. "
        "Proceedings of the National Academy of Sciences, 113(27), 7353-7360."
    )
    db["wager|athey|2018"] = (
        "Wager, S. & Athey, S. (2018). "
        "Estimation and inference of heterogeneous treatment effects using random forests. "
        "Journal of the American Statistical Association, 113(523), 1228-1242."
    )
    db["chernozhukov|chetverikov|demirer|duflo|hansen|newey|robins|2018"] = (
        "Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., "
        "Newey, W. & Robins, J. (2018). "
        "Double/debiased machine learning for treatment and structural parameters. "
        "The Econometrics Journal, 21(1), C1-C68."
    )

    # ── Sensitivity / Robustness ──
    db["oster|2019"] = (
        "Oster, E. (2019). "
        "Unobservable selection and coefficient stability: Theory and evidence. "
        "Journal of Business & Economic Statistics, 37(2), 187-204."
    )
    db["rosenbaum|2002"] = (
        "Rosenbaum, P.R. (2002). "
        "Observational Studies (2nd ed.). Springer."
    )
    db["roth|2022"] = (
        "Roth, J. (2022). "
        "Pretest with caution: Event-study estimates after testing for parallel trends. "
        "American Economic Review: Insights, 4(3), 305-322."
    )
    db["rambachan|roth|2023"] = (
        "Rambachan, A. & Roth, J. (2023). "
        "A more credible approach to parallel trends. "
        "Review of Economic Studies, 90(5), 2555-2591."
    )

    # ── Handbooks / Textbooks ──
    db["angrist|pischke|2009"] = (
        "Angrist, J.D. & Pischke, J.S. (2009). "
        "Mostly Harmless Econometrics: An Empiricist's Companion. "
        "Princeton University Press."
    )
    db["cunningham|2021"] = (
        "Cunningham, S. (2021). "
        "Causal Inference: The Mixtape. Yale University Press."
    )
    db["imbens|rubin|2015"] = (
        "Imbens, G.W. & Rubin, D.B. (2015). "
        "Causal Inference for Statistics, Social, and Biomedical Sciences. "
        "Cambridge University Press."
    )

    # ── Placebo / Permutation ──
    db["abadie|athey|imbens|wooldridge|2023"] = (
        "Abadie, A., Athey, S., Imbens, G.W. & Wooldridge, J.M. (2023). "
        "When should you adjust standard errors for clustering? "
        "Quarterly Journal of Economics, 138(1), 1-35."
    )
    db["bertrand|duflo|mullainathan|2004"] = (
        "Bertrand, M., Duflo, E. & Mullainathan, S. (2004). "
        "How much should we trust differences-in-differences estimates? "
        "Quarterly Journal of Economics, 119(1), 249-275."
    )
    db["mackinnon|webb|2020"] = (
        "MacKinnon, J.G. & Webb, M.D. (2020). "
        "Randomization inference for difference-in-differences with few treated clusters. "
        "Journal of Econometrics, 218(2), 435-450."
    )

    # ── IV ──
    db["angrist|imbens|rubin|1996"] = (
        "Angrist, J.D., Imbens, G.W. & Rubin, D.B. (1996). "
        "Identification of causal effects using instrumental variables. "
        "Journal of the American Statistical Association, 91(434), 444-455."
    )
    db["montielolea|pflueger|2013"] = (
        "Montiel Olea, J.L. & Pflueger, C. (2013). "
        "A robust test for weak instruments. "
        "Journal of Business & Economic Statistics, 31(3), 358-369."
    )

    return db


def _extract_references(sec: dict) -> list[dict]:
    """Extract references from literature sections + canonical database.

    Returns a list of dicts:
      {"text": "Full citation...", "complete": true}   — matched in DB
      {"text": "Author (Year)", "complete": false,
       "search_query": "Author Year paper", "authors": "...", "year": "..."}
         — needs LLM web search to complete

    Incomplete entries are flagged so the Stage 9 LLM translation step can
    search the web and fill in the full citation before rendering.
    """
    lit = sec.get("intro_literature", "")
    if not lit:
        lit = sec.get("intro_background", "")
    institution = sec.get("institution", "")
    theory_text = sec.get("theory", "")
    all_text = " ".join([lit, institution, theory_text])

    ref_db = _build_reference_database()

    # Extract author-year citations from prose
    author_year = re.findall(
        r'([A-Z][a-z]*(?:\s+(?:&|and)\s+[A-Z][a-z]*)*(?:\s+et\s+al\.?)?)\s*[\(（](\d{4})[\)）]',
        all_text
    )

    refs = []
    seen = set()

    for authors, year in author_year:
        authors = authors.strip()
        if len(authors) < 2:
            continue
        bare_key = f"{authors} ({year})"
        if bare_key in seen:
            continue
        seen.add(bare_key)

        # Try exact match against canonical database
        lookup = _normalize_author_year(authors, year)
        if lookup in ref_db:
            full_ref = ref_db[lookup]
            if full_ref not in seen:
                seen.add(full_ref)
                refs.append({"text": full_ref, "complete": True})
        else:
            # Try partial match
            lookup_parts = set(lookup.split("|"))
            best_match = None
            for db_key, db_ref in ref_db.items():
                db_parts = set(db_key.split("|"))
                if year in db_parts and len(lookup_parts & db_parts) >= 2:
                    if db_ref not in seen:
                        best_match = db_ref
                        break
            if best_match:
                seen.add(best_match)
                refs.append({"text": best_match, "complete": True})
            else:
                # Flag for LLM web search
                search_query = f"{authors} {year} paper"
                refs.append({
                    "text": bare_key,
                    "complete": False,
                    "search_query": search_query,
                    "authors": authors,
                    "year": year,
                })

    # Add explicitly listed references, remove incomplete regex duplicates
    explicit_refs = sec.get("references", [])
    if isinstance(explicit_refs, list) and explicit_refs:
        # Extract author-year from each explicit ref to dedup against regex ones
        explicit_keys = set()
        for er in explicit_refs:
            m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*).*?\((\d{4})\)', er)
            if m:
                explicit_keys.add(f"{m.group(1)} ({m.group(2)})")
        # Remove regex-extracted incomplete refs that match explicit refs
        refs = [r for r in refs if not (
            not r.get("complete", True) and
            any(r.get("text", "").startswith(k) for k in explicit_keys)
        )]
        # Prepend explicit refs
        for er in reversed(explicit_refs):
            refs.insert(0, {"text": er, "complete": True})

    # Add canonical method references not cited in text but essential
    canonical_must_include = [
        "callaway|santanna|2021", "goodman-bacon|2021",
        "sun|abraham|2021", "dechaisemartin|dhaultfoeuille|2020",
        "angrist|pischke|2009", "roth|2022",
    ]
    for ck in canonical_must_include:
        cr = ref_db.get(ck, "")
        if cr and cr not in seen:
            seen.add(cr)
            refs.append({"text": cr, "complete": True})

    return refs


# ═══════════════════════════════════════════════════════════════════════════
# Structured data builder
# ═══════════════════════════════════════════════════════════════════════════

# Variable name → (Chinese label, definition, data source)
CTRL_VARDEFS = {
    # Outcome variables
    "birth_rate":            ("出生率", "每千人口出生人数（‰）", "中国城市统计年鉴"),
    "log_birth_rate":        ("出生率（对数）", "ln(出生率)，取对数以解释弹性效应", "中国城市统计年鉴"),
    "log_so2":               ("SO₂排放（对数）", "ln(工业SO₂排放量/万吨)", "中国环境统计年鉴"),
    # Control variables
    "gdp_per_capita":        ("人均GDP", "地区生产总值/常住人口（元/人）", "中国城市统计年鉴"),
    "population_10k":        ("人口规模（万人）", "年末户籍人口或常住人口", "中国城市统计年鉴"),
    "elderly_ratio":         ("老龄化率", "65岁以上人口占总人口比例", "中国城市统计年鉴"),
    "urbanization_rate":     ("城镇化率", "城镇人口占总人口比例", "中国城市统计年鉴"),
    "fiscal_revenue_pc":     ("人均财政收入", "地方一般公共预算收入/常住人口（元/人）", "中国城市统计年鉴"),
    "female_labor_participation": ("女性劳动参与率", "女性就业人口占女性劳动年龄人口比例", "中国劳动统计年鉴"),
    "industrial_rate":       ("工业化率", "工业增加值占GDP比例", "中国城市统计年鉴"),
    "n_hospitals":           ("医院数量", "辖区内医院总数（个）", "中国城市统计年鉴"),
    "energy_consumption":    ("能源消耗", "单位GDP能耗（吨标准煤/万元）", "中国能源统计年鉴"),
    "log_gdp_pc":            ("人均GDP（对数）", "ln(地区生产总值/常住人口)", "中国城市统计年鉴"),
    "log_population":        ("人口规模（对数）", "ln(年末常住人口)", "中国城市统计年鉴"),
}


def build_report_data(
    policy: str,
    outcome: str,
    stage3: dict = None,
    stage6: dict = None,
    stage7: dict = None,
    stage8: dict = None,
    stage8_placebo: dict = None,
    stage8_summary: dict = None,
    stage8_alt_windows: dict = None,
    data_source: str = "",
    data_span: str = "",
    n_obs: str = "",
    data_status: dict = None,
    stage2_sections: dict = None,
    data_path: str = None,
    event_study_path: str = None,
) -> dict:
    """Build a unified report data dict for downstream rendering."""

    data = {
        "policy": policy,
        "outcome": outcome,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # ── Method chain ──
    theory = ""
    final = ""
    changed = False
    gap = ""
    chain_list = []

    if stage6:
        theory = stage6.get("theoretical_method", "")
        final = stage6.get("final_method", "")
        changed = stage6.get("method_changed", False)
        gap = stage6.get("gap_explanation", "")
        chain_list = stage6.get("chain", [theory, final] if theory and final else [])

    if not theory and stage3:
        theory = stage3.get("primary_method",
                 stage3.get("recommendation", {}).get("primary_method", ""))

    data["method_chain"] = {
        "theoretical": theory,
        "final": final,
        "changed": changed,
        "gap_explanation": gap,
        "chain": chain_list,
    }

    # ── Data metadata ──
    method_name = ""
    obs_str = n_obs
    if stage7:
        method_name = _extract_method_name(stage7)
        if not obs_str:
            r = _unwrap_stage7(stage7)
            n = r.get("n_obs", 0)
            entities = r.get("n_entities", 0)
            periods = r.get("n_periods", 0) or r.get("n_total_periods", 0)
            obs_str = f"{n}"
            if entities and periods:
                obs_str += f" ({entities} 个城市 × {periods} 年)"

    # Try to extract a meaningful data source description
    src_label = data_source
    if not src_label:
        if data_path:
            src_label = f"面板数据（{Path(data_path).name}）"
        elif data_status:
            fetched = [v.get("description", k) for k, v in data_status.items()
                       if v.get("status") == "fetched"]
            if fetched:
                src_label = "、".join(fetched[:5])
        else:
            src_label = method_name if method_name else "面板数据"

    data["data_meta"] = {
        "source": src_label,
        "data_path": data_path or "",
        "span": data_span,
        "n_obs": obs_str,
    }

    # ── Main result ──
    main_result = {}
    if stage7:
        # Multi-spec format (from --baseline mode)
        if "specifications" in stage7:
            specs = stage7["specifications"]
            main_result["specifications"] = specs
            last = specs[-1]
            main_result["coefficient"] = last["coefficient"]
            main_result["std_error"] = last["std_error"]
            main_result["p_value"] = last["p_value"]
            main_result["method"] = stage7.get("method", "")
            mech = (stage6 or {}).get("mechanism", "") or (stage3 or {}).get("mechanism", "")
            main_result["treatment_var"] = _treatment_var_label(mech)
            main_result["controls"] = last.get("controls", [])
            main_result["n_obs"] = last.get("n_obs", 0)
            main_result["n_entities"] = last.get("n_entities", 0)
            main_result["n_periods"] = last.get("n_periods", 0)
            main_result["r2"] = last.get("r2")
            main_result["direction"] = _direction(last["coefficient"])
            main_result["effect_magnitude"] = _effect_magnitude(stage7, last["coefficient"])
            main_result["entity_fe"] = _has_entity_fe(mech)
            main_result["time_fe"] = _has_time_fe(mech)
        else:
            # Legacy single-spec format
            coef = _extract_coef(stage7)
            se = _extract_se(stage7)
            pval = _extract_pval(stage7)
            r = _unwrap_stage7(stage7)
            controls = r.get("controls", [])

            main_result["coefficient"] = coef
            main_result["std_error"] = se
            main_result["p_value"] = pval
            main_result["method"] = r.get("method", "")
            main_result["outcome"] = r.get("outcome", outcome)
            mech = (stage6 or {}).get("mechanism", "") or (stage3 or {}).get("mechanism", "")
            main_result["treatment_var"] = _treatment_var_label(mech)
            main_result["controls"] = controls
            main_result["n_obs"] = r.get("n_obs", 0)
            main_result["n_entities"] = r.get("n_entities", 0)
            main_result["n_periods"] = r.get("n_periods", 0) or r.get("n_total_periods", 0)
            main_result["r2"] = r.get("r2_within", r.get("r2"))
            main_result["direction"] = _direction(coef)
            main_result["effect_magnitude"] = _effect_magnitude(stage7, coef)
            main_result["entity_fe"] = _has_entity_fe(mech)
            main_result["time_fe"] = _has_time_fe(mech)

    data["main_result"] = main_result

    # Fix data_meta from main_result (built after data_meta)
    if main_result:
        if data["data_meta"]["n_obs"] in ("0", ""):
            if main_result.get("specifications"):
                last_spec = main_result["specifications"][-1]
                n = last_spec.get("n_obs", 0)
                entities = last_spec.get("n_entities", 0)
                periods = last_spec.get("n_periods", 0)
                obs_str = str(n) if n else ""
                if entities and periods:
                    obs_str += f" ({entities} 个城市 × {periods} 年)"
                data["data_meta"]["n_obs"] = obs_str
            elif main_result.get("n_obs"):
                data["data_meta"]["n_obs"] = str(main_result["n_obs"])
            elif data["data_meta"].get("n_cities") and data["data_meta"].get("n_years"):
                data["data_meta"]["n_obs"] = str(data["data_meta"]["n_cities"] * data["data_meta"]["n_years"])
        if not data["data_meta"].get("n_cities"):
            data["data_meta"]["n_cities"] = main_result.get("n_entities", 0)
        if not data["data_meta"].get("n_years"):
            data["data_meta"]["n_years"] = main_result.get("n_periods", 0)

    # ── Assumption verdicts (directly from Stage 6) ──
    data["assumptions"] = stage6.get("assumption_verdicts", []) if stage6 else []

    # ── Event study ──
    event_study = {}
    if event_study_path and Path(event_study_path).exists():
        with open(event_study_path, encoding="utf-8") as f:
            event_study = json.load(f)
    elif stage7:
        # First check: embedded event study data in unwrapped stage7 result
        r = _unwrap_stage7(stage7)
        if "event_study" in r and isinstance(r["event_study"], dict):
            event_study = r["event_study"]
        # Second check: event study references an external file
        es_file = stage7.get("event_study") or r.get("event_study")
        if not event_study and es_file and isinstance(es_file, dict) and es_file.get("output_file"):
            es_path = es_file["output_file"]
            if Path(es_path).exists():
                with open(es_path, encoding="utf-8") as f:
                    event_study = json.load(f)
    # Normalize event study to the format render_report.py expects:
    # {"coefficients": {t: {coefficient, std_error, p_value}}, "pre_trends_test": {f_stat, p_value}}
    if event_study and "coefficients" not in event_study:
        if "periods" in event_study and "coefs" in event_study:
            periods = event_study.get("periods", [])
            coefs = event_study.get("coefs", [])
            ses = event_study.get("ses", [])
            pvals = event_study.get("pvals", [])
            coeff_dict = {}
            for i, t in enumerate(periods):
                coeff_dict[str(t)] = {
                    "coefficient": coefs[i] if i < len(coefs) else 0.0,
                    "std_error": ses[i] if i < len(ses) else 0.0,
                    "p_value": pvals[i] if i < len(pvals) else 1.0,
                }
            event_study["coefficients"] = coeff_dict
        if "pre_trends_f" in event_study and "pre_trends_test" not in event_study:
            event_study["pre_trends_test"] = {
                "f_stat": event_study.get("pre_trends_f"),
                "p_value": event_study.get("pre_trends_p"),
            }
    data["event_study"] = event_study

    # ── Fallback attempts ──
    data["fallback_attempts"] = stage6.get("fallback_attempts", []) if stage6 else []

    # ── Robustness ──
    data["robustness"] = _normalize_checks(stage8, stage8_placebo, stage8_summary, stage8_alt_windows)

    # ── Data quality ──
    data["data_quality"] = stage6.get("data_quality_summary", {}) if stage6 else {}

    # ── Limitations ──
    limitations = []
    if stage6:
        limitations.extend(stage6.get("limitations", []))
    if stage8:
        limitations.extend(stage8.get("limitations", []))
    data["limitations"] = limitations

    # ── Warnings ──
    data["warnings"] = stage6.get("warnings", []) if stage6 else []

    # ── Causal claim strength ──
    data["causal_claim_strength"] = (
        stage6.get("causal_claim_strength", "not assessed") if stage6 else "not assessed"
    )

    # ── Appendix ──
    appendix = {}

    # A.1 Policy background (from stage2 sections if available, else from stage3 basics)
    bg_lines = []
    if stage3:
        mech_label = stage3.get("mechanism_label", "")
        mech = stage3.get("mechanism", "")
        if mech_label:
            bg_lines.append(f"Assignment mechanism: {mech_label}")
        if mech:
            bg_lines.append(f"Mechanism type: {mech}")
        rec = stage3.get("recommendation", {})
        src_var = rec.get("source_of_variation", "")
        if src_var:
            bg_lines.append(f"Source of variation: {src_var}")
    if not bg_lines:
        bg_lines.append(f"Policy evaluation of {policy} on {outcome}.")
    appendix["policy_background"] = bg_lines

    # A.2 Method selection rationale — full Stage 3 output
    method_lines = []
    if stage3:
        method_lines.append(f"Assignment mechanism: {stage3.get('mechanism_label', '')}")
        method_lines.append(f"Method family: {stage3.get('method_family', '')}")
        method_lines.append("")
        rec = stage3.get("recommendation", {})
        primary = rec.get("primary_method", "")
        method_lines.append(f"Primary method: {primary}")
        src = rec.get("source_of_variation", "")
        if src:
            method_lines.append(f"Source of variation: {src}")
        method_lines.append("")

        why = rec.get("why", [])
        if why:
            method_lines.append("Why this method:")
            for i, w in enumerate(why, 1):
                method_lines.append(f"  {i}. {w}")
            method_lines.append("")

        assumptions = rec.get("assumptions", [])
        if assumptions:
            method_lines.append("Required assumptions:")
            for a in assumptions:
                testable = "可检验" if a.get("testable") else "需论证"
                method_lines.append(f"  • {a.get('name', '')} [{testable}]")
                if a.get("description"):
                    method_lines.append(f"    {a['description']}")
                if a.get("test_method"):
                    method_lines.append(f"    检验方法: {a['test_method']}")
            method_lines.append("")

        fallbacks_list = rec.get("fallbacks", [])
        if fallbacks_list:
            method_lines.append("Fallback strategies (in priority order):")
            for i, fb in enumerate(fallbacks_list, 1):
                method_lines.append(f"  {i}. {fb.get('method', 'Unknown')}")
                method_lines.append(f"     Condition: {fb.get('condition', '')}")
                method_lines.append(f"     Relaxes: {fb.get('assumption_relaxed', '')}")

    if not method_lines:
        method_lines.append("Method selection not documented.")
    appendix["method_rationale"] = method_lines

    # A.3 Data sources — auto-build from available pipeline info
    data_lines = []

    # 1. Main panel data file
    if data_path:
        p = Path(data_path)
        data_lines.append(f"分析数据文件：{p.name}（{p.suffix.upper().lstrip('.')}格式）")

    # 2. Data span and coverage
    if data_span:
        data_lines.append(f"数据时间跨度：{data_span}")
    elif "time_span" in data.get("data_meta", {}):
        data_lines.append(f"数据时间跨度：{data['data_meta']['time_span']}")

    # 3. Key variables from estimation
    if stage7:
        r = _unwrap_stage7(stage7)
        outcome_name = r.get("outcome", stage7.get("outcome", outcome))
        data_lines.append(f"被解释变量：{outcome_name}")

        controls = main_result.get("controls", [])
        if controls:
            data_lines.append(f"控制变量：{', '.join(controls)}")

    # 4. Method and software
    data_lines.append(f"估计方法：{method_name or final}")

    # 5. Tier A fetched data (if data_status available)
    if data_status:
        fetched_vars = []
        for var_name, info in data_status.items():
            tier = info.get("tier", "?")
            status = info.get("status", "?")
            desc = info.get("description", var_name)
            if status == "fetched":
                fetched_vars.append(f"[Tier {tier}] {desc}")
            elif status in ("requested", "received"):
                fetched_vars.append(f"[Tier {tier}] {desc} — {status}")
        if fetched_vars:
            data_lines.append("自动获取的数据（Tier A）：")
            data_lines.extend(f"  • {v}" for v in fetched_vars)

        # Tier B/C pending
        pending_vars = [(var_name, info) for var_name, info in data_status.items()
                        if info.get("status") in ("pending", "requested")]
        if pending_vars:
            data_lines.append("手动收集的数据（Tier B/C）：")
            for pv_name, pv_info in pending_vars:
                data_lines.append(f"  • {pv_info.get('description', pv_name)} — {pv_info.get('status', 'pending')}")

    # 6. Data quality from stage6
    if stage6:
        dq = stage6.get("data_quality_summary", {})
        if dq:
            data_lines.append(f"数据质量评估：{dq.get('overall', '?')}（发现{dq.get('total_issues', 0)}个问题）")
            critical = dq.get("critical_issues", [])
            if critical:
                data_lines.append(f"  严重问题：{len(critical)}个")

    # 6. Fallback: auto-generate source info from variable names
    if not data_status and stage7:
        controls = main_result.get("controls", [])
        sources_used = set()
        for ctrl in controls:
            src = CTRL_VARDEFS.get(ctrl, ("", "", ""))[2]
            if src:
                sources_used.add(src)
        if sources_used:
            data_lines.append("主要数据来源（根据变量自动推断）：")
            for src in sorted(sources_used):
                data_lines.append(f"  • {src}")
        if len(data_lines) <= 4:  # Only fallback info, no real data
            data_lines.append("提示：运行 Stage 5（数据获取）可生成完整的数据来源记录。")

    if not data_lines:
        data_lines.append("数据来源未记录。")

    appendix["data_sources"] = data_lines

    data["appendix"] = appendix

    # ── Narrative sections (from Stage 2 LLM output) ──
    if stage2_sections:
        data["sections"] = {
            "intro_background": stage2_sections.get("intro_background", ""),
            "intro_literature": stage2_sections.get("intro_literature", ""),
            "institution": stage2_sections.get("institution", ""),
            "theory": stage2_sections.get("theory", ""),
            "references": stage2_sections.get("references", []),
        }
    if stage2_sections and stage2_sections.get("conclusion"):
        data["sections"]["conclusion"] = stage2_sections["conclusion"]
    elif not data.get("sections"):
        data["sections"] = {}

    # Data description always comes from Stage 4+5 (not Stage 2)
    data["sections"]["data_desc"] = _build_data_desc(
        data_path, data.get("data_meta", {}), main_result, data_status, data_span
    )

    # ── References (extract from sections text + canonical DB) ──
    data["references"] = _extract_references(data.get("sections", {}))

    # ── Model specification (always built, from Stage 7 method + data) ──
    if not data.get("model_spec"):
        method_name = main_result.get("method", "")
        r = _unwrap_stage7(stage7) if stage7 else {}
        actual_controls = main_result.get("controls", [])
        n_treated = r.get("n_treated_units", data["data_meta"].get("n_treated", 0))
        n_control = r.get("n_control_units", data["data_meta"].get("n_control", 0))
        n_cities = n_treated + n_control or data["data_meta"].get("n_cities", 0)
        n_years = data["data_meta"].get("n_years", 0)
        n_periods = main_result.get("n_periods", n_years)
        n_cohorts = r.get("n_cohorts", 0)

        # Detect method type from method name
        is_cs = "Callaway" in method_name or "Sant'Anna" in method_name
        is_did = "DID" in method_name or "TWFE" in method_name or "difference-in-differences" in method_name.lower() or is_cs

        # Equation
        if is_cs:
            eq = r"Y_{it} = \alpha + \sum_{g} \sum_{t \ge g} \beta_{gt} \cdot \mathbb{1}(G_i=g, t \ge g) + \gamma X_{it} + \mu_i + \lambda_t + \varepsilon_{it}"
            eq_desc = "其中 Y 为被解释变量，G_i 表示城市 i 所属的处理批次（cohort），β_{gt} 为批次 g 在第 t 期的组别-时间特定处理效应（ATT(g,t)）。通过按队列规模加权平均得到总体ATT。控制变量向量 X 包含可能影响结果的时变特征。μ_i 和 λ_t 分别为城市固定效应和年份固定效应。"
        else:
            eq = r"Y_{it} = \alpha + \beta (Treated_i \times Post_{it}) + \gamma X_{it} + \mu_i + \lambda_t + \varepsilon_{it}"
            eq_desc = "其中 Y 为被解释变量，核心解释变量为处理组与处理后的交互项。控制变量向量 X 包含可能影响结果的时变特征。μ_i 和 λ_t 分别为城市固定效应和年份固定效应。"

        # Variable table
        DEP = "被解释变量"
        EXP = "核心解释变量"
        CTRL = "控制变量"
        variables = []
        oname = r.get("outcome", data.get("outcome", "被解释变量"))
        olabel, odef, osrc = CTRL_VARDEFS.get(str(oname), (str(oname), str(oname), ""))
        variables.append((DEP, olabel, odef, osrc))
        if is_cs:
            variables.append((EXP, "ATT(g,t)", "组别-时间特定平均处理效应（staggered treatment）", ""))
        else:
            variables.append((EXP, "Treated × Post", "处理组虚拟变量与处理后虚拟变量的交互项", ""))

        for ctrl in actual_controls:
            label, definition, source = CTRL_VARDEFS.get(ctrl, (ctrl, "控制变量", ""))
            variables.append((CTRL, label, definition, source))

        # ── Detailed identification narrative (generic, parameterized) ──
        ident_parts = []
        entity_label = "省份" if "province" in str(data_path or "").lower() else "城市" if "city" in str(data_path or "").lower() else "单元"

        # 1. Policy design → identification logic
        if is_cs:
            ident_parts.append(
                f"本研究利用{policy}分批推广的政策设计特征，构建因果识别策略。"
                f"该政策在不同{entity_label}分批次实施，形成了\"交错处理\"结构："
                f"不同{entity_label}在不同时点进入处理状态，且存在未受处理的对照组单元，"
                f"为利用交错型双重差分策略识别因果效应提供了准自然实验条件。"
            )
        else:
            ident_parts.append(
                f"本研究利用{policy}的政策设计特征，构建因果识别策略。"
                f"该政策在全国范围内实施，但不同{entity_label}受到的\"处理强度\"存在差异，"
                f"形成了高处理强度与低处理强度（或未处理）单元之间的可比变异。"
                f"本文以政策实施时间为断点，以处理强度较高（或受政策直接影响）的{entity_label}为处理组，"
                f"以处理强度较低（或不受政策直接影响）的{entity_label}为对照组，"
                f"构造双重差分（Difference-in-Differences, DID）识别框架。"
            )

        # 2. Why this specific method
        if is_cs:
            ident_parts.append(
                f"本文采用Callaway & Sant'Anna（2021）提出的交错型双重差分估计量（staggered DID）"
                f"作为基准识别策略。选择这一方法的原因是：处理时点交错的情境下，"
                f"传统双向固定效应（TWFE）估计量可能产生\"负权重\"问题（Goodman-Bacon, 2021）。"
                f"C&S估计量通过先估计组别-时间特定的ATT(g,t)，再按组别规模加权平均，避免了这一问题。"
                f"本文以{n_control}个未受处理{entity_label}作为对照组。"
            )
        else:
            method_label = main_result.get("method", "DID")
            ident_parts.append(
                f"本文采用{method_label}作为基准识别策略，利用处理组和对照组在政策实施前后"
                "的结果差异，通过双重差分消除不随时间变化的未观测混杂因素。"
            )

        # 3. Source of identification variation
        if is_cs:
            ident_parts.append(
                f"核心识别变异的来源为：同一时点上，不同处理批次{entity_label}与从未受处理{entity_label}之间"
                f"{outcome}变化的系统性差异。{entity_label}固定效应吸收了所有不随时间变化的"
                f"{entity_label}层面特征，年份固定效应吸收了所有{entity_label}共同面临的"
                f"时间趋势冲击。Callaway & Sant'Anna（2021）估计量通过先估计组别-时间特定"
                f"ATT(g,t)，再按队列规模加权平均得到总体ATT，这一系数可以解释为"
                f"{policy}对{outcome}的因果效应。"
            )
        else:
            ident_parts.append(
                f"核心识别变异的来源为：同一时点上，处理组{entity_label}与对照组{entity_label}之间"
                f"{outcome}变化的系统性差异。{entity_label}固定效应吸收了所有不随时间变化的"
                f"{entity_label}层面特征，年份固定效应吸收了所有{entity_label}共同面临的"
                f"时间趋势冲击，二者的联合控制使得\"处理组×处理后\"的交互项系数可以解释为"
                f"{policy}对{outcome}的因果效应。"
            )

        # 4. Key identifying assumptions
        ident_parts.append(
            "该方法依赖以下核心识别假设：\n"
            "  • 平行趋势假设（Parallel Trends）【可检验】：在不受处理的反事实情境下，"
            f"处理组和对照组的{outcome}变化趋势应当相同。通过事件研究法中处理前各期系数的联合F检验进行评估。\n"
            "  • 无预期效应（No Anticipation）【可检验】：受处理单元在政策正式实施前，"
            "不应因预期到政策变化而提前调整行为。通过检验处理前各期系数是否接近零来验证。\n"
            "  • 稳定单元处理值假设（SUTVA）【需论证】：一个单元的处理状态不应影响其他单元的结果变量。"
            "该假设的合理性取决于政策的具体设计，需根据制度背景进行论证。\n"
            "  • 条件独立性假设（CIA/Unconfoundedness）【需论证】：在控制固定效应"
            "及可观测时变控制变量的条件下，处理分配与潜在结果独立。"
            "通过纳入相关控制变量和进行Oster界限分析来增强该假设的可信度。"
        )

        # 6. Data and dimensions
        ident_parts.append(
            f"本文使用的数据集为{n_years}年（2010-2022）中国{n_cities}个地级及以上城市的面板数据，"
            f"其中处理组{n_treated}个城市（分{n_cohorts}个处理批次）、对照组{n_control}个城市。"
            f"共{n_periods}个时期，合计{data['data_meta'].get('n_obs', '')}个观测值。"
            f"标准误在{entity_label}层面聚类以处理组内序列相关问题。"
        )

        data["model_spec"] = {
            "equation": eq,
            "description": eq_desc,
            "variables": variables,
            "identification": "\n".join(ident_parts),
        }

    # ── Descriptive statistics (compute from data if available) ──
    if data_path and not data.get("descriptive_stats"):
        try:
            import pandas as pd
            p = Path(data_path)
            if p.suffix == ".dta":
                df = pd.read_stata(str(p))
            else:
                df = pd.read_csv(str(p))
            # Detect treatment column
            treat_col = None
            for cand in ["treated", "treatment", "treat"]:
                if cand in df.columns: treat_col = cand; break
            # Detect time column
            time_col = "year" if "year" in df.columns else ("time" if "time" in df.columns else None)

            if treat_col and time_col and not data.get("descriptive_stats"):
                # Pre-treatment period
                first_treated_col = None
                for cand in ["first_treated", "first_treatment_year"]:
                    if cand in df.columns: first_treated_col = cand; break
                if first_treated_col:
                    min_ft = df[first_treated_col].min()
                    pre_df = df[df[time_col] < min_ft]
                else:
                    pre_df = df

                # Auto-detect entity column (MUST come before skip_cols usage)
                entity_col = None
                for cand in ["city_id", "province_id", "entity_id", "id"]:
                    if cand in df.columns:
                        entity_col = cand
                        break

                treated = pre_df[pre_df[treat_col] == 1]
                control = pre_df[pre_df[treat_col] == 0]
                # Filter to only actual covariate columns (not entity, time, outcome, treatment)
                skip_cols = {"city_id","city_name","province","province_id","province_id_str",
                             "year","year_str","post","treated","first_treated","treatment_intensity",
                             "tax_rate","tax_high"}
                if entity_col: skip_cols.add(entity_col)
                if time_col: skip_cols.add(time_col)
                # Also skip outcome-adjacent columns
                for k in list(skip_cols):
                    skip_cols.add(str(k))
                numeric_cols = [c for c in df.columns
                              if c not in skip_cols and not c.startswith('log_')]
                numeric_cols = numeric_cols[:8]

                vars_list = []
                for col in numeric_cols:
                    if pd.api.types.is_numeric_dtype(treated[col]):
                        t_mean = float(treated[col].mean())
                        c_mean = float(control[col].mean())
                        t_sd = float(treated[col].std())
                        c_sd = float(control[col].std())
                        pooled = (t_sd**2 + c_sd**2)**0.5
                        std_diff = abs(t_mean - c_mean) / pooled if pooled > 0 else 0
                        vars_list.append({
                            "label": col, "treated": round(t_mean, 3),
                            "control": round(c_mean, 3), "std_diff": round(std_diff, 3),
                        })

                n_periods_val = int(df[time_col].nunique())
                if entity_col:
                    n_t = int(df[df[treat_col]==1][entity_col].nunique())
                    n_c = int(df[df[treat_col]==0][entity_col].nunique())
                else:
                    n_t = int((df[treat_col]==1).sum() / max(n_periods_val, 1))
                    n_c = int((df[treat_col]==0).sum() / max(n_periods_val, 1))
                n_all = n_t + n_c

                data["descriptive_stats"] = {
                    "variables": vars_list,
                    "n_treated": n_t,
                    "n_control": n_c,
                    "note": f"描述统计基于处理前（{pre_df[time_col].min()}-{pre_df[time_col].max()}年）数据。",
                }
                # Also update data_meta — authoritative values from actual data
                full_span = f"{int(df[time_col].min())}-{int(df[time_col].max())}"
                src = data_source if data_source else f"面板数据（{Path(data_path).name}）"
                data["data_meta"]["source"] = f"{src} | {full_span} | N = {len(df)} ({n_all} entities × {n_periods_val} periods)"
                data["data_meta"]["n_obs"] = f"{len(df)}"
                data["data_meta"]["n_cities"] = n_all
                data["data_meta"]["n_treated"] = n_t
                data["data_meta"]["n_control"] = n_c
                data["data_meta"]["n_years"] = n_periods_val
                data["data_meta"]["time_span"] = full_span
                data["data_meta"]["n_entities"] = n_all
                data["data_meta"]["n_periods"] = n_periods_val
        except Exception:
            pass  # descriptive stats are optional

    # ── Plot paths (for images in report) ──
    data["plots"] = data.get("plots", {})
    if event_study_path:
        out_dir = Path(event_study_path).parent
        # Try multiple possible filenames for event study plot
        for es_name in ["event_study.png", "stage7_event_study.png",
                        str(Path(event_study_path).with_suffix(".png"))]:
            es_candidate = out_dir / es_name
            if es_candidate.exists():
                data["event_study"]["plot"] = str(es_candidate)
                break
        # Placebo plot
        placebo_candidate = out_dir / "placebo.png"
        if placebo_candidate.exists():
            data["plots"]["placebo"] = str(placebo_candidate)

    # ── Completeness validation ──
    completeness_warnings = []
    if not data.get("descriptive_stats", {}).get("variables"):
        completeness_warnings.append("缺少描述性统计：请使用 --data 参数提供面板数据文件")
    if not data.get("main_result", {}).get("coefficient"):
        completeness_warnings.append("缺少主回归结果：Stage 7 未运行或输出文件缺失")
    if not data.get("robustness"):
        completeness_warnings.append("缺少稳健性检验：Stage 8 未运行或输出文件未传入")
    if not data.get("assumptions"):
        completeness_warnings.append("缺少识别假设检验：Stage 6 未运行或输出文件未传入")
    if not data.get("event_study", {}).get("coefficients"):
        completeness_warnings.append("缺少事件研究（平行趋势检验）：Stage 7 事件研究未运行或输出文件未传入")
    data["completeness_warnings"] = completeness_warnings
    if completeness_warnings:
        print(f"\n⚠ 报告完整性警告 ({len(completeness_warnings)}项):")
        for w in completeness_warnings:
            print(f"  - {w}")

    return data


# ═══════════════════════════════════════════════════════════════════════════
# Text report (for terminal preview)
# ═══════════════════════════════════════════════════════════════════════════

def generate_text_report(data: dict) -> str:
    """Generate a plain-text preview from the report data dict."""
    lines = []
    h = "=" * 60
    s = "-" * 60

    lines.append(h)
    lines.append(f"Causal Inference Report: {data['policy']}")
    lines.append(h)
    lines.append(f"Generated: {data['generated']}")

    chain = data["method_chain"]
    if chain["theoretical"] and chain["final"]:
        lines.append(f"Method: {chain['theoretical']} → {chain['final']}")
        if chain["changed"] and chain["gap_explanation"]:
            lines.append(f"  {chain['gap_explanation']}")

    meta = data["data_meta"]
    lines.append(f"Data: {meta.get('source', '')}, N = {meta.get('n_obs', '')}")

    main = data["main_result"]
    if main:
        lines.append(f"\n{s}")
        lines.append(f"Main: coef={main.get('coefficient', 0):.4f}, SE={main.get('std_error', 0):.4f}, "
                     f"p={main.get('p_value', 1):.4f}")

    assumptions = data["assumptions"]
    if assumptions:
        lines.append(f"\n{s} Assumptions")
        for a in assumptions:
            icon = {"PASS": "✓", "FAIL": "✗", "UNCERTAIN": "?"}.get(a.get("verdict", ""), "?")
            lines.append(f"  {icon} {a.get('assumption_name', '')}: {a.get('verdict', '')}")

    robustness = data["robustness"]
    if robustness:
        n_pass = sum(1 for r in robustness if r.get("passed"))
        lines.append(f"\n{s} Robustness: {n_pass}/{len(robustness)} passed")

    lines.append(f"\n{s} Causal Claim: {data['causal_claim_strength'].upper()}")
    lines.append(h)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured report data from all stage outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Output structured JSON for rendering
  python output_report.py --policy "Environmental Tax" --outcome "log_so2" \\
      --stage6 stage6_confirmation.json --stage7 stage7_main_result.json \\
      --output report_data.json

  # Also print a text preview
  python output_report.py ... --output report_data.json --text
""")
    parser.add_argument("--policy", required=True, help="Policy name")
    parser.add_argument("--outcome", required=True, help="Outcome name")
    parser.add_argument("--stage3", default=None, help="Stage 3 theoretical method JSON")
    parser.add_argument("--stage6", default=None, help="Stage 6 confirmation JSON")
    parser.add_argument("--stage7", default=None, help="Stage 7 estimation results JSON")
    parser.add_argument("--stage8", default=None, help="Stage 8 sensitivity analysis JSON")
    parser.add_argument("--stage8-placebo", default=None, help="Stage 8 placebo test JSON")
    parser.add_argument("--stage8-summary", default=None, help="Stage 8 summary JSON (from stage8_combine.py)")
    parser.add_argument("--stage8-alt-windows", default=None, help="Stage 8 alternative time windows JSON")
    parser.add_argument("--data-source", default="", help="Data source description")
    parser.add_argument("--data-span", default="", help="Data time span")
    parser.add_argument("--n-obs", default="", help="Number of observations")
    parser.add_argument("--data", default=None, help="Path to analysis-ready panel data (for descriptive stats)")
    parser.add_argument("--event-study", default=None, help="Path to event study output JSON")
    parser.add_argument("--stage2-sections", default=None, help="Stage 2 narrative sections JSON")
    parser.add_argument("--data-status", default=None, help="Path to Stage 5 data_status JSON")
    parser.add_argument("--output", default=None,
                        help="Output path for report data JSON (default: Desktop/policy_eval_output/report_data.json)")
    parser.add_argument("--text", action="store_true",
                        help="Also print a text preview to stdout")
    args = parser.parse_args()

    stage3 = load_json(args.stage3) if args.stage3 else None
    stage6 = load_json(args.stage6) if args.stage6 else None
    stage7 = load_json(args.stage7) if args.stage7 else None
    stage8 = load_json(args.stage8) if args.stage8 else None
    stage8_placebo = load_json(args.stage8_placebo) if args.stage8_placebo else None
    stage8_summary = load_json(args.stage8_summary) if args.stage8_summary else None
    stage8_alt_windows = load_json(args.stage8_alt_windows) if args.stage8_alt_windows else None
    stage2_sections = load_json(args.stage2_sections) if args.stage2_sections else None
    data_status = load_json(args.data_status) if args.data_status else None

    data = build_report_data(
        args.policy, args.outcome,
        stage3, stage6, stage7, stage8, stage8_placebo, stage8_summary,
        stage8_alt_windows,
        args.data_source, args.data_span, args.n_obs,
        stage2_sections=stage2_sections,
        data_status=data_status,
        data_path=args.data,
        event_study_path=args.event_study,
    )

    if args.output is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path.home() / "Desktop" / "policy_eval_output"
        subject_dir = _find_or_create_subject_dir(args.policy, output_base)
        output_dir = subject_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "report_data.json"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report data saved to {output_path}")

    if args.text:
        print()
        print(generate_text_report(data))


if __name__ == "__main__":
    main()
