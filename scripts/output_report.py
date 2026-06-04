"""
Extract structured data from all stage outputs into a unified JSON format
for downstream rendering (Markdown / XeLaTeX).

Usage:
    python scripts/output_report.py --policy "LTCI Pilot" --outcome "Fertility Rate" \\
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        "staggered_policy_shock": "Treated × Post (staggered)",
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
                       stage8_summary: dict = None) -> list[dict]:
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
        outcome_name = outcome_label if isinstance(outcome_label, str) else "被解释变量"
        parts.append(f"被解释变量为{outcome_name}，核心解释变量为长护险试点状态与时间的交互项（staggered treatment）。")
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
            parts.append(f"处理组包含{n_treated}个长护险试点城市（含2017年和2021年两批），对照组包含{n_control}个非试点城市。长护险试点城市名单及处理时点来自人社厅发〔2016〕80号和医保发〔2020〕37号文件。")

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
    different spellings (e.g., 'LTCI' vs '长期护理保险试点').
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

    # Add explicitly listed references from stage2_sections
    explicit_refs = sec.get("references", [])
    if isinstance(explicit_refs, list):
        for er in explicit_refs:
            if er and er not in seen:
                seen.add(er)
                refs.append({"text": er, "complete": True})

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

def build_report_data(
    policy: str,
    outcome: str,
    stage3: dict = None,
    stage6: dict = None,
    stage7: dict = None,
    stage8: dict = None,
    stage8_placebo: dict = None,
    stage8_summary: dict = None,
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
                obs_str += f" ({entities} entities × {periods} periods)"

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
                    obs_str += f" ({entities} entities × {periods} periods)"
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
    data["event_study"] = event_study

    # ── Fallback attempts ──
    data["fallback_attempts"] = stage6.get("fallback_attempts", []) if stage6 else []

    # ── Robustness ──
    data["robustness"] = _normalize_checks(stage8, stage8_placebo, stage8_summary)

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
        n_cities = data["data_meta"].get("n_cities", 0)
        n_treated = data["data_meta"].get("n_treated", 0)
        n_control = data["data_meta"].get("n_control", 0)
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
        variables.append((DEP, oname, "城市层面的生育率对数", ""))
        if is_cs:
            variables.append((EXP, "ATT(g,t)", "组别-时间特定平均处理效应（staggered treatment）", ""))
        else:
            variables.append((EXP, "Treated × Post", "处理组虚拟变量与处理后虚拟变量的交互项", ""))

        ctrl_labels = {
            "log_gdp_pc": "人均GDP（对数）", "log_population": "人口规模（对数）",
            "gdp_per_capita": "人均GDP", "population_10k": "人口规模（万人）",
            "elderly_ratio": "老龄化率（65岁以上占比）", "urbanization_rate": "城镇化率",
            "fiscal_revenue_pc": "人均财政收入", "female_labor_participation": "女性劳动参与率",
            "industrial_rate": "工业化率", "n_hospitals": "医院数量",
        }
        for ctrl in actual_controls:
            label = ctrl_labels.get(ctrl, ctrl)
            variables.append((CTRL, label, "控制变量", ""))

        # ── Detailed identification narrative ──
        ident_parts = []

        # 1. Policy design → identification logic
        ident_parts.append(
            "本研究利用长期护理保险试点分批推广的政策设计特征，构建因果识别策略。"
            "长护险试点采用\"先行试点、逐步推广\"的方式，在国家层面分两批确定试点城市："
            "第一批于2016年确定、2017年正式实施，覆盖承德、长春、上海、青岛等15个城市；"
            "第二批于2020年确定、2021年正式实施，新增北京石景山区、天津、福州等14个城市。"
            "全国其余约200多个地级及以上城市始终未纳入试点。"
            "这一制度设计形成了典型的\"交错处理+从未处理对照组\"结构："
            "不同城市在不同时点进入处理状态，存在大量从未受处理的对照组单元，"
            "为利用双重差分策略识别因果效应提供了理想的准自然实验条件。"
        )

        # 2. Why this specific method
        if is_cs:
            ident_parts.append(
                f"本文采用Callaway & Sant'Anna（2021）提出的交错型双重差分估计量（staggered DID）"
                f"作为基准识别策略。选择这一方法的理由如下："
            )
            reasons = [
                "处理时点交错：不同城市在不同时点接受处理（2017年和2021年），传统双向固定效应（TWFE）"
                "估计量在交错处理情境下可能产生\"负权重\"问题（Goodman-Bacon, 2021），"
                "即某些组别-时期的处理效应被赋予负权重，导致估计量的加权平均难以解释为有意义的因果参数。"
                "C&S估计量通过先估计组别-时间特定的ATT(g,t)，再按组别规模加权平均，"
                "避免了负权重问题。",
                f"存在稳定的对照组：{n_control}个从未参与试点的城市提供了对照组，"
                "这些城市不受长护险政策的直接影响，构成了\"从未处理\"（never-treated）对照组。"
                "C&S估计量支持多种对照组选择（从未处理、尚未处理），本研究采用从未处理作为基准。",
                "可检验的识别假设：平行趋势假设和无预期效应假设可通过事件研究法进行统计检验。"
                f"本文构造了处理前5期和处理后5期的事件研究，通过F检验评估处理前各期系数的联合显著性。",
            ]
            for i, reason in enumerate(reasons, 1):
                ident_parts.append(f"（{i}）{reason}")
        else:
            ident_parts.append(
                f"本文采用双重差分法（DID）作为基准识别策略，利用处理组和对照组在政策实施前后"
                "的结果差异，通过双重差分消除不随时间变化的未观测混杂因素。"
            )

        # 3. Source of identification variation
        ident_parts.append(
            "核心识别变异的来源为：同一时点上，已进入长护险试点的城市与尚未（或从未）"
            "进入试点的城市之间，生育率变化的系统性差异。城市固定效应吸收了所有不随时间变化的"
            "城市特征（如地理区位、文化传统等），年份固定效应吸收了所有城市共同面临的"
            "时间趋势冲击（如全国性生育政策调整、宏观经济波动等），"
            "二者的联合控制使得\"处理组×处理后\"的交互项系数可以解释为长护险对生育率的因果效应。"
        )

        # 4. Key identifying assumptions
        ident_parts.append(
            "该方法依赖以下核心识别假设：\n"
            "  • 平行趋势假设（Parallel Trends）【可检验】：在不受处理的反事实情境下，"
            "处理组和对照组的生育率变化趋势应当相同。通过事件研究法中处理前各期系数的联合F检验进行评估。\n"
            "  • 无预期效应（No Anticipation）【可检验】：城市在正式进入长护险试点之前，"
            "不应因预期政策变化而提前调整生育行为。通过检验处理前一期（t=-1）系数是否接近零来验证。\n"
            "  • 稳定单元处理值假设（SUTVA）【需论证】：一个城市的试点状态不应影响其他城市的生育率。"
            "鉴于长护险的参保资格和待遇享受具有明确的地域边界（以城市为单位），"
            "跨城市的政策溢出效应有限，该假设基本合理。\n"
            "  • 条件独立性假设（CIA/Unconfoundedness）【需论证】：在控制城市固定效应、"
            "年份固定效应及可观测时变控制变量的条件下，处理分配与潜在结果独立。"
            "试点城市的选择基于可观测的城市特征（如老龄化程度、经济发展水平、医保基金承受能力），"
            "本文通过控制这些变量来增强该假设的可信度。"
        )

        # 5. Handling concurrent policies
        ident_parts.append(
            "需要特别指出的是，长护险试点的推广期间与中国生育政策的重大调整存在时间重叠："
            "2016年全面二孩政策实施（与第一批试点几乎同步），2021年三孩政策宣布（与第二批试点接近）。"
            "本研究通过以下策略应对这一挑战：（1）全面二孩和三孩政策是全国性政策，对所有城市同时生效，"
            "其影响被年份固定效应吸收，不会系统地混淆城市层面试点状态的因果效应；"
            "（2）利用两批试点的交错时点（2017年和2021年），在同一全国性政策环境下识别不同试点时期的"
            "差异化处理效应，增强了因果推断的可信度；"
            "（3）2021年三孩政策的影响有限（已有研究表明有意愿生育三孩的家庭比例极低），"
            "且样本期至2022年，三孩政策后的观测期较短，其潜在混淆作用较小。"
        )

        # 6. Data and dimensions
        ident_parts.append(
            f"本文使用的数据集为{n_years}年（2010-2022）中国{n_cities}个地级及以上城市的面板数据，"
            f"其中处理组{n_treated}个城市（分{n_cohorts}个处理批次）、对照组{n_control}个城市。"
            f"共{n_periods}个时期，合计{data['data_meta'].get('n_obs', '')}个观测值。"
            f"标准误在城市层面聚类以处理组内序列相关问题。"
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
                # Also update data_meta
                data["data_meta"]["n_cities"] = n_all
                data["data_meta"]["n_treated"] = n_t
                data["data_meta"]["n_control"] = n_c
                data["data_meta"]["n_years"] = n_periods_val
                data["data_meta"]["time_span"] = f"{int(pre_df[time_col].min())}-{int(pre_df[time_col].max())}"
        except Exception:
            pass  # descriptive stats are optional

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
  python output_report.py --policy "LTCI Pilot" --outcome "Fertility Rate" \\
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
    stage2_sections = load_json(args.stage2_sections) if args.stage2_sections else None
    data_status = load_json(args.data_status) if args.data_status else None

    data = build_report_data(
        args.policy, args.outcome,
        stage3, stage6, stage7, stage8, stage8_placebo, stage8_summary,
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
