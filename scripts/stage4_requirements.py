"""
Stage 4: Generate structured data requirements from Stage 3 output.

Reads Stage 3 method recommendation, cross-references with variable_map.json,
and produces a structured requirements file that Stage 5 can execute.

Usage:
    python scripts/stage4_requirements.py --stage3 data/auto/stage3_result.json \
        --output data/auto/stage4_requirements.json

    # Also print a human-readable table
    python scripts/stage4_requirements.py ... --text

    # With user-provided variable mappings (interactive)
    python scripts/stage4_requirements.py ... --mappings mappings.json
"""

import argparse
import json
import re
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_variable_map() -> dict:
    """Load variable_map.json, stripping metadata keys."""
    path = SKILL_ROOT / "references" / "variable_map.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════════
# Concept → variable_map keyword matching
# ═══════════════════════════════════════════════════════════════════════

# Maps common concept variable names to likely variable_map keys
_CONCEPT_HINTS = {
    "outcome": ["outcome", "dependent_variable"],
    "entity_id": ["entity_id", "city_id", "province_id"],
    "time": ["year", "date", "time"],
    "treatment_assignment": ["treated", "treatment", "treat"],
    "first_treated": ["first_treated", "first_treatment_year"],
    "running_variable": ["running_variable", "score", "range"],
    "cutoff_value": ["cutoff", "threshold"],
    "instrument": ["instrument", "iv"],
    "treatment_intensity": ["treatment_intensity", "dose", "intensity"],
    "baseline_covariates": ["covariate"],
    "compliance_indicator": ["compliance"],
    "never_treated_indicator": ["never_treated"],
    "post": ["post", "after"],
    "gdp": ["gdp", "gdp_per_capita", "gdp_growth"],
    "population": ["population", "population_10k"],
    "urbanization": ["urbanization_rate", "urban_population_pct"],
    "fiscal": ["fiscal_revenue", "fiscal_revenue_pc", "fiscal_expenditure"],
    "employment": ["employment", "unemployment_rate"],
    "education": ["education", "college_edu_pct", "school_enrollment"],
    "health": ["hospital_beds", "doctors", "life_expectancy", "infant_mortality"],
    "income": ["urban_disposable_income", "rural_disposable_income", "avg_wage"],
    "consumption": ["urban_consumption", "rural_consumption", "retail_sales"],
    "investment": ["fixed_investment", "fdi_pct_gdp"],
    "trade": ["foreign_trade", "exports", "imports", "trade_pct_gdp"],
    "environment": ["aqi", "co2_emissions", "energy_consumption"],
    "age": ["elderly_ratio", "age_65_plus_pct", "age_0_14_pct"],
    "birth": ["birth_rate", "fertility_rate", "natural_growth_rate"],
    "death": ["death_rate", "infant_mortality"],
    "insurance": ["medical_insurance", "pension_insurance", "unemployment_insurance"],
    "transport": ["car_ownership", "public_transport", "road_area", "highway_mileage"],
    "innovation": ["patents", "rd_expenditure"],
}


def _keyword_in_text(keywords, text):
    """Check if any keyword appears in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def search_variable_map(concept: str, var_map: dict) -> list[dict]:
    """Search variable_map.json for entries matching a concept name.

    Returns a list of candidate matches with relevance scores.
    """
    candidates = []
    concept_lower = concept.lower().replace("_", " ").replace("-", " ")

    # Get hint keywords for this concept
    hints = _CONCEPT_HINTS.get(concept, [concept_lower])

    for key, entry in var_map.items():
        if key.startswith("_"):
            continue
        desc = entry.get("description", "")
        keywords_str = entry.get("keywords", "")
        notes = entry.get("notes", "")
        source = entry.get("source", "")

        # Build search text
        search_text = f"{key} {desc} {keywords_str} {notes} {source}".lower()

        score = 0
        # Exact key match
        if key == concept or key.replace("_", "") == concept.replace("_", ""):
            score += 100
        # Key contains concept or vice versa
        if concept_lower in key.lower() or key.lower() in concept_lower:
            score += 50
        # Description contains concept
        if concept_lower in desc.lower():
            score += 40
        # Hint keywords match
        if _keyword_in_text(hints, search_text):
            score += 30
        # Concept words appear in search text
        concept_words = set(concept_lower.split())
        matching_words = concept_words & set(search_text.split())
        score += len(matching_words) * 5

        if score > 0:
            candidates.append({
                "key": key,
                "description": desc,
                "source": source,
                "score": score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Build requirements
# ═══════════════════════════════════════════════════════════════════════

def build_requirements(
    stage3: dict,
    var_map: dict = None,
    user_mappings: dict = None,
) -> dict:
    """Build a structured Stage 4 requirements JSON from Stage 3 output.

    Parameters
    ----------
    stage3 : dict
        Stage 3 output (from stage3_analyze.py).
    var_map : dict
        variable_map.json contents.
    user_mappings : dict
        User-provided concept → variable_map_key overrides.

    Returns
    -------
    dict with structure:
        {
            "method": "...",
            "mechanism": "...",
            "entity_level": "...",
            "time_unit": "...",
            "variables": [
                {
                    "concept": "outcome",
                    "description": "被解释变量",
                    "essential": true,
                    "matched_key": "gdp_per_capita",
                    "tier": "A",
                    "source_label": "世界银行API",
                    "acquisition": "auto",
                    "alternatives": [...]
                },
                ...
            ],
            "unmatched": [...],
            "summary": {...}
        }
    """
    if var_map is None:
        var_map = load_variable_map()
    if user_mappings is None:
        user_mappings = {}

    rec = stage3.get("recommendation", {})
    mechanism = stage3.get("mechanism", "unknown")
    method = rec.get("primary_method", stage3.get("primary_method", ""))

    required = rec.get("required_variables", [])
    optional = rec.get("optional_variables", [])

    # Stage 3 also provides required_vars/optional_vars at top level
    top_required = stage3.get("required_variables", [])
    top_optional = stage3.get("optional_variables", [])

    # Merge
    all_required = list(dict.fromkeys(required + top_required))
    all_optional = list(dict.fromkeys(optional + top_optional))

    variables = []

    def process_var(concept, essential):
        # Check user-provided mapping first
        if concept in user_mappings:
            mapping = user_mappings[concept]
            matched_key = mapping.get("key", "")
            source_label = mapping.get("source", "")
            tier = mapping.get("tier", "B")
            return {
                "concept": concept,
                "description": mapping.get("description", concept),
                "essential": essential,
                "matched_key": matched_key,
                "tier": tier,
                "source_label": source_label,
                "acquisition": "auto" if tier == "A" and matched_key else "manual",
                "user_mapped": True,
                "alternatives": [],
            }

        # Auto-search variable_map
        candidates = search_variable_map(concept, var_map)
        best = candidates[0] if candidates else None

        if best and best["score"] >= 50:
            entry = var_map[best["key"]]
            source_key = entry.get("source", "")
            # Get tier from SOURCE_LABELS
            source_label, tier = _get_tier(source_key, best["description"])

            return {
                "concept": concept,
                "description": best["description"],
                "essential": essential,
                "matched_key": best["key"],
                "tier": tier,
                "source_label": source_label,
                "acquisition": "auto" if tier == "A" else "manual",
                "user_mapped": False,
                "alternatives": [
                    {"key": c["key"], "description": c["description"],
                     "score": c["score"]}
                    for c in candidates[1:4]
                ],
            }
        elif candidates:
            # Low-confidence matches — list as suggestions
            return {
                "concept": concept,
                "description": f"需要匹配: {concept}",
                "essential": essential,
                "matched_key": None,
                "tier": "未知",
                "source_label": "",
                "acquisition": "manual",
                "user_mapped": False,
                "alternatives": [
                    {"key": c["key"], "description": c["description"],
                     "score": c["score"]}
                    for c in candidates[:5]
                ],
            }
        else:
            return {
                "concept": concept,
                "description": f"需要匹配: {concept}",
                "essential": essential,
                "matched_key": None,
                "tier": "未知",
                "source_label": "",
                "acquisition": "manual",
                "user_mapped": False,
                "alternatives": [],
            }

    for v in all_required:
        variables.append(process_var(v, True))
    for v in all_optional:
        variables.append(process_var(v, False))

    # Summary
    matched = [v for v in variables if v["matched_key"]]
    unmatched = [v for v in variables if not v["matched_key"]]
    auto_fetch = [v for v in matched if v["acquisition"] == "auto"]
    manual = [v for v in variables if v["acquisition"] == "manual"]

    return {
        "method": method,
        "mechanism": mechanism,
        "entity_level": "city-year",
        "time_unit": "year",
        "variables": variables,
        "unmatched": [v["concept"] for v in unmatched],
        "summary": {
            "total": len(variables),
            "essential": len([v for v in variables if v["essential"]]),
            "optional": len([v for v in variables if not v["essential"]]),
            "auto_fetch": len(auto_fetch),
            "manual_or_request": len(manual),
            "unmatched": len(unmatched),
        },
    }


def _get_tier(source_key: str, description: str = "") -> tuple[str, str]:
    """Map source key to (source_label, tier)."""
    # Import from fetch_data if possible, else use inline
    labels = {
        "world_bank": ("世界银行API (World Bank)", "A"),
        "akshare_city": ("中国城市统计年鉴 (via akshare)", "A"),
        "akshare_province": ("中国省级统计年鉴 (via akshare)", "A"),
        "akshare": ("akshare 公开数据", "A"),
        "akshare_stock_index": ("akshare 股票指数数据", "A"),
        "akshare_stock_individual": ("akshare 个股数据", "A"),
        "akshare_custom": ("akshare 数据", "A"),
        "yfinance": ("Yahoo Finance", "A"),
        "yfinance_global_index": ("Yahoo Finance 全球指数", "A"),
        "census": ("中国人口普查数据 (via akshare)", "A"),
        "nbs": ("国家统计局 API", "A"),
        "fred": ("FRED 美联储经济数据库", "A"),
        "oecd": ("OECD API", "A"),
        "eurostat": ("Eurostat API", "A"),
    }
    return labels.get(source_key, (f"{source_key} 数据", "B"))


# ═══════════════════════════════════════════════════════════════════════
# Text output
# ═══════════════════════════════════════════════════════════════════════

def print_requirements_table(req: dict):
    """Print a human-readable requirements table."""
    print(f"\n{'='*70}")
    print(f"Stage 4: Data Requirements — {req['method']}")
    print(f"{'='*70}")
    print(f"Mechanism: {req['mechanism']}")
    print(f"Summary: {req['summary']['total']} variables "
          f"({req['summary']['essential']} essential, "
          f"{req['summary']['optional']} optional) | "
          f"{req['summary']['auto_fetch']} auto-fetch, "
          f"{req['summary']['manual_or_request']} manual, "
          f"{req['summary']['unmatched']} unmatched")

    print(f"\n{'─'*70}")
    print(f"{'概念变量':<25} {'variable_map Key':<25} {'Tier':<6} {'获取':<10} {'来源'}")
    print(f"{'─'*70}")

    for v in req["variables"]:
        essential_mark = "★" if v["essential"] else " "
        concept = f"{essential_mark} {v['concept']}"
        key = v["matched_key"] or "—"
        tier = v["tier"]
        acq = v["acquisition"]
        src = v["source_label"][:35] if v["source_label"] else "需指定"
        print(f"{concept:<25} {key:<25} {tier:<6} {acq:<10} {src}")

    if req["unmatched"]:
        print(f"\n⚠ 未匹配变量: {', '.join(req['unmatched'])}")
        print("  这些变量需要在 variable_map.json 中添加，或由用户手动提供（Tier B/C）。")
        print("  对于每个未匹配变量，请确认：")
        print("  1. 该变量的具体数据来源")
        print("  2. 获取方式（API申请 / 手动收集 / 不可获取）")
        print("  3. 是否可以使用已匹配的变量作为替代")

    print(f"{'─'*70}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Generate structured data requirements from Stage 3 output",
    )
    parser.add_argument("--stage3", required=True,
                        help="Path to Stage 3 output JSON")
    parser.add_argument("--output", default=None,
                        help="Output path for stage4_requirements.json")
    parser.add_argument("--mappings", default=None,
                        help="User-provided concept→key mappings JSON")
    parser.add_argument("--text", action="store_true",
                        help="Print human-readable table")
    args = parser.parse_args()

    stage3 = load_json(args.stage3)
    var_map = load_variable_map()
    user_mappings = load_json(args.mappings) if args.mappings else {}

    req = build_requirements(stage3, var_map, user_mappings)

    if args.output is None:
        args.output = str(SKILL_ROOT / "data" / "auto" / "stage4_requirements.json")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage 4 requirements saved to {output_path}")

    if args.text:
        print_requirements_table(req)


if __name__ == "__main__":
    main()
