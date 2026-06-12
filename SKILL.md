---
name: policy-eval
description: End-to-end causal inference system for policy evaluation. Guides the user through problem definition, automatic policy research, data acquisition, method selection, regression analysis, robustness checks, and result reporting. Use when user asks about "政策评估系统", or to "evaluate a policy", "assess policy impact", "causal inference", "政策评估", "因果识别", "政策效果", "DID analysis", "RDD analysis", "treatment effect estimation", or asks whether a specific policy had an effect on an outcome.
---

# Policy Evaluation System

## Core principles

1. **Default to automatic.** The system searches, fetches, and analyzes automatically. Only interrupt when genuinely stuck.
2. **When uncertain, ask.** Never guess. If information is ambiguous, output a clear question with options.
3. **Theory before data.** Identify the theoretically correct method from the assignment mechanism first (Stage 3). Data constraints only enter at Stage 6. Never reverse-engineer a method from available data.
4. **All decisions are explained.** Every method choice comes with a justification written in plain language.

---

## The 9 stages

```
Stage 1: Problem definition        →  What policy? What outcome?
Stage 2: Policy research           →  Auto-search policy details
Stage 3: Theoretical method        →  What method does the assignment mechanism imply?
Stage 4: Data requirements         →  What data does the theoretical method demand?
Stage 5: Data acquisition          →  Fetch or request
Stage 6: Final method confirmation →  Does the actual data support the theoretical method?
                                       If not, what is the best feasible alternative?
Stage 7: Estimation                →  Run the regressions
Stage 8: Robustness checks         →  Is the result real?
Stage 9: Result report             →  What did we find?
```

---

## Stage 1: Problem definition

You only need to tell the system one sentence. For example:

> "Evaluate the impact of long-term care insurance pilots on fertility rates in China."

The system will parse this into:

| Element | Extraction |
|---|---|
| Policy | Long-term care insurance pilot |
| Outcome | Fertility rate |
| Geography | China (city-level, implied) |
| Time frame | 2016-2020 (to be confirmed in Stage 2) |

If any element is ambiguous, ask ONE clarifying question at a time. Do not bombard the user.

---

## Stage 2: Policy research

### What the system does automatically

Search for the following in order:

1. **Policy timeline**: When was the policy announced? When did it take effect? Was it rolled out in batches?
2. **Policy content**: What exactly does the policy do? Who does it cover? What's the treatment intensity?
3. **Coverage/eligibility**: Which cities/regions/individuals are treated? Which are not? Why?
4. **Existing research**: What methods have other researchers used? What data sources? What results?
5. **Concurrent policies**: During the same period, were there other policies that could affect the same outcome?

### How to search

Use web search for each question above. Read government announcements, policy documents, academic papers, and news reports. Synthesize findings into a **policy background report**.

### Policy background report format

```
═══════════════════════════════════
Policy Background Report: [Policy Name]
═══════════════════════════════════

1. Policy overview
   [2-3 sentences describing the policy]

2. Timeline and batches
   Batch 1 (YYYY-MM): N cities — [list]
   Batch 2 (YYYY-MM): N cities — [list]
   ...

3. Policy design features
   ├── Staggered rollout? → Yes/No → Implication for methods
   ├── Clear treatment/control groups? → Yes/No → Implication
   ├── Random assignment? → Yes/No → Implication
   ├── Threshold-based eligibility? → Yes/No → Implication
   └── Concurrent policies? → [List if any]

4. Existing literature
   [Key papers, their methods, data sources, findings]

5. Uncertainties
   [Specific questions the system cannot answer from public sources]
═══════════════════════════════════
```

Note: Do NOT propose method candidates at this stage. That is Stage 3.

### Structured facts output (required before Stage 3)

After the narrative report, the LLM MUST produce a structured facts file. The file goes into the per-analysis output directory derived from the pipeline state file name (e.g., `--state ltci_state.json` → `data/auto/ltci_state/stage2_facts.json`). This ensures different analyses never overwrite each other. The facts file contains ONLY observable facts about the policy design — no methodological judgments. The facts answer 7 questions that any policy researcher can determine from public documents:

```json
{
  "policy_name": "Long-term Care Insurance Pilot",
  "outcome": "Fertility Rate",

  "q1_assignment": {
    "how_is_treatment_assigned": "by_policy_timing",
    "_options": ["by_lottery", "by_threshold_score", "by_observed_characteristics", "by_policy_timing"],
    "evidence": "The central government selected pilot cities in two batches based on city-level economic and demographic characteristics. Not random, no single eligibility score."
  },

  "q2_threshold": {
    "has_eligibility_threshold": false,
    "_note": "Only fill if q1='by_threshold_score'. Was there a continuous score with a sharp cutoff? (e.g., exam score ≥ 600, income ≤ poverty line, vehicle range ≥ 400km)",
    "threshold_variable": null,
    "cutoff_value": null,
    "compliance_is_perfect": null
  },

  "q3_timing": {
    "has_known_start_time": true,
    "treatment_starts_at_different_times": true,
    "first_treatment_time": 2016,
    "last_treatment_time": 2020,
    "_note": "Was the policy announced and implemented at a known point in time, creating a before/after comparison? Were all treated units treated at the same time, or in batches?",
    "timing_detail": "Batch 1: 15 cities in 2016. Batch 2: 14 cities in 2020. Some cities never participated."
  },

  "q4_control_group": {
    "has_never_treated_units": true,
    "all_units_eventually_treated": false,
    "why_some_never_treated": "The pilot program only covered specific cities. Many cities were never selected for the pilot.",
    "_note": "Are there units that NEVER receive treatment? Or does every unit eventually get treated, just at different times?"
  },

  "q5_treatment_type": {
    "is_binary": true,
    "has_intensity_variation": false,
    "intensity_detail": null,
    "_note": "Does every treated unit get the same treatment (binary)? Or do different units get different doses/intensities? (e.g., different subsidy amounts, different coverage levels)"
  },

  "q6_concurrent_policies": {
    "has_overlapping_policies": false,
    "overlapping_list": [],
    "can_be_separated": null,
    "_note": "During the same period, are there OTHER policies that could affect the SAME outcome? If yes, can we separate their effects (e.g., by region or timing)?"
  },

  "q7_instrument": {
    "has_plausible_instrument": false,
    "instrument_description": null,
    "_note": "Is there a variable that affects whether a unit is treated, but does NOT directly affect the outcome? (e.g., distance to college for education; wind direction for pollution)"
  }
}
```

Each question is answered from observable policy facts. The `_note` and `_options` fields are documentation for the LLM — they guide the LLM toward correct answers without requiring causal inference expertise. The actual values are factual:

- **how_is_treatment_assigned**: pick one of the four options from what you observe about the policy design
- **q3_timing**: look at when the policy was announced and implemented — does it create before/after?
- **q4_control_group**: look at who is covered — are there units left out? permanently or temporarily?
- **q5_treatment_type**: look at what the treatment actually is — same for everyone, or different doses?
- **q6_concurrent_policies**: search for other policies in the same domain during the same period
- **q7_instrument**: this one requires some economic reasoning, but the question is factual: is there a known external factor that shifted treatment exposure?

Save this file:
```bash
# The LLM writes this file after completing Stage 2 research
# Path pattern: data/auto/<state_stem>/stage2_facts.json
# <state_stem> = the state file name without .json extension
# Example: --state ltci_state.json → data/auto/ltci_state/stage2_facts.json
```

Then pass it to Stage 3:
```bash
# Example (replace <state_stem> with actual state file stem):
python scripts/stage3_analyze.py --from-facts data/auto/<state_stem>/stage2_facts.json
```

This replaces the previous approach where the LLM manually translated policy research into CLI flags. Now Stage 3 reads the facts and determines both the mechanism (Level 1) and the specific method (Level 2) deterministically.

### When to ask the user

Only ask when the information is genuinely unavailable from public sources. For example:

- "Qingdao appears to have started LTCI as early as 2012, 4 years before the national pilot. Should we treat it as a 2012 or 2016 treatment city?"
- "Several provinces expanded pilots on their own after 2020, but the exact city list is not published. Do you have access to provincial policy documents?"

Format each question with:
- What the ambiguity is
- Why it matters
- Concrete options (A/B/C) with implications
- A recommended default

---

### Narrative sections output (required before Stage 9)

After the structured facts JSON, the LLM MUST also produce narrative prose for the final academic report, saved to the same per-analysis directory (e.g., `data/auto/<state_stem>/stage2_sections.json`). These sections are independent of method choice — they describe the policy context and economic logic, which applies regardless of which estimator is used.

**CRITICAL: These sections are the primary content for the final report's 引言 and 制度背景与理论分析 chapters. Each section MUST be 400-800 Chinese characters of substantive, publication-quality prose. Short placeholder text is unacceptable.** The content written here directly determines the quality of the final academic paper. Write as if drafting a real journal submission.

#### Section 1: `intro_background` (研究背景, 400-600字)

This section opens the paper. Structure it as THREE paragraphs:

**Paragraph 1 — Problem setting (150-200字):**
- Open with a striking data point, policy announcement event, or social problem fact. Use a specific number, date, or quote.
- Describe the socioeconomic context that motivated the policy.
- State what problem the policy was designed to solve.

**Paragraph 2 — Policy introduction (100-150字):**
- Name the policy, its issuance date, issuing authority, and basic mechanism.
- State its geographic scope and rollout pattern (unified vs. staggered).
- Mention who is covered and what the intervention consists of.

**Paragraph 3 — Research motivation (150-250字):**
- Why is causal evaluation of this policy important? (Academic debate? Policy uncertainty? Scale of expenditure?)
- What is the research question in one clear sentence.
- Preview the paper's approach without naming the specific method (e.g., "利用政策分批推广的准自然实验变异" not "采用Callaway & Sant'Anna (2021) staggered DID").

Example of good opening: "2016年6月，人力资源和社会保障部发布《关于开展长期护理保险制度试点的指导意见》（人社厅发〔2016〕xx号），选定15个城市启动长期护理保险首批试点。截至2020年第二批试点扩面，全国共有49个城市参与。这一政策覆盖约1.2亿参保人群，年度基金支出规模超过100亿元，是'十三五'期间中国社会保障领域最重要的制度创新之一。"

#### Section 2: `intro_literature` (文献综述, 400-600字)

**Paragraph 1 — Related literature (200-250字):**
- Cite 4-6 real papers with author surnames and years. Group them thematically (not one-by-one).
- Include BOTH: (a) papers studying this specific policy or similar policies in other countries, AND (b) methodological papers relevant to the identification challenge.
- For each paper or group, mention: what they studied, what method they used, key finding.
- DO NOT invent papers. Use papers discovered during web searches in Stage 2.

**Paragraph 2 — Gap identification (100-150字):**
- What do existing studies miss? Common gaps: limited data (short panels, few cities), weak identification (no control group, no staggered timing), narrow outcomes, lack of robustness checks.
- Be specific about which gap YOUR study fills. Don't say "there are few studies" — say "existing studies on this policy use provincial-level data and cannot exploit city-level variation in treatment timing."

**Paragraph 3 — Contribution (100-200字):**
- State 2-3 concrete marginal contributions. Use numbered points.
- Contributions should cover: data (broader coverage, longer panel), method (exploiting specific policy design feature), and substance (new outcome variable, new channel).

Example: "已有文献主要使用省级面板数据评估该政策效果（Zhang, 2020; Li & Wang, 2021），无法利用城市层面的受处理时间变异..."

#### Section 3: `institution` (制度背景, 500-800字)

This is the LONGEST and MOST DETAILED section. It provides the institutional knowledge that justifies the identification strategy.

**Paragraph 1 — Policy origins and legal basis (100-150字):**
- Issuing authority, key document numbers, legislative/regulatory basis.
- Policy objectives as stated in official documents.

**Paragraph 2 — Rollout timeline and geography (200-300字):**
- List EACH batch: year, number of cities, selection criterion (if known).
- List at least 8-10 pilot city names grouped by batch.
- Mention any voluntary opt-in or mandatory participation features.
- INCLUDE a table-like structure in prose: "第一批（2016年）：青岛市、上海市、...等15个城市；第二批（2020年）：北京市、天津市、...等14个城市。"

**Paragraph 3 — Policy design features relevant to identification (150-200字):**
- Is treatment binary or continuous? Same for all treated units?
- Are there never-treated units? Why were they not selected?
- Was assignment based on observable characteristics? Which ones?
- Any threshold rules or eligibility criteria?

**Paragraph 4 — Concurrent policies and caveats (100-150字):**
- List other policies affecting the same outcome during the same period.
- Discuss whether they are separable (by region, timing, or data).

#### Section 4: `theory` (理论机制, 400-600字)

**Paragraph 1 — Channel framework (150-200字):**
- State 2-3 competing channels through which the policy could affect the outcome.
- Label each channel clearly (e.g., "收入效应", "替代效应", "一般均衡效应").
- For EACH channel: explain the logic chain linking policy → intermediate variable → outcome.

**Paragraph 2 — Channel analysis (150-200字):**
- For each channel, discuss its expected direction and magnitude.
- Cite established economic theory or prior empirical work supporting each channel (e.g., "根据Becker (1965)的家庭生产模型...").
- State whether channels reinforce or offset each other.

**Paragraph 3 — Net prediction (100-200字):**
- Is the net effect theoretically determinate or ambiguous?
- If ambiguous: state that the empirical analysis will resolve the direction.
- If determinate: state the predicted sign and explain why one channel is expected to dominate.

#### Quality checklist (verify BEFORE writing the JSON file)

Before saving the sections JSON file, verify:

```
[ ] intro_background: Opens with a concrete number/fact/event?    □
[ ] intro_background: States the research question explicitly?    □
[ ] intro_literature: Cites 4+ real papers with author+year?     □
[ ] intro_literature: States 2+ specific contributions?           □
[ ] institution: Lists specific policy document numbers?          □
[ ] institution: Names 8+ pilot cities grouped by batch?          □
[ ] institution: Discusses identification-relevant design?        □
[ ] theory: Presents 2+ competing channels?                       □
[ ] theory: Cites economic theory for each channel?               □
[ ] ALL sections: 400+ Chinese characters (not just spaces)?      □
```

If any checkbox is unchecked, rewrite that section before saving. A section under 350 characters is a HARD FAIL — expand it with more specific details from the Stage 2 research findings.

Save this file:
```bash
# The LLM writes this file after completing Stage 2 research
# Path pattern: data/auto/<state_stem>/stage2_sections.json
# Example: --state ltci_state.json → data/auto/ltci_state/stage2_sections.json
```

The JSON schema:
```json
{
  "intro_background": "段落1-3的完整文本，共400-600字",
  "intro_literature": "段落1-3的完整文本，共400-600字",
  "institution": "段落1-4的完整文本，共500-800字",
  "theory": "段落1-3的完整文本，共400-600字"
}
```

A fifth section `conclusion` is written AFTER Stage 7 estimation completes — the LLM generates it based on the actual results, with the pipeline state as context. The other four sections are written during Stage 2 and do not depend on estimation results.

---
## Stage 3: Theoretical method analysis

### What this stage does

**This is the core of the system.** Before looking at any data, determine what identification strategy is theoretically appropriate. The goal is to answer:

> *Given the treatment assignment mechanism discovered in Stage 2, what source of variation can credibly identify a causal effect?*

This stage does NOT consider data availability. It answers: what should we do, in an ideal world?

### Two-level decision tree

The decision runs in two levels, both implemented deterministically in `stage3_analyze.py`:

**Level 1 — Facts → Mechanism**: Reads the Stage 2 structured facts (`data/auto/<state_stem>/stage2_facts.json`). Classifies the assignment mechanism based on observable policy features. No methodological knowledge needed — pure rule-based classification.

**Level 2 — Mechanism → Method**: Given a mechanism type and data structure flags, selects the specific identification strategy, enumerates assumptions, lists fallbacks, and previews required data.

Priority order for Level 1 classification (early rules dominate):
1. Random assignment → randomization inference
2. Known eligibility threshold → RDD (sharp or fuzzy)
3. Plausible instrument identified → IV / SCM (time-varying unobservables)
4. No time dimension → selection on observables (matching / ML)
5. Multiple overlapping policies → DDD
6. Continuous treatment intensity → intensity DID
7. Known policy timing + staggered → C&S staggered DID
8. Known policy timing + single → standard DID

Rule 3 is new: when Stage 2 research identifies a plausible instrument (q7), the system
routes to the IV/SCM family regardless of whether DID timing features are present — an
instrument provides a stronger source of identifying variation than staggered or single DID.
When no instrument is found, DID-family classifications carry an `unobservables_risk` flag
that surfaces in the Stage 3 output and strengthens SCM fallback recommendations.

### Mixed mechanisms (secondary features)

When a policy has multiple features but only the highest-priority one determines the
primary mechanism, the remaining features are NOT discarded. `classify_mechanism()` runs
`_detect_secondary_features()` at each return point, scanning q2-q7 for features that
exist but were overridden by the primary rule. These are returned as `secondary_features`
and automatically appended as additional fallback strategies in the Level 2 output.

For example, a new energy vehicle subsidy with both a range threshold (≥400km) and
staggered phase-out timing will classify as `threshold_rule` → Sharp RDD as primary,
but the Stage 3 output will also list C&S staggered DID, Intensity DID, and never-treated
control as fallbacks — ready for Stage 6 if the McCrary test fails.

This means:
- Primary method stays deterministic (same input → same output)
- Secondary features are surfaced transparently in the Stage 3 report
- Stage 6 has ready-made alternative strategies without re-running Stage 2-3
- No LLM reasoning is involved — feature detection is purely rule-based

### Running Stage 3

```bash
# Full two-level: facts → mechanism → method (preferred path)
# Replace <state_stem> with your state file stem (e.g., ltci_state)
python scripts/stage3_analyze.py --from-facts data/auto/<state_stem>/stage2_facts.json \
    --output data/auto/<state_stem>/stage3_result.json

# Level 2 only: mechanism already known (backward compatibility)
python scripts/stage3_analyze.py --mechanism staggered_policy_shock \
    --has-control-group --output data/auto/<state_stem>/stage3_result.json
```

The script outputs:
- Level 1 classification reasoning (which rules fired and why)
- Primary theoretical recommendation with full justification
- All identifying assumptions (testable vs. argument-based)
- Ranked fallback strategies with conditions
- Required and optional data variables (preview for Stage 4)
- Heterogeneity analysis recommendation (Causal Forest) as supplement
- Data compatibility warnings (e.g., "DID requires panel data")

The script is **deterministic**: same facts always produce the same mechanism, same mechanism + same flags always produce the same method. This guarantees consistency across sessions.

### Why theory comes before data

Reversing the order — looking at available data first, then choosing a method — is a common error. It leads to:

1. **Method shopping**: picking a method that produces "significant" results rather than one that is theoretically appropriate.
2. **Ignoring the assignment mechanism**: the most important driver of method choice.
3. **Opaque compromises**: the reader cannot distinguish "we chose this method because it's correct" from "we chose this method because it's what our data allowed."

By fixing the theoretical method first, any later deviation due to data constraints is explicit and traceable. Stage 6 will document the gap between "should have used" and "actually used."

Do not proceed until the user confirms.

---

## Stage 4: Data requirements

### What this stage does

Translates Stage 3's concept-level variable requirements (e.g., "outcome", "entity_id") into concrete variable_map.json keys with tier assignments and acquisition plans. Produces a structured JSON that Stage 5 can execute directly.

### Running Stage 4

```bash
# Auto-generate requirements by cross-referencing Stage 3 with variable_map.json
python scripts/stage4_requirements.py --stage3 data/auto/<state_stem>/stage3_result.json \
    --output data/auto/<state_stem>/stage4_requirements.json --text
```

The script:
1. Reads Stage 3's `required_variables` and `optional_variables` (concept-level names)
2. Searches `variable_map.json` for matching entries by keyword, description, and concept hints
3. Assigns tier (A/B/C) based on source type
4. Flags unmatched variables that need manual handling
5. Outputs `stage4_requirements.json` — a structured file listing every variable with its `matched_key`, `tier`, `source_label`, and `acquisition` plan

### Output format (`stage4_requirements.json`)

```json
{
  "method": "Callaway & Sant'Anna (2021) staggered DID",
  "mechanism": "staggered_policy_shock",
  "entity_level": "city-year",
  "time_unit": "year",
  "variables": [
    {
      "concept": "outcome",
      "description": "Fertility rate (births per woman)",
      "essential": true,
      "matched_key": "fertility_rate",
      "tier": "B",
      "source_label": "World Bank — 仅国家层面，城市层面需CFPS",
      "acquisition": "manual",
      "alternatives": [
        {"key": "birth_rate", "description": "出生率", "score": 55}
      ]
    },
    {
      "concept": "gdp",
      "description": "GDP per capita (current US$)",
      "essential": true,
      "matched_key": "gdp_per_capita",
      "tier": "A",
      "source_label": "世界银行API (World Bank)",
      "acquisition": "auto",
      "alternatives": []
    }
  ],
  "unmatched": ["custom_policy_index"],
  "summary": {
    "total": 8, "essential": 5, "optional": 3,
    "auto_fetch": 4, "manual_or_request": 3, "unmatched": 1
  }
}
```

### What the LLM does in Stage 4

1. **Review auto-matches**: The script suggests variable_map keys. The LLM reviews each match for correctness — does `gdp_per_capita` from World Bank actually match the required "GDP at city level"? If not, flag it.

2. **Handle unmatched variables**: For variables the script couldn't match, the LLM must:
   - Search `variable_map.json` manually for alternatives
   - Determine if the variable is Tier A (available via a different key), Tier B (micro survey), Tier C (manual collection), or Tier D (unavailable)
   - Write a `--mappings` JSON file with the correct key and source for each unmatched variable, then re-run `stage4_requirements.py --mappings mappings.json`

3. **Present gaps to the user**: For Tier B/C/D variables, tell the user exactly what they need to provide. Only ask about genuinely ambiguous choices.

4. **Save the final requirements**: The completed `stage4_requirements.json` is the single source of truth that Stage 5 reads.

### User mappings format (`--mappings`)

When the auto-match is wrong or a variable isn't matched, the LLM writes a mappings JSON:

```json
{
  "outcome": {
    "key": "city_birth_rate",
    "description": "城市层面出生率 — 需从统计年鉴手动收集",
    "tier": "C",
    "source": "中国城市统计年鉴 2015-2020"
  },
  "running_variable": {
    "key": "vehicle_range_km",
    "description": "新能源车续航里程",
    "tier": "B",
    "source": "工信部新能源汽车推广应用推荐车型目录"
  }
}
```

Then re-run:
```bash
python scripts/stage4_requirements.py --stage3 data/auto/<state_stem>/stage3_result.json \
    --mappings data/auto/<state_stem>/stage4_mappings.json --output data/auto/<state_stem>/stage4_requirements.json
```

### Confirmation step

After presenting the final requirements table, ask the user:

```
Stage 4 complete. Summary:
  - {N} variables auto-matched (Tier A) → Stage 5 will fetch automatically
  - {M} variables need manual handling (Tier B/C)
  - {K} variables unmatched

For Tier B/C data: "Place the files in data/raw/ and the pipeline will pick them up."
Proceed to Stage 5?
```

**Do not proceed to Stage 5 until the user confirms.** Gaps identified here drive the data acquisition plan.

---

## Stage 5: Data acquisition

### Data source tiers

| Tier | Source type | Action |
|---|---|---|
| **A** | Public API / open data | Look up variable in `variable_map.json`, call pre-written function in `fetch_data.py` |
| **B** | Requires registration (micro surveys, proprietary databases) | Generate instructions; wait for user to place data in `data/raw/` |
| **C** | Manual collection (policy documents, local yearbooks) | Provide Excel template, tell user exactly what to fill in |
| **D** | Does not exist / inaccessible | Honestly report and suggest alternatives |

### Tier A: Configuration-driven (no ad-hoc code)

`scripts/fetch_data.py` is **configuration-driven**. Most akshare data sources are
fetched by a single generic engine `fetch_akshare(entry)` — the entry dict comes
directly from `variable_map.json`. Custom functions exist only for non-trivial cases
(year-loop, different API):

| Function | When used | Example config entries |
|---|---|---|
| `fetch_akshare(entry)` | Most akshare calls — function + rename + transform | `shanghai_index`, `cny_usd`, `pmi`, `bond_yield_curve`, `money_supply`, `shibor`, `lpr`, `house_price_70cities` |
| `fetch_wb_indicator(code, countries, years)` | World Bank API | `gdp`, `population`, `fertility_rate` |
| `fetch_cn_city_macro(indicator, years)` | City-level data (year-loop) | `city_gdp`, `city_population`, `city_fiscal_revenue` |
| `fetch_cn_province_macro(indicator, years)` | Province-level data (year-loop) | `province_gdp`, `province_population`, `province_cpi` |
| `fetch_cn_aqi(city, start, end)` | Air quality (city-loop) | `aqi_city` |
| `fetch_global_stock_index(name, start, end)` | yfinance | `sp500`, `hang_seng`, `nikkei225` |
| `fetch_from_variable_map(variables, ...)` | Batch from variable_map.json | One-call fetch for multiple variables |

**Adding a new data source is a JSON config change, not a Python code change:**
```json
{
  "new_variable": {
    "description": "描述",
    "source": "akshare",
    "level": "daily",
    "akshare": {
      "func": "akshare_function_name",
      "kwargs": {"param": "value"},
      "rename": {"中文列名": "english_name"},
      "entity_id": "constant_id",
      "output_cols": ["entity_id", "date", "value"]
    }
  }
}
```

If the data needs reshaping (wide→long, column detection), add a `"transform"`
field referencing one of the 3 built-in transforms: `wide_to_long`, `exchange_rate`,
`pmi`.

### Variable map: the Stage 4 → Stage 5 bridge

`references/variable_map.json` contains 52+ entries covering macro, financial,
and environmental data. The LLM's job in Stage 5 is:

1. Read the Stage 4 data requirements (essential + optional variables)
2. Match each variable to an entry in `variable_map.json`
3. Call the corresponding function in `fetch_data.py`
4. Variables without a match → Tier B/C gap report

```bash
# Single indicator
python scripts/fetch_data.py --source wb --indicator gdp_per_capita \
    --countries CN --start 2010 --end 2020

# China city-level data
python scripts/fetch_data.py --source akshare_city --indicator gdp \
    --start 2015 --end 2020

# Batch from variable map
python scripts/fetch_data.py --from-map gdp,population,fertility_rate \
    --region cn --start 2010 --end 2020

# Stock index
python scripts/fetch_data.py --source akshare_stock_index --indicator shanghai_composite

# Individual stock
python scripts/fetch_data.py --source akshare_stock_individual --indicator 600519
```

### Gap report format

After auto-fetching, output what's still missing:

```
═══════════════════════════════════
Data Gap Report
═══════════════════════════════════

Auto-fetched (✓):
├── gdp → fetch_wb_indicator("NY.GDP.MKTP.CD") → 280 rows ✓
├── population → fetch_wb_indicator("SP.POP.TOTL") → 280 rows ✓
├── city_gdp → fetch_cn_city_macro("gdp") → 3200 rows ✓
└── fertility_rate → not in variable_map.json

Still needed (manual):
├── ✗ fertility_rate → Tier B (micro survey data: CFPS)
│     Not available from public APIs at city level.
│     Application URL: cfps.pku.edu.cn
│     Once approved, place data in data/raw/fertility.csv
│     Expected format: city_id, year, fertility_rate
│
├── ✗ pilot_city_list → Tier C (manual collection)
│     Template: data/manual/pilot_cities.xlsx (pre-populated with known dates)
│     Instructions: Fill in the "?" cells only
│
└── ✗ Exact treatment dates for batch 2 → Tier C
      Template: data/manual/treatment_dates.xlsx (pre-populated with known data)
      Instructions: Confirm or correct each date

═══════════════════════════════════
```

### Template generation

For every Tier C item, generate an Excel template in `data/manual/` using `openpyxl` or `pandas`. Pre-fill all known data, mark unknown cells with "?", and include a "source" column.

### Post-acquisition validation

After data is acquired (auto-fetched or manually provided), run data validation before proceeding:

```bash
python scripts/validate_data.py --data data/merged/panel.dta \
    --entity city_id --time year \
    --outcome log_fertility \
    --treated treated --first-treated first_treated \
    --controls gdp population \
    --output data/auto/<state_stem>/validation_report.json
```

This checks: panel balance, missing values, outliers, duplicate entity-time rows, variable type consistency, treatment variable logic, and pre-treatment data sufficiency. Fix critical issues before proceeding to Stage 6.

### Tier B: Standard application instructions

For variables that require registration (micro surveys, proprietary databases),
output a structured application brief — not free-form text:

```
─────────────────────────────────
Data Request: [Variable Name]
─────────────────────────────────
Data source:   [e.g., CFPS 2020 wave]
Portal URL:    [application website]
Requirements:  [e.g., institutional email, research proposal]
Timeline:      [e.g., 2-4 weeks for approval]
Access level:  [public microdata / restricted / on-site only]
Variables needed: [list specific variable names from the codebook]
Expected format after receipt: [entity_id, year, value]
Target path:   data/raw/[filename].csv
─────────────────────────────────
```

After presenting all Tier B items, ask the user: "Which of these can you apply for?"
Do not proceed until the user confirms they've placed data in `data/raw/`.

### Tier C: Standard Excel template

For manually collected data (policy dates, city lists, yearbook values),
generate one Excel file per variable in `data/manual/`:

| Column | Content | Example |
|---|---|---|
| `entity_id` | Unit identifier | 110100 |
| `year` | Time period | 2020 |
| `value` | Variable value | 1234.56 |
| `source` | Where this number came from | 北京市统计年鉴2021 p.45 |

Pre-fill rules:
- **Known data from Stage 2**: fill in directly (e.g., pilot city names, confirmed dates)
- **Unknown cells**: leave blank (not "?" — blanks are easier to fill and won't break pandas)
- **entity_id**: pre-populate from the entity map if available
- **year**: pre-populate the full year range

### Pipeline state tracking

The pipeline state JSON (`--state` in `run_pipeline.py`) tracks each variable's
acquisition status in `stages.stage5.data_status`.

**IMPORTANT: After calling `fetch_from_variable_map()`, always convert results with `results_to_data_status()` to auto-populate source metadata.** This function reads `description`, `source`, and `tier` from each `FetchResult` (which pulls them from `variable_map.json`) — no manual guessing required:

```python
from fetch_data import fetch_from_variable_map, results_to_data_status

results = fetch_from_variable_map(["gdp", "population", "fertility_rate"])
data_status = results_to_data_status(results)
# Then write data_status into the pipeline state JSON under stages.stage5.data_status
```

Each entry in `data_status` MUST include all fields below:

```json
{
  "stages": {
    "stage5": {
      "data_status": {
        "gdp": {
          "tier": "A",
          "status": "fetched",
          "description": "GDP per capita (current US$)",
          "source": "世界银行API (World Bank)",
          "path": "data/auto/gdp.json"
        },
        "population": {
          "tier": "A",
          "status": "fetched",
          "description": "Total population",
          "source": "世界银行API (World Bank)",
          "path": "data/auto/population.json"
        },
        "fertility_rate": {
          "tier": "B",
          "status": "requested",
          "description": "Fertility rate (births per woman)",
          "source": "CFPS 2020 wave (需申请)",
          "path": null
        },
        "pilot_city_list": {
          "tier": "C",
          "status": "pending",
          "description": "长期护理保险试点城市名单",
          "source": "人社厅发〔2016〕xx号 + 医保发〔2020〕xx号",
          "path": "data/manual/pilot_cities.xlsx"
        }
      }
    }
  }
}
      }
    }
  }
}
```

Status values: `fetched` | `cached` | `requested` | `received` | `pending` | `unavailable`

This enables:
- **Resume**: `run_pipeline.py --from-stage 5` checks status and only fetches missing variables
- **Cache**: Tier A variables with status `fetched` are skipped on re-run (unless `--force`)
- **Gap tracking**: The gap report is generated by filtering `data_status` for non-`fetched` entries

### Key principle

Stage 5 reads `stage4_requirements.json` to know what to fetch. For each variable:
- `acquisition: "auto"` → call `fetch_from_variable_map()` with the `matched_key`
- `acquisition: "manual"` → follow Tier B/C templates (application instructions or Excel templates)
- After fetching, call `results_to_data_status(results)` to auto-populate source metadata — every variable gets `description`, `source`, `tier` from `variable_map.json`, no manual guessing.

Tier A uses pre-written `fetch_data.py` functions — no ad-hoc code. Tier B/C
follow fixed templates so the data contract between stages is predictable.
The pipeline state file is the single source of truth for what data exists,
what's pending, and what's unavailable.

Wait for the user to say "data is ready" before proceeding to Stage 6.

---

## Stage 6: Final method confirmation

### What this stage does

**This is where theory meets reality.** The system automatically verifies every testable identifying assumption from Stage 3 against the actual data. If the primary method's assumptions fail, it walks the pre-registered fallback chain from Stage 3 in ranked order — and every decision is traceable.

Stage 6 is automated via `scripts/stage6_confirm.py`. It is **deterministic**: same Stage 3 output + same data always produces the same confirmation.

### Running Stage 6

```bash
# From a pipeline state file (recommended)
python scripts/run_pipeline.py --state my_analysis.json --from-stage 6 --data data/merged/panel.dta

# Standalone
python scripts/stage6_confirm.py --stage3 data/auto/<state_stem>/stage3_result.json \
    --data data/merged/panel.dta --output data/auto/<state_stem>/stage6_confirmation.json

# With cached validation report
python scripts/stage6_confirm.py --stage3 data/auto/<state_stem>/stage3_result.json \
    --data data/merged/panel.dta \
    --validate-report data/auto/<state_stem>/validation_report.json \
    --output data/auto/<state_stem>/stage6_confirmation.json
```

### How it works

1. **Reads Stage 3 output**: extracts the primary method, all identifying assumptions, and ranked fallbacks.
2. **Runs data validation**: calls `validate_data.py` (or loads a cached report) to check panel structure, missing values, outliers, and treatment consistency.
3. **Dispatches diagnostics**: for each testable assumption, runs the appropriate test (see mapping below). For non-testable assumptions (SUTVA, exclusion restriction, unconfoundedness), marks them as UNCERTAIN — these require human judgment.
4. **Evaluates results**: each assumption gets PASS / FAIL / UNCERTAIN.
5. **If FAIL exists**: walks the fallback chain from Stage 3 in ranked order. For each fallback, checks data availability, runs its own diagnostics, and selects the first fallback that passes.
6. **Produces structured output**: `stage6_confirmation.json` with assumption verdicts, final method decision, specification for Stage 7, and causal claim strength.

### Diagnostic coverage by mechanism

| Mechanism | Testable assumptions | Diagnostic |
|---|---|---|
| staggered_policy_shock | Parallel trends, No anticipation | Event study F-test + t-1 coefficient |
| staggered_policy_shock | Limited heterogeneity / Negative weights | Goodman-Bacon decomposition |
| single_policy_shock | Parallel trends, No anticipation | Event study F-test + t-1 coefficient |
| single_policy_shock | Stable unit composition | Panel balance check |
| threshold_rule | No manipulation of running variable | McCrary (2008) density test |
| threshold_rule | Continuity of potential outcomes | Covariate balance at cutoff |
| threshold_rule | First-stage strength (fuzzy RDD) | F-statistic at cutoff |
| time_varying_unobservables | Relevance (IV path) | Montiel Olea & Pflueger effective F |
| time_varying_unobservables | Pre-treatment fit (SCM path) | Pre-treatment RMSE |
| time_varying_unobservables | No post-treatment confounders | In-space placebo |
| selection_on_observables | Overlap / positivity | Propensity score support |
| selection_on_observables | ML model quality (DML) | Nuisance model CV R² |
| continuous_intensity | Parallel trends in dose-response | Dose-response event study |
| multiple_overlapping_policies | Group exclusivity | Policy exposure overlap check |
| random_assignment | Attrition, Compliance | Attrition rate comparison, compliance rate |

Non-testable assumptions (SUTVA, exclusion restriction, unconfoundedness/CIA, monotonicity, no policy interactions, exogenous intensity, never-treated validity) are always UNCERTAIN — they must be argued from institutional knowledge in Stage 9.

### Output format (`stage6_confirmation.json`)

```json
{
  "mechanism": "staggered_policy_shock",
  "theoretical_method": "Callaway & Sant'Anna (2021) — never-treated as control",
  "final_method": "Callaway & Sant'Anna (2021) — never-treated as control",
  "method_changed": false,
  "chain": ["Callaway & Sant'Anna (2021) — never-treated as control"],
  "gap_explanation": "",
  "assumption_verdicts": [
    {
      "assumption_name": "Parallel trends",
      "description": "...",
      "testable": true,
      "test_method": "Event study: pre-treatment coefficients jointly zero (F-test).",
      "diagnostic": {
        "diagnostic_name": "event_study_parallel_trends",
        "status": "PASS",
        "values": {"f_stat": 1.23, "p_value": 0.31},
        "threshold": "p_value > 0.05",
        "interpretation": "Pre-trends F-test p=0.3104. Parallel trends supported."
      },
      "verdict": "PASS",
      "reasoning": "Pre-trends F-test p=0.3104. Parallel trends supported."
    }
  ],
  "fallback_attempts": [],
  "data_quality_summary": {"overall": "PASS", "total_issues": 0, "critical_issues": []},
  "specification": {
    "entity_col": "city_id",
    "time_col": "year",
    "outcome": "log_fertility",
    "first_treated_col": "first_treated",
    "control_type": "never-treated",
    "method": "cs",
    "no_estimation_script": false
  },
  "warnings": [],
  "limitations": [],
  "causal_claim_strength": "strong"
}
```

### Fallback chain logic

When the primary method fails any diagnostic:

1. Fallbacks are tried in the exact ranked order from Stage 3's `fallbacks` list.
2. For each fallback: check if required data columns exist → run method-specific diagnostics → if all pass, select this fallback.
3. If no fallback passes, keep the theoretical method but downgrade causal claim strength and flag major limitations.
4. If a fallback has no estimation script (e.g., PSM, DDD), it is selected but marked `no_estimation_script: true` — Stage 7 will report this to the user.

### Verdict semantics

| Verdict | Meaning |
|---|---|
| PASS | The diagnostic confirms the assumption is supported by the data. |
| FAIL | The diagnostic contradicts the assumption. A fallback is triggered if available. |
| UNCERTAIN | The assumption cannot be tested with available data, or requires institutional knowledge. |

### Causal claim strength

| Rating | Conditions |
|---|---|
| strong | All testable assumptions pass, no method downgrade |
| moderate | All pass but method was downgraded, or some UNCERTAIN |
| suggestive | 1-2 failures after downgrade, or most assumptions UNCERTAIN |
| not identifiable | Multiple failures with no viable fallback |

If you disagree with a verdict, re-run with corrected data or manually override the `stage6` key in the pipeline state file before Stage 7.

---

## Stage 7: Estimation

### What the system does

Run the main specification chosen in Stage 6. The estimation scripts support both analytic standard errors (influence-function based for doubly-robust estimators) and cluster-bootstrap inference.

```bash
# Standard DID
python scripts/run_did.py --data data/merged/panel.dta \
    --outcome log_fertility --entity city_id --time year \
    --treated treated --post post --controls gdp population

# Callaway & Sant'Anna (2021) staggered DID with doubly-robust estimation
python scripts/run_staggered_did.py --data data/merged/panel.dta \
    --outcome log_fertility --entity city_id --time year \
    --first-treated first_treated --method cs --control never-treated \
    --controls gdp population --bootstrap 200
```

The staggered DID script implements:
- **Propensity score estimation** via logistic regression (P(cohort_g | covariates))
- **Doubly-robust ATT(g,t)** combining IPW reweighting with outcome regression
- **Influence-function standard errors** for valid inference
- **Bootstrap option** for cluster-robust confidence intervals
- **Event study** aggregation by relative time
- **Cohort-specific and overall ATT** with size-based weighting

Output the key result in plain language:

```
Main result:
├── Coefficient on treated_post: β = [value]
├── Standard error: [value] (clustered at city level)
├── p-value: [value] → [significant at 1%/5%/10% or not significant]
├── Interpretation: The policy [increased/decreased/had no effect on] the outcome by [X]%
├── R²: [value]
└── Number of observations: [N] cities × [T] years = [total]
```

Also generate:
- Event study plot (coefficient by relative time, with confidence intervals)
- Raw means comparison plot (treated vs control over time)

### Machine Learning Methods

Two ML-based scripts complement the traditional econometric methods:

**Double/Debiased Machine Learning (DML):**
```bash
python scripts/run_dml.py --data data/merged/panel.dta \
    --outcome log_wage --treatment treated \
    --controls age education experience occupation sector \
    --ml-model gradient_boosting --cv 5
```
DML uses flexible ML models (RandomForest, GradientBoosting, Lasso) for nuisance function estimation, with:
- Neyman-orthogonal scores that debias ML predictions
- K-fold cross-fitting to prevent overfitting bias
- Valid inference under weak conditions (ML models don't need to be correctly specified)
- CATE estimation as a byproduct
- Preferred over PSM/IPW when the number of controls is large (>15-20)

**Causal Forest (Heterogeneity Analysis):**
```bash
python scripts/run_causal_forest.py --data data/merged/panel.dta \
    --outcome log_wage --treatment treated \
    --features age education experience occupation sector \
    --num-trees 2000 --plot cf_heterogeneity.png
```
Causal Forest (Athey & Imbens 2016, Wager & Athey 2018) estimates CATE for each unit:
- Honest estimation with sample splitting
- Variable importance for treatment effect heterogeneity drivers
- Best Linear Projection (BLP) for systematic heterogeneity patterns
- Quantile comparison (top vs. bottom quintile of CATE)
- Run as a complement to any primary method in Stage 7

**Synthetic Difference-in-Differences:**
```bash
python scripts/run_synthetic_did.py --data data/merged/panel.dta \
    --outcome log_fertility --entity city_id --time year \
    --treated-unit 110100 --first-treated 2016 \
    --output data/auto/sd_result.json
```
Synthetic DID (Arkhangelsky et al. 2021) combines SCM-style unit weighting with DID time differencing:
- Finds optimal control unit weights that match pre-treatment outcomes
- Applies DID to the synthetic control, providing valid standard errors
- Placebo-based inference on each donor unit builds a null distribution
- Unlike traditional SCM, provides p-values and confidence intervals
- Recommended when there is no untreated control group but many donor units exist
- For multiple treated units, use `--treated-units` (comma-separated); results are aggregated

---

## Stage 8: Robustness checks

**Note**: If Stage 6 confirmed a method that relies on unconfoundedness (CIA) — such as PSM, IPW, DML, or Causal Forest — the Oster bounds and coefficient stability checks below are especially important. CIA cannot be empirically tested; these sensitivity tests assess how strong unobservables would need to be to explain away the result.

### Automatic checks (run all, report which pass)

| Check | What it tests | How |
|---|---|---|
| Placebo test | Is the effect real or just noise? | Randomly reassign treatment 1000 times. Plot distribution of fake coefficients vs actual. |
| Alternative window | Is the result sensitive to time period? | Re-run with ±1 year, ±2 year windows around treatment. |
| Leave-one-out | Is the result driven by one city? | Re-run dropping one treated city at a time. Report min/max coefficient. |
| Alternative outcome | Is the result sensitive to measurement? | If multiple outcome measures exist, test each. |
| Bacon-Decomp (staggered DID) | Are there negative weights? | Run Goodman-Bacon decomposition. Flag if negative weights > 10%. |
| Parallel trends (pre-treatment) | Did treated and control diverge before treatment? | Event study: are pre-treatment coefficients jointly zero? |
| **Sensitivity analysis** | Could unobservables explain the result away? | Oster bounds, coefficient stability, Rosenbaum bounds, placebo-in-time |
| **Placebo-in-time** | Would we find an "effect" before the policy actually happened? | Shift treatment to earlier time periods; test if fake effects appear |

For each check, report:
```
✓ [Check name]: Passed — [one-line interpretation]
✗ [Check name]: Failed — [what this means and what to do]
```

### Sensitivity analysis (run after primary robustness checks)

```bash
python scripts/sensitivity_analysis.py --data data/merged/panel.dta \
    --outcome log_fertility --entity city_id --time year \
    --treated treated --post post \
    --controls gdp population \
    --treatment-col treated --first-treated 2016 \
    --output data/auto/sensitivity.json
```

This runs five tests:

1. **Oster (2019) bounds**: How strong would unobservables need to be (relative to observables) to explain away the treatment effect? Reports δ (delta) — the ratio of unobservable-to-observable selection strength needed to nullify the coefficient. δ > 1 means the result is robust.

2. **Coefficient stability**: Tracks how the treatment coefficient changes as controls are added incrementally. A stable coefficient across specifications suggests robustness.

3. **Rosenbaum bounds**: For matched/weighted designs, reports Γ (gamma) — the odds ratio of hidden bias needed to overturn significance. Γ > 2 is considered robust.

4. **Placebo-in-time**: Shifts the treatment date to earlier periods. If significant "effects" appear before the real policy, the identifying assumptions are suspect.

5. **Leave-one-out influence**: Re-estimates the model dropping one entity at a time. Identifies whether any single unit drives the result.

### When checks fail

Don't just report failure. Explain:

- What specifically failed?
- Does it invalidate the main result or just weaken it?
- What can be done? (method switch, data fix, caveat added)

---

## Stage 9: Result report

### What this stage does

Generates a comprehensive academic paper draft in three formats:
1. **Structured JSON** (`report_data.json`) — machine-readable, all data extracted and normalized
2. **Markdown** (`paper.md`) — pipe tables with section numbering, GitHub/VS Code preview, convertible to PDF/Word via pandoc
3. **XeLaTeX** (`paper.tex`) — three-line `booktabs` tables with `siunitx` number alignment, `xeCJK` Chinese support, compiles directly to PDF

The report follows standard academic paper structure: Abstract → Introduction → Institution & Theory → Research Design → Empirical Results → Robustness → Conclusion → References → Appendix.

### Architecture

```
output_report.py     →  report_data.json     (data extraction + rough auto-generation)
       ↓
LLM translation step  →  report_data.json    (⚠ CRITICAL: translate identification, abstract, conclusion
       │                                        into natural Chinese academic prose)
       ↓
render_report.py     →  paper.md / paper.tex (rendering + optional PDF compilation)
```

`output_report.py` handles all format differences between Stage 7 estimation scripts and produces a unified JSON. The auto-generated fields (`model_spec.identification`, abstract, conclusion) use raw Stage 3/6 data which contains English terminology.

**The LLM MUST then read `report_data.json` and do the following:**

**A. Complete incomplete references (do this FIRST):**

References are stored in `data["references"]` as a list of dicts:
```json
{"text": "Full citation...", "complete": true}     // done
{"text": "Author (Year)", "complete": false,
 "search_query": "Author Year paper", "authors": "...", "year": "..."}  // needs web search
```

For every entry with `"complete": false`:
1. Search the web using `search_query` (or construct a query like `"{authors} {year} paper title journal"`)
2. Find the full citation: authors, title, journal, volume, pages, DOI if available
3. Replace the entry with `{"text": "<full citation string>", "complete": true}`

**B. Rewrite these fields into natural Chinese academic prose:**

| Field | What to translate | Source |
|---|---|---|
| `model_spec.identification` | Method rationale, assumption names, mechanism labels, fallback descriptions | Stage 3 + Stage 6 |
| `sections.conclusion` | Auto-generated conclusion paragraph | Stage 7 results |
| `data_meta` → abstract fields | Abstract text (rendered live from main_result) | Stage 7 results |

Translation guidelines:
- **Method names**: Translate but keep the English reference in parentheses on first use. E.g., "交错型双重差分估计量（Callaway & Sant'Anna, 2021）"
- **Assumption names**: Use standard Chinese econometrics terminology. E.g., "平行趋势假设" not "Parallel trends", "无预期效应" not "No anticipation"
- **Mechanism labels**: Translate the concept, not the code. E.g., "政策分批推广形成的交错处理" not "staggered_policy_shock"
- **Fallback descriptions**: Make them readable sentences, not key-value dumps
- **Abstract**: Rewrite as a flowing academic abstract paragraph, not bullet points

After completing both reference completion and field translation, save the updated `report_data.json`, then run `render_report.py`.

### Running Stage 9

```bash
# Step 1: Extract data
# Replace <state_stem> with your state file stem (e.g., ltci_state)
python scripts/output_report.py --policy "LTCI Pilot" --outcome "Fertility Rate" \
    --stage3 data/auto/<state_stem>/stage3_result.json \
    --stage6 data/auto/<state_stem>/stage6_confirmation.json \
    --stage7 data/auto/<state_stem>/stage7_main_result.json \
    --stage8 data/auto/<state_stem>/stage8_sensitivity.json \
    --stage8-placebo data/auto/<state_stem>/stage8_placebo.json \
    --stage2-sections data/auto/<state_stem>/stage2_sections.json \
    --data data/merged/panel.dta \
    --data-status data/auto/<state_stem>/stage5_data_status.json \
    --event-study data/auto/<state_stem>/stage7_event_study.json \
    --output data/auto/<state_stem>/report_data.json

# Step 2: LLM completes references (web search) + translates identification/abstract/conclusion → natural Chinese
# (Read report_data.json, web-search incomplete refs, translate fields, save back)

# Step 3: Render
python scripts/render_report.py --data data/auto/<state_stem>/report_data.json --compile
```

When using the pipeline orchestrator, the LLM must perform Step 2 between
`output_report.py` and `render_report.py`.

### CLI flags for output_report.py

| Flag | Required | Description |
|---|---|---|
| `--policy` | Yes | Policy name |
| `--outcome` | Yes | Outcome name |
| `--stage3` | No | Stage 3 theoretical method JSON |
| `--stage6` | No | Stage 6 confirmation JSON |
| `--stage7` | No | Stage 7 estimation results JSON |
| `--stage8` | No | Stage 8 sensitivity analysis JSON |
| `--stage8-placebo` | No | Stage 8 placebo test JSON |
| `--stage2-sections` | No | Stage 2 narrative sections JSON |
| `--data` | No | Path to panel data (.dta/.csv) for descriptive stats |
| `--data-status` | No | Stage 5 data_status JSON for data source appendix |
| `--event-study` | No | Path to event study output JSON |
| `--data-source` | No | Data source description |
| `--data-span` | No | Data time span |
| `--n-obs` | No | Number of observations |
| `--output` | No | Output path for report_data.json |

### Report sections

| Section | Source | Content |
|---|---|---|
| 摘要 (Abstract) | Auto-generated from results | Policy, method, main finding, robustness summary, causal claim strength |
| 一、引言 | Stage 2 sections | 1.1 研究背景, 1.2 文献综述 |
| 二、制度背景与理论分析 | Stage 2 sections | 2.1 制度背景, 2.2 理论机制 |
| 三、研究设计 | Stage 4 + Stage 7 + data | 3.1 数据来源与样本, 3.2 变量定义, 3.3 描述性统计, 3.4 识别策略与模型设定 |
| 四、实证结果 | Stage 7 + Stage 8 | 4.1 基准回归结果, 4.2 事件研究（平行趋势检验）|
| 五、稳健性检验 | Stage 8 | Consolidated checks: pass/fail + plain-language interpretation |
| 六、方法选择链 | Stage 6 | Methods tried, outcomes, rejection reasons (only if method changed) |
| 七、因果推断可信度评估 | Stage 6 | Strong/moderate/suggestive/not identifiable with justification |
| 八、结论与政策建议 | Stage 2 sections or auto-generated | Findings, policy implications, limitations |
| 参考文献 | Extracted from literature + canonical | Author-year references |
| 附录 | Stage 3 + Stage 6 | A. 识别假设检验, B. 方法选择与识别策略, C. 数据来源明细, D. 数据质量评估 |

### LaTeX compilation

```bash
xelatex paper.tex
# Or for a polished output:
xelatex paper.tex && xelatex paper.tex
```

Requires: `booktabs`, `siunitx`, `amsmath`, `fontspec`, `xeCJK` packages. System fonts: SimSun (宋体), SimHei (黑体), Times New Roman.

---

## Folder structure

```
policy-eval/
├── SKILL.md
├── scripts/
│   ├── run_pipeline.py            # Pipeline orchestrator (chain all stages)
│   ├── stage3_analyze.py          # Deterministic method recommendation (Stage 3)
│   ├── validate_data.py           # Data validation and quality checks (Stage 5→6)
│   ├── fetch_data.py              # Tier A data fetch (pre-written, not ad-hoc)
│   ├── clean_panel.py             # Merge and clean
│   ├── run_did.py                 # Standard DID
│   ├── run_staggered_did.py       # Staggered DID — doubly-robust C&S / S&A
│   ├── run_event_study.py         # Event study
│   ├── run_scm.py                 # Traditional synthetic control
│   ├── run_synthetic_did.py       # Synthetic DID (Arkhangelsky et al. 2021)
│   ├── run_rdd.py                 # Regression discontinuity
│   ├── run_iv.py                  # Instrumental variables (2SLS / LIML)
│   ├── placebo_test.py            # Placebo permutation test
│   ├── bacon_decomp.py            # Goodman-Bacon decomposition
│   ├── sensitivity_analysis.py    # Oster bounds, Rosenbaum, coefficient stability
│   ├── run_dml.py                 # Double/Debiased Machine Learning
│   ├── run_causal_forest.py       # Causal Forest for heterogeneity analysis
│   └── output_report.py           # Generate final report
├── references/
│   ├── data_sources.md            # Known data sources and access methods
│   ├── variable_map.json          # Stage 4→5 bridge: variable → fetch function
│   └── method_guide.md            # When to use each method
├── assets/
│   └── data_request_template.xlsx
└── data/
    ├── auto/                      # Auto-fetched data + stage outputs
    ├── manual/                    # User-filled templates
    ├── raw/                       # Raw files user drops in
    └── merged/                    # Final analysis-ready data
```

## Pipeline orchestration

Use `run_pipeline.py` to chain stages automatically:

```bash
# Start a new pipeline (runs Stage 3 automatically, prompts for interactive stages)
python scripts/run_pipeline.py --policy "LTCI Pilot" --outcome "Fertility Rate" \
    --state my_analysis.json

# Resume from a specific stage after completing interactive phases
python scripts/run_pipeline.py --state my_analysis.json --from-stage 7 \
    --data data/merged/panel.dta

# Preview what will be run (dry run)
python scripts/run_pipeline.py --policy "LTCI Pilot" --outcome "Fertility Rate" --dry-run

# Check pipeline status
python scripts/run_pipeline.py --state my_analysis.json --status
```

Stages 1-2 (problem definition and policy research) and Stages 4-5 (data requirements and acquisition) require user interaction. Stages 3, 6, 7, 8, and 9 run automatically when their inputs are available.

The pipeline maintains a JSON state file tracking which stages are completed and what outputs each stage produced. This enables restartability — if a stage fails, fix the input and resume from that stage.

---

## Interaction protocol

### The system's rule: default automatic, ask when stuck

```
Can the system do it alone?
├── Yes → Do it silently. Report only the result.
└── No → Is the ambiguity critical?
    ├── No → Make a reasonable assumption, note it, proceed.
    └── Yes → Ask the user one question at a time.
```

### Question format

When asking the user:

```
QUESTION:
[Clear, specific question]

Why this matters: [One sentence on how this affects the analysis]

Options:
A. [Option] → [Consequence]
B. [Option] → [Consequence]
C. [Option] → [Consequence]

Recommendation: [Option X] because [reason].
```

### Data handoff

When requesting manual data:

1. Generate an Excel template with known data pre-filled
2. Mark unknown cells with "?"
3. Place in `data/manual/`
4. Tell the user: "The template is at [path]. Fill the cells marked '?' and say 'done'."

---

## Examples

### Example 1: Theory and data agree

User: "用 DID 评估 2016 年长护险试点对城市生育率的影响"

System:
1. (Stage 2) Searches for policy details → outputs background report
2. (Stage 3) Assignment mechanism: staggered policy shock across cities. Recommends Callaway & Sant'Anna staggered DID. User approves.
3. (Stage 4) Determines data needs: city-level fertility, GDP, pilot status, pilot timing
4. (Stage 5) Fetches GDP from CNBS API. Cannot find city-level fertility → outputs gap report with template
5. User fills template, says "done"
6. (Stage 6) Verifies parallel pre-trends pass, no negative weight problem → confirms C&S method
7. (Stage 7) Runs C&S staggered DID → outputs ATTs by group and aggregate
8. (Stage 8) Runs placebo, alternative window, leave-one-out → all pass
9. (Stage 9) Outputs final report. Method chain: C&S (Stage 3) → C&S (Stage 6), no gap.

### Example 2: Theory and data disagree (method switch)

User: "评估新能源汽车补贴对空气质量的影响"

System:
1. Searches → finds subsidies are partially based on range threshold (≥400km gets max subsidy)
2. (Stage 3) Assignment mechanism: threshold-based eligibility. Recommends RDD as primary, DID as fallback.
3. (Stage 4) Data requirements: running variable (vehicle range), cutoff (400km), air quality, etc.
4. (Stage 5) Fetches air quality data, vehicle registrations
5. (Stage 6) Runs McCrary test → manipulation suspected near cutoff. Additionally, pre-treatment trends fail for DID comparison. Reports: "RDD invalid (manipulation), DID invalid (non-parallel trends). Falling back to SCM."
6. User approves the compromise.
7. (Stage 7) Runs SCM → finds significant effect
8. (Stage 9) Final report explains: "Theoretically preferred RDD was rejected because... Final method SCM identifies effect via synthetic counterfactual rather than local randomization at threshold. Causal claim: Moderate."
