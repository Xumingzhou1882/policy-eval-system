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

After the narrative report, the LLM MUST produce a structured facts file at `data/auto/stage2_facts.json`. This file contains ONLY observable facts about the policy design — no methodological judgments. The facts answer 7 questions that any policy researcher can determine from public documents:

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
# Path: data/auto/stage2_facts.json
```

Then pass it to Stage 3:
```bash
python scripts/stage3_analyze.py --from-facts data/auto/stage2_facts.json
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

## Stage 3: Theoretical method analysis

### What this stage does

**This is the core of the system.** Before looking at any data, determine what identification strategy is theoretically appropriate. The goal is to answer:

> *Given the treatment assignment mechanism discovered in Stage 2, what source of variation can credibly identify a causal effect?*

This stage does NOT consider data availability. It answers: what should we do, in an ideal world?

### Two-level decision tree

The decision runs in two levels, both implemented deterministically in `stage3_analyze.py`:

**Level 1 — Facts → Mechanism**: Reads the Stage 2 structured facts (`data/auto/stage2_facts.json`). Classifies the assignment mechanism based on observable policy features. No methodological knowledge needed — pure rule-based classification.

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
python scripts/stage3_analyze.py --from-facts data/auto/stage2_facts.json \
    --output data/auto/stage3_result.json

# Level 2 only: mechanism already known (backward compatibility)
python scripts/stage3_analyze.py --mechanism staggered_policy_shock \
    --has-control-group --output data/auto/stage3_result.json
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

### What the system does

Based on the theoretical method from Stage 3, produce a concrete data requirements checklist — what variables are needed, at what level, from what source, and whether each is essential or optional. Then **wait for the user to confirm** what they can provide before proceeding to data acquisition.

### Output format

Present data needs in structured tables, not prose. Distinguish **essential** (required by the identification strategy) from **optional** (improves precision or enables robustness checks).

#### Essential variables

| 变量 | 含义 | 层级 | 来源 |
|---|---|---|---|
| `outcome` | [What to measure] | [Individual? City-year?] | [Expected source] |
| `entity_id` | [Unit identifier] | [Entity level] | [Expected source] |
| `time` | [Time period] | [Year? Quarter?] | [Expected source] |
| `first_treated` | [When each unit is first treated] | [Entity level] | Policy documents |

Add identification-specific requirements from Stage 3 (e.g., running variable for RDD, instrument for IV, never-treated indicator for staggered DID).

#### Optional variables

| 变量 | 含义 | 层级 | 来源 |
|---|---|---|---|
| ... | ... | ... | ... |

#### Policy variables (from Stage 2)

List policy data already collected (pilot city names, dates, batch info).

### Confirmation step

After presenting the checklist, explicitly ask the user:

```
Which of these data sources do you have access to?
A. [Source 1] — I can provide this
B. [Source 2] — I can provide this
C. [Source 3] — not available

If Tier B/C data is available: "Place the files in data/raw/ and say 'data is ready'."
If Tier A: "I will auto-fetch this in Stage 5."
```

**Do not proceed to Stage 5 until the user confirms what data is available.** Gaps identified here drive the data acquisition plan.

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
    --output data/auto/validation_report.json
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
acquisition status in `stages.stage5.data_status`:

```json
{
  "stages": {
    "stage5": {
      "data_status": {
        "gdp":              {"tier": "A", "status": "fetched",  "path": "data/auto/gdp.json"},
        "population":       {"tier": "A", "status": "fetched",  "path": "data/auto/population.json"},
        "fertility_rate":   {"tier": "B", "status": "requested","path": null},
        "pilot_city_list":  {"tier": "C", "status": "pending",  "path": "data/manual/pilot_cities.xlsx"},
        "education_level":  {"tier": "D", "status": "unavailable", "path": null}
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

Tier A uses pre-written `fetch_data.py` functions — no ad-hoc code. Tier B/C
follow fixed templates so the data contract between stages is predictable.
The pipeline state file is the single source of truth for what data exists,
what's pending, and what's unavailable.

Wait for the user to say "data is ready" before proceeding to Stage 6.

---

## Stage 6: Final method confirmation

### What this stage does

**This is where theory meets reality.** With actual data in hand, verify whether the theoretical method from Stage 3 is feasible. If not, fall back to the pre-registered alternatives — and document the compromise.

This stage has two possible outcomes:

```
Outcome A: Data supports the theoretical method   → Proceed as planned
Outcome B: Data contradicts an assumption          → Switch to fallback, document why
```

### Step 1: Verify theoretical method against actual data

For each key identifying assumption from Stage 3, check against the real data:

```
Assumption verification:
├── [Assumption 1]: [How to test / How to argue]
│   ├── Data check: [Specific test result or data characteristic]
│   ├── Verdict: ✓ Holds / ✗ Violated / ? Cannot verify
│   └── If violated: Is there a fallback that relaxes this assumption?
├── [Assumption 2]: [How to test / How to argue]
│   ...
└── [Assumption 3]: ...
```

For DID-family methods, pay special attention to:

- **Parallel pre-trends**: Generate an event study plot. Test whether pre-treatment coefficients are jointly zero. If pre-treatment trends diverge, DID is not valid — switch to the fallback.
- **Staggered DID negative weights**: Run Goodman-Bacon decomposition. If the negative weight share exceeds 10%, TWFE is unreliable — switch to C&S / S&A / BJS.
- **No-anticipation**: Are there effects in t-1? If yes, units may have adjusted before the policy — flag as a caveat.
- **Treatment variation**: Is there sufficient within-unit treatment variation? If treatment is near-constant (e.g., one batch covers 90% of units), staggered DID has low power.

For RDD:

- **McCrary test**: Is there a discontinuity in the density of the running variable at the cutoff? If yes, manipulation is suspected — use donut-hole RDD or switch to DID.
- **Baseline covariate balance at cutoff**: Are pre-treatment covariates smooth across the cutoff? If not, the RDD design is suspect.
- **First-stage strength (fuzzy RDD)**: Is the jump in treatment probability at the cutoff statistically significant? If the F-statistic < 10, the instrument is weak.

For IV:

- **First-stage F-statistic**: Montiel Olea & Pflueger (2013) effective F-statistic. If below the critical value, the instrument is weak — use LIML or switch strategy.
- **Overidentification test** (if multiple instruments): Hansen J-test. Rejection means at least one instrument is invalid.

For SCM:

- **Pre-treatment RMSE**: Is the synthetic control closely tracking the treated unit before treatment? If fit is poor, SCM is unreliable — consider interactive fixed effects instead.
- **Donor pool adequacy**: Are there enough untreated units with similar characteristics? If not, SCM extrapolates too far.
- **In-space placebo**: Does the effect for the treated unit stand out when you apply SCM to each donor unit?

### Step 2: Make the final decision

```
Can the theoretical method be applied to this data?
├── YES → Use the theoretical method.
│   Document that all testable assumptions hold.
│
└── NO → Which assumption fails?
    ├── Assumption is testable and clearly violated
    │   → Switch to the pre-registered fallback from Stage 3.
    │   → Record what was violated and why the fallback is valid.
    │
    ├── Data is missing a required variable
    │   → Return to Stage 5, request the missing data.
    │
    └── No fallback available
        → Best feasible method given constraints.
        → Downgrade causal claim strength.
        → Flag as a major limitation in Stage 9.
```

### Final method confirmation report

```
═══════════════════════════════════
Final Method Confirmation
═══════════════════════════════════

Theoretical method (Stage 3): [Method A]
Final method (Stage 6): [Method B]

[If A = B]:
All testable assumptions of [Method A] are supported by the data. Proceeding with
the theoretically preferred specification.

[If A ≠ B]:
[Method A] was theoretically preferred, but the data revealed that:
├── [Specific assumption violation or data limitation]
├── [Why this matters for identification]
└── Therefore switching to [Method B], which:
    ├── [Relaxes assumption X]
    ├── [Requires assumption Y instead — is this plausible?]
    └── [Implication for causal claim strength: weaker / different interpretation]

Key specification details:
├── Model: [e.g., Callaway & Sant'Anna (2021) group-time ATT, never-treated control]
├── Outcome transformation: [e.g., log, IHS, none]
├── Fixed effects: [Entity, Time, both, or neither]
├── Standard errors: [e.g., Clustered at city level, wild bootstrap]
└── Covariates: [List of controls included in the main specification]

Known limitations (to be surfaced in Stage 9):
├── [Limitation 1]
└── [Limitation 2]

Do you approve the final method?
═══════════════════════════════════
```

Do not proceed until the user confirms.

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

### Output format

```
═══════════════════════════════════
Causal Inference Report
═══════════════════════════════════

Policy: [Policy name]
Outcome: [Outcome name]
Method chain: [Theoretical method (Stage 3)] → [Final method (Stage 6)]
  [If they differ, explain the gap here in one sentence]
Data: [Source, time span, N]

─── Main Result ───
The policy [increased/decreased/had no effect on] [outcome].
Effect size: [X]%, p = [value] (significant at [level])

─── Robustness ───
[X/Y] robustness checks passed.
[List each check and result, one line each]

─── Key Assumption ───
[Assumption]: [Did it hold? How do we know?]

─── Limitations ───
1. [Limitation — including any gap between theoretical and final method]
2. [Limitation]

─── Causal Claim Strength ───
[Strong / Moderate / Suggestive]
[One sentence justification, referencing assumption verification and the
method chain from Stage 3 → Stage 6]

═══════════════════════════════════
```

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

Stages 1-2 (problem definition and policy research) and Stages 4-6 (data requirements, acquisition, and method confirmation) require user interaction. Stages 3, 7, 8, and 9 run automatically when their inputs are available.

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
