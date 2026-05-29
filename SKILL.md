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

### Step 1: Characterize the assignment mechanism

From the policy research in Stage 2, identify which of the following best describes how treatment was assigned:

| Assignment mechanism | What it means | Implied variation |
|---|---|---|
| **Random assignment** | Lottery, randomized experiment, or natural randomization | Random — direct comparison is valid |
| **Threshold rule** | A continuous score determines eligibility with a sharp cutoff | Discontinuity at the threshold |
| **Selection on observables** | Treatment depends on characteristics we can measure and control for | Within-group comparisons after conditioning |
| **Selection on time-invariant unobservables** | Units differ in fixed ways; treatment is a policy shock at a known time | Within-unit change over time, across-unit difference in timing |
| **Selection on time-varying unobservables** | Confounding factors change over time and differ across units | Instrument-based, or synthetic counterfactual |
| **Continuous exposure intensity** | Different units receive different doses/intensities of the same policy | Dose-response variation |
| **Multiple overlapping policies** | Several policies affect the same units concurrently | Differential exposure across policies |

### Step 2: Select the theoretical identification strategy

Start from the assignment mechanism. Trace through the decision logic:

```
Q1: How is treatment assigned?

A. Random assignment
  └── Experimental analysis (randomization inference, Fisher exact test)
      Rationale: Randomization breaks the link between treatment and confounders.
      Key assumption: No selective attrition, no spillover (SUTVA).

B. A known continuous threshold determines eligibility
  ├── Sharp RDD: treatment probability jumps from 0 to 1 at cutoff
  └── Fuzzy RDD: treatment probability jumps by less than 1 at cutoff
      Rationale: Units just above and below the threshold are as-good-as-random.
      Key assumption: No precise manipulation of the running variable (McCrary test).
      If manipulation suspected → donut-hole RDD or switch to another strategy.

C. Treatment depends on observables (no time dimension, or cross-section only)
  ├── Treatment/control overlap is good → PSM, IPW, or doubly-robust (AIPW)
  ├── Covariate balance is the priority → CEM, entropy balancing
  └── No overlap (extrapolation required) → Not identifiable; flag as major limitation
      Rationale: If we can measure what drives selection, we can condition on it.
      Key assumption: Unconfoundedness (CIA / selection on observables).
      Cannot be directly tested — must be argued from institutional knowledge.

D. Treatment is a policy shock at a known time, affecting some units but not others
  ├── Single treatment time, clear treated/untreated groups
  │   └── Standard DID (two-way fixed effects)
  │       Key assumption: Parallel trends — in the absence of treatment, treated
  │       and untreated would have followed the same path.
  ├── Staggered adoption (different units treated at different times)
  │   └── Heterogeneity-robust DiD estimators:
  │       - Callaway & Sant'Anna (2021): group-time average treatment effects
  │       - Sun & Abraham (2021): cohort-specific ATT, never-treated as control
  │       - Borusyak, Jaravel & Spiess (2024): imputation-based
  │       - de Chaisemartin & D'Haultfoeuille (2020): instantaneous switchers
  │       Note: Traditional TWFE is NOT recommended for staggered designs.
  │       It can produce negative weights when treatment effects are heterogeneous.
  │       Key assumption: Parallel trends, no anticipation, no spillover.
  └── No clear untreated group (everyone gets treated eventually)
      └── Consider the earlier-treated as control for later-treated (C&S approach),
          but recognize this only identifies effects for the later-treated group
          over the window before they themselves are treated.

E. Selection on unobservables that may be time-varying
  ├── A valid instrument exists (affects treatment but not outcome directly)
  │   └── IV / 2SLS (with weak-instrument diagnostics: Montiel Olea & Pflueger 2013)
  │       Estimates LATE for compliers.
  │       Key assumption: Exclusion restriction — the instrument only affects the
  │       outcome through treatment. Must be argued, cannot be formally tested.
  ├── No instrument available
  │   ├── A few treated units, many untreated → Synthetic Control Method (SCM)
  │   │     Constructs a weighted combination of untreated units as counterfactual.
  │   │     Key assumption: Pre-treatment fit quality, no unobserved time-varying
  │   │     confounders during the post-treatment period.
  │   ├── Panel data with many units → Interactive fixed effects (Bai 2009),
  │   │     Generalized SCM (Xu 2017), Matrix completion (Athey et al. 2021)
  │   │     Key assumption: Factor structure captures unobserved confounding.
  │   └── No panel → Not identifiable without stronger assumptions.
  └── A threshold exists but compliance is imperfect → Fuzzy RDD = IV

F. Continuous treatment intensity (everyone is treated, but at different doses)
  └── Intensity DID, or IV if a variable shifts intensity exogenously.
      Careful: cross-sectional variation in intensity is often endogenous.
      Need an argument for why the intensity variation is exogenous.
      Key assumption: Parallel trends in the dose-response relationship.

G. Multiple policies overlapping in time
  ├── Policies affect different groups differently → Triple difference (DDD)
  │     Compares the DID estimate in a group exposed to both policies vs.
  │     the DID estimate in a group exposed to only one.
  ├── Policies are separable → Control for the other policy as a covariate
  └── Policies are inseparable → Flag as fundamental identification problem
```

### Step 3: The theoretical method report

Output a formal justification:

```
═══════════════════════════════════
Theoretical Method Analysis
═══════════════════════════════════

Policy: [Policy name]
Assignment mechanism: [Which type from Step 1]

Why this mechanism:
[2-3 sentences connecting the policy design to the mechanism type.
 Cite specific institutional facts from Stage 2.]

Primary theoretical recommendation: [Method name]

Why:
├── Source of variation: [What variation identifies the causal effect]
├── Theoretically addresses: [What bias would otherwise exist]
└── Literature basis: [This method is standard for this assignment mechanism]

Key identifying assumptions (theoretical — not yet checked against data):
├── [Assumption 1] → Testable with data? → [Yes (method) / No (argument required)]
├── [Assumption 2] → Testable with data? → [Yes (method) / No (argument required)]
└── [Assumption 3] → Testable with data? → [Yes (method) / No (argument required)]

If the primary strategy fails (e.g., assumption violated by data),
fallback strategies:
├── Fallback 1: [Method] — applicable if [condition]
├── Fallback 2: [Method] — applicable if [condition]
└── Worst case: [Method] — weaker but still informative

What this method requires from the data (preview for Stage 4):
├── Essential: [Variable 1, Variable 2, ...]
└── Optional: [Variable 3, Variable 4, ...]

Do you approve this theoretical framework?
═══════════════════════════════════
```

### Why theory comes before data

Reversing the order — looking at available data first, then choosing a method — is a common error. It leads to:

1. **Method shopping**: picking a method that produces "significant" results rather than one that is theoretically appropriate.
2. **Ignoring the assignment mechanism**: the most important driver of method choice.
3. **Opaque compromises**: the reader cannot distinguish "we chose this method because it's correct" from "we chose this method because it's what our data allowed."

By fixing the theoretical method first, any later deviation due to data constraints is explicit and traceable. Stage 6 will document the gap between "should have used" and "actually used."

### Step 4: Run the deterministic decision engine

After the LLM characterizes the assignment mechanism, run the decision engine to produce a deterministic recommendation:

```bash
python scripts/stage3_analyze.py \
    --mechanism staggered_policy_shock \
    --staggered \
    --has-control-group \
    --policy "Policy Name" \
    --outcome "Outcome Variable" \
    --output stage3_result.json
```

The script is **deterministic**: same inputs always produce the same output. This guarantees consistency across sessions and prevents the LLM's reasoning from drifting over time.

The LLM's role is to:
1. Extract the assignment mechanism type and flags from Stage 2 research
2. Feed them into the script
3. Present the output to the user for approval

The script's role is to:
1. Apply the fixed decision tree to select the primary method
2. Enumerate assumptions, fallbacks, and required variables
3. Output structured JSON for later stages to consume

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
| **A** | Public API / open data | LLM writes fetch code on-the-fly using `akshare`, `pandas_datareader`, `requests`, etc. |
| **B** | Requires registration (micro surveys, proprietary databases) | Generate instructions; wait for user to place data in `data/raw/` |
| **C** | Manual collection (policy documents, local yearbooks) | Provide Excel template, tell user exactly what to fill in |
| **D** | Does not exist / inaccessible | Honestly report and suggest alternatives |

### Tier A: LLM writes fetch code on-the-fly

No fixed fetch scripts. For each data source identified in Stage 4, the LLM writes a short Python script tailored to that specific source:

```python
# Example: fetch data from a public API (package varies by data source)
import pandas as pd
# from package import function  # depends on the data source

df = some_api_fetch()  # LLM writes this per project
df.to_json("data/auto/variable_name.json", orient="records", force_ascii=False)
```

```

After fetching, output what was obtained and what's still missing.

### Gap report format

After auto-fetching, output what's still missing:

```
═══════════════════════════════════
Data Gap Report
═══════════════════════════════════

Auto-fetched (✓):
├── Variable A from public API
├── Variable B from statistical database
└── Geographic identifier mapping

Still needed (manual):
├── ✗ Variable C → Tier B (micro survey data)
│     Application URL: [relevant data portal]
│     System will draft the application text for you.
│     Once approved, place data in data/raw/
│
├── ✗ Variable D → Tier C (manual collection)
│     Template: data/manual/variable_d.xlsx
│     Instructions: [specific instructions for the user]
│
└── ✗ Exact treatment dates → Tier C
      Template: data/manual/treatment_dates.xlsx (pre-populated with known data)
      Instructions: Fill in the "?" cells only

═══════════════════════════════════
```

### Template generation

For every Tier C item, generate an Excel template in `data/manual/` using `openpyxl` or `pandas`. Pre-fill all known data, mark unknown cells with "?", and include a "source" column.

### Key principle

Data acquisition is project-specific. The LLM writes, runs, and discards fetch code for each project. Fixed scripts are reserved for the estimation pipeline (Stages 6-8), where the logic is stable across projects.

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

Run the main specification chosen in Stage 6:

```python
# Core regression (example for Callaway & Sant'Anna staggered DID)
# Use the did package (R) or csdid Python port
import pandas as pd

df = pd.read_stata("data/merged/analysis_ready.dta")

# Example: standard DID with TWFE (only if TWFE is valid — confirmed in Stage 6)
from linearmodels import PanelOLS

df["log_outcome"] = np.log(df["outcome"])
df["treated_post"] = df["treated"] * df["post"]

model = PanelOLS.from_formula(
    "log_outcome ~ treated_post + control1 + control2 + EntityEffects + TimeEffects",
    data=df.set_index(["city_id", "year"])
)
results = model.fit(cov_type="clustered", cluster_entity=True)
```

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

For each check, report:
```
✓ [Check name]: Passed — [one-line interpretation]
✗ [Check name]: Failed — [what this means and what to do]
```

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
│   ├── stage3_analyze.py        # Deterministic method recommendation (Stage 3)
│   ├── clean_panel.py           # Merge and clean
│   ├── run_did.py               # Standard DID
│   ├── run_staggered_did.py     # Staggered DID (C&S / S&A / BJS)
│   ├── run_event_study.py       # Event study
│   ├── run_scm.py               # Synthetic control
│   ├── run_rdd.py               # Regression discontinuity
│   ├── run_iv.py                # Instrumental variables
│   ├── placebo_test.py          # Placebo permutation test
│   ├── bacon_decomp.py          # Goodman-Bacon decomposition
│   └── output_report.py         # Generate final report
├── references/
│   ├── data_sources.md          # Known data sources and access methods
│   └── method_guide.md          # When to use each method
├── assets/
│   └── data_request_template.xlsx
└── data/
    ├── auto/                    # Auto-fetched data
    ├── manual/                  # User-filled templates
    ├── raw/                     # Raw files user drops in
    └── merged/                  # Final analysis-ready data
```

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
