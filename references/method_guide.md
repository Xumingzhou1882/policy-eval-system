# Method Selection Guide

## How to read this guide

This guide mirrors the deterministic decision tree in `stage3_analyze.py`. Start from the **assignment mechanism** (how was treatment assigned?), then follow the tree.

---

## Method overview

### Randomization Inference / Fisher Exact Test

**When**: Treatment is randomly assigned (lottery, RCT, natural experiment).

**What it does**: Compares outcomes between treated and control using exact permutation tests. No distributional assumptions needed.

**Key assumption**: SUTVA (no spillover), no selective attrition.

**When it fails**: Compliance is imperfect → use IV with assignment as instrument.

**Script**: (use `scipy.stats` directly)

---

### Regression Discontinuity Design (Sharp RDD)

**When**: A continuous score determines treatment with a sharp cutoff (e.g., exam score, income threshold, vehicle range).

**What it does**: Compares units just above vs. just below the cutoff using local linear regression.

**Key assumption**: No manipulation of the running variable at the cutoff.

**Test**: McCrary (2008) density test — check for discontinuity in density of running variable.

**When it fails**: Manipulation detected → donut-hole RDD (drop observations at cutoff) or switch to DID.

**Script**: `run_rdd.py --type sharp`

---

### Regression Discontinuity Design (Fuzzy RDD)

**When**: A threshold affects treatment probability but does not determine it perfectly (partial compliance).

**What it does**: Uses the threshold as an instrument for treatment. Estimates LATE for compliers near the cutoff.

**Key assumption**: First-stage must be strong (F > 10 at cutoff).

**Script**: `run_rdd.py --type fuzzy --treatment-var <name>`

---

### Propensity Score Matching / IPW / Doubly-Robust (AIPW)

**When**: Treatment depends on observable characteristics; cross-section or pooled data.

**What it does**: Matches or reweights units so treated and control are comparable on observables.

**Key assumption**: Unconfoundedness (CIA) — all confounders are measured. **Cannot be tested.**

**When it fails**: Unmeasured confounders exist → switch to panel methods (DID, FE).

**Script**: (use `statsmodels` or `causalinference` directly; CEM via `cem` package)

---

### Standard Difference-in-Differences (TWFE)

**When**: Single treatment time, clear treated/untreated groups, panel data available.

**What it does**: Differences out time-invariant unit effects and period effects.

**Key assumption**: Parallel trends — treated and control would have followed same path absent treatment.

**Test**: Event study — pre-treatment coefficients should be jointly zero.

**When it fails**: Pre-trends diverge → PSM-DID or SCM.

**Script**: `run_did.py`

---

### Callaway & Sant'Anna (2021) / Sun & Abraham (2021)

**When**: **Staggered policy rollout** — different units treated at different times.

**What it does**: Estimates group-time ATTs for each treatment cohort, then aggregates. Avoids the negative-weight problem of traditional staggered TWFE (Goodman-Bacon 2021).

**Key assumption**: Parallel trends conditional on covariates; no anticipation.

**Test**: Event study by cohort; Bacon decomposition (negative weight share < 10%).

**When it fails**: Never-treated units not comparable → use not-yet-treated as control (C&S with `not-yet-treated`).

**Script**: `run_staggered_did.py`, `bacon_decomp.py`

---

### Instrumental Variables (2SLS / LIML)

**When**: A valid instrument exists — a variable that affects treatment but not the outcome directly.

**What it does**: Uses exogenous variation in the instrument to identify the causal effect of treatment. Estimates LATE for compliers.

**Key assumption**: Exclusion restriction — instrument only affects outcome through treatment. **Cannot be tested.** Relevance must be strong (Montiel Olea & Pflueger 2013 effective F > critical value).

**When it fails**: Weak instruments → use LIML (more robust). Invalid instruments → find a better instrument or use SCM.

**Script**: `run_iv.py`

---

### Synthetic Difference-in-Differences (Synthetic DID)

**When**: No untreated control group but many untreated donor units exist; panel data available; want valid standard errors and p-values.

**What it does**: Finds optimal unit weights for donor pool that match pre-treatment outcomes, then applies DID differencing. Combines SCM-style weighting with DID-style time differencing to produce valid inference.

**Key assumption**: Good pre-treatment fit; synthetic counterfactual captures time-varying confounders.

**Test**: Pre-treatment RMSE; placebo-based inference (apply Synthetic DID to each donor unit → null distribution).

**When it fails**: Poor pre-treatment fit → traditional SCM or interactive fixed effects.

**Script**: `run_synthetic_did.py`

**Literature**: Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), "Synthetic Difference-in-Differences." AER.

---

### Traditional Synthetic Control Method (SCM)

**When**: Few treated units, many untreated units, no valid instrument, time-varying unobservables present. Use when Synthetic DID placebo inference is unstable.

**What it does**: Constructs a weighted combination of untreated donor units to create a counterfactual that matches the treated unit's pre-treatment trajectory.

**Key assumption**: Good pre-treatment fit; no unobserved time-varying confounders in post-treatment period.

**Test**: In-space placebo (apply SCM to untreated units); pre-treatment RMSE.

**When it fails**: Poor pre-treatment fit → use interactive fixed effects or generalized SCM (Xu 2017).

**Script**: `run_scm.py`

---

### Intensity DID / Continuous-Treatment DiD

**When**: All units are treated but at different doses/intensities.

**What it does**: Replaces binary treatment with continuous dose in the DID framework.

**Key assumption**: Parallel trends in the dose-response relationship; intensity variation is exogenous.

**When it fails**: Endogenous intensity → IV with an instrument for intensity.

**Script**: (use panel regression with continuous treatment interaction)

---

### Double/Debiased Machine Learning (DML)

**When**: Selection on observables with many controls (>15-20), or when the functional form of confounding is unknown. Modern upgrade over PSM/IPW.

**What it does**: Uses flexible ML models (RandomForest, GradientBoosting, Lasso) to estimate nuisance functions E[Y|X] and E[T|X], then constructs Neyman-orthogonal scores that debias the ML estimates. Cross-fitting prevents overfitting bias.

**Key assumption**: Unconfoundedness (CIA) — all confounders are measured. ML can model them flexibly but cannot fix omitted variable bias.

**Test**: CV R² of nuisance models (should both be > 0.1); check for extreme propensity scores.

**When it fails**: Very poor overlap → try trimming or AIPW. Missing confounders → no method fixes this; use Oster bounds (sensitivity_analysis.py).

**Script**: `run_dml.py --ml-model gradient_boosting --cv 5`

**Literature**: Chernozhukov et al. (2018), "Double/Debiased Machine Learning for Treatment and Structural Parameters." Econometrics Journal.

---

### Causal Forest (Generalized Random Forest)

**When**: Treatment effect heterogeneity is of substantive interest — "who benefits most from the policy?" Not a primary identification strategy but a complement to any method.

**What it does**: Honest random forest that estimates CATE for each unit. Uses sample splitting: one half for tree structure, the other for leaf estimates. Provides valid confidence intervals for individual-level CATE.

**Key outputs**:
- CATE distribution (mean, SD, quantiles, share positive/negative)
- Variable importance (which features drive heterogeneity)
- Best Linear Projection (which variables systematically predict CATE)
- Quantile comparison (characteristics of most- vs. least-affected units)

**Key assumption**: Unconfoundedness + overlap (same as DML). The causal forest inherits the identification assumptions of the underlying method.

**When it fails**: Very small sample (< 500) → trees won't split enough for useful heterogeneity. Very few features → limited heterogeneity to discover.

**Script**: `run_causal_forest.py --num-trees 2000 --features x1 x2 x3`

**Literature**: Wager & Athey (2018), "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." JASA. Athey, Tibshirani & Wager (2019), "Generalized Random Forests." Annals of Statistics.

---

### Triple Difference (DDD)

**When**: Multiple policies affect the same outcome concurrently.

**What it does**: Adds a third difference dimension — compares DID estimates across groups differentially exposed to each policy.

**Key assumption**: Policies are separable (no interaction effects); parallel trends in the triple-differenced sense.

**When it fails**: Policies inseparable → flag as fundamentally unidentified for single-policy effects.

**Script**: (use interaction of three dimensions in panel regression)

---

## Quick reference: assumption testability

| Assumption | Testable? | How to test |
|---|---|---|
| Parallel trends (DID) | Yes | Event study pre-trend coefficients |
| No manipulation (RDD) | Yes | McCrary density test |
| First-stage strength (IV/Fuzzy RDD) | Yes | F-statistic > 10 (Montiel Olea & Pflueger) |
| Overlap/common support (matching) | Yes | Propensity score distribution check |
| Pre-treatment fit (SCM) | Yes | Pre-treatment RMSE |
| No anticipation (DID) | Yes | t-1 coefficient in event study |
| Negative weights (Staggered DID) | Yes | Goodman-Bacon decomposition |
| Unconfoundedness (CIA) | **No** | Must be argued from institutional knowledge |
| Exclusion restriction (IV) | **No** | Must be argued; partial test via overidentification (Hansen J) |
| SUTVA (no spillover) | **No** | Argue from study design; spatial placebo as partial check |
| Monotonicity (IV) | **No** | Plausible if instrument is policy eligibility rule |
