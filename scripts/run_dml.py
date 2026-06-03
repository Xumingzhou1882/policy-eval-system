"""
Double/Debiased Machine Learning (DML) for causal inference.

Implements Chernozhukov et al. (2018) DML with:
  - Neyman-orthogonal scores (partialling out confounding)
  - K-fold cross-fitting to avoid overfitting bias
  - Flexible ML models for nuisance functions (RandomForest, GradientBoosting, Lasso)
  - Valid inference under weak conditions (slower-than-n^-1/4 convergence of ML models)

Supports:
  - Continuous outcomes (PartialLinearDML)
  - Binary treatments (standard DML for ATE)
  - Heterogeneous treatment effects via CATE estimation

Usage:
    python scripts/run_dml.py --data data/merged/panel.dta \\
                              --outcome log_wage \\
                              --treatment education_years \\
                              --controls age experience occupation \\
                              --ml-model gradient_boosting \\
                              --cv 5 --output dml_result.json

Reference:
    Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018).
    "Double/Debiased Machine Learning for Treatment and Structural Parameters."
    The Econometrics Journal, 21(1), C1-C68.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Check for econml
try:
    from econml.dml import LinearDML, SparseLinearDML, NonParamDML
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False

# Check for sklearn
try:
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
    )
    from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".dta":
        return pd.read_stata(str(p))
    elif p.suffix == ".csv":
        return pd.read_csv(str(p))
    raise ValueError(f"Unsupported format: {p.suffix}")


# ═══════════════════════════════════════════════════════════════════════
# ML model factory
# ═══════════════════════════════════════════════════════════════════════

def _get_ml_models(model_type: str, discrete_treatment: bool = False):
    """Return (outcome_model, treatment_model) for DML nuisance functions."""
    if model_type == "random_forest":
        model_y = RandomForestRegressor(n_estimators=200, max_depth=10,
                                         min_samples_leaf=20, random_state=42,
                                         n_jobs=-1)
        model_t = (RandomForestClassifier(n_estimators=200, max_depth=10,
                                          min_samples_leaf=20, random_state=42,
                                          n_jobs=-1)
                   if discrete_treatment
                   else RandomForestRegressor(n_estimators=200, max_depth=10,
                                              min_samples_leaf=20, random_state=42,
                                              n_jobs=-1))
    elif model_type == "gradient_boosting":
        model_y = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                             min_samples_leaf=20, random_state=42)
        model_t = (GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                               min_samples_leaf=20, random_state=42)
                   if discrete_treatment
                   else GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   min_samples_leaf=20, random_state=42))
    elif model_type == "lasso":
        model_y = LassoCV(cv=5, random_state=42, max_iter=5000)
        model_t = (LogisticRegressionCV(cv=5, penalty='l1', solver='saga',
                                        max_iter=5000, random_state=42)
                   if discrete_treatment
                   else LassoCV(cv=5, random_state=42, max_iter=5000))
    elif model_type == "linear":
        model_y = LinearRegression()
        model_t = (LogisticRegressionCV(cv=5, max_iter=2000, random_state=42)
                   if discrete_treatment
                   else LinearRegression())
    else:
        model_y = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                             min_samples_leaf=20, random_state=42)
        model_t = (GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                               min_samples_leaf=20, random_state=42)
                   if discrete_treatment
                   else GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   min_samples_leaf=20, random_state=42))
    return model_y, model_t


# ═══════════════════════════════════════════════════════════════════════
# Manual DML implementation (fallback when econml not available)
# ═══════════════════════════════════════════════════════════════════════

def _manual_dml(X: np.ndarray, y: np.ndarray, T: np.ndarray,
                model_y, model_t, n_folds: int = 5,
                discrete_treatment: bool = True) -> dict:
    """
    Manual DML with K-fold cross-fitting.

    Steps:
      1. Split data into K folds
      2. For each fold k:
         a. Train model_y on folds != k to predict y (outcome nuisance)
         b. Train model_t on folds != k to predict T (treatment nuisance)
         c. Compute residuals: y_tilde = y - E[y|X], T_tilde = T - E[T|X]
         d. Regress y_tilde on T_tilde to get theta_k
      3. Average theta_k across folds → ATE
      4. Compute standard error via influence functions
    """
    n = len(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    thetas = []
    if_scores = np.zeros(n)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        # Step 1: Outcome nuisance E[Y | X]
        model_y_clone = _clone_model(model_y)
        model_y_clone.fit(X_train, y_train)
        y_pred = model_y_clone.predict(X_test)

        # Step 2: Treatment nuisance E[T | X]
        model_t_clone = _clone_model(model_t)
        if discrete_treatment:
            model_t_clone.fit(X_train, T_train.ravel())
            T_pred = model_t_clone.predict_proba(X_test)[:, 1]
        else:
            model_t_clone.fit(X_train, T_train.ravel())
            T_pred = model_t_clone.predict(X_test).ravel()

        # Step 3: Residuals
        y_tilde = y_test - y_pred
        T_tilde = T_test.ravel() - T_pred

        # Step 4: ATE for this fold (IV-type estimator)
        # theta = E[T_tilde * y_tilde] / E[T_tilde^2]
        numerator = np.mean(T_tilde * y_tilde)
        denominator = np.mean(T_tilde ** 2)
        theta_k = numerator / denominator if abs(denominator) > 1e-10 else 0
        thetas.append(theta_k)

        # Influence function scores for SE
        # psi = (T_tilde * (y_tilde - theta * T_tilde)) / E[T_tilde^2]
        psi = (T_tilde * (y_tilde - theta_k * T_tilde)) / denominator
        if_scores[test_idx] = psi

    # Aggregate
    ate = float(np.mean(thetas))
    # SE via influence function: sqrt(Var(psi) / n)
    se = float(np.sqrt(np.var(if_scores) / n))
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - _normal_cdf(abs(t_stat))))

    return {
        "ate": ate,
        "std_error": se,
        "t_statistic": t_stat,
        "p_value": p_value,
        "fold_estimates": [float(t) for t in thetas],
        "n_folds": n_folds,
    }


def _clone_model(model):
    """Clone an sklearn model."""
    from sklearn.base import clone
    return clone(model)


# ═══════════════════════════════════════════════════════════════════════
# econml-based DML
# ═══════════════════════════════════════════════════════════════════════

def _econml_dml(X: np.ndarray, y: np.ndarray, T: np.ndarray,
                model_y, model_t, n_folds: int = 5,
                discrete_treatment: bool = True) -> dict:
    """Use econml's LinearDML for robust estimation."""
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        cv=n_folds,
        random_state=42,
    )
    dml.fit(y, T, X=X, inference="auto")

    ate = float(dml.ate_[0]) if hasattr(dml, 'ate_') else float(dml.coef_)
    se = float(dml.ate__se[0]) if hasattr(dml, 'ate__se') else float(dml.coef__se)
    t_stat = ate / se if se > 0 else 0
    p_value = float(2 * (1 - _normal_cdf(abs(t_stat))))

    ci_lower = float(dml.ate__interval()[0][0]) if hasattr(dml, 'ate__interval') else ate - 1.96 * se
    ci_upper = float(dml.ate__interval()[1][0]) if hasattr(dml, 'ate__interval') else ate + 1.96 * se

    # Best linear predictor of CATE (heterogeneity summary)
    cate_summary = None
    try:
        blp = dml.summary()
        cate_summary = str(blp)
    except Exception:
        pass

    return {
        "ate": ate,
        "std_error": se,
        "t_statistic": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_folds": n_folds,
        "cate_summary": cate_summary,
    }


# ═══════════════════════════════════════════════════════════════════════
# CATE estimation via DML
# ═══════════════════════════════════════════════════════════════════════

def estimate_cate(X: np.ndarray, y: np.ndarray, T: np.ndarray,
                  model_y, model_t, n_folds: int = 5,
                  discrete_treatment: bool = True,
                  feature_names: list[str] = None) -> dict:
    """Estimate conditional average treatment effects using DML."""
    if HAS_ECONML:
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            cv=n_folds,
            random_state=42,
        )
        dml.fit(y, T, X=X)

        # CATE for each observation
        cate = dml.effect(X)
        cate_mean = float(np.mean(cate))
        cate_std = float(np.std(cate))
        cate_min = float(np.min(cate))
        cate_max = float(np.max(cate))
        cate_p25 = float(np.percentile(cate, 25))
        cate_p50 = float(np.percentile(cate, 50))
        cate_p75 = float(np.percentile(cate, 75))

        # Best linear predictor of CATE
        blp_result = {}
        try:
            blp = dml.summary()
            blp_result = {"summary": str(blp)}
        except Exception:
            pass

        # Heterogeneity by feature (group-level CATE)
        heterogeneity = {}
        if feature_names and len(feature_names) > 0:
            for i, name in enumerate(feature_names):
                if i >= X.shape[1]:
                    break
                x_col = X[:, i]
                if len(np.unique(x_col)) <= 20:  # categorical or discretizable
                    # Split by median for continuous, by category for discrete
                    if len(np.unique(x_col)) <= 10:
                        groups = {}
                        for val in np.unique(x_col):
                            mask = x_col == val
                            if mask.sum() >= 10:
                                groups[str(val)] = {
                                    "cate_mean": float(np.mean(cate[mask])),
                                    "n": int(mask.sum()),
                                }
                        if len(groups) > 1:
                            heterogeneity[name] = groups
                    else:
                        median = np.median(x_col)
                        low_mask = x_col <= median
                        high_mask = x_col > median
                        heterogeneity[name] = {
                            "low": {"cate_mean": float(np.mean(cate[low_mask])), "n": int(low_mask.sum())},
                            "high": {"cate_mean": float(np.mean(cate[high_mask])), "n": int(high_mask.sum())},
                        }

        return {
            "cate_mean": cate_mean,
            "cate_std": cate_std,
            "cate_min": cate_min,
            "cate_p25": cate_p25,
            "cate_median": cate_p50,
            "cate_p75": cate_p75,
            "cate_max": cate_max,
            "heterogeneity_by_feature": heterogeneity,
            "blp_summary": blp_result,
        }
    else:
        # Manual CATE via residualized outcome
        n = len(y)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cate = np.zeros(n)
        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]

            model_y_clone = _clone_model(model_y)
            model_y_clone.fit(X_train, y_train)
            y_pred = model_y_clone.predict(X_test)

            model_t_clone = _clone_model(model_t)
            if discrete_treatment:
                model_t_clone.fit(X_train, T_train.ravel())
                T_pred = model_t_clone.predict_proba(X_test)[:, 1]
            else:
                model_t_clone.fit(X_train, T_train.ravel())
                T_pred = model_t_clone.predict(X_test).ravel()

            y_tilde = y_test - y_pred
            T_tilde = T_test.ravel() - T_pred

            # Simple CATE: T_tilde * y_tilde / E[T_tilde^2] (approximate per-unit)
            denom = np.mean(T_tilde ** 2)
            cate[test_idx] = T_tilde * y_tilde / denom if abs(denom) > 1e-10 else 0

        cate_mean = float(np.mean(cate))
        cate_std = float(np.std(cate))

        return {
            "cate_mean": cate_mean,
            "cate_std": cate_std,
            "cate_min": float(np.min(cate)),
            "cate_median": float(np.median(cate)),
            "cate_max": float(np.max(cate)),
            "heterogeneity_by_feature": {},
        }


# ═══════════════════════════════════════════════════════════════════════
# Main DML function
# ═══════════════════════════════════════════════════════════════════════

def run_dml(df: pd.DataFrame, outcome: str, treatment: str,
            controls: list[str], ml_model: str = "gradient_boosting",
            n_folds: int = 5, estimate_heterogeneity: bool = True) -> dict:
    """
    Run Double/Debiased Machine Learning for ATE estimation.

    Parameters
    ----------
    df : DataFrame
    outcome : str — outcome variable name
    treatment : str — treatment variable name
    controls : list[str] — control variable names
    ml_model : str — "random_forest", "gradient_boosting", "lasso", or "linear"
    n_folds : int — number of cross-fitting folds (default 5)
    estimate_heterogeneity : bool — also estimate CATE distribution
    """
    if not HAS_SKLEARN:
        return {"error": "scikit-learn is required. Install with: pip install scikit-learn"}

    # Prepare data
    df = df.dropna(subset=[outcome, treatment] + controls)
    X = df[controls].values.astype(float)
    y = df[outcome].values.astype(float)
    T = df[treatment].values.astype(float).reshape(-1, 1)

    # Detect if treatment is binary
    unique_T = np.unique(T)
    discrete_treatment = len(unique_T) <= 2 or set(unique_T).issubset({0, 1, 0.0, 1.0})

    # Get ML models
    model_y, model_t = _get_ml_models(ml_model, discrete_treatment)

    # Run DML
    if HAS_ECONML:
        result = _econml_dml(X, y, T, model_y, model_t, n_folds, discrete_treatment)
        engine = "econml"
    else:
        result = _manual_dml(X, y, T, model_y, model_t, n_folds, discrete_treatment)
        engine = "manual (sklearn)"

    result["method"] = "Double/Debiased Machine Learning"
    result["ml_model"] = ml_model
    result["engine"] = engine
    result["discrete_treatment"] = discrete_treatment
    result["n_obs"] = len(y)
    result["n_controls"] = len(controls)
    result["outcome"] = outcome
    result["treatment"] = treatment

    # Significance label
    p = result.get("p_value", 1.0)
    if p < 0.01:
        result["significant"] = "*** (1%)"
    elif p < 0.05:
        result["significant"] = "** (5%)"
    elif p < 0.1:
        result["significant"] = "* (10%)"
    else:
        result["significant"] = "not significant"

    # CATE estimation
    if estimate_heterogeneity and controls:
        try:
            cate_result = estimate_cate(X, y, T, model_y, model_t, n_folds,
                                        discrete_treatment, controls)
            result["cate"] = cate_result
        except Exception as e:
            result["cate"] = {"error": str(e)}

    # Model diagnostics: CV R² for nuisance functions
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y_cv_pred = cross_val_predict(
            _clone_model(model_y), X_scaled, y, cv=min(n_folds, 5), n_jobs=-1
        )
        r2_y = 1 - np.sum((y - y_cv_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        if discrete_treatment:
            T_flat = T.ravel()
            T_cv_pred = cross_val_predict(
                _clone_model(model_t), X_scaled, T_flat, cv=min(n_folds, 5), n_jobs=-1,
                method='predict_proba'
            )[:, 1] if hasattr(model_t, 'predict_proba') else cross_val_predict(
                _clone_model(model_t), X_scaled, T_flat, cv=min(n_folds, 5), n_jobs=-1
            )
            T_baseline = np.mean(T_flat)
            r2_t = 1 - np.sum((T_flat - T_cv_pred) ** 2) / np.sum((T_flat - T_baseline) ** 2)
        else:
            T_flat = T.ravel()
            T_cv_pred = cross_val_predict(
                _clone_model(model_t), X_scaled, T_flat, cv=min(n_folds, 5), n_jobs=-1
            )
            T_baseline = np.mean(T_flat)
            r2_t = 1 - np.sum((T_flat - T_cv_pred) ** 2) / np.sum((T_flat - T_baseline) ** 2)

        result["nuisance_scores"] = {
            "outcome_cv_r2": float(max(0, r2_y)),
            "treatment_cv_r2": float(max(0, r2_t)),
            "interpretation": (
                f"Outcome model CV R²={r2_y:.3f}, Treatment model CV R²={r2_t:.3f}. "
                + ("Both models have reasonable fit." if r2_y > 0.1 and r2_t > 0.1
                   else "Low nuisance model fit — consider more flexible ML models or more controls.")
            ),
        }
    except Exception:
        result["nuisance_scores"] = {"note": "Could not compute CV R²."}

    # Interpretation
    ate = result.get("ate", 0)
    se = result.get("std_error", 0)
    ci_lower = result.get("ci_lower", ate - 1.96 * se)
    ci_upper = result.get("ci_upper", ate + 1.96 * se)

    result["interpretation"] = (
        f"DML estimates that {treatment} {'increases' if ate > 0 else 'decreases'} {outcome} "
        f"by {ate:.4f} on average (SE={se:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]). "
        + (f"The effect is statistically significant (p={p:.4f})."
           if p < 0.05 else f"The effect is not statistically significant (p={p:.4f}).")
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Double/Debiased Machine Learning (DML) for causal inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gradient boosting DML with 5-fold cross-fitting
  python run_dml.py --data panel.dta --outcome log_wage \\
      --treatment education_years --controls age experience occupation \\
      --ml-model gradient_boosting --cv 5

  # Lasso-based DML for high-dimensional controls
  python run_dml.py --data panel.dta --outcome log_wage \\
      --treatment treated --controls $(cat control_list.txt) \\
      --ml-model lasso --cv 10

  # Without econml (uses manual sklearn implementation)
  python run_dml.py --data panel.dta --outcome log_wage \\
      --treatment treated --controls age experience --ml-model random_forest
""")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--treatment", required=True, help="Treatment variable name")
    parser.add_argument("--controls", nargs="+", required=True,
                        help="Control variable names")
    parser.add_argument("--ml-model", default="gradient_boosting",
                        choices=["random_forest", "gradient_boosting", "lasso", "linear"],
                        help="ML model for nuisance functions")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-fitting folds")
    parser.add_argument("--no-heterogeneity", action="store_true",
                        help="Skip CATE estimation")
    parser.add_argument("--output", default=None, help="Output path (.json)")
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    if not HAS_SKLEARN:
        print("Error: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    if HAS_ECONML:
        print("Using econml for DML estimation.")
    else:
        print("econml not found — using manual sklearn implementation.")
        print("  For full functionality, install: pip install econml")

    result = run_dml(
        df, args.outcome, args.treatment, args.controls,
        ml_model=args.ml_model, n_folds=args.cv,
        estimate_heterogeneity=not args.no_heterogeneity,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\n═══════════════════════════════════")
    print("Double/Debiased Machine Learning")
    print("═══════════════════════════════════")
    print(f"Method:      {result['method']}")
    print(f"ML model:    {result['ml_model']}")
    print(f"Engine:      {result.get('engine', 'unknown')}")
    print(f"Treatment:   {result['treatment']} ({'binary' if result['discrete_treatment'] else 'continuous'})")
    print(f"Outcome:     {result['outcome']}")
    print(f"N obs:       {result['n_obs']}")
    print(f"N controls:  {result['n_controls']}")
    print(f"CV folds:    {result.get('n_folds', args.cv)}")
    print(f"\n─── ATE Estimate ───")
    print(f"ATE:         {result['ate']:.6f}")
    print(f"Std Error:   {result['std_error']:.6f}")
    print(f"t-statistic: {result['t_statistic']:.3f}")
    print(f"p-value:     {result['p_value']:.4f}  {result['significant']}")
    if "ci_lower" in result:
        print(f"95% CI:      [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    if result.get("nuisance_scores"):
        ns = result["nuisance_scores"]
        if "outcome_cv_r2" in ns:
            print(f"\n─── Nuisance Model Fit (CV R²) ───")
            print(f"Outcome model:    {ns['outcome_cv_r2']:.3f}")
            print(f"Treatment model:  {ns['treatment_cv_r2']:.3f}")
            print(f"  {ns.get('interpretation', '')}")

    if result.get("cate") and "error" not in result["cate"]:
        cate = result["cate"]
        print(f"\n─── CATE Distribution ───")
        print(f"Mean:    {cate['cate_mean']:.4f}")
        print(f"Median:  {cate.get('cate_median', 'N/A')}")
        print(f"Std:     {cate['cate_std']:.4f}")
        print(f"25th:    {cate.get('cate_p25', 'N/A')}")
        print(f"75th:    {cate.get('cate_p75', 'N/A')}")
        print(f"Min:     {cate.get('cate_min', 'N/A')}")
        print(f"Max:     {cate.get('cate_max', 'N/A')}")

        if cate.get("heterogeneity_by_feature"):
            print(f"\nHeterogeneity by feature:")
            for feat, groups in cate["heterogeneity_by_feature"].items():
                print(f"  {feat}:")
                for group, info in groups.items():
                    print(f"    {group}: CATE={info['cate_mean']:.4f} (n={info['n']})")

    print(f"\n{result['interpretation']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
