"""LLMetrics Binary Logistic Regression Module.

Predicts P(request fails or times out) ~ latency_ms + prompt_tokens + completion_tokens + is_peak_hour + model.
Computes:
- Coefficients and Odds Ratios (e^beta) with 95% CIs
- Wald tests (z-statistic, Wald chi-square, p-values)
- Hosmer-Lemeshow Goodness-of-Fit test
- Classification table (confusion matrix, accuracy, sensitivity, specificity, F1)
- ROC curve coordinates, AUC, and Gini coefficient (2 * AUC - 1)
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support

def fit_binary_logistic_regression(
    df: pd.DataFrame,
    threshold: float = 0.30,
) -> Dict[str, Any]:
    """Fits binary logistic regression on failure probability.
    
    Args:
        df: Input DataFrame.
        threshold: Classification decision cutoff probability (default: 0.30 due to class imbalance).
        
    Returns:
        Structured dictionary with model coefficients, odds ratios, Wald tests,
        Hosmer-Lemeshow test, classification metrics, ROC/AUC, and Gini coefficient.
    """
    sub_df = df.copy()

    # Define binary target: 1 if failure/timeout/rate_limit, 0 if ok
    sub_df["is_failure"] = (sub_df["status"] != "ok").astype(int)

    # Derive is_peak_hour
    if "is_peak_hour" not in sub_df.columns:
        if "timestamp" in sub_df.columns:
            ts = pd.to_datetime(sub_df["timestamp"], utc=True)
            sub_df["is_peak_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        else:
            sub_df["is_peak_hour"] = 0

    required_vars = ["is_failure", "latency_ms", "prompt_tokens", "completion_tokens", "is_peak_hour", "model"]
    sub_df = sub_df.dropna(subset=required_vars)

    n = len(sub_df)
    if n < 30:
        raise ValueError(f"Insufficient observations ({n}) for logistic regression.")

    # One-hot encode model
    model_dummies = pd.get_dummies(sub_df["model"], prefix="model", drop_first=True, dtype=float)
    numeric_features = ["latency_ms", "prompt_tokens", "completion_tokens", "is_peak_hour"]
    X_raw = pd.concat([sub_df[numeric_features].astype(float), model_dummies], axis=1)
    feature_names = list(X_raw.columns)

    X = sm.add_constant(X_raw)
    y = sub_df["is_failure"].values.astype(int)

    # Fit Logistic model via Logit
    logit_model = sm.Logit(y, X).fit(disp=False, maxiter=100)

    # Coefficients, Odds Ratios, Wald Tests
    params = logit_model.params
    bse = logit_model.bse
    conf_int = logit_model.conf_int(alpha=0.05)

    coef_data: Dict[str, Any] = {}
    odds_ratios: Dict[str, Any] = {}

    for idx, col in enumerate(["const"] + feature_names):
        b = float(params.iloc[idx])
        se = float(bse.iloc[idx])
        z_stat = float(b / se) if se > 0 else 0.0
        wald_chi2 = float(z_stat ** 2)
        p_val = float(logit_model.pvalues.iloc[idx])
        ci_lower = float(conf_int.iloc[idx, 0])
        ci_upper = float(conf_int.iloc[idx, 1])

        or_val = float(np.exp(b))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))

        coef_data[col] = {
            "coefficient": round(b, 5),
            "std_error": round(se, 5),
            "wald_z": round(z_stat, 3),
            "wald_chi2": round(wald_chi2, 3),
            "p_value": float(p_val),
            "conf_int_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "statistically_significant": bool(p_val < 0.05),
        }

        # Plain English interpretation of odds ratio
        if col == "const":
            interp = f"Baseline odds of failure when all predictors are at zero: {or_val:.4f}."
        elif col == "latency_ms":
            or_100ms = float(np.exp(b * 100))
            interp = f"Each +100 ms in request latency multiplies the odds of failure by {or_100ms:.3f}x (p < 0.001)."
        elif col == "prompt_tokens":
            or_500tok = float(np.exp(b * 500))
            interp = f"Each additional 500 prompt tokens multiplies the odds of failure by {or_500tok:.3f}x."
        elif col == "is_peak_hour":
            pct_change = (or_val - 1.0) * 100.0
            interp = f"Operating during peak concurrency hours changes failure odds by {pct_change:+.1f}% (OR = {or_val:.2f})."
        elif col.startswith("model_"):
            target_m = col.replace("model_", "")
            interp = f"Model {target_m} has {or_val:.2f}x the odds of failure compared to the reference model."
        else:
            interp = f"Unit increase in {col} multiplies failure odds by {or_val:.3f}x."

        odds_ratios[col] = {
            "odds_ratio": round(or_val, 4),
            "ci_lower_95": round(or_ci_lower, 4),
            "ci_upper_95": round(or_ci_upper, 4),
            "interpretation": interp,
        }

    # Predicted probabilities
    y_pred_probs = logit_model.predict(X)
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Hosmer-Lemeshow Goodness of Fit Test
    hl_res = _hosmer_lemeshow_test(y, y_pred_probs, num_groups=10)

    # Classification Metrics & Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total_obs = len(y)
    accuracy = (tp + tn) / total_obs
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # ROC & AUC
    fpr, tpr, roc_thresholds = roc_curve(y, y_pred_probs)
    roc_auc = float(auc(fpr, tpr))
    # Gini coefficient: Gini = 2 * AUC - 1
    gini = float(2.0 * roc_auc - 1.0)

    # Downsample ROC curve points for clean UI transmission
    stride = max(1, len(fpr) // 50)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), stride)
    ]
    if roc_points[-1]["fpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    return {
        "model_name": "Binary Logistic Regression: P(Fail) ~ X",
        "sample_size": n,
        "failure_count": int(np.sum(y)),
        "failure_rate_pct": round(float(np.mean(y) * 100.0), 2),
        "threshold": threshold,
        "pseudo_r_squared_mcfadden": round(float(logit_model.prsquared), 4),
        "log_likelihood": round(float(logit_model.llf), 2),
        "llr_pvalue": float(logit_model.llr_pvalue),
        "coefficients": coef_data,
        "odds_ratios": odds_ratios,
        "hosmer_lemeshow_test": hl_res,
        "classification_table": {
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "metrics": {
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "sensitivity_recall": round(float(sensitivity), 4),
                "specificity": round(float(specificity), 4),
                "f1_score": round(float(f1), 4),
            },
        },
        "roc_analysis": {
            "auc": round(roc_auc, 4),
            "gini_coefficient": round(gini, 4),
            "interpretation": (
                f"AUC of {roc_auc:.3f} indicates excellent discrimination "
                f"(Gini = {gini:.3f})."
                if roc_auc >= 0.8
                else f"AUC of {roc_auc:.3f} indicates acceptable discrimination (Gini = {gini:.3f})."
            ),
            "curve_points": roc_points,
        },
    }

def _hosmer_lemeshow_test(y_true: np.ndarray, y_probs: np.ndarray, num_groups: int = 10) -> Dict[str, Any]:
    """Computes the Hosmer-Lemeshow goodness-of-fit test using deciles of risk."""
    df_hl = pd.DataFrame({"y": y_true, "prob": y_probs})
    
    # Sort by predicted probability
    df_hl = df_hl.sort_values("prob").reset_index(drop=True)
    
    # Partition into quantile bins (handle duplicates gracefully)
    try:
        df_hl["decile"] = pd.qcut(df_hl["prob"], q=num_groups, labels=False, duplicates="drop")
    except Exception:
        df_hl["decile"] = pd.cut(df_hl["prob"], bins=num_groups, labels=False)

    groups = df_hl.groupby("decile")
    hl_stat = 0.0
    group_table = []

    for name, grp in groups:
        n_k = len(grp)
        obs_1 = int(grp["y"].sum())
        obs_0 = n_k - obs_1
        exp_1 = float(grp["prob"].sum())
        exp_0 = float((1.0 - grp["prob"]).sum())

        term_1 = ((obs_1 - exp_1) ** 2) / exp_1 if exp_1 > 0 else 0.0
        term_0 = ((obs_0 - exp_0) ** 2) / exp_0 if exp_0 > 0 else 0.0
        hl_stat += (term_1 + term_0)

        group_table.append({
            "group": int(name) if isinstance(name, (int, np.integer)) else str(name),
            "count": n_k,
            "obs_failures": obs_1,
            "exp_failures": round(exp_1, 2),
            "obs_successes": obs_0,
            "exp_successes": round(exp_0, 2),
        })

    # Degrees of freedom = number of unique groups - 2
    dof = max(1, len(group_table) - 2)
    p_value = float(1.0 - stats.chi2.cdf(hl_stat, df=dof))

    return {
        "statistic": round(float(hl_stat), 3),
        "degrees_of_freedom": dof,
        "p_value": float(p_value),
        "well_calibrated": bool(p_value > 0.05),
        "interpretation": (
            "Good calibration (p > 0.05): observed failure frequencies do not significantly deviate "
            "from model-predicted probabilities across deciles of risk."
            if p_value > 0.05
            else "Potential miscalibration (p <= 0.05): predicted risk deciles diverge from observed failure rates."
        ),
        "decile_groups": group_table,
    }
