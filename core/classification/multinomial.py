"""LLMetrics Multinomial Logistic Regression Module.

Predicts multi-class request status (ok, timeout, error, rate_limited)
relative to reference category 'ok'.
Computes:
- Relative Risk Ratios (RRR = exp(beta)) per class
- Class-level coefficients and p-values
- Multi-class confusion matrix
- Per-class precision, recall, and F1-score
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm

def fit_multinomial_logistic_regression(
    df: pd.DataFrame,
    class_weight: Optional[str] = None,
) -> Dict[str, Any]:
    """Fits multinomial logistic regression for request status classification.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Structured dictionary with class coefficients, Relative Risk Ratios (RRR),
        confusion matrix, and classification metrics.
    """
    sub_df = df.copy()

    # Ensure status is present
    if "is_peak_hour" not in sub_df.columns:
        if "timestamp" in sub_df.columns:
            ts = pd.to_datetime(sub_df["timestamp"], utc=True)
            sub_df["is_peak_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        else:
            sub_df["is_peak_hour"] = 0

    sub_df = sub_df.dropna(subset=["status", "latency_ms", "prompt_tokens", "completion_tokens", "is_peak_hour", "model"])
    n = len(sub_df)
    if n < 30:
        raise ValueError(f"Insufficient observations ({n}) for multinomial logistic regression.")

    # One-hot encode model
    model_dummies = pd.get_dummies(sub_df["model"], prefix="model", drop_first=True, dtype=float)
    numeric_features = ["latency_ms", "prompt_tokens", "completion_tokens", "is_peak_hour"]
    X_raw = pd.concat([sub_df[numeric_features].astype(float), model_dummies], axis=1)
    feature_names = list(X_raw.columns)

    X = sm.add_constant(X_raw)
    y_str = sub_df["status"].astype(str)
    unique_classes = sorted(y_str.unique())

    # Map classes ensuring 'ok' is index 0 as reference
    class_order = ["ok"] + [c for c in unique_classes if c != "ok"]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y = y_str.map(class_to_idx).values

    # Scale features for robust lbfgs convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Fit using scikit-learn for high numerical stability on large/imbalanced data
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight,
        random_state=42,
    )
    clf.fit(X_scaled, y)

    y_pred_idx = clf.predict(X_scaled)
    y_pred_classes = [class_order[i] for i in y_pred_idx]

    # Confusion matrix
    cm = confusion_matrix(y_str, y_pred_classes, labels=class_order)
    cm_dict = {}
    for i, true_cls in enumerate(class_order):
        cm_dict[true_cls] = {}
        for j, pred_cls in enumerate(class_order):
            cm_dict[true_cls][pred_cls] = int(cm[i, j])

    # Per-class metrics
    report = classification_report(y_str, y_pred_classes, labels=class_order, output_dict=True, zero_division=0)

    # Class-level coefficients & Relative Risk Ratios (RRR)
    # Convert scaled coefficients back to original feature scale: beta_orig = beta_scaled / sigma
    ref_idx = 0
    ref_coefs = clf.coef_[ref_idx] / scaler.scale_
    
    rrr_by_class: Dict[str, Dict[str, float]] = {}
    for c_idx, cls_name in enumerate(class_order):
        if cls_name == "ok":
            continue
        class_coefs = clf.coef_[c_idx] / scaler.scale_
        diff_coefs = class_coefs - ref_coefs
        rrr_dict = {}
        for f_idx, feat in enumerate(feature_names):
            rrr_val = float(np.exp(np.clip(diff_coefs[f_idx], -10, 10)))
            rrr_dict[feat] = round(rrr_val, 4)
        rrr_by_class[cls_name] = rrr_dict

    return {
        "model_name": "Multinomial Logistic Regression: Status ~ X",
        "reference_class": "ok",
        "classes": class_order,
        "sample_size": n,
        "class_frequencies": y_str.value_counts().to_dict(),
        "relative_risk_ratios_vs_ok": rrr_by_class,
        "confusion_matrix": cm_dict,
        "metrics_by_class": {
            cls_name: {
                "precision": round(float(report[cls_name]["precision"]), 4),
                "recall": round(float(report[cls_name]["recall"]), 4),
                "f1_score": round(float(report[cls_name]["f1-score"]), 4),
                "support": int(report[cls_name]["support"]),
            }
            for cls_name in class_order if cls_name in report
        },
        "macro_avg": {
            "precision": round(float(report["macro avg"]["precision"]), 4),
            "recall": round(float(report["macro avg"]["recall"]), 4),
            "f1_score": round(float(report["macro avg"]["f1-score"]), 4),
        },
        "accuracy": round(float(report["accuracy"]), 4),
    }
