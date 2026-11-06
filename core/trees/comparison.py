"""LLMetrics Model Comparison Module: CART vs CHAID vs Logistic Regression.

Compares CART, CHAID, and Logistic Regression on the exact same hold-out test set
for the binary request failure prediction task.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from core.trees.chaid import CHAIDClassifier

def compare_failure_models(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
    decision_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Evaluates Logistic Regression, CART, and CHAID on a held-out test split.
    
    Args:
        df: Input DataFrame.
        test_size: Proportion of holdout test set (default: 0.30).
        random_state: Seed for reproducibility.
        decision_threshold: Classification decision cutoff.
        
    Returns:
        Structured dictionary comparing all three models with metrics and rankings.
    """
    sub_df = df.copy()
    sub_df["is_failure"] = (sub_df["status"] != "ok").astype(int)

    if "is_peak_hour" not in sub_df.columns:
        if "timestamp" in sub_df.columns:
            ts = pd.to_datetime(sub_df["timestamp"], utc=True)
            sub_df["is_peak_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        else:
            sub_df["is_peak_hour"] = 0

    feature_cols = ["latency_ms", "prompt_tokens", "completion_tokens", "model", "is_peak_hour", "cache_hit"]
    sub_df = sub_df.dropna(subset=feature_cols + ["is_failure"])

    # Prepare stratified train/test split
    train_df, test_df = train_test_split(
        sub_df,
        test_size=test_size,
        random_state=random_state,
        stratify=sub_df["is_failure"],
    )

    y_train = train_df["is_failure"].values
    y_test = test_df["is_failure"].values

    # One-hot encoded features for Logistic & CART
    model_dummies_train = pd.get_dummies(train_df["model"], prefix="model", drop_first=True, dtype=float)
    model_dummies_test = pd.get_dummies(test_df["model"], prefix="model", drop_first=True, dtype=float)
    # Align dummy columns
    model_dummies_test = model_dummies_test.reindex(columns=model_dummies_train.columns, fill_value=0.0)

    numeric_cols = ["latency_ms", "prompt_tokens", "completion_tokens", "is_peak_hour", "cache_hit"]
    X_train_encoded = pd.concat([train_df[numeric_cols].astype(float), model_dummies_train], axis=1)
    X_test_encoded = pd.concat([test_df[numeric_cols].astype(float), model_dummies_test], axis=1)

    # 1. Model: Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state)
    lr.fit(X_train_encoded, y_train)
    lr_probs = lr.predict_proba(X_test_encoded)[:, 1]
    lr_preds = (lr_probs >= decision_threshold).astype(int)

    # 2. Model: CART
    cart = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=random_state)
    cart.fit(X_train_encoded, y_train)
    cart_probs = cart.predict_proba(X_test_encoded)[:, 1]
    cart_preds = (cart_probs >= decision_threshold).astype(int)

    # 3. Model: CHAID
    chaid = CHAIDClassifier(max_depth=3, min_samples_split=30, min_samples_leaf=15)
    chaid.fit(train_df, target_col="is_failure", feature_cols=feature_cols)
    chaid_probs = chaid.predict_proba(test_df)
    chaid_preds = (chaid_probs >= decision_threshold).astype(int)

    # Evaluate all three models
    models_eval = {
        "Logistic Regression": (lr_preds, lr_probs),
        "Decision Tree (CART)": (cart_preds, cart_probs),
        "CHAID Tree": (chaid_preds, chaid_probs),
    }

    comparison_table: List[Dict[str, Any]] = []

    for name, (preds, probs) in models_eval.items():
        acc = float(accuracy_score(y_test, preds))
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        try:
            auc_v = float(roc_auc_score(y_test, probs))
        except Exception:
            auc_v = 0.5
        gini_v = 2.0 * auc_v - 1.0

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        comparison_table.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "specificity": round(float(spec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(auc_v, 4),
            "gini_coefficient": round(gini_v, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        })

    # Sort ranking by ROC-AUC
    comparison_table.sort(key=lambda x: x["roc_auc"], reverse=True)
    best_model = comparison_table[0]["model"]

    return {
        "task": "Binary LLM Request Failure Prediction (ok vs failure/timeout/rate_limited)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_failure_prevalence_pct": round(float(np.mean(y_test) * 100.0), 2),
        "decision_threshold": decision_threshold,
        "comparison_table": comparison_table,
        "best_model_by_auc": best_model,
        "recommendation": (
            f"{best_model} achieved the strongest discrimination on the holdout test set. "
            "CART provides explicit binary split rules, CHAID handles categorical and interaction splits directly, "
            "and Logistic Regression delivers calibrated odds ratios."
        ),
    }

