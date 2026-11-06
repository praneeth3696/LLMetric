"""LLMetrics Classification & Regression Trees (CART) Module.

Uses scikit-learn DecisionTreeClassifier and DecisionTreeRegressor with pruning,
generating human-readable decision rules and feature importances.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, r2_score, root_mean_squared_error

def fit_cart_classifier(
    df: pd.DataFrame,
    max_depth: int = 3,
    min_samples_leaf: int = 25,
) -> Dict[str, Any]:
    """Fits a pruned CART classification tree to predict request failure.
    
    Args:
        df: Input DataFrame.
        max_depth: Maximum tree depth for interpretability (default: 3).
        min_samples_leaf: Minimum samples required at leaf node (default: 25).
        
    Returns:
        Structured dictionary with tree rules, feature importances, and evaluation metrics.
    """
    sub_df = df.copy()
    sub_df["is_failure"] = (sub_df["status"] != "ok").astype(int)

    if "is_peak_hour" not in sub_df.columns:
        if "timestamp" in sub_df.columns:
            ts = pd.to_datetime(sub_df["timestamp"], utc=True)
            sub_df["is_peak_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        else:
            sub_df["is_peak_hour"] = 0

    model_dummies = pd.get_dummies(sub_df["model"], prefix="model", drop_first=True, dtype=float)
    numeric_features = ["latency_ms", "prompt_tokens", "completion_tokens", "is_peak_hour", "cache_hit"]
    X = pd.concat([sub_df[numeric_features].astype(float), model_dummies], axis=1)
    y = sub_df["is_failure"].values.astype(int)

    feature_names = list(X.columns)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    acc = float(accuracy_score(y, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    try:
        auc_val = float(roc_auc_score(y, y_prob))
    except Exception:
        auc_val = 0.5

    # Human-readable decision rules
    rules_text = export_text(clf, feature_names=feature_names)
    structured_rules = _extract_cart_rules(clf, feature_names, class_names=["Success", "Failure"])

    # Feature importances
    importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_names, clf.feature_importances_)
        if imp > 0.001
    }
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    return {
        "model_name": "CART Classification Tree (Failure Prediction)",
        "hyperparameters": {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "criterion": "gini",
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(auc_val, 4),
        },
        "feature_importances": importances,
        "human_readable_rules": structured_rules,
        "ascii_tree": rules_text,
    }

def fit_cart_regressor(
    df: pd.DataFrame,
    target_col: str = "latency_ms",
    max_depth: int = 3,
    min_samples_leaf: int = 25,
) -> Dict[str, Any]:
    """Fits a pruned CART regression tree to predict response latency.
    
    Args:
        df: Input DataFrame.
        target_col: Dependent variable (default: 'latency_ms').
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        
    Returns:
        Structured dictionary with tree rules, R^2, RMSE, and feature importances.
    """
    sub_df = df.copy()
    if "is_peak_hour" not in sub_df.columns:
        if "timestamp" in sub_df.columns:
            ts = pd.to_datetime(sub_df["timestamp"], utc=True)
            sub_df["is_peak_hour"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        else:
            sub_df["is_peak_hour"] = 0

    model_dummies = pd.get_dummies(sub_df["model"], prefix="model", drop_first=True, dtype=float)
    numeric_features = ["prompt_tokens", "completion_tokens", "is_peak_hour", "cache_hit"]
    X = pd.concat([sub_df[numeric_features].astype(float), model_dummies], axis=1)
    y = sub_df[target_col].values.astype(float)

    feature_names = list(X.columns)

    reg = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    reg.fit(X, y)

    y_pred = reg.predict(X)
    r2 = float(r2_score(y, y_pred))
    rmse = float(root_mean_squared_error(y, y_pred))

    # Feature importances
    importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_names, reg.feature_importances_)
        if imp > 0.001
    }
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    rules_text = export_text(reg, feature_names=feature_names)

    return {
        "model_name": f"CART Regression Tree ({target_col} Prediction)",
        "metrics": {
            "r_squared": round(r2, 4),
            "rmse": round(rmse, 2),
        },
        "feature_importances": importances,
        "ascii_tree": rules_text,
    }

def _extract_cart_rules(tree_model: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]) -> List[str]:
    """Generates clean human-readable text rules from scikit-learn decision tree."""
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node: int, current_rule: List[str]) -> None:
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left child: <= threshold
            left_rule = current_rule + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], left_rule)

            # Right child: > threshold
            right_rule = current_rule + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], right_rule)
        else:
            # Leaf node
            val = tree_.value[node][0]
            total = np.sum(val)
            pred_class_idx = int(np.argmax(val))
            pred_class = class_names[pred_class_idx]
            fail_prob = val[1] / total if total > 0 else 0.0
            condition_str = " AND ".join(current_rule) if current_rule else "All Requests"
            rules.append(
                f"IF {condition_str} THEN Class = {pred_class} (Fail Prob = {fail_prob:.1%}, n = {int(total)})"
            )

    recurse(0, [])
    return rules

