"""Unit tests for CHAID, CART, Text Features, and Model Comparison."""

import pytest
import numpy as np
import pandas as pd
from core.trees.chaid import CHAIDClassifier, fit_chaid_classifier
from core.trees.cart import fit_cart_classifier, fit_cart_regressor
from core.trees.text_features import extract_prompt_features, analyze_text_features
from core.trees.comparison import compare_failure_models

def test_chaid_classifier_induction_and_prediction():
    """Unit test for custom CHAID algorithm: builds tree and predicts on unseen rows."""
    rng = np.random.default_rng(505)
    n = 1000

    latency = rng.uniform(50, 2000, size=n)
    is_peak = rng.choice([0, 1], size=n)
    model = rng.choice(["A", "B"], size=n)

    # Strong dependency: high latency + peak hour -> high failure
    p_fail = np.where((latency > 1200) & (is_peak == 1), 0.85, 0.05)
    y = (rng.uniform(size=n) < p_fail).astype(int)

    df = pd.DataFrame({
        "latency_ms": latency,
        "is_peak_hour": is_peak,
        "model": model,
        "is_failure": y,
    })

    chaid = CHAIDClassifier(max_depth=3, min_samples_split=20, min_samples_leaf=10)
    chaid.fit(df, target_col="is_failure", feature_cols=["latency_ms", "is_peak_hour", "model"])

    assert chaid.root is not None
    assert chaid.root.is_leaf is False
    assert chaid.root.split_feature is not None  # Best feature picked by chi-square test

    rules = chaid.get_rules()
    assert len(rules) >= 2
    for r in rules:
        assert "IF " in r and "THEN " in r

    # Test predictions
    probs = chaid.predict_proba(df.head(20))
    preds = chaid.predict(df.head(20), threshold=0.5)

    assert len(probs) == 20
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert len(preds) == 20
    assert set(preds).issubset({0, 1})

def test_cart_rules_and_metrics():
    """Verify CART classification and regression produce readable rules and metrics."""
    rng = np.random.default_rng(606)
    n = 800

    df = pd.DataFrame({
        "latency_ms": rng.uniform(100, 1500, size=n),
        "prompt_tokens": rng.uniform(100, 1000, size=n),
        "completion_tokens": rng.uniform(50, 400, size=n),
        "is_peak_hour": rng.choice([0, 1], size=n),
        "cache_hit": rng.choice([0, 1], size=n),
        "model": rng.choice(["m1", "m2"], size=n),
        "status": rng.choice(["ok", "timeout", "error"], p=[0.8, 0.1, 0.1], size=n),
    })

    clf_res = fit_cart_classifier(df, max_depth=3)
    assert clf_res["metrics"]["accuracy"] > 0.60
    assert len(clf_res["human_readable_rules"]) > 0

    reg_res = fit_cart_regressor(df, target_col="latency_ms", max_depth=3)
    assert reg_res["metrics"]["r_squared"] is not None
    assert "feature_importances" in reg_res

def test_unstructured_text_features():
    """Verify text feature extraction identifies code blocks and keywords."""
    df = pd.DataFrame({
        "prompt_text": [
            "Please run this python script: ```def hello(): pass```",
            "SELECT * FROM users WHERE status = 'urgent';",
            "Can you explain why I got an exception error in production?",
            "Just a simple plain question.",
        ],
        "status": ["ok", "timeout", "error", "ok"],
    })

    extracted = extract_prompt_features(df)
    assert extracted.loc[0, "has_code_block"] == 1
    assert extracted.loc[1, "has_sql_keyword"] == 1
    assert extracted.loc[1, "has_urgent_keyword"] == 1
    assert extracted.loc[2, "has_error_keyword"] == 1
    assert extracted.loc[3, "has_code_block"] == 0

    res = analyze_text_features(df)
    assert len(res["text_feature_effects"]) > 0

def test_model_comparison_holdout():
    """Verify holdout comparison evaluates CART vs CHAID vs Logistic on identical test set."""
    rng = np.random.default_rng(707)
    n = 1000

    df = pd.DataFrame({
        "latency_ms": rng.uniform(100, 2500, size=n),
        "prompt_tokens": rng.uniform(100, 1200, size=n),
        "completion_tokens": rng.uniform(50, 500, size=n),
        "is_peak_hour": rng.choice([0, 1], size=n),
        "cache_hit": rng.choice([0, 1], size=n),
        "model": rng.choice(["gpt-4o-mini", "claude-3-haiku"], size=n),
        "status": rng.choice(["ok", "timeout", "rate_limited"], p=[0.75, 0.15, 0.10], size=n),
    })

    comp = compare_failure_models(df, test_size=0.30)
    assert len(comp["comparison_table"]) == 3
    methods = [row["model"] for row in comp["comparison_table"]]
    assert "Logistic Regression" in methods
    assert "Decision Tree (CART)" in methods
    assert "CHAID Tree" in methods

    for row in comp["comparison_table"]:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert -1.0 <= row["gini_coefficient"] <= 1.0
