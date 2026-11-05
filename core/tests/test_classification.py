"""Unit tests for Binary and Multinomial Logistic Regression."""

import pytest
import numpy as np
import pandas as pd
from core.classification.logistic import fit_binary_logistic_regression
from core.classification.multinomial import fit_multinomial_logistic_regression

def test_binary_logistic_regression():
    """Verify binary logistic regression recovers odds ratios and computes HL, AUC, and Gini."""
    rng = np.random.default_rng(303)
    n = 2000

    latency = rng.normal(800, 300, size=n)
    prompt_tok = rng.normal(500, 200, size=n)
    comp_tok = rng.normal(250, 80, size=n)
    is_peak = rng.choice([0, 1], size=n)
    model = rng.choice(["gpt-4o-mini", "claude-3-haiku"], size=n)

    # Logistic DGP: logit(p) = -3.5 + 0.003 * latency + 0.8 * is_peak
    z = -3.5 + 0.003 * latency + 0.8 * is_peak
    p = 1.0 / (1.0 + np.exp(-z))
    is_failed = rng.uniform(size=n) < p

    status = ["timeout" if f else "ok" for f in is_failed]

    df = pd.DataFrame({
        "latency_ms": latency,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "is_peak_hour": is_peak,
        "model": model,
        "status": status,
    })

    res = fit_binary_logistic_regression(df)

    # 1. Check AUC and Gini
    auc_v = res["roc_analysis"]["auc"]
    gini_v = res["roc_analysis"]["gini_coefficient"]
    assert auc_v > 0.70
    assert gini_v == pytest.approx(2.0 * auc_v - 1.0, abs=1e-3)

    # 2. Check Wald tests & Odds Ratios
    odds_r = res["odds_ratios"]["latency_ms"]["odds_ratio"]
    assert odds_r > 1.0  # Higher latency increases failure odds

    # 3. Check Hosmer-Lemeshow Goodness-of-Fit Test
    hl = res["hosmer_lemeshow_test"]
    assert "statistic" in hl
    assert "p_value" in hl
    assert len(hl["decile_groups"]) > 0

    # 4. Check Confusion Matrix Metrics
    cm = res["classification_table"]["metrics"]
    assert 0.0 <= cm["accuracy"] <= 1.0
    assert 0.0 <= cm["sensitivity_recall"] <= 1.0
    assert 0.0 <= cm["specificity"] <= 1.0

def test_multinomial_logistic_regression():
    """Verify multinomial logistic regression computes relative risk ratios across classes."""
    rng = np.random.default_rng(404)
    n = 1500

    latency = rng.uniform(100, 3000, size=n)
    prompt_tok = rng.uniform(100, 2000, size=n)
    comp_tok = rng.uniform(50, 800, size=n)
    is_peak = rng.choice([0, 1], size=n)
    model = rng.choice(["model_a", "model_b"], size=n)

    # Status distribution
    statuses = []
    for lat in latency:
        if lat > 2400:
            statuses.append("timeout")
        elif lat > 1800 and rng.uniform() < 0.4:
            statuses.append("rate_limited")
        elif rng.uniform() < 0.08:
            statuses.append("error")
        else:
            statuses.append("ok")

    df = pd.DataFrame({
        "latency_ms": latency,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "is_peak_hour": is_peak,
        "model": model,
        "status": statuses,
    })

    res = fit_multinomial_logistic_regression(df)

    assert "ok" == res["reference_class"]
    assert "relative_risk_ratios_vs_ok" in res
    assert "timeout" in res["relative_risk_ratios_vs_ok"]
    assert "confusion_matrix" in res
    assert res["accuracy"] > 0.60
