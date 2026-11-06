"""Unit tests for Simple and Multiple Linear Regression with known-answer verification."""

import pytest
import numpy as np
import pandas as pd
from core.regression.simple import fit_simple_regression
from core.regression.multiple import fit_multiple_regression

def test_simple_regression_known_coefficients():
    """Verify simple linear regression recovers known synthetic slope and intercept within tolerance."""
    rng = np.random.default_rng(101)
    n = 2000
    true_intercept = 150.0
    true_slope = 1.25

    x = rng.uniform(20, 800, size=n)
    noise = rng.normal(0, 15.0, size=n)
    y = true_intercept + true_slope * x + noise

    df_known = pd.DataFrame({
        "completion_tokens": x,
        "latency_ms": y,
        "cache_hit": 0,
    })

    res = fit_simple_regression(df_known, filter_cache_hits=False)

    recovered_intercept = res["coefficients"]["intercept"]["value"]
    recovered_slope = res["coefficients"]["completion_tokens"]["value"]

    # Verify recovery within 5% relative tolerance
    assert abs(recovered_intercept - true_intercept) / true_intercept < 0.05
    assert abs(recovered_slope - true_slope) / true_slope < 0.05

    # Check statistical tests
    assert res["coefficients"]["completion_tokens"]["p_value"] < 0.001
    assert res["metrics"]["r_squared"] > 0.90
    assert len(res["intervals"]) > 0

    # Verify 95% confidence interval wraps the true parameter
    ci = res["coefficients"]["completion_tokens"]["conf_interval_95"]
    assert ci[0] <= true_slope <= ci[1]

def test_multiple_regression_suite_and_diagnostics():
    """Verify multiple linear regression computes VIF, Breusch-Pagan, Cook's distance, and stepwise AIC."""
    rng = np.random.default_rng(202)
    n = 1500

    prompt_tok = rng.uniform(100, 1500, size=n)
    comp_tok = rng.uniform(50, 600, size=n)
    is_peak = rng.choice([0, 1], size=n)
    cache = rng.choice([0, 1], p=[0.8, 0.2], size=n)
    model = rng.choice(["model_a", "model_b", "model_c"], size=n)

    # Intentionally inject heteroscedastic noise that scales with comp_tok
    noise_sigma = 5.0 + 0.15 * comp_tok
    noise = rng.normal(0, noise_sigma)

    latency = 120.0 + 1.15 * comp_tok + 0.04 * prompt_tok + 80.0 * is_peak - 100.0 * cache + noise

    df = pd.DataFrame({
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "is_peak_hour": is_peak,
        "cache_hit": cache,
        "model": model,
        "latency_ms": latency,
    })

    res = fit_multiple_regression(df, target_col="latency_ms")

    # 1. R^2 and Adjusted R^2
    assert res["metrics"]["adjusted_r_squared"] > 0.80

    # 2. Breusch-Pagan test should detect the injected heteroscedasticity (p < 0.05)
    bp_p = res["diagnostics"]["heteroscedasticity_breusch_pagan"]["p_value"]
    assert bp_p < 0.05
    assert res["diagnostics"]["heteroscedasticity_breusch_pagan"]["has_heteroscedasticity"] is True

    # 3. Multicollinearity VIF checks
    vifs = res["diagnostics"]["multicollinearity_vif"]
    assert len(vifs) >= 4
    for v in vifs:
        assert v["vif"] >= 1.0

    # 4. Cook's distance
    cooks = res["diagnostics"]["outliers_and_influence"]
    assert "cooks_distance_threshold" in cooks
    assert len(cooks["top_influential_cases"]) > 0

    # 5. Durbin-Watson check
    dw = res["diagnostics"]["autocorrelation_durbin_watson"]["statistic"]
    assert 1.5 < dw < 2.5  # independent i.i.d errors should be near 2.0

    # 6. Stepwise AIC history
    stepwise = res["model_building_procedure"]["step_history"]
    assert len(stepwise) >= 2

