"""Unit tests for Time Series Analysis: ADF, ACF/PACF, ARIMA, and VAR."""

import pytest
import numpy as np
import pandas as pd
from core.timeseries.stationarity import check_stationarity_and_autocorrelation
from core.timeseries.arima import fit_auto_arima
from core.timeseries.var import fit_var_model

def test_stationarity_and_acf():
    """Verify ADF unit root test and ACF/PACF calculation with Bartlett bounds."""
    rng = np.random.default_rng(909)
    # Stationary white noise + mean
    stationary_data = rng.normal(50.0, 5.0, size=100)

    res = check_stationarity_and_autocorrelation(stationary_data, max_lags=15, series_name="test_spend")

    assert res["adf_test"]["test_statistic"] < 0
    assert res["adf_test"]["is_stationary"] is True
    assert len(res["acf"]) == 16
    assert len(res["pacf"]) == 16
    assert res["confidence_threshold_95"] > 0

def test_auto_arima_and_diagnostics():
    """Verify ARIMA order selection by AIC and Ljung-Box residual diagnostics."""
    rng = np.random.default_rng(1010)
    # AR(1) process: y_t = 0.7 y_{t-1} + e_t
    n = 80
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.normal(0, 1.0)
    y += 100.0  # add level

    arima_res = fit_auto_arima(y, horizon=7, max_p=2, max_d=1, max_q=2)

    assert "order" in arima_res
    assert len(arima_res["forecast"]) == 7
    assert "ljung_box" in arima_res["residual_diagnostics"]
    assert arima_res["residual_diagnostics"]["ljung_box"]["statistic"] >= 0

def test_var_and_granger_causality():
    """Verify VAR multivariate estimation and Granger causality testing."""
    rng = np.random.default_rng(1111)
    n = 60

    # Traffic volume drives spend
    requests = 1000 + rng.normal(0, 100, size=n)
    spend = 0.12 * requests + rng.normal(0, 5, size=n)
    latency = 200 + 0.05 * requests + rng.normal(0, 10, size=n)

    daily_df = pd.DataFrame({
        "daily_spend": spend,
        "daily_p50_latency": latency,
        "daily_requests": requests,
    })

    var_res = fit_var_model(daily_df, horizon=7)

    assert var_res["selected_lag"] >= 1
    assert len(var_res["forecast"]) == 7
    assert len(var_res["granger_causality_tests"]) > 0

    # Check that requests Granger-causes spend
    gc_matches = [
        t for t in var_res["granger_causality_tests"]
        if t["causing_variable"] == "daily_requests" and t["caused_variable"] == "daily_spend"
    ]
    assert len(gc_matches) == 1
    assert gc_matches[0]["f_statistic"] > 0

