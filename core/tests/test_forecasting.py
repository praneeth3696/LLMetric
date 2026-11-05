"""Unit tests for Forecasting Models and Rolling-Origin Backtesting."""

import pytest
import numpy as np
import pandas as pd
from core.forecasting.baselines import naive_forecast, seasonal_naive_forecast
from core.forecasting.smoothing import (
    moving_average_forecast,
    simple_exponential_smoothing_forecast,
    holt_linear_forecast,
    holt_winters_forecast,
)
from core.forecasting.causal import causal_regression_forecast
from core.forecasting.backtest import (
    prepare_daily_series,
    run_rolling_origin_backtest,
    generate_forecast_with_intervals,
)

def test_baselines_and_smoothing():
    """Verify point forecast shapes and seasonal behaviors."""
    history = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0])
    horizon = 7

    naive_f = naive_forecast(history, horizon)
    assert len(naive_f) == horizon
    assert np.all(naive_f == 23.0)

    s_naive_f = seasonal_naive_forecast(history, horizon, season_length=7)
    assert len(s_naive_f) == horizon
    # Should repeat the last 7 items
    assert np.all(s_naive_f == history[-7:])

    ma_f = moving_average_forecast(history, horizon, window=7)
    assert len(ma_f) == horizon
    assert abs(ma_f[0] - np.mean(history[-7:])) < 1e-4

    ses_f = simple_exponential_smoothing_forecast(history, horizon)
    assert len(ses_f) == horizon

    holt_f = holt_linear_forecast(history, horizon)
    assert len(holt_f) == horizon

    hw_f = holt_winters_forecast(history, horizon, season_length=7)
    assert len(hw_f) == horizon

def test_rolling_origin_backtest():
    """Verify rolling-origin cross-validation computes MAE, RMSE, and MAPE across methods."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    rng = np.random.default_rng(808)
    
    # 7-day seasonal spend
    season = np.tile([100, 110, 105, 115, 120, 60, 55], 9)[:60]
    spend = season + rng.normal(0, 5, size=60)
    requests = (spend * 10).astype(int)

    daily_df = pd.DataFrame({
        "date": dates,
        "daily_spend": spend,
        "daily_requests": requests,
        "daily_p50_latency": rng.uniform(200, 400, size=60),
    })

    res = run_rolling_origin_backtest(daily_df, target_col="daily_spend", horizon=7, min_train_size=35, stride=5)

    assert len(res["comparison_table"]) >= 6
    assert res["best_method"] in [row["method"] for row in res["comparison_table"]]

    for row in res["comparison_table"]:
        assert row["mae"] > 0
        assert row["rmse"] > 0
        assert row["mape"] > 0

def test_forecast_with_intervals():
    """Verify full forecast output contains historical series, future projection, and intervals."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    spend = 100.0 + np.sin(np.arange(60)) * 20.0
    daily_df = pd.DataFrame({
        "date": dates,
        "daily_spend": spend,
        "daily_requests": np.full(60, 1000),
        "daily_p50_latency": np.full(60, 300.0),
    })

    fc = generate_forecast_with_intervals(daily_df, target_col="daily_spend", horizon=14, method="auto")

    assert fc["horizon_days"] == 14
    assert len(fc["forecast"]) == 14
    assert len(fc["history"]) == 60

    for item in fc["forecast"]:
        assert item["lower_95"] <= item["lower_80"] <= item["forecast"] <= item["upper_80"] <= item["upper_95"]
