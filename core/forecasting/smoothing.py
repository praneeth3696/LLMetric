"""LLMetrics Exponential Smoothing and Moving Average Forecasting Module.

Implements:
- Trailing Moving Average
- Simple Exponential Smoothing (SES)
- Holt's Linear Trend Exponential Smoothing
- Holt-Winters Seasonal Exponential Smoothing (additive)
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

def moving_average_forecast(history: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    """Computes trailing moving average forecast."""
    if len(history) < window:
        window = max(1, len(history))
    ma_val = float(np.mean(history[-window:]))
    return np.full(horizon, fill_value=ma_val)

def simple_exponential_smoothing_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Fits Simple Exponential Smoothing (SES) and generates out-of-sample forecasts."""
    if len(history) < 3:
        return naive_forecast_fallback(history, horizon)
    try:
        model = SimpleExpSmoothing(history, initialization_method="estimated").fit(optimized=True)
        preds = model.forecast(horizon)
        return np.array(preds)
    except Exception:
        return moving_average_forecast(history, horizon, window=min(7, len(history)))

def holt_linear_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Fits Holt's Linear Trend model and generates forecasts."""
    if len(history) < 5:
        return simple_exponential_smoothing_forecast(history, horizon)
    try:
        model = Holt(history, initialization_method="estimated").fit(optimized=True)
        preds = model.forecast(horizon)
        # Prevent negative values for token or spend quantities
        return np.maximum(0.0, np.array(preds))
    except Exception:
        return simple_exponential_smoothing_forecast(history, horizon)

def holt_winters_forecast(history: np.ndarray, horizon: int, season_length: int = 7) -> np.ndarray:
    """Fits Holt-Winters Exponential Smoothing with additive trend and seasonality."""
    if len(history) < 2 * season_length:
        return holt_linear_forecast(history, horizon)
    try:
        model = ExponentialSmoothing(
            history,
            seasonal_periods=season_length,
            trend="add",
            seasonal="add",
            initialization_method="estimated",
        ).fit(optimized=True)
        preds = model.forecast(horizon)
        return np.maximum(0.0, np.array(preds))
    except Exception:
        return holt_linear_forecast(history, horizon)

def naive_forecast_fallback(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, fill_value=history[-1] if len(history) > 0 else 0.0)
