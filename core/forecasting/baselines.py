"""LLMetrics Baseline Forecasting Models.

Implements Naive and Seasonal Naive (7-day lag) baselines for benchmark comparisons.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Naive persistence forecast: y_{t+h} = y_t.
    
    Args:
        history: 1D array of historical values.
        horizon: Forecast horizon (steps ahead).
        
    Returns:
        1D array of length horizon.
    """
    if len(history) == 0:
        raise ValueError("History array cannot be empty.")
    last_val = history[-1]
    return np.full(horizon, fill_value=last_val)

def seasonal_naive_forecast(history: np.ndarray, horizon: int, season_length: int = 7) -> np.ndarray:
    """Seasonal Naive forecast: y_{t+h} = y_{t+h - season_length}.
    
    Args:
        history: 1D array of historical values.
        horizon: Forecast horizon.
        season_length: Seasonal lag period (default: 7 for daily data).
        
    Returns:
        1D array of length horizon.
    """
    if len(history) < season_length:
        # Fall back to simple naive if history is shorter than a season
        return naive_forecast(history, horizon)
    
    forecast = []
    for h in range(horizon):
        # Index back into the most recent completed seasonal cycle
        idx = -season_length + (h % season_length)
        forecast.append(history[idx])
    return np.array(forecast)
