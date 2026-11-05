"""LLMetrics Causal Forecasting Module.

Implements causal regression forecasting using calendar and operational exogenous variables:
- is_weekend
- day_of_week (dummy-coded)
- expected request volume
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def causal_regression_forecast(
    daily_df: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> np.ndarray:
    """Generates causal forecast using calendar and traffic variables.
    
    Args:
        daily_df: DataFrame of daily aggregated stats with 'date' and target_col.
        target_col: Target series ('daily_spend' or 'daily_tokens').
        horizon: Steps ahead to forecast.
        
    Returns:
        1D numpy array of forecasted values.
    """
    df = daily_df.copy().reset_index(drop=True)
    if "date" not in df.columns:
        raise ValueError("daily_df must contain 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["time_idx"] = np.arange(len(df)).astype(float)

    # Historical moving average volume as feature
    if "daily_requests" in df.columns:
        df["volume_feat"] = df["daily_requests"].rolling(7, min_periods=1).mean()
    else:
        df["volume_feat"] = 1000.0

    # Build features: time trend, is_weekend, volume_feat, day-of-week dummies
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=True, dtype=float)
    X = pd.concat([df[["time_idx", "is_weekend", "volume_feat"]], dow_dummies], axis=1)
    y = df[target_col].values

    # Fit regularized Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Construct future features
    last_date = df["date"].max()
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]
    future_time_idx = np.arange(len(df), len(df) + horizon).astype(float)
    future_dow = [d.weekday() for d in future_dates]
    future_is_weekend = [1.0 if dow >= 5 else 0.0 for dow in future_dow]
    
    # Future volume estimated from recent seasonal pattern
    recent_vol = df["volume_feat"].iloc[-1]
    future_volume = [recent_vol * (0.6 if is_w else 1.0) for is_w in future_is_weekend]

    future_df = pd.DataFrame({
        "time_idx": future_time_idx,
        "is_weekend": future_is_weekend,
        "volume_feat": future_volume,
        "day_of_week": future_dow,
    })
    future_dow_dummies = pd.get_dummies(future_df["day_of_week"], prefix="dow", drop_first=True, dtype=float)
    future_dow_dummies = future_dow_dummies.reindex(columns=dow_dummies.columns, fill_value=0.0)

    X_future = pd.concat([future_df[["time_idx", "is_weekend", "volume_feat"]], future_dow_dummies], axis=1)
    preds = model.predict(X_future)
    return np.maximum(0.0, preds)
