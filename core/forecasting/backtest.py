"""LLMetrics Rolling-Origin Backtesting & Multi-Model Forecasting Engine.

Executes rolling-origin time series cross-validation across:
1. Naive baseline
2. Seasonal Naive baseline (7-day period)
3. Moving Average (7-day trailing)
4. Simple Exponential Smoothing (SES)
5. Holt's Linear Trend
6. Holt-Winters Additive Seasonal Exponential Smoothing
7. Causal Regression Model (calendar + traffic regressors)
8. ARIMA (auto-order selected by AIC)

Evaluates out-of-sample MAE, RMSE, and MAPE across rolling horizons,
selects the top-performing model, and computes forecasts with prediction intervals.
"""

from typing import Dict, List, Any, Tuple, Optional
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
from core.timeseries.arima import fit_auto_arima

def prepare_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates raw request logs into a daily time series."""
    df_copy = df.copy()
    if "timestamp" not in df_copy.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")

    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], utc=True)
    df_copy["date"] = df_copy["timestamp"].dt.date

    daily = (
        df_copy.groupby("date")
        .agg(
            daily_spend=("cost_usd", "sum"),
            daily_tokens=("prompt_tokens", lambda p: int(p.sum() + df_copy.loc[p.index, "completion_tokens"].sum())),
            daily_requests=("request_id", "count"),
            daily_p50_latency=("latency_ms", "median"),
            daily_p95_latency=("latency_ms", lambda x: np.percentile(x, 95)),
            daily_error_rate=("status", lambda s: float((s != "ok").mean() * 100.0)),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily

def run_rolling_origin_backtest(
    daily_df: pd.DataFrame,
    target_col: str = "daily_spend",
    horizon: int = 7,
    min_train_size: int = 40,
    stride: int = 3,
) -> Dict[str, Any]:
    """Runs rolling-origin backtesting across all candidate forecasting methods.
    
    Args:
        daily_df: Aggregated daily DataFrame.
        target_col: 'daily_spend' or 'daily_tokens'.
        horizon: Forecast horizon to evaluate at each origin.
        min_train_size: Minimum number of historical observations before first forecast.
        stride: Step size between evaluation origins (default: 3 days).
        
    Returns:
        Structured evaluation report with error metrics per model, rankings,
        and final chosen model.
    """
    y_full = daily_df[target_col].values.astype(float)
    n = len(y_full)

    if n < (min_train_size + horizon):
        raise ValueError(
            f"Insufficient daily observations ({n}) for backtesting with min_train_size={min_train_size} and horizon={horizon}."
        )

    model_names = [
        "Naive",
        "Seasonal Naive (7d)",
        "Moving Average (7d)",
        "Simple Exp Smoothing",
        "Holt's Linear Trend",
        "Holt-Winters (Additive)",
        "Causal Model",
        "ARIMA (Auto)",
    ]

    errors: Dict[str, Dict[str, List[float]]] = {
        m: {"mae": [], "rmse": [], "mape": []} for m in model_names
    }

    # Generate roll origins
    origins = list(range(min_train_size, n - horizon + 1, stride))

    for orig in origins:
        train_y = y_full[:orig]
        test_y = y_full[orig : orig + horizon]
        sub_daily = daily_df.iloc[:orig].copy()

        # Forecast each model
        preds: Dict[str, np.ndarray] = {
            "Naive": naive_forecast(train_y, horizon),
            "Seasonal Naive (7d)": seasonal_naive_forecast(train_y, horizon, season_length=7),
            "Moving Average (7d)": moving_average_forecast(train_y, horizon, window=7),
            "Simple Exp Smoothing": simple_exponential_smoothing_forecast(train_y, horizon),
            "Holt's Linear Trend": holt_linear_forecast(train_y, horizon),
            "Holt-Winters (Additive)": holt_winters_forecast(train_y, horizon, season_length=7),
            "Causal Model": causal_regression_forecast(sub_daily, target_col, horizon),
        }

        # ARIMA forecast
        try:
            arima_out = fit_auto_arima(train_y, horizon=horizon)
            preds["ARIMA (Auto)"] = np.array([item["forecast"] for item in arima_out["forecast"]])
        except Exception:
            preds["ARIMA (Auto)"] = preds["Moving Average (7d)"]

        # Calculate error metrics
        for m_name in model_names:
            p_vec = preds[m_name]
            mae = float(np.mean(np.abs(test_y - p_vec)))
            rmse = float(np.sqrt(np.mean((test_y - p_vec) ** 2)))
            
            # Non-zero safe MAPE
            denom = np.maximum(1e-4, np.abs(test_y))
            mape = float(np.mean(np.abs((test_y - p_vec) / denom)) * 100.0)

            errors[m_name]["mae"].append(mae)
            errors[m_name]["rmse"].append(rmse)
            errors[m_name]["mape"].append(mape)

    # Average metrics across all rolling origins
    summary_table: List[Dict[str, Any]] = []
    for m_name in model_names:
        mean_mae = float(np.mean(errors[m_name]["mae"]))
        mean_rmse = float(np.mean(errors[m_name]["rmse"]))
        mean_mape = float(np.mean(errors[m_name]["mape"]))

        summary_table.append({
            "method": m_name,
            "mae": round(mean_mae, 3),
            "rmse": round(mean_rmse, 3),
            "mape": round(mean_mape, 2),
            "num_origins_evaluated": len(origins),
        })

    # Sort by lowest MAE
    summary_table.sort(key=lambda x: x["mae"])
    best_method = summary_table[0]["method"]

    return {
        "target": target_col,
        "eval_horizon_days": horizon,
        "num_origins_evaluated": len(origins),
        "comparison_table": summary_table,
        "best_method": best_method,
        "justification": (
            f"{best_method} achieved the lowest average out-of-sample error across rolling evaluation origins "
            f"(MAE: {summary_table[0]['mae']}, MAPE: {summary_table[0]['mape']}%)."
        ),
    }

def generate_forecast_with_intervals(
    daily_df: pd.DataFrame,
    target_col: str = "daily_spend",
    horizon: int = 14,
    method: str = "auto",
) -> Dict[str, Any]:
    """Generates out-of-sample forecast with intervals using chosen or best method.
    
    Args:
        daily_df: Daily aggregated data.
        target_col: 'daily_spend' or 'daily_tokens'.
        horizon: Number of days forward to project (7, 14, 30).
        method: 'auto', 'ma', 'ets', 'arima', 'causal', etc.
        
    Returns:
        Structured forecast dictionary with historical series, future series,
        confidence and prediction intervals, and evaluation backtest metrics.
    """
    y_full = daily_df[target_col].values.astype(float)
    backtest = run_rolling_origin_backtest(daily_df, target_col=target_col, horizon=min(7, horizon))
    best_name = backtest["best_method"]

    if method == "auto":
        chosen_method = best_name
    elif method == "ma":
        chosen_method = "Moving Average (7d)"
    elif method == "ets":
        chosen_method = "Holt-Winters (Additive)"
    elif method == "arima":
        chosen_method = "ARIMA (Auto)"
    elif method == "causal":
        chosen_method = "Causal Model"
    else:
        chosen_method = best_name

    # Compute chosen forecast
    if chosen_method == "ARIMA (Auto)":
        arima_res = fit_auto_arima(y_full, horizon=horizon)
        forecast_points = arima_res["forecast"]
        mean_preds = np.array([p["forecast"] for p in forecast_points])
        lower_80 = np.array([p["lower_80"] for p in forecast_points])
        upper_80 = np.array([p["upper_80"] for p in forecast_points])
        lower_95 = np.array([p["lower_95"] for p in forecast_points])
        upper_95 = np.array([p["upper_95"] for p in forecast_points])
    elif chosen_method == "Causal Model":
        mean_preds = causal_regression_forecast(daily_df, target_col, horizon)
        # Approximate prediction intervals using residual std
        resid_std = float(backtest["comparison_table"][0]["rmse"])
        lower_80 = np.maximum(0.0, mean_preds - 1.28 * resid_std)
        upper_80 = mean_preds + 1.28 * resid_std
        lower_95 = np.maximum(0.0, mean_preds - 1.96 * resid_std)
        upper_95 = mean_preds + 1.96 * resid_std
    elif chosen_method == "Holt-Winters (Additive)":
        mean_preds = holt_winters_forecast(y_full, horizon)
        resid_std = float(backtest["comparison_table"][0]["rmse"])
        lower_80 = np.maximum(0.0, mean_preds - 1.28 * resid_std)
        upper_80 = mean_preds + 1.28 * resid_std
        lower_95 = np.maximum(0.0, mean_preds - 1.96 * resid_std)
        upper_95 = mean_preds + 1.96 * resid_std
    elif chosen_method == "Holt's Linear Trend":
        mean_preds = holt_linear_forecast(y_full, horizon)
        resid_std = float(backtest["comparison_table"][0]["rmse"])
        lower_80 = np.maximum(0.0, mean_preds - 1.28 * resid_std)
        upper_80 = mean_preds + 1.28 * resid_std
        lower_95 = np.maximum(0.0, mean_preds - 1.96 * resid_std)
        upper_95 = mean_preds + 1.96 * resid_std
    elif chosen_method == "Simple Exp Smoothing":
        mean_preds = simple_exponential_smoothing_forecast(y_full, horizon)
        resid_std = float(backtest["comparison_table"][0]["rmse"])
        lower_80 = np.maximum(0.0, mean_preds - 1.28 * resid_std)
        upper_80 = mean_preds + 1.28 * resid_std
        lower_95 = np.maximum(0.0, mean_preds - 1.96 * resid_std)
        upper_95 = mean_preds + 1.96 * resid_std
    elif chosen_method == "Seasonal Naive (7d)":
        mean_preds = seasonal_naive_forecast(y_full, horizon, season_length=7)
        resid_std = float(backtest["comparison_table"][0]["rmse"])
        lower_80 = np.maximum(0.0, mean_preds - 1.28 * resid_std)
        upper_80 = mean_preds + 1.28 * resid_std
        lower_95 = np.maximum(0.0, mean_preds - 1.96 * resid_std)
        upper_95 = mean_preds + 1.96 * resid_std
    else:  # Moving Average
        mean_preds = moving_average_forecast(y_full, horizon)
        resid_std = float(backtest["comparison_table"][0]["rmse"])
        lower_80 = np.maximum(0.0, mean_preds - 1.28 * resid_std)
        upper_80 = mean_preds + 1.28 * resid_std
        lower_95 = np.maximum(0.0, mean_preds - 1.96 * resid_std)
        upper_95 = mean_preds + 1.96 * resid_std

    last_date = pd.to_datetime(daily_df["date"].max())
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]

    forecast_items = [
        {
            "date": future_dates[i].strftime("%Y-%m-%d"),
            "forecast": round(float(mean_preds[i]), 2),
            "lower_80": round(float(lower_80[i]), 2),
            "upper_80": round(float(upper_80[i]), 2),
            "lower_95": round(float(lower_95[i]), 2),
            "upper_95": round(float(upper_95[i]), 2),
        }
        for i in range(horizon)
    ]

    # Historical series for charting
    history_items = [
        {
            "date": pd.to_datetime(daily_df["date"].iloc[i]).strftime("%Y-%m-%d"),
            "actual": round(float(daily_df[target_col].iloc[i]), 2),
        }
        for i in range(len(daily_df))
    ]

    return {
        "target": target_col,
        "horizon_days": horizon,
        "chosen_method": chosen_method,
        "backtest_evaluation": backtest,
        "history": history_items,
        "forecast": forecast_items,
    }
