"""LLMetrics Time Series Stationarity & Autocorrelation Module.

Performs:
- Augmented Dickey-Fuller (ADF) Unit Root Test
- Autocorrelation Function (ACF) with 95% confidence bounds
- Partial Autocorrelation Function (PACF) with 95% confidence bounds
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf

def check_stationarity_and_autocorrelation(
    series: np.ndarray,
    max_lags: int = 20,
    series_name: str = "daily_spend",
) -> Dict[str, Any]:
    """Runs ADF stationarity test and computes ACF/PACF series.
    
    Args:
        series: 1D array of time series values.
        max_lags: Number of lags to compute for ACF/PACF.
        series_name: Label for reports.
        
    Returns:
        Structured dictionary with test statistics, critical values,
        ACF/PACF series with 95% confidence intervals, and stationarity verdict.
    """
    clean_series = np.asarray(series, dtype=float)
    clean_series = clean_series[~np.isnan(clean_series)]
    n = len(clean_series)

    if n < 15:
        raise ValueError(f"Insufficient time steps ({n}) for stationarity analysis.")

    # 1. Augmented Dickey-Fuller (ADF) Test
    # Null hypothesis: Presence of unit root (non-stationary)
    # Alternative: Stationary
    adf_res = adfuller(clean_series, autolag="AIC")
    adf_stat = float(adf_res[0])
    adf_pvalue = float(adf_res[1])
    used_lag = int(adf_res[2])
    n_obs = int(adf_res[3])
    crit_vals = {k: float(v) for k, v in adf_res[4].items()}

    is_stationary = bool(adf_pvalue < 0.05)

    # 2. ACF & PACF with 95% Bartlett confidence intervals (+- 1.96 / sqrt(N))
    actual_lags = min(max_lags, n // 2 - 1)
    acf_vals, acf_confint = acf(clean_series, nlags=actual_lags, alpha=0.05)
    pacf_vals, pacf_confint = pacf(clean_series, nlags=actual_lags, alpha=0.05, method="yw")

    conf_bound = 1.96 / np.sqrt(n)

    acf_data = [
        {
            "lag": lag,
            "correlation": round(float(acf_vals[lag]), 4),
            "ci_lower": round(float(acf_confint[lag][0]), 4),
            "ci_upper": round(float(acf_confint[lag][1]), 4),
            "statistically_significant": bool(abs(acf_vals[lag]) > conf_bound),
        }
        for lag in range(len(acf_vals))
    ]

    pacf_data = [
        {
            "lag": lag,
            "correlation": round(float(pacf_vals[lag]), 4),
            "ci_lower": round(float(pacf_confint[lag][0]), 4),
            "ci_upper": round(float(pacf_confint[lag][1]), 4),
            "statistically_significant": bool(abs(pacf_vals[lag]) > conf_bound),
        }
        for lag in range(len(pacf_vals))
    ]

    return {
        "series_name": series_name,
        "observations": n,
        "adf_test": {
            "test_statistic": round(adf_stat, 4),
            "p_value": float(adf_pvalue),
            "lags_used": used_lag,
            "critical_values": crit_vals,
            "is_stationary": is_stationary,
            "verdict": (
                f"Stationary (ADF p = {adf_pvalue:.4f} < 0.05). Reject the unit root null hypothesis."
                if is_stationary
                else f"Non-stationary (ADF p = {adf_pvalue:.4f} >= 0.05). Differencing (d=1) required for ARIMA."
            ),
        },
        "acf": acf_data,
        "pacf": pacf_data,
        "confidence_threshold_95": round(float(conf_bound), 4),
    }
