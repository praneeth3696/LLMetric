"""LLMetrics Vector Autoregression (VAR) Module.

Fits multivariate VAR model on [daily_spend, daily_p50_latency, daily_requests].
Computes:
- Optimal lag order selection (AIC, BIC, FPE, HQIC)
- Granger Causality tests between system variables
- Multivariate out-of-sample forecasts
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def fit_var_model(
    daily_df: pd.DataFrame,
    horizon: int = 14,
    max_lags: int = 4,
) -> Dict[str, Any]:
    """Fits Vector Autoregressive (VAR) model across cost, latency, and request volume.
    
    Args:
        daily_df: DataFrame containing daily_spend, daily_p50_latency, daily_requests.
        horizon: Forecast horizon.
        max_lags: Maximum lag order to consider.
        
    Returns:
        Structured dictionary with lag order selection, Granger causality tests,
        and multivariate forecasts.
    """
    vars_to_model = ["daily_spend", "daily_p50_latency", "daily_requests"]
    for v in vars_to_model:
        if v not in daily_df.columns:
            raise ValueError(f"daily_df missing required VAR variable '{v}'")

    data = daily_df[vars_to_model].dropna().astype(float)
    n = len(data)
    if n < 20:
        raise ValueError(f"Insufficient daily observations ({n}) for VAR modeling.")

    var_model = VAR(data)
    lag_order_res = var_model.select_order(maxlags=min(max_lags, n // 5))
    selected_lag = int(lag_order_res.aic) if lag_order_res.aic > 0 else 1

    fitted_var = var_model.fit(selected_lag)

    # Granger Causality Tests
    # e.g., Does daily_requests Granger-cause daily_p50_latency?
    granger_results = []
    for caused in vars_to_model:
        for causing in vars_to_model:
            if caused != causing:
                try:
                    gc_test = fitted_var.test_causality(caused, [causing], kind="f")
                    granger_results.append({
                        "causing_variable": causing,
                        "caused_variable": caused,
                        "f_statistic": round(float(gc_test.test_statistic), 3),
                        "p_value": float(gc_test.pvalue),
                        "granger_causes": bool(gc_test.pvalue < 0.05),
                        "interpretation": (
                            f"{causing} Granger-causes {caused} (p < 0.05): past values contain predictive information."
                            if gc_test.pvalue < 0.05
                            else f"No significant Granger causality from {causing} to {caused} (p >= 0.05)."
                        ),
                    })
                except Exception:
                    continue

    # Out of sample forecasts
    forecast_values = fitted_var.forecast(data.values[-selected_lag:], steps=horizon)
    forecast_list = [
        {
            "step": h + 1,
            "daily_spend": round(float(np.maximum(0.0, forecast_values[h, 0])), 2),
            "daily_p50_latency": round(float(np.maximum(0.0, forecast_values[h, 1])), 2),
            "daily_requests": round(float(np.maximum(0.0, forecast_values[h, 2])), 1),
        }
        for h in range(horizon)
    ]

    return {
        "model_name": f"VAR({selected_lag}) Vector Autoregression",
        "variables": vars_to_model,
        "selected_lag": selected_lag,
        "lag_order_criteria": {
            "aic_lag": int(lag_order_res.aic),
            "bic_lag": int(lag_order_res.bic),
            "fpe_lag": int(lag_order_res.fpe),
            "hqic_lag": int(lag_order_res.hqic),
        },
        "granger_causality_tests": granger_results,
        "forecast": forecast_list,
    }
