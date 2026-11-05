"""LLMetrics ARIMA Time Series Modeling Module.

Automated order selection (p, d, q) by AIC minimization, model fitting,
residual diagnostics (Ljung-Box test for white noise), and interval forecasts.
"""

from typing import Dict, List, Any, Tuple, Optional
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

def fit_auto_arima(
    series: np.ndarray,
    horizon: int = 14,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
) -> Dict[str, Any]:
    """Selects best ARIMA(p,d,q) order by AIC, fits model, and evaluates residual diagnostics.
    
    Args:
        series: 1D array of historical values.
        horizon: Forecast horizon.
        max_p: Max AR order.
        max_d: Max differencing order.
        max_q: Max MA order.
        
    Returns:
        Structured dictionary with chosen order, AIC/BIC, coefficients,
        Ljung-Box test, and forecast with prediction intervals.
    """
    clean_series = np.asarray(series, dtype=float)
    clean_series = clean_series[~np.isnan(clean_series)]
    n = len(clean_series)

    best_aic = float("inf")
    best_order = (1, 1, 1)
    best_model = None

    search_history = []

    # Grid search across candidate orders
    for d in range(0, max_d + 1):
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(clean_series, order=(p, d, q)).fit()
                        aic_val = float(model.aic)
                        search_history.append({"order": [p, d, q], "aic": round(aic_val, 2)})
                        if aic_val < best_aic:
                            best_aic = aic_val
                            best_order = (p, d, q)
                            best_model = model
                except Exception:
                    continue

    if best_model is None:
        # Fallback to simple random walk with drift
        best_order = (0, 1, 0)
        best_model = ARIMA(clean_series, order=best_order).fit()
        best_aic = float(best_model.aic)

    # Residuals & Ljung-Box test for independence
    residuals = best_model.resid
    lb_res = acorr_ljungbox(residuals, lags=[min(10, n // 5)], return_df=True)
    lb_stat = float(lb_res["lb_stat"].iloc[0])
    lb_pvalue = float(lb_res["lb_pvalue"].iloc[0])
    is_white_noise = bool(lb_pvalue > 0.05)

    # Out-of-sample forecast with intervals
    forecast_res = best_model.get_forecast(steps=horizon)
    mean_forecast = np.maximum(0.0, forecast_res.predicted_mean)
    ci_80 = forecast_res.conf_int(alpha=0.20)
    ci_95 = forecast_res.conf_int(alpha=0.05)

    forecast_items = [
        {
            "step": h + 1,
            "forecast": round(float(mean_forecast[h]), 2),
            "lower_80": round(float(np.maximum(0.0, ci_80[h, 0])), 2),
            "upper_80": round(float(ci_80[h, 1]), 2),
            "lower_95": round(float(np.maximum(0.0, ci_95[h, 0])), 2),
            "upper_95": round(float(ci_95[h, 1]), 2),
        }
        for h in range(horizon)
    ]

    # Parameter estimates
    params_dict = {
        name: {
            "value": round(float(best_model.params[i]), 4),
            "std_error": round(float(best_model.bse[i]), 4) if len(best_model.bse) > i else 0.0,
            "p_value": float(best_model.pvalues[i]) if len(best_model.pvalues) > i else 0.0,
        }
        for i, name in enumerate(best_model.param_names)
    }

    return {
        "model_name": f"ARIMA{best_order}",
        "order": list(best_order),
        "aic": round(float(best_model.aic), 2),
        "bic": round(float(best_model.bic), 2),
        "parameters": params_dict,
        "residual_diagnostics": {
            "ljung_box": {
                "lags": min(10, n // 5),
                "statistic": round(lb_stat, 3),
                "p_value": float(lb_pvalue),
                "residuals_are_white_noise": is_white_noise,
                "interpretation": (
                    "Residuals pass Ljung-Box test (p > 0.05): no remaining autocorrelations, "
                    "indicating adequate model specification."
                    if is_white_noise
                    else "Residuals exhibit remaining autocorrelation (p <= 0.05). Additional seasonal/MA order may be needed."
                ),
            },
        },
        "forecast": forecast_items,
        "search_history": sorted(search_history, key=lambda x: x["aic"])[:5],
    }
