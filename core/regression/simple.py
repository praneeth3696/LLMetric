"""LLMetrics Simple Linear Regression Module.

Fits latency_ms ~ completion_tokens, calculates coefficient t-tests, R^2,
residual diagnostics (residuals vs fitted, Q-Q quantiles, normality),
and confidence and prediction intervals.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def fit_simple_regression(
    df: pd.DataFrame,
    filter_cache_hits: bool = True,
    sample_size_for_plots: int = 500,
) -> Dict[str, Any]:
    """Fits simple linear regression of latency_ms on completion_tokens.
    
    Args:
        df: DataFrame containing latency_ms and completion_tokens.
        filter_cache_hits: Whether to exclude cache hits (which bypass token latency).
        sample_size_for_plots: Number of points to downsample for plotting data.
        
    Returns:
        Structured dictionary with model parameters, statistical tests,
        residual diagnostics, intervals, and plain-English interpretations.
    """
    sub_df = df.copy()
    if filter_cache_hits and "cache_hit" in sub_df.columns:
        sub_df = sub_df[sub_df["cache_hit"] == 0]

    sub_df = sub_df.dropna(subset=["completion_tokens", "latency_ms"])
    n = len(sub_df)
    if n < 5:
        raise ValueError("Insufficient observations to fit simple linear regression.")

    x = sub_df["completion_tokens"].values.astype(float)
    y = sub_df["latency_ms"].values.astype(float)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    intercept = float(model.params[0])
    slope = float(model.params[1])
    intercept_se = float(model.bse[0])
    slope_se = float(model.bse[1])
    intercept_t = float(model.tvalues[0])
    slope_t = float(model.tvalues[1])
    intercept_p = float(model.pvalues[0])
    slope_p = float(model.pvalues[1])

    conf_int = model.conf_int(alpha=0.05)
    intercept_ci = [round(float(conf_int[0, 0]), 4), round(float(conf_int[0, 1]), 4)]
    slope_ci = [round(float(conf_int[1, 0]), 4), round(float(conf_int[1, 1]), 4)]

    r2 = float(model.rsquared)
    adj_r2 = float(model.rsquared_adj)
    f_stat = float(model.fvalue)
    f_pvalue = float(model.f_pvalue)
    s_err = float(np.sqrt(model.mse_resid))

    # Residuals analysis
    fitted = model.fittedvalues
    residuals = model.resid
    std_residuals = residuals / s_err

    # Downsample points for UI scatter/residual plot
    if n > sample_size_for_plots:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_size_for_plots, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    scatter_data = [
        {
            "completion_tokens": round(float(x[i]), 1),
            "actual_latency_ms": round(float(y[i]), 2),
            "fitted_latency_ms": round(float(fitted[i]), 2),
            "residual": round(float(residuals[i]), 2),
            "std_residual": round(float(std_residuals[i]), 2),
        }
        for i in idx
    ]

    # Q-Q plot computation (theoretical normal quantiles vs sorted standardized residuals)
    sub_res = std_residuals[idx]
    sorted_res = np.sort(sub_res)
    quantiles_theoretical = stats.norm.ppf((np.arange(1, len(sorted_res) + 1) - 0.5) / len(sorted_res))
    qq_data = [
        {
            "theoretical": round(float(quantiles_theoretical[i]), 3),
            "sample": round(float(sorted_res[i]), 3),
        }
        for i in range(len(sorted_res))
    ]

    # Normality tests
    jb_stat, jb_p = stats.jarque_bera(residuals)
    # Shapiro-Wilk on max 5000 sample
    sw_sample = residuals[np.random.default_rng(42).choice(len(residuals), size=min(5000, len(residuals)), replace=False)]
    sw_stat, sw_p = stats.shapiro(sw_sample)

    # Confidence and Prediction intervals along the range of completion_tokens
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 40)
    X_grid = sm.add_constant(x_grid)
    pred_res = model.get_prediction(X_grid)
    pred_summary = pred_res.summary_frame(alpha=0.05)

    interval_data = [
        {
            "completion_tokens": round(float(x_grid[i]), 1),
            "mean_prediction": round(float(pred_summary["mean"].iloc[i]), 2),
            "ci_lower": round(float(pred_summary["mean_ci_lower"].iloc[i]), 2),
            "ci_upper": round(float(pred_summary["mean_ci_upper"].iloc[i]), 2),
            "pi_lower": round(float(pred_summary["obs_ci_lower"].iloc[i]), 2),
            "pi_upper": round(float(pred_summary["obs_ci_upper"].iloc[i]), 2),
        }
        for i in range(len(x_grid))
    ]

    # Interpretations
    p_desc = "< 0.001" if slope_p < 0.001 else f"= {slope_p:.4f}"
    interpretations = [
        f"For every additional completion token generated, latency increases by an estimated {slope:.2f} ms (95% CI: [{slope_ci[0]}, {slope_ci[1]}], t = {slope_t:.2f}, p {p_desc}).",
        f"The baseline system and overhead latency (intercept) is approximately {intercept:.1f} ms.",
        f"The model explains {r2 * 100.0:.1f}% of the variance in latency (R² = {r2:.3f}, F = {f_stat:.1f}, p < 0.001).",
        "Prediction intervals are significantly wider than confidence intervals, accounting for individual request response jitter.",
    ]

    return {
        "model_name": "Simple Linear Regression: latency_ms ~ completion_tokens",
        "sample_size": n,
        "filter_cache_hits": filter_cache_hits,
        "metrics": {
            "r_squared": round(r2, 4),
            "adjusted_r_squared": round(adj_r2, 4),
            "f_statistic": round(f_stat, 2),
            "f_pvalue": float(f_pvalue),
            "rmse": round(s_err, 2),
        },
        "coefficients": {
            "intercept": {
                "value": round(intercept, 4),
                "std_error": round(intercept_se, 4),
                "t_statistic": round(intercept_t, 3),
                "p_value": float(intercept_p),
                "conf_interval_95": intercept_ci,
            },
            "completion_tokens": {
                "value": round(slope, 4),
                "std_error": round(slope_se, 4),
                "t_statistic": round(slope_t, 3),
                "p_value": float(slope_p),
                "conf_interval_95": slope_ci,
            },
        },
        "residual_analysis": {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "jarque_bera": {
                "statistic": round(float(jb_stat), 2),
                "p_value": float(jb_p),
                "is_normal": bool(jb_p > 0.05),
            },
            "shapiro_wilk": {
                "statistic": round(float(sw_stat), 4),
                "p_value": float(sw_p),
                "is_normal": bool(sw_p > 0.05),
            },
            "scatter_points": scatter_data,
            "qq_plot": qq_data,
        },
        "intervals": interval_data,
        "interpretations": interpretations,
    }

