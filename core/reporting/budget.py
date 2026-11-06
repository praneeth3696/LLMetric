"""LLMetrics Budget Tracking and Breach Probability Module.

Projects month-end spend using time-series forecasts and computes the analytical
probability of breaching a user-configured monthly budget.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

def calculate_budget_projection(
    daily_df: pd.DataFrame,
    monthly_budget: float = 1200.0,
    forecast_mean_daily: Optional[float] = None,
    forecast_daily_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculates month-end projected spend, intervals, and breach probability.
    
    Args:
        daily_df: Daily aggregated historical data with 'daily_spend'.
        monthly_budget: Monthly budget limit in USD.
        forecast_mean_daily: Expected daily spend for future days (if precomputed).
        forecast_daily_std: Standard deviation of daily forecast errors.
        
    Returns:
        Structured budget alert object.
    """
    total_days_in_month = 30
    n_days_hist = len(daily_df)
    
    # Days elapsed in current cycle (modulo 30 or min(30, n))
    days_elapsed = (n_days_hist % total_days_in_month) if (n_days_hist % total_days_in_month) > 0 else 15
    days_remaining = total_days_in_month - days_elapsed

    recent_slice = daily_df.tail(days_elapsed)
    current_spend = float(recent_slice["daily_spend"].sum())

    if forecast_mean_daily is None:
        forecast_mean_daily = float(daily_df["daily_spend"].tail(14).mean())
    if forecast_daily_std is None:
        forecast_daily_std = float(daily_df["daily_spend"].tail(14).std())
        if forecast_daily_std == 0:
            forecast_daily_std = 1.0

    expected_future_spend = days_remaining * forecast_mean_daily
    projected_month_end = current_spend + expected_future_spend

    # Cumulative variance of remaining days assuming independent daily errors: Var = days_remaining * sigma^2
    cum_std = np.sqrt(days_remaining) * forecast_daily_std

    # Prediction intervals (95%)
    pi_lower = max(current_spend, projected_month_end - 1.96 * cum_std)
    pi_upper = projected_month_end + 1.96 * cum_std

    # Breach probability: P(Spend > Budget)
    if cum_std > 0:
        z_score = (monthly_budget - projected_month_end) / cum_std
        breach_prob = float(1.0 - stats.norm.cdf(z_score))
    else:
        breach_prob = 1.0 if projected_month_end > monthly_budget else 0.0

    breach_prob_pct = round(breach_prob * 100.0, 1)

    status = (
        "CRITICAL" if breach_prob_pct >= 75.0
        else ("WARNING" if breach_prob_pct >= 30.0 else "HEALTHY")
    )

    return {
        "monthly_budget": round(monthly_budget, 2),
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
        "current_spend_usd": round(current_spend, 2),
        "projected_month_end_spend": round(projected_month_end, 2),
        "prediction_interval_95": [round(pi_lower, 2), round(pi_upper, 2)],
        "breach_probability_pct": breach_prob_pct,
        "budget_status": status,
        "interpretation": (
            f"Current spend is ${current_spend:,.2f} at day {days_elapsed}. "
            f"Projected month-end spend is ${projected_month_end:,.2f} (95% PI: [${pi_lower:,.2f}, ${pi_upper:,.2f}]). "
            f"Breach risk is {breach_prob_pct}% ({status})."
        ),
    }

