"""LLMetrics Model-Swap / Distribution Shift Detector Module.

Uses a rolling two-sample Kolmogorov-Smirnov (K-S) test on completion token length
and latency distributions to detect unannounced upstream model swaps or routing mutations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

def detect_model_swap_ks(
    df: pd.DataFrame,
    target_col: str = "completion_tokens",
    target_model: str = "gpt-4o-mini",
    baseline_days: int = 25,
    window_days: int = 3,
    alpha: float = 0.01,
    min_ks_stat: float = 0.15,
) -> Dict[str, Any]:
    """Applies rolling two-sample Kolmogorov-Smirnov test against baseline distribution.
    
    Args:
        df: Cleaned request logs DataFrame with 'timestamp'.
        target_col: 'completion_tokens' or 'latency_ms'.
        target_model: Model name to inspect (or 'all').
        baseline_days: Number of initial days to establish baseline empirical CDF.
        window_days: Size of the sliding test window.
        alpha: Significance threshold for rejection of distribution equality.
        min_ks_stat: Minimum D statistic to prevent trivial sample-size alerts.
        
    Returns:
        Structured report with rolling K-S statistics, p-values, flagged shift windows,
        and detection points.
    """
    sub_df = df.copy()
    if target_model != "all" and "model" in sub_df.columns:
        sub_df = sub_df[sub_df["model"] == target_model]

    if "timestamp" not in sub_df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")

    sub_df["timestamp"] = pd.to_datetime(sub_df["timestamp"], utc=True)
    sub_df = sub_df.sort_values("timestamp").reset_index(drop=True)

    start_time = sub_df["timestamp"].min()
    baseline_cutoff = start_time + pd.Timedelta(days=baseline_days)

    baseline_data = sub_df[sub_df["timestamp"] < baseline_cutoff][target_col].values.astype(float)
    if len(baseline_data) < 30:
        raise ValueError("Insufficient baseline observations for model-swap detector.")

    # Daily sliding window test
    daily_groups = sub_df.groupby(sub_df["timestamp"].dt.date)
    unique_dates = sorted(list(daily_groups.groups.keys()))

    test_series = []
    alarms = []

    for d_idx, d_curr in enumerate(unique_dates):
        d_ts = pd.Timestamp(d_curr, tz="UTC")
        if d_ts < baseline_cutoff:
            continue

        # Collect observations in sliding window [d_curr - window_days + 1, d_curr]
        window_start = d_ts - pd.Timedelta(days=window_days - 1)
        window_mask = (sub_df["timestamp"] >= window_start) & (sub_df["timestamp"] <= d_ts + pd.Timedelta(days=1))
        window_data = sub_df[window_mask][target_col].values.astype(float)

        if len(window_data) < 20:
            continue

        ks_res = stats.ks_2samp(baseline_data, window_data)
        d_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        is_shift = bool(p_val < alpha and d_stat >= min_ks_stat)

        test_point = {
            "date": str(d_curr),
            "window_sample_size": len(window_data),
            "ks_statistic_d": round(d_stat, 4),
            "p_value": float(p_val),
            "distribution_shift_alarm": is_shift,
        }
        test_series.append(test_point)

        if is_shift:
            alarms.append({
                "date": str(d_curr),
                "ks_statistic_d": round(d_stat, 4),
                "p_value": float(p_val),
                "type": "Model Distribution Shift Detected",
            })

    return {
        "detector": "Rolling Two-Sample Kolmogorov-Smirnov Test",
        "target_model": target_model,
        "metric_analyzed": target_col,
        "baseline_sample_size": len(baseline_data),
        "significance_alpha": alpha,
        "min_d_threshold": min_ks_stat,
        "windows_evaluated": len(test_series),
        "alarm_count": len(alarms),
        "alarms": alarms,
        "time_series": test_series,
    }
