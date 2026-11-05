"""LLMetrics Statistical Quality Control (SQC) Charts Module.

Implements:
1. Shewhart Control Chart (Individual and Moving Range, 3-sigma limits)
2. EWMA (Exponentially Weighted Moving Average) Chart
3. CUSUM (Cumulative Sum) Chart with slack k and decision threshold h
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

def compute_shewhart_chart(
    series: np.ndarray,
    timestamps: List[str],
    metric_name: str = "latency_p95",
    baseline_window: int = 40,
    sigma_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """Computes Shewhart Individual Control Chart with 3-sigma limits."""
    vals = np.asarray(series, dtype=float)
    n = len(vals)
    if n < 10:
        raise ValueError("Insufficient points for Shewhart chart.")

    # Estimate baseline mean and sigma from initial in-control period
    base_n = min(baseline_window, n // 2)
    base_slice = vals[:base_n]
    mu0 = float(np.mean(base_slice))

    # Moving range estimation for sigma: MR = |x_t - x_{t-1}|, sigma = mean(MR) / d2 (d2 = 1.128 for n=2)
    mr = np.abs(np.diff(base_slice))
    mean_mr = float(np.mean(mr)) if len(mr) > 0 else float(np.std(base_slice))
    sigma0 = mean_mr / 1.128 if mean_mr > 0 else float(np.std(base_slice))
    if sigma0 == 0:
        sigma0 = 1.0

    ucl = mu0 + sigma_multiplier * sigma0
    lcl = max(0.0, mu0 - sigma_multiplier * sigma0)

    points = []
    alarms = []

    for i in range(n):
        v = float(vals[i])
        is_alarm = bool(v > ucl or v < lcl)
        points.append({
            "timestamp": timestamps[i],
            "value": round(v, 2),
            "center_line": round(mu0, 2),
            "ucl": round(ucl, 2),
            "lcl": round(lcl, 2),
            "alarm": is_alarm,
        })
        if is_alarm:
            alarms.append({
                "index": i,
                "timestamp": timestamps[i],
                "value": round(v, 2),
                "threshold": round(ucl if v > ucl else lcl, 2),
                "type": "Upper Breach" if v > ucl else "Lower Breach",
            })

    return {
        "chart_type": "Shewhart Individual Chart",
        "metric": metric_name,
        "center_line": round(mu0, 2),
        "ucl": round(ucl, 2),
        "lcl": round(lcl, 2),
        "sigma": round(sigma0, 2),
        "total_points": n,
        "alarm_count": len(alarms),
        "points": points,
        "alarms": alarms,
    }

def compute_ewma_chart(
    series: np.ndarray,
    timestamps: List[str],
    metric_name: str = "latency_p95",
    baseline_window: int = 40,
    lam: float = 0.20,
    L: float = 3.0,
) -> Dict[str, Any]:
    """Computes Exponentially Weighted Moving Average (EWMA) chart."""
    vals = np.asarray(series, dtype=float)
    n = len(vals)
    if n < 10:
        raise ValueError("Insufficient points for EWMA chart.")

    base_n = min(baseline_window, n // 2)
    base_slice = vals[:base_n]
    mu0 = float(np.mean(base_slice))
    sigma0 = float(np.std(base_slice)) if float(np.std(base_slice)) > 0 else 1.0

    z = mu0
    points = []
    alarms = []

    for t in range(n):
        x_t = float(vals[t])
        z = lam * x_t + (1.0 - lam) * z

        # Time-varying EWMA standard error
        t_step = t + 1
        factor = np.sqrt((lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * t_step)))
        ucl_t = mu0 + L * sigma0 * factor
        lcl_t = max(0.0, mu0 - L * sigma0 * factor)

        is_alarm = bool(z > ucl_t or z < lcl_t)
        points.append({
            "timestamp": timestamps[t],
            "raw_value": round(x_t, 2),
            "ewma_value": round(z, 2),
            "center_line": round(mu0, 2),
            "ucl": round(float(ucl_t), 2),
            "lcl": round(float(lcl_t), 2),
            "alarm": is_alarm,
        })
        if is_alarm:
            alarms.append({
                "index": t,
                "timestamp": timestamps[t],
                "ewma_value": round(z, 2),
                "threshold": round(float(ucl_t if z > ucl_t else lcl_t), 2),
                "type": "EWMA Upper Alarm" if z > ucl_t else "EWMA Lower Alarm",
            })

    return {
        "chart_type": "EWMA Control Chart",
        "metric": metric_name,
        "lambda": lam,
        "L_multiplier": L,
        "center_line": round(mu0, 2),
        "total_points": n,
        "alarm_count": len(alarms),
        "points": points,
        "alarms": alarms,
    }

def compute_cusum_chart(
    series: np.ndarray,
    timestamps: List[str],
    metric_name: str = "latency_p95",
    baseline_window: int = 40,
    k_factor: float = 0.5,
    h_factor: float = 5.0,
) -> Dict[str, Any]:
    """Computes Tabular CUSUM chart for drift detection."""
    vals = np.asarray(series, dtype=float)
    n = len(vals)
    if n < 10:
        raise ValueError("Insufficient points for CUSUM chart.")

    base_n = min(baseline_window, n // 2)
    base_slice = vals[:base_n]
    mu0 = float(np.mean(base_slice))
    sigma0 = float(np.std(base_slice)) if float(np.std(base_slice)) > 0 else 1.0

    k = k_factor * sigma0
    h = h_factor * sigma0

    c_plus = 0.0
    c_minus = 0.0

    points = []
    alarms = []

    for t in range(n):
        x_t = float(vals[t])
        c_plus = max(0.0, c_plus + (x_t - mu0 - k))
        c_minus = max(0.0, c_minus + (mu0 - k - x_t))

        is_alarm = bool(c_plus > h or c_minus > h)
        points.append({
            "timestamp": timestamps[t],
            "raw_value": round(x_t, 2),
            "cusum_plus": round(c_plus, 2),
            "cusum_minus": round(c_minus, 2),
            "decision_interval_h": round(h, 2),
            "alarm": is_alarm,
        })
        if is_alarm:
            alarms.append({
                "index": t,
                "timestamp": timestamps[t],
                "raw_value": round(x_t, 2),
                "cusum_stat": round(c_plus if c_plus > h else c_minus, 2),
                "decision_interval_h": round(h, 2),
                "type": "CUSUM Positive Shift" if c_plus > h else "CUSUM Negative Shift",
            })

    return {
        "chart_type": "Tabular CUSUM Chart",
        "metric": metric_name,
        "slack_k": round(k, 2),
        "decision_interval_h": round(h, 2),
        "center_line": round(mu0, 2),
        "total_points": n,
        "alarm_count": len(alarms),
        "points": points,
        "alarms": alarms,
    }
