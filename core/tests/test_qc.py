"""Unit tests for Statistical Quality Control (Shewhart, EWMA, CUSUM, Model-Swap K-S)."""

import pytest
import numpy as np
import pandas as pd
from core.qc.charts import compute_shewhart_chart, compute_ewma_chart, compute_cusum_chart
from core.qc.model_swap import detect_model_swap_ks
from core.qc.evaluation import evaluate_qc_detectors

def test_shewhart_fires_on_injected_spike():
    """Verify Shewhart chart flags a known 4-sigma excursion."""
    rng = np.random.default_rng(1212)
    n = 60
    series = rng.normal(100.0, 5.0, size=n)
    # Inject isolated 5-sigma spike at step 45
    series[45] = 135.0  # +7 sigma above mean

    timestamps = [f"2026-01-{i+1:02d}" for i in range(n)]
    res = compute_shewhart_chart(series, timestamps, baseline_window=40)

    assert res["alarm_count"] >= 1
    alarm_indices = [a["index"] for a in res["alarms"]]
    assert 45 in alarm_indices

def test_cusum_and_ewma_fire_on_persistent_drift():
    """Verify CUSUM and EWMA fire within a controlled number of steps after a persistent mean shift."""
    rng = np.random.default_rng(1313)
    n = 70
    # In-control period (mean 100, std 5)
    base = rng.normal(100.0, 5.0, size=40)
    # Shifted period starting at step 40 (mean shifted to 112, +2.4 sigma)
    shifted = rng.normal(112.0, 5.0, size=30)
    series = np.concatenate([base, shifted])
    timestamps = [f"Day_{i}" for i in range(n)]

    # CUSUM test
    cusum_res = compute_cusum_chart(series, timestamps, baseline_window=35, k_factor=0.5, h_factor=5.0)
    assert cusum_res["alarm_count"] > 0
    # First alarm must happen soon after step 40
    first_cusum_alarm = cusum_res["alarms"][0]["index"]
    assert 40 <= first_cusum_alarm <= 48

    # EWMA test
    ewma_res = compute_ewma_chart(series, timestamps, baseline_window=35, lam=0.2, L=3.0)
    assert ewma_res["alarm_count"] > 0
    first_ewma_alarm = ewma_res["alarms"][0]["index"]
    assert 40 <= first_ewma_alarm <= 50

def test_model_swap_ks_detector():
    """Verify Kolmogorov-Smirnov detector flags an unannounced distribution shift in output tokens."""
    rng = np.random.default_rng(1414)
    # 40 days of traffic
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    records = []

    for d_idx, dt in enumerate(dates):
        # Days 0-25: Baseline distribution ~ Normal(200, 30)
        # Days 26-39: Swapped model distribution ~ Normal(380, 50)
        is_swapped = d_idx >= 26
        comp_mean = 380 if is_swapped else 200

        for _ in range(50):
            records.append({
                "timestamp": dt.isoformat(),
                "model": "gpt-4o-mini",
                "completion_tokens": max(10, int(rng.normal(comp_mean, 35))),
            })

    df = pd.DataFrame(records)
    res = detect_model_swap_ks(
        df,
        target_col="completion_tokens",
        target_model="gpt-4o-mini",
        baseline_days=20,
        window_days=3,
        alpha=0.01,
        min_ks_stat=0.15,
    )

    assert res["alarm_count"] > 0
    assert any(a["date"] >= "2026-01-26" for a in res["alarms"])

