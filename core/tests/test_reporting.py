"""Unit tests for EDA, Weekly Insights, and Budget Tracking."""

import pytest
import numpy as np
import pandas as pd
from core.eda.analysis import run_eda
from core.reporting.insights import generate_weekly_insights
from core.reporting.budget import calculate_budget_projection

def test_eda_analysis():
    """Verify EDA generates univariate statistics, histograms, correlations, and cross-tabs."""
    rng = np.random.default_rng(1515)
    n = 300
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "latency_ms": rng.normal(300, 50, size=n),
        "cost_usd": rng.uniform(0.001, 0.05, size=n),
        "prompt_tokens": rng.normal(400, 100, size=n),
        "completion_tokens": rng.normal(200, 50, size=n),
        "retry_count": rng.choice([0, 1], size=n),
        "model": rng.choice(["gpt-4o-mini", "claude-3-haiku"], size=n),
        "feature": rng.choice(["chat", "summarize"], size=n),
        "status": rng.choice(["ok", "timeout"], p=[0.9, 0.1], size=n),
        "cache_hit": rng.choice([0, 1], size=n),
    })

    eda = run_eda(df)
    assert "univariate" in eda
    assert "bivariate" in eda
    assert "multivariate" in eda
    assert eda["univariate"]["numeric"]["latency_ms"]["median"] > 0
    assert len(eda["univariate"]["numeric"]["latency_ms"]["histogram"]) > 0
    assert "pearson" in eda["bivariate"]["correlations"]

def test_budget_projection_and_breach():
    """Verify budget alert computes month-end spend, prediction intervals, and breach risk."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    daily_spend = np.full(20, 50.0)  # $50/day * 30 days = $1500 projected

    daily_df = pd.DataFrame({"date": dates, "daily_spend": daily_spend})

    # Case 1: Budget = 1200, Projected = 1500 -> High risk of breach
    res_tight = calculate_budget_projection(daily_df, monthly_budget=1200.0)
    assert res_tight["projected_month_end_spend"] > 1400.0
    assert res_tight["breach_probability_pct"] > 90.0
    assert res_tight["budget_status"] == "CRITICAL"

    # Case 2: Budget = 2500, Projected = 1500 -> Low risk of breach
    res_safe = calculate_budget_projection(daily_df, monthly_budget=2500.0)
    assert res_safe["breach_probability_pct"] < 10.0
    assert res_safe["budget_status"] == "HEALTHY"

def test_deterministic_insights():
    """Verify weekly insights generates formatted bullet points without external API."""
    stats = {
        "total_cost_usd": 1420.50,
        "total_requests": 115000,
        "p95_latency_ms": 1240.2,
        "error_rate_pct": 3.8,
    }
    insights = generate_weekly_insights(stats)

    assert "summary_text" in insights
    assert len(insights["bullets"]) >= 3
    assert "LLMetrics Executive Weekly Summary" in insights["summary_text"]
    assert insights["is_llm_rewritten"] is False

