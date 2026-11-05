"""LLMetrics Analytics Service Layer.

Orchestrates analytical execution, caching, background execution state tracking,
and data queries for all FastAPI routes.
"""

from typing import Dict, List, Any, Optional
import threading
import numpy as np
import pandas as pd

from backend.db.database import (
    load_dataframe_from_db,
    load_incidents_from_db,
    get_cached_analysis,
    set_cached_analysis,
    get_setting,
    set_setting,
    DEFAULT_DB_PATH,
)
from core.reporting.summary import run_full_analytics_suite
from core.reporting.budget import calculate_budget_projection
from core.reporting.insights import generate_weekly_insights
from core.forecasting.backtest import generate_forecast_with_intervals, prepare_daily_series

# In-memory background task progress tracker
_TASK_LOCK = threading.Lock()
_TASK_STATE: Dict[str, Any] = {
    "status": "idle",
    "progress_pct": 100,
    "current_step": "Ready",
    "error": None,
}

def get_current_task_status() -> Dict[str, Any]:
    with _TASK_LOCK:
        return dict(_TASK_STATE)

def update_task_status(status: str, progress_pct: int, current_step: str, error: Optional[str] = None) -> None:
    with _TASK_LOCK:
        _TASK_STATE["status"] = status
        _TASK_STATE["progress_pct"] = progress_pct
        _TASK_STATE["current_step"] = current_step
        _TASK_STATE["error"] = error

def get_or_compute_full_analysis(db_path: Optional[str] = None, force_recompute: bool = False) -> Dict[str, Any]:
    """Retrieves full analysis suite from cache or computes and caches it."""
    ver = get_setting("dataset_version", default="none", db_path=db_path)
    if ver == "none":
        raise ValueError("No dataset currently ingested. Please call /data/generate or /data/upload first.")

    cached = get_cached_analysis("full_suite", db_path=db_path)
    if cached and not force_recompute:
        return cached

    # Cold compute
    update_task_status("running", 10, "Loading data from database")
    df = load_dataframe_from_db(db_path=db_path)
    if len(df) == 0:
        raise ValueError("Database table 'requests' is empty.")
    if len(df) < 30:
        raise ValueError(f"Insufficient observations ({len(df)}). Minimum 30 requests required for analysis.")

    incidents = load_incidents_from_db(db_path=db_path)
    monthly_budget = float(get_setting("monthly_budget", default="1200.0", db_path=db_path))

    update_task_status("running", 30, "Computing regression, reliability & time series models")
    results = run_full_analytics_suite(df, incidents=incidents, monthly_budget=monthly_budget)

    update_task_status("running", 85, "Caching computed analytics in SQLite")
    set_cached_analysis("full_suite", results, db_path=db_path)

    update_task_status("completed", 100, "Analytics ready")
    return results

def compute_overview_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Computes high-level overview KPIs and period-over-period comparison."""
    df = load_dataframe_from_db(db_path=db_path)
    if len(df) == 0:
        raise ValueError("No dataset available.")

    full_suite = get_or_compute_full_analysis(db_path=db_path)
    ver = get_setting("dataset_version", default="none", db_path=db_path)

    # Calculate period comparison (last 14 days vs prior 14 days)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    max_date = df["timestamp"].max()
    p2_start = max_date - pd.Timedelta(days=14)
    p1_start = max_date - pd.Timedelta(days=28)

    p1_df = df[(df["timestamp"] >= p1_start) & (df["timestamp"] < p2_start)]
    p2_df = df[df["timestamp"] >= p2_start]

    def _calc_stats(sub: pd.DataFrame) -> Dict[str, float]:
        if len(sub) == 0:
            return {"spend": 0.0, "latency_p95": 0.0, "error_rate": 0.0, "requests": 0}
        return {
            "spend": round(float(sub["cost_usd"].sum()), 2),
            "latency_p95": round(float(np.percentile(sub["latency_ms"], 95)), 1),
            "error_rate": round(float((sub["status"] != "ok").mean() * 100.0), 2),
            "requests": int(len(sub)),
        }

    p1_stats = _calc_stats(p1_df)
    p2_stats = _calc_stats(p2_df)

    active_alerts = len(full_suite["syllabus_8_quality_control"]["control_charts"]["shewhart"]["alarms"])

    return {
        "total_requests": len(df),
        "total_cost_usd": full_suite["overview"]["total_cost_usd"],
        "p50_latency_ms": full_suite["overview"]["p50_latency_ms"],
        "p95_latency_ms": full_suite["overview"]["p95_latency_ms"],
        "error_rate_pct": full_suite["overview"]["error_rate_pct"],
        "active_alerts_count": active_alerts,
        "period_comparison": {
            "prior_period_14d": p1_stats,
            "recent_period_14d": p2_stats,
            "spend_change_pct": round(
                ((p2_stats["spend"] - p1_stats["spend"]) / p1_stats["spend"] * 100.0)
                if p1_stats["spend"] > 0 else 0.0, 1
            ),
            "latency_p95_change_ms": round(p2_stats["latency_p95"] - p1_stats["latency_p95"], 1),
        },
        "dataset_version": ver,
    }

def get_forecast_for_params(
    target: str = "daily_spend",
    horizon: int = 14,
    method: str = "auto",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates on-demand forecast for specific target, horizon, and method."""
    df = load_dataframe_from_db(db_path=db_path)
    if len(df) == 0:
        raise ValueError("No dataset available.")

    daily_df = prepare_daily_series(df)
    target_col = "daily_spend" if target in ["spend", "daily_spend"] else "daily_tokens"

    fc_res = generate_forecast_with_intervals(
        daily_df,
        target_col=target_col,
        horizon=horizon,
        method=method,
    )
    return fc_res

def update_and_project_budget(
    new_budget: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Updates budget setting and computes month-end projection."""
    if new_budget is not None:
        set_setting("monthly_budget", str(new_budget), db_path=db_path)

    monthly_budget = float(get_setting("monthly_budget", default="1200.0", db_path=db_path))
    df = load_dataframe_from_db(db_path=db_path)
    if len(df) == 0:
        return {
            "monthly_budget": monthly_budget,
            "days_elapsed": 0,
            "days_remaining": 30,
            "current_spend_usd": 0.0,
            "projected_month_end_spend": 0.0,
            "prediction_interval_95": [0.0, 0.0],
            "breach_probability_pct": 0.0,
            "budget_status": "HEALTHY",
            "interpretation": "No data ingested yet.",
        }

    daily_df = prepare_daily_series(df)
    return calculate_budget_projection(daily_df, monthly_budget=monthly_budget)
