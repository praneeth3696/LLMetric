"""LLMetrics Consolidated Analysis Summary Module.

Executes all 8 syllabus statistical analyses in sequence and packages them into
a single comprehensive, JSON-serializable dictionary.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from core.eda.analysis import run_eda
from core.regression.simple import fit_simple_regression
from core.regression.multiple import fit_multiple_regression
from core.classification.logistic import fit_binary_logistic_regression
from core.classification.multinomial import fit_multinomial_logistic_regression
from core.trees.cart import fit_cart_classifier, fit_cart_regressor
from core.trees.chaid import fit_chaid_classifier
from core.trees.text_features import analyze_text_features
from core.trees.comparison import compare_failure_models
from core.forecasting.backtest import (
    prepare_daily_series,
    run_rolling_origin_backtest,
    generate_forecast_with_intervals,
)
from core.timeseries.stationarity import check_stationarity_and_autocorrelation
from core.timeseries.arima import fit_auto_arima
from core.timeseries.var import fit_var_model
from core.qc.evaluation import evaluate_qc_detectors
from core.reporting.insights import generate_weekly_insights
from core.reporting.budget import calculate_budget_projection

def run_full_analytics_suite(
    df: pd.DataFrame,
    incidents: Optional[List[Dict[str, Any]]] = None,
    monthly_budget: float = 1200.0,
    forecast_horizon: int = 14,
) -> Dict[str, Any]:
    """Runs all 8 syllabus modules on the provided logs DataFrame."""
    # 1. EDA
    eda_results = run_eda(df)

    # 2. Simple Linear Regression: latency_ms ~ completion_tokens
    simple_reg = fit_simple_regression(df)

    # 3. Multiple Linear Regression: latency_ms ~ prompt + completion + model + peak + cache
    mult_reg = fit_multiple_regression(df)

    # 4. Logistic Regression: Binary and Multinomial
    bin_logit = fit_binary_logistic_regression(df)
    multi_logit = fit_multinomial_logistic_regression(df)

    # 5. Decision Trees: CART, CHAID, Unstructured Text, Comparison
    cart_clf = fit_cart_classifier(df)
    cart_reg = fit_cart_regressor(df)
    chaid_clf = fit_chaid_classifier(df)
    text_feat_res = analyze_text_features(df)
    tree_comparison = compare_failure_models(df)

    # Prepare daily time series for forecasting & time series items
    daily_df = prepare_daily_series(df)

    # 6. Forecasting: Rolling-origin backtest & horizon projection
    forecast_eval = generate_forecast_with_intervals(daily_df, target_col="daily_spend", horizon=forecast_horizon)

    # 7. Time Series Analysis: Stationarity, ARIMA, VAR
    stationarity_res = check_stationarity_and_autocorrelation(daily_df["daily_spend"].values, series_name="daily_spend")
    arima_res = fit_auto_arima(daily_df["daily_spend"].values, horizon=forecast_horizon)
    var_res = fit_var_model(daily_df, horizon=forecast_horizon)

    # 8. Statistical Quality Control: Shewhart, EWMA, CUSUM, Model Swap, Benchmark
    qc_res = evaluate_qc_detectors(df, incidents=incidents)

    # Extensions
    budget_res = calculate_budget_projection(daily_df, monthly_budget=monthly_budget)
    overview_stats = {
        "total_cost_usd": eda_results["dataset_summary"]["total_cost_usd"],
        "total_requests": eda_results["dataset_summary"]["total_requests"],
        "p95_latency_ms": eda_results["univariate"]["numeric"]["latency_ms"]["p95"],
        "p50_latency_ms": eda_results["univariate"]["numeric"]["latency_ms"]["median"],
        "error_rate_pct": eda_results["dataset_summary"]["overall_error_rate_pct"],
    }
    insights_res = generate_weekly_insights(
        overview_stats,
        qc_stats={"total_alarms": sum(t["total_alarms"] for t in qc_res["benchmark_table"][:3])},
        forecast_stats=forecast_eval,
        budget_stats=budget_res,
    )

    return {
        "metadata": {
            "application": "LLMetrics",
            "version": "1.0.0",
            "course": "23XWA8 Predictive Analytics (M.Sc Software Systems)",
            "total_requests_analyzed": len(df),
            "daily_timesteps": len(daily_df),
        },
        "overview": overview_stats,
        "syllabus_1_eda": eda_results,
        "syllabus_2_simple_linear_regression": simple_reg,
        "syllabus_3_multiple_linear_regression": mult_reg,
        "syllabus_4_logistic_regression": {
            "binary": bin_logit,
            "multinomial": multi_logit,
        },
        "syllabus_5_decision_trees_and_unstructured_data": {
            "cart_classification": cart_clf,
            "cart_regression": cart_reg,
            "chaid_classification": chaid_clf,
            "unstructured_text_features": text_feat_res,
            "model_comparison": tree_comparison,
        },
        "syllabus_6_forecasting": {
            "forecast_evaluation": forecast_eval,
        },
        "syllabus_7_timeseries_analysis": {
            "stationarity_and_acf_pacf": stationarity_res,
            "arima_model": arima_res,
            "vector_autoregression_var": var_res,
        },
        "syllabus_8_quality_control": qc_res,
        "extensions": {
            "budget_alerting": budget_res,
            "weekly_insights": insights_res,
        },
    }
