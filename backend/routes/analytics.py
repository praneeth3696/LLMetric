"""LLMetrics Core Analytics & Syllabus Modules Endpoints."""

from typing import Dict, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas.api_models import (
    OverviewResponse,
    EdaResponse,
    SimpleRegressionResponse,
    MultipleRegressionResponse,
    ReliabilityResponse,
    ForecastResponse,
    AlertsResponse,
    InsightsResponse,
)
from backend.services.analytics_service import (
    get_or_compute_full_analysis,
    compute_overview_stats,
    get_forecast_for_params,
)

router = APIRouter(prefix="", tags=["Syllabus Analytics Modules"])

@router.get(
    "/overview",
    response_model=OverviewResponse,
    summary="High-level operational overview & KPI metrics",
)
def get_overview() -> OverviewResponse:
    """Returns top-level KPIs (spend, p50/p95 latency, error rate, active alerts)

    and 14-day period-over-period comparison.
    """
    try:
        data = compute_overview_stats()
        return OverviewResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get(
    "/eda",
    response_model=EdaResponse,
    summary="Syllabus Item 1: Exploratory Data Analysis & Cleaning Diagnostics",
)
def get_eda() -> EdaResponse:
    """Univariate distributions (histograms, IQR, skewness), bivariate correlations,

    and multivariate breakdowns by model and peak hour.
    """
    try:
        suite = get_or_compute_full_analysis()
        eda_data = suite["syllabus_1_eda"]
        return EdaResponse(**eda_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get(
    "/regression/simple",
    response_model=SimpleRegressionResponse,
    summary="Syllabus Item 2: Simple Linear Regression (latency ~ tokens)",
)
def get_simple_regression() -> SimpleRegressionResponse:
    """Simple linear regression of latency on completion tokens.

    Reports R^2, t-tests, residual Q-Q data, confidence and prediction intervals.
    """
    try:
        suite = get_or_compute_full_analysis()
        reg_data = suite["syllabus_2_simple_linear_regression"]
        return SimpleRegressionResponse(**reg_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get(
    "/regression/multiple",
    response_model=MultipleRegressionResponse,
    summary="Syllabus Item 3: Multiple Linear Regression with Econometric Diagnostics",
)
def get_multiple_regression() -> MultipleRegressionResponse:
    """Multiple regression with Breusch-Pagan heteroscedasticity test, VIF multicollinearity,

    Cook's distance, Durbin-Watson autocorrelation, log-transform comparison, and stepwise AIC trace.
    """
    try:
        suite = get_or_compute_full_analysis()
        mreg_data = suite["syllabus_3_multiple_linear_regression"]
        return MultipleRegressionResponse(**mreg_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get(
    "/reliability",
    response_model=ReliabilityResponse,
    summary="Syllabus Items 4 & 5: Failure Prediction (Logistic, Multinomial, CART, CHAID)",
)
def get_reliability() -> ReliabilityResponse:
    """Binary logistic regression (AUC, Gini, Wald, Hosmer-Lemeshow), multinomial classification,

    decision trees (CART, CHAID), prompt text feature extraction, and holdout model comparison.
    """
    try:
        suite = get_or_compute_full_analysis()
        bin_log = suite["syllabus_4_logistic_regression"]["binary"]
        multi_log = suite["syllabus_4_logistic_regression"]["multinomial"]
        trees = suite["syllabus_5_decision_trees_and_unstructured_data"]

        return ReliabilityResponse(
            binary_logistic=bin_log,
            multinomial_logistic=multi_log,
            cart_classification=trees["cart_classification"],
            chaid_classification=trees["chaid_classification"],
            unstructured_text_features=trees["unstructured_text_features"],
            model_comparison=trees["model_comparison"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get(
    "/forecast",
    response_model=ForecastResponse,
    summary="Syllabus Items 6 & 7: Time Series Forecasting with Rolling-Origin Backtesting",
)
def get_forecast(
    target: Literal["spend", "tokens"] = Query(default="spend", description="Target metric to project"),
    horizon: int = Query(default=14, ge=1, le=90, description="Forecast horizon in days (e.g. 7, 14, 30)"),
    method: Literal["auto", "ma", "ets", "arima", "causal"] = Query(
        default="auto", description="Forecasting algorithm"
    ),
) -> ForecastResponse:
    """Rolling-origin backtesting (MAE, RMSE, MAPE) across baselines and advanced forecasting models,

    with 80% and 95% out-of-sample prediction intervals.
    """
    try:
        target_name = "daily_spend" if target == "spend" else "daily_tokens"
        fc_data = get_forecast_for_params(target=target_name, horizon=horizon, method=method)
        return ForecastResponse(**fc_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="Syllabus Item 8: Statistical Quality Control Charts & Incident Evaluation",
)
def get_alerts() -> AlertsResponse:
    """Shewhart, EWMA, and CUSUM control charts, rolling K-S model-swap detector,

    and empirical evaluation against injected ground-truth incidents.
    """
    try:
        suite = get_or_compute_full_analysis()
        qc_data = suite["syllabus_8_quality_control"]
        return AlertsResponse(
            control_charts={
                "shewhart": qc_data["control_charts"]["shewhart"],
                "ewma": qc_data["control_charts"]["ewma"],
                "cusum": qc_data["control_charts"]["cusum"],
            },
            model_swap_detector=qc_data["control_charts"]["model_swap"],
            benchmark_evaluation={
                "benchmark_table": qc_data["benchmark_table"],
                "ground_truth_incidents": qc_data["ground_truth_incidents"],
                "summary": qc_data["summary"],
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get(
    "/insights",
    response_model=InsightsResponse,
    summary="Extension: Plain-Language Weekly Insights Summary",
)
def get_insights() -> InsightsResponse:
    """Deterministic natural language summary of operational posture,

    with optional Anthropic API rewrite if key is configured in environment.
    """
    try:
        suite = get_or_compute_full_analysis()
        insights_data = suite["extensions"]["weekly_insights"]
        return InsightsResponse(**insights_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
