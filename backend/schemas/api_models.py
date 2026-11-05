"""LLMetrics API Request and Response Pydantic Schemas."""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field

# Data Generation & Upload Schemas
class GenerateDataRequest(BaseModel):
    days: int = Field(default=90, ge=1, le=365, description="Number of days of traffic to synthesize")
    daily_volume: int = Field(default=1200, ge=50, le=10000, description="Average requests per day")
    seed: int = Field(default=42, description="Random seed for reproducible generation")

class GenerateDataResponse(BaseModel):
    status: str
    message: str
    total_records: int
    days_simulated: int
    dataset_version: str

class UploadDataResponse(BaseModel):
    status: str
    message: str
    total_records: int
    cleaning_report: Dict[str, Any]
    dataset_version: str

# Health & Task Status
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    database_connected: bool
    dataset_loaded: bool
    total_requests: int
    current_dataset_version: str

class TaskStatusResponse(BaseModel):
    status: Literal["idle", "running", "completed", "failed"]
    progress_pct: int
    current_step: str
    error: Optional[str] = None

# Budget Schemas
class BudgetUpdateRequest(BaseModel):
    monthly_budget: float = Field(gt=0.0, description="Target monthly budget in USD")

class BudgetResponse(BaseModel):
    monthly_budget: float
    days_elapsed: int
    days_remaining: int
    current_spend_usd: float
    projected_month_end_spend: float
    prediction_interval_95: List[float]
    breach_probability_pct: float
    budget_status: str
    interpretation: str

# Overview Schemas
class OverviewResponse(BaseModel):
    total_requests: int
    total_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    error_rate_pct: float
    active_alerts_count: int
    period_comparison: Dict[str, Any]
    dataset_version: str

# Syllabus Analytics Response Schemas
class EdaResponse(BaseModel):
    syllabus_mapping: str = "Syllabus Item 1: Data Wrangling & EDA"
    dataset_summary: Dict[str, Any]
    univariate: Dict[str, Any]
    bivariate: Dict[str, Any]
    multivariate: Dict[str, Any]

class SimpleRegressionResponse(BaseModel):
    syllabus_mapping: str = "Syllabus Item 2: Simple Linear Regression (latency_ms ~ completion_tokens)"
    model_name: str
    sample_size: int
    metrics: Dict[str, Any]
    coefficients: Dict[str, Any]
    residual_analysis: Dict[str, Any]
    intervals: List[Dict[str, Any]]
    interpretations: List[str]

class MultipleRegressionResponse(BaseModel):
    syllabus_mapping: str = "Syllabus Item 3: Multiple Linear Regression with Full Econometric Diagnostics"
    model_name: str
    target_variable: str
    sample_size: int
    metrics: Dict[str, Any]
    coefficients: Dict[str, Any]
    interpretations: List[str]
    diagnostics: Dict[str, Any]
    log_transform_comparison: Dict[str, Any]
    model_building_procedure: Dict[str, Any]

class ReliabilityResponse(BaseModel):
    syllabus_mapping: str = "Syllabus Items 4 & 5: Logistic Regression & Decision Trees (CART, CHAID)"
    binary_logistic: Dict[str, Any]
    multinomial_logistic: Dict[str, Any]
    cart_classification: Dict[str, Any]
    chaid_classification: Dict[str, Any]
    unstructured_text_features: Dict[str, Any]
    model_comparison: Dict[str, Any]

class ForecastResponse(BaseModel):
    syllabus_mapping: str = "Syllabus Item 6: Time-Series Forecasting & Rolling-Origin Backtesting"
    target: str
    horizon_days: int
    chosen_method: str
    backtest_evaluation: Dict[str, Any]
    history: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]

class AlertsResponse(BaseModel):
    syllabus_mapping: str = "Syllabus Item 8: Statistical Quality Control Charts & Incident Detectors"
    control_charts: Dict[str, Any]
    model_swap_detector: Dict[str, Any]
    benchmark_evaluation: Dict[str, Any]

class InsightsResponse(BaseModel):
    summary_text: str
    is_llm_rewritten: bool
    engine: str
    bullets: List[str]
