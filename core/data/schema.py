"""LLMetrics Log Data Schema and Validation Models."""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "timestamp",
    "request_id",
    "feature",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "status",
    "retry_count",
    "cache_hit",
    "prompt_text",
    "cost_usd",
]

class RequestLogEntry(BaseModel):
    timestamp: datetime
    request_id: str
    feature: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    status: Literal["ok", "timeout", "error", "rate_limited"]
    retry_count: int = Field(default=0, ge=0)
    cache_hit: int = Field(default=0, ge=0, le=1)
    prompt_text: str = Field(default="")
    cost_usd: float = Field(ge=0.0)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"ok", "timeout", "error", "rate_limited"}
        if v not in allowed:
            raise ValueError(f"status '{v}' must be one of {allowed}")
        return v

def validate_dataframe_schema(df: pd.DataFrame) -> None:
    """Validates that a pandas DataFrame contains all required columns and valid types.
    
    Raises:
        ValueError: If columns are missing or values are out of bounds.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty.")

    # Check numeric bounds
    if (df["prompt_tokens"] < 0).any():
        raise ValueError("Found negative values in prompt_tokens")
    if (df["completion_tokens"] < 0).any():
        raise ValueError("Found negative values in completion_tokens")
    if (df["latency_ms"] < 0).any():
        raise ValueError("Found negative values in latency_ms")
    if (df["cost_usd"] < 0).any():
        raise ValueError("Found negative values in cost_usd")
    
    valid_statuses = {"ok", "timeout", "error", "rate_limited"}
    actual_statuses = set(df["status"].dropna().unique())
    invalid_statuses = actual_statuses - valid_statuses
    if invalid_statuses:
        raise ValueError(f"Invalid status values found: {invalid_statuses}. Allowed: {valid_statuses}")
