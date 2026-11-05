"""Unit tests for synthetic log generator and cleaner."""

import pytest
import pandas as pd
import numpy as np
from core.data.generator import generate_synthetic_logs
from core.data.cleaner import clean_request_logs
from core.data.schema import validate_dataframe_schema, REQUIRED_COLUMNS

def test_generator_determinism():
    """Verify that the generator produces identical DataFrames for a fixed seed."""
    df1, inc1 = generate_synthetic_logs(days=3, daily_volume=100, seed=123)
    df2, inc2 = generate_synthetic_logs(days=3, daily_volume=100, seed=123)

    assert len(df1) == len(df2)
    assert df1.equals(df2)
    assert inc1 == inc2

def test_generator_schema_and_fields():
    """Verify generated data complies strictly with the schema."""
    df, incidents = generate_synthetic_logs(days=5, daily_volume=150, seed=42)
    
    assert len(df) > 0
    validate_dataframe_schema(df)

    for col in REQUIRED_COLUMNS:
        assert col in df.columns

    assert set(df["status"].unique()).issubset({"ok", "timeout", "error", "rate_limited"})
    assert (df["latency_ms"] > 0).all()
    assert (df["cost_usd"] >= 0).all()
    assert len(incidents) >= 4

def test_cleaner_handles_missing_and_duplicates():
    """Verify data cleaner removes duplicates and handles missing values."""
    df_raw, _ = generate_synthetic_logs(days=2, daily_volume=50, seed=42)
    
    # Introduce duplicate row
    dup_row = df_raw.iloc[0:1].copy()
    df_dirty = pd.concat([df_raw, dup_row], ignore_index=True)

    # Introduce missing value in numeric column
    df_dirty.loc[10, "latency_ms"] = np.nan
    df_dirty.loc[12, "feature"] = np.nan

    cleaned_df, report = clean_request_logs(df_dirty, handle_missing="impute")

    assert report["duplicates_removed"] >= 1
    assert "latency_ms" in report["missing_values_detected"]
    assert "feature" in report["missing_values_detected"]
    assert cleaned_df["latency_ms"].isnull().sum() == 0
    assert cleaned_df["feature"].isnull().sum() == 0
    assert len(cleaned_df) == len(df_raw)

def test_cleaner_iqr_outlier_detection():
    """Verify cleaner detects artificial extreme outliers."""
    df_raw, _ = generate_synthetic_logs(days=2, daily_volume=50, seed=42)
    df_dirty = df_raw.copy()
    
    # Inject extreme outlier
    df_dirty.loc[0, "latency_ms"] = 99999.0

    _, report = clean_request_logs(df_dirty, iqr_multiplier=3.0)
    assert report["outliers_detected"]["latency_ms"]["count"] >= 1
