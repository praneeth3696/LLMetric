"""LLMetrics Data Cleaning & Preprocessing Module.

Handles missing values, duplicate detection, outlier detection (IQR & Z-score),
and produces an audit report of cleaning transformations.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from core.data.schema import validate_dataframe_schema, REQUIRED_COLUMNS

def clean_request_logs(
    df: pd.DataFrame,
    iqr_multiplier: float = 3.0,
    handle_missing: str = "impute",
    remove_outliers: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans raw LLM request logs and generates an audit report.
    
    Args:
        df: Raw pandas DataFrame.
        iqr_multiplier: Multiplier for IQR outlier thresholding (default 3.0 for extreme outliers).
        handle_missing: 'impute' or 'drop'.
        remove_outliers: If True, filters out rows with extreme outliers; if False, flags them.
        
    Returns:
        Tuple of (cleaned pd.DataFrame, cleaning audit report dictionary).
    """
    initial_rows = len(df)
    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "missing_values_detected": {},
        "missing_values_handled": handle_missing,
        "duplicates_removed": 0,
        "outliers_detected": {},
        "outliers_removed": remove_outliers,
        "final_rows": 0,
    }

    clean_df = df.copy()

    # 1. Missing Value Detection & Handling
    missing_counts = clean_df.isnull().sum().to_dict()
    report["missing_values_detected"] = {k: int(v) for k, v in missing_counts.items() if v > 0}

    if handle_missing == "drop":
        clean_df = clean_df.dropna(subset=REQUIRED_COLUMNS)
    else:  # 'impute'
        # Impute numeric columns with median, categorical with mode
        numeric_cols = ["prompt_tokens", "completion_tokens", "latency_ms", "cost_usd", "retry_count", "cache_hit"]
        for col in numeric_cols:
            if col in clean_df.columns and clean_df[col].isnull().any():
                median_val = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(median_val)
        
        categorical_cols = ["feature", "model", "status", "prompt_text"]
        for col in categorical_cols:
            if col in clean_df.columns and clean_df[col].isnull().any():
                mode_val = clean_df[col].mode()[0] if not clean_df[col].mode().empty else "unknown"
                clean_df[col] = clean_df[col].fillna(mode_val)

    # 2. Duplicate Detection & Removal
    if "request_id" in clean_df.columns:
        dups = clean_df.duplicated(subset=["request_id"]).sum()
        report["duplicates_removed"] = int(dups)
        if dups > 0:
            clean_df = clean_df.drop_duplicates(subset=["request_id"], keep="first")
    else:
        dups = clean_df.duplicated().sum()
        report["duplicates_removed"] = int(dups)
        clean_df = clean_df.drop_duplicates(keep="first")

    # Coerce timestamp
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
        clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)

    # 3. Outlier Diagnostics (IQR and Z-score)
    numeric_target_cols = ["latency_ms", "cost_usd", "prompt_tokens", "completion_tokens"]
    outlier_mask = pd.Series(False, index=clean_df.index)

    for col in numeric_target_cols:
        if col in clean_df.columns:
            series = clean_df[col].dropna()
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = max(0.0, q25 - iqr_multiplier * iqr)
            upper_bound = q75 + iqr_multiplier * iqr

            # Z-scores
            mean_val = float(series.mean())
            std_val = float(series.std()) if float(series.std()) > 0 else 1.0
            z_scores = ((series - mean_val) / std_val).abs()

            col_outliers = (clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)
            outlier_count = int(col_outliers.sum())
            outlier_mask = outlier_mask | col_outliers

            report["outliers_detected"][col] = {
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
                "count": outlier_count,
                "percentage": round(100.0 * outlier_count / max(1, len(clean_df)), 2),
                "z_score_gt_3_count": int((z_scores > 3.0).sum()),
            }

    if remove_outliers:
        clean_df = clean_df[~outlier_mask].reset_index(drop=True)

    # Validate final schema
    validate_dataframe_schema(clean_df)

    report["final_rows"] = len(clean_df)
    return clean_df, report
