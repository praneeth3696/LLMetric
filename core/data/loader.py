"""LLMetrics Data Ingestion & File Loader Module.

Supports CSV and JSON format ingestion with strict schema verification and
graceful error reporting.
"""

from io import StringIO, BytesIO
from typing import Tuple, Dict, Any, Union
import json
import pandas as pd
from core.data.schema import validate_dataframe_schema, REQUIRED_COLUMNS
from core.data.cleaner import clean_request_logs

def load_data_from_filepath(
    path: str,
    clean: bool = True,
    **clean_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads dataset from file path (CSV or JSON).
    
    Args:
        path: Path to CSV or JSON file.
        clean: Whether to apply data cleaning pipeline.
        
    Returns:
        Tuple of (pd.DataFrame, cleaning/audit report).
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        # Attempt CSV first, then JSON
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_json(path)

    return _process_loaded_df(df, clean=clean, **clean_kwargs)

def load_data_from_buffer(
    content: Union[str, bytes],
    file_type: str = "csv",
    clean: bool = True,
    **clean_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads dataset from an in-memory buffer or uploaded file bytes/string.
    
    Args:
        content: String or bytes containing CSV or JSON data.
        file_type: 'csv' or 'json'.
        clean: Whether to apply data cleaning.
        
    Returns:
        Tuple of (pd.DataFrame, cleaning/audit report).
    """
    if isinstance(content, bytes):
        content_str = content.decode("utf-8")
    else:
        content_str = content

    if file_type.lower() == "csv":
        df = pd.read_csv(StringIO(content_str))
    elif file_type.lower() == "json":
        df = pd.read_json(StringIO(content_str))
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Supported types: 'csv', 'json'")

    return _process_loaded_df(df, clean=clean, **clean_kwargs)

def _process_loaded_df(
    df: pd.DataFrame,
    clean: bool = True,
    **clean_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validates and optionally cleans an ingested DataFrame."""
    # Check for basic column requirements
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Uploaded data schema error: Missing required columns: {missing_cols}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    if clean:
        return clean_request_logs(df, **clean_kwargs)
    else:
        validate_dataframe_schema(df)
        return df, {"rows": len(df), "columns": list(df.columns)}

