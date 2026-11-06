"""LLMetrics Data Management Endpoints (Generation & Ingestion)."""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
import pandas as pd

from backend.schemas.api_models import (
    GenerateDataRequest,
    GenerateDataResponse,
    UploadDataResponse,
    TaskStatusResponse,
)
from backend.db.database import (
    save_dataframe_to_db,
    save_incidents_to_db,
    get_setting,
)
from backend.services.analytics_service import (
    get_or_compute_full_analysis,
    get_current_task_status,
    update_task_status,
)
from core.data.generator import generate_synthetic_logs
from core.data.loader import load_data_from_buffer
from core.data.cleaner import clean_request_logs

router = APIRouter(prefix="", tags=["Data Ingestion & Management"])

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit

@router.post(
    "/data/generate",
    response_model=GenerateDataResponse,
    summary="Generate reproducible synthetic LLM API logs with injected incidents",
)
def generate_data(
    req: GenerateDataRequest,
    background_tasks: BackgroundTasks,
) -> GenerateDataResponse:
    """Generates synthetic LLM access logs with diurnal/weekly seasonality,

    heteroscedastic noise, logistic failure rates, and injected anomaly incidents.
    """
    try:
        update_task_status("running", 5, "Generating synthetic logs")
        df_raw, incidents = generate_synthetic_logs(
            days=req.days,
            daily_volume=req.daily_volume,
            seed=req.seed,
        )

        df_clean, _ = clean_request_logs(df_raw)
        total_saved = save_dataframe_to_db(df_clean)
        save_incidents_to_db(incidents)

        # Trigger analytical compute in background
        background_tasks.add_task(get_or_compute_full_analysis, force_recompute=True)

        ver = get_setting("dataset_version", default="none")
        return GenerateDataResponse(
            status="success",
            message=f"Generated {total_saved:,} synthetic requests spanning {req.days} days.",
            total_records=total_saved,
            days_simulated=req.days,
            dataset_version=ver,
        )
    except Exception as e:
        update_task_status("failed", 0, "Generation error", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post(
    "/data/upload",
    response_model=UploadDataResponse,
    summary="Upload user CSV or JSON request log dataset",
)
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadDataResponse:
    """Ingests, validates, and cleans uploaded CSV or JSON request logs.

    Enforces schema validity and file size restrictions (< 50MB).
    """
    filename = file.filename or "uploaded_data.csv"
    ext = filename.split(".")[-1].lower()
    if ext not in ["csv", "json"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Please upload a valid CSV or JSON file.",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 50 MB upload limit.",
        )

    try:
        update_task_status("running", 5, "Parsing uploaded data")
        df_clean, clean_report = load_data_from_buffer(content, file_type=ext, clean=True)
        total_saved = save_dataframe_to_db(df_clean)

        # Trigger analytical recomputation in background
        background_tasks.add_task(get_or_compute_full_analysis, force_recompute=True)

        ver = get_setting("dataset_version", default="none")
        return UploadDataResponse(
            status="success",
            message=f"Successfully ingested and cleaned {total_saved:,} records.",
            total_records=total_saved,
            cleaning_report=clean_report,
            dataset_version=ver,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as e:
        update_task_status("failed", 0, "Upload error", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data ingestion failed: {str(e)}",
        )

@router.get(
    "/task/status",
    response_model=TaskStatusResponse,
    summary="Check analytical processing progress status",
)
def check_task_status() -> TaskStatusResponse:
    """Returns background analytics processing status and progress percentage."""
    st = get_current_task_status()
    return TaskStatusResponse(**st)

