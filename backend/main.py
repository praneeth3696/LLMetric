"""LLMetrics FastAPI Application Entrypoint.

Provides a unified REST API for LLM cost, latency, reliability monitoring,
and predictive analytics across all 8 syllabus topics.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.db.database import init_db, get_db_connection, get_setting
from backend.routes.data import router as data_router
from backend.routes.analytics import router as analytics_router
from backend.routes.budget import router as budget_router
from backend.schemas.api_models import HealthResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llmetrics.api")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle handler."""
    logger.info("Initializing LLMetrics SQLite database schema...")
    init_db()
    logger.info("LLMetrics backend ready.")
    yield
    logger.info("LLMetrics backend shutting down.")

app = FastAPI(
    title="LLMetrics API",
    description=(
        "Cost, Latency & Reliability Monitor for LLM Applications.\n\n"
        "Course Project for 23XWA8 Predictive Analytics (M.Sc Software Systems).\n"
        "Visibly implements and documents all 8 course syllabus modules: "
        "EDA, Simple Linear Regression, Multiple Linear Regression, Logistic Regression, "
        "Decision Trees (CART & CHAID), Forecasting, Time Series Analysis (ARIMA & VAR), "
        "and Statistical Quality Control Charts."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration restricted to frontend origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Structured request timing & logging middleware
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f} ms)"
    )
    return response

# Standardized Exception Handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "status_code": 500,
            "path": request.url.path,
        },
    )

# Register Routers
app.include_router(data_router)
app.include_router(analytics_router)
app.include_router(budget_router)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Health check & database connectivity probe",
)
def health_check() -> HealthResponse:
    """Verifies service health, database connection, and whether an active dataset is loaded."""
    try:
        conn = get_db_connection()
        count_row = conn.execute("SELECT COUNT(*) as cnt FROM requests;").fetchone()
        req_count = int(count_row["cnt"]) if count_row else 0
        ver = get_setting("dataset_version", default="none")
        conn.close()

        return HealthResponse(
            status="ok",
            version="1.0.0",
            database_connected=True,
            dataset_loaded=(req_count > 0),
            total_requests=req_count,
            current_dataset_version=ver,
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            database_connected=False,
            dataset_loaded=False,
            total_requests=0,
            current_dataset_version=f"error: {str(e)}",
        )
