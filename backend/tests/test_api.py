"""Integration tests for LLMetrics FastAPI Backend."""

import os
import tempfile
import time
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from backend.main import app
from backend.db.database import init_db

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient connected to an isolated temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        temp_db_path = tmp.name

    os.environ["LLMETRICS_DB_PATH"] = temp_db_path
    init_db(temp_db_path)

    with TestClient(app) as test_client:
        yield test_client

    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

def test_health_on_fresh_db(client):
    """GET /health on a brand-new database returns healthy status with 0 requests."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database_connected"] is True
    assert data["dataset_loaded"] is False
    assert data["total_requests"] == 0

def test_empty_database_behavior(client):
    """GET endpoints return 404 when no data has been generated or uploaded yet."""
    endpoints = [
        "/overview",
        "/eda",
        "/regression/simple",
        "/regression/multiple",
        "/reliability",
        "/forecast",
        "/alerts",
        "/insights",
    ]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 404
        assert "detail" in resp.json()

def test_bad_upload_validation(client):
    """POST /data/upload rejects invalid file extensions and malformed schemas."""
    # 1. Invalid file extension (.txt)
    resp = client.post(
        "/data/upload",
        files={"file": ("test.txt", b"some text content", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]

    # 2. CSV missing required columns
    bad_csv = "id,name,value\n1,test,100\n"
    resp = client.post(
        "/data/upload",
        files={"file": ("bad_schema.csv", bad_csv.encode("utf-8"), "text/csv")},
    )
    assert resp.status_code == 400
    assert "Missing required columns" in resp.json()["detail"]

def test_generate_data_and_verify_all_get_endpoints(client):
    """POST /data/generate followed by verified calls to every GET endpoint."""
    # 1. Generate data (25 days, 100 req/day for robust integration test)
    gen_payload = {"days": 25, "daily_volume": 100, "seed": 42}
    resp_gen = client.post("/data/generate", json=gen_payload)
    assert resp_gen.status_code == 200
    gen_data = resp_gen.json()
    assert gen_data["status"] == "success"
    assert gen_data["total_records"] > 0
    assert gen_data["dataset_version"] != "none"

    # 2. GET /overview
    resp = client.get("/overview")
    assert resp.status_code == 200
    ov = resp.json()
    assert ov["total_requests"] > 0
    assert ov["total_cost_usd"] > 0
    assert ov["p95_latency_ms"] > 0
    assert "period_comparison" in ov

    # 3. GET /eda
    resp = client.get("/eda")
    assert resp.status_code == 200
    eda = resp.json()
    assert "univariate" in eda
    assert "bivariate" in eda
    assert "multivariate" in eda

    # 4. GET /regression/simple
    resp = client.get("/regression/simple")
    assert resp.status_code == 200
    s_reg = resp.json()
    assert s_reg["metrics"]["r_squared"] > 0
    assert "completion_tokens" in s_reg["coefficients"]
    assert len(s_reg["intervals"]) > 0

    # 5. GET /regression/multiple
    resp = client.get("/regression/multiple")
    assert resp.status_code == 200
    m_reg = resp.json()
    assert m_reg["metrics"]["adjusted_r_squared"] > 0
    assert "heteroscedasticity_breusch_pagan" in m_reg["diagnostics"]
    assert "multicollinearity_vif" in m_reg["diagnostics"]
    assert "log_transform_comparison" in m_reg

    # 6. GET /reliability
    resp = client.get("/reliability")
    assert resp.status_code == 200
    rel = resp.json()
    assert "binary_logistic" in rel
    assert "multinomial_logistic" in rel
    assert "cart_classification" in rel
    assert "chaid_classification" in rel
    assert "model_comparison" in rel

    # 7. GET /forecast (default: spend, 14d, auto)
    resp = client.get("/forecast")
    assert resp.status_code == 200
    fc = resp.json()
    assert fc["target"] == "daily_spend"
    assert len(fc["forecast"]) == 14
    assert len(fc["history"]) > 0

    # 8. GET /forecast with custom parameters (tokens, 7d, ma)
    resp = client.get("/forecast?target=tokens&horizon=7&method=ma")
    assert resp.status_code == 200
    fc_custom = resp.json()
    assert fc_custom["target"] == "daily_tokens"
    assert len(fc_custom["forecast"]) == 7

    # 9. GET /alerts
    resp = client.get("/alerts")
    assert resp.status_code == 200
    al = resp.json()
    assert "control_charts" in al
    assert "shewhart" in al["control_charts"]
    assert "ewma" in al["control_charts"]
    assert "cusum" in al["control_charts"]
    assert "benchmark_evaluation" in al

    # 10. GET /insights
    resp = client.get("/insights")
    assert resp.status_code == 200
    ins = resp.json()
    assert len(ins["bullets"]) >= 3
    assert "summary_text" in ins

    # 11. POST /budget & GET /budget
    resp = client.post("/budget", json={"monthly_budget": 1500.0})
    assert resp.status_code == 200
    assert resp.json()["monthly_budget"] == 1500.0

    resp = client.get("/budget")
    assert resp.status_code == 200
    assert resp.json()["monthly_budget"] == 1500.0
    assert "breach_probability_pct" in resp.json()

    # 12. GET /task/status
    resp = client.get("/task/status")
    assert resp.status_code == 200
    assert resp.json()["status"] in ["idle", "running", "completed"]

def test_cache_speed_under_500ms(client):
    """Subsequent requests hit the SQLite analysis cache and respond in < 500 ms."""
    start_t = time.time()
    resp = client.get("/regression/multiple")
    elapsed_ms = (time.time() - start_t) * 1000.0

    assert resp.status_code == 200
    assert elapsed_ms < 500.0  # Must be fast once cached

def test_openapi_documentation(client):
    """OpenAPI schema is exposed at /openapi.json and documents all routes."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()

    assert "paths" in schema
    paths = schema["paths"]
    expected_paths = [
        "/data/generate",
        "/data/upload",
        "/overview",
        "/eda",
        "/regression/simple",
        "/regression/multiple",
        "/reliability",
        "/forecast",
        "/alerts",
        "/insights",
        "/budget",
        "/health",
        "/task/status",
    ]
    for p in expected_paths:
        assert p in paths, f"Path {p} is missing from OpenAPI documentation!"

def test_valid_csv_upload(client):
    """Upload valid CSV request log file and verify ingestion."""
    # Generate 15 days of valid requests
    from core.data.generator import generate_synthetic_logs
    df_sample, _ = generate_synthetic_logs(days=15, daily_volume=40, seed=99)
    csv_bytes = df_sample.to_csv(index=False).encode("utf-8")

    resp = client.post(
        "/data/upload",
        files={"file": ("real_logs.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_records"] == len(df_sample)
    assert data["status"] == "success"
