"""LLMetrics End-to-End Smoke Test.

Runs an automated end-to-end verification of the backend API:
1. Health check & DB verification
2. Synthetic data generation (90 days, seed=42)
3. Complete analytical retrieval across all 8 syllabus endpoints
4. Verification of numerical soundness and schema compliance
"""

import sys
import os
import time

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient

from backend.main import app
from backend.db.database import init_db

def run_e2e_smoke_test() -> None:
    print("=" * 70)
    print("  LLMetrics End-to-End Smoke Test")
    print("=" * 70)

    init_db()
    client = TestClient(app)

    # 1. Health Probe
    print("\n[1/5] Checking service health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health = resp.json()
    print(f"      Service status: {health['status']}, DB connected: {health['database_connected']}")

    # 2. Data Generation
    print("\n[2/5] Generating 90-day synthetic dataset (~1200 req/day, seed=42)...")
    t0 = time.time()
    resp = client.post("/data/generate", json={"days": 90, "daily_volume": 1200, "seed": 42})
    assert resp.status_code == 200, f"Generation failed: {resp.text}"
    gen_data = resp.json()
    gen_time = time.time() - t0
    print(f"      Generated {gen_data['total_records']:,} requests in {gen_time:.2f}s (Version: {gen_data['dataset_version']})")

    # 3. Query All 8 Syllabus Endpoints
    print("\n[3/5] Querying all 8 syllabus analytical endpoints...")
    endpoints = [
        ("/overview", "System Overview & 14d Comparison"),
        ("/eda", "Syllabus 1: EDA & Cleaning"),
        ("/regression/simple", "Syllabus 2: Simple Linear Regression"),
        ("/regression/multiple", "Syllabus 3: Multiple Regression & Diagnostics"),
        ("/reliability", "Syllabus 4 & 5: Failure Prediction & Decision Trees"),
        ("/forecast?target=spend&horizon=14&method=auto", "Syllabus 6 & 7: Forecasting & Backtesting"),
        ("/alerts", "Syllabus 8: SQC Charts & Incident Evaluation"),
        ("/insights", "Extension: Weekly Insights Summary"),
        ("/budget", "Extension: Budget Tracking & Breach Risk"),
    ]

    results_cache = {}
    for path, label in endpoints:
        t_start = time.time()
        r = client.get(path)
        t_ms = (time.time() - t_start) * 1000.0
        assert r.status_code == 200, f"Endpoint {path} failed: {r.text}"
        data = r.json()
        results_cache[path] = data
        print(f"      ✓ {label:<45} [{t_ms:>6.1f} ms]")

    # 4. Numerical Verifications
    print("\n[4/5] Verifying statistical results against syllabus requirements...")
    mreg = results_cache["/regression/multiple"]
    sreg = results_cache["/regression/simple"]
    rel = results_cache["/reliability"]
    fc = results_cache["/forecast?target=spend&horizon=14&method=auto"]
    alerts = results_cache["/alerts"]
    ov = results_cache["/overview"]

    # Simple regression checks
    r2_simple = sreg["metrics"]["r_squared"]
    assert 0.40 <= r2_simple <= 0.85, f"Unexpected simple R²: {r2_simple}"

    # Multiple regression checks
    adj_r2 = mreg["metrics"]["adjusted_r_squared"]
    bp_p = mreg["diagnostics"]["heteroscedasticity_breusch_pagan"]["p_value"]
    assert adj_r2 > 0.75, f"Unexpected multiple adj R²: {adj_r2}"
    assert bp_p < 0.05, f"Breusch-Pagan failed to detect heteroscedasticity: {bp_p}"

    # Logistic regression checks
    auc_v = rel["binary_logistic"]["roc_analysis"]["auc"]
    gini_v = rel["binary_logistic"]["roc_analysis"]["gini_coefficient"]
    assert auc_v > 0.60, f"Unexpected AUC: {auc_v}"

    # QC Detector checks
    benchmark_table = alerts["benchmark_evaluation"]["benchmark_table"]
    assert len(benchmark_table) >= 3

    print(f"      • Simple Linear Regression R²:              {r2_simple:.4f}")
    print(f"      • Multiple Linear Regression Adj R²:         {adj_r2:.4f}")
    print(f"      • Breusch-Pagan Heteroscedasticity p-value:  {bp_p:.4e} (Detected)")
    print(f"      • Failure Prediction ROC-AUC:               {auc_v:.4f} (Gini: {gini_v:.4f})")
    print(f"      • Best Forecasting Model:                   {fc['chosen_method']}")
    print(f"      • Total Quality Control Incident Alarms:    {benchmark_table[0]['total_alarms']}")

    # 5. Cache speed test
    print("\n[5/5] Testing cache retrieval speed...")
    t_cache_start = time.time()
    r_cache = client.get("/regression/multiple")
    cache_elapsed_ms = (time.time() - t_cache_start) * 1000.0
    assert r_cache.status_code == 200
    assert cache_elapsed_ms < 500.0, f"Cache too slow: {cache_elapsed_ms} ms"
    print(f"      ✓ Cached multiple regression response: {cache_elapsed_ms:.2f} ms (< 500 ms SLA)")

    print("\n" + "=" * 70)
    print("  ALL END-TO-END SMOKE TESTS PASSED!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_e2e_smoke_test()
