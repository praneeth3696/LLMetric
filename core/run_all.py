"""LLMetrics Core CLI Driver (python -m core.run_all).

Generates synthetic data, executes all 8 syllabus statistical analyses,
evaluates detectors against ground-truth incidents, and exports:
- Consolidated JSON report (data/results.json)
- Ground truth incidents (data/incidents_truth.json)
- Analytical figures (data/figures/*.png)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

from core.data.generator import generate_synthetic_logs
from core.data.cleaner import clean_request_logs
from core.reporting.summary import run_full_analytics_suite
from core.reporting.plots import generate_analytical_figures

def main() -> None:
    parser = argparse.ArgumentParser(description="LLMetrics End-to-End Analytics Engine CLI")
    parser.add_argument("--days", type=int, default=90, help="Number of simulated days (default: 90)")
    parser.add_argument("--volume", type=int, default=1200, help="Average daily request volume (default: 1200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for data and results (default: data)")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figure generation")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("  LLMetrics: Cost, Latency & Reliability Monitor for LLM Applications")
    print("  23XWA8 Predictive Analytics Course Project (M.Sc Software Systems)")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")

    # Step 1: Generate Synthetic Data
    print(f"\n[1/4] Generating synthetic logs ({args.days} days, ~{args.volume} req/day, seed={args.seed})...")
    df_raw, incidents = generate_synthetic_logs(days=args.days, daily_volume=args.volume, seed=args.seed)
    print(f"      Synthesized {len(df_raw):,} raw requests spanning {df_raw['timestamp'].min()[:10]} to {df_raw['timestamp'].max()[:10]}.")

    # Save small sample of raw CSV for inspection/demo
    sample_csv_path = os.path.join(args.output_dir, "sample_requests.csv")
    df_raw.head(100).to_csv(sample_csv_path, index=False)
    print(f"      Saved sample raw records to: {sample_csv_path}")

    # Save Ground Truth Incidents
    incidents_path = os.path.join(args.output_dir, "incidents_truth.json")
    with open(incidents_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2)
    print(f"      Saved ground-truth incidents to: {incidents_path}")

    # Step 2: Clean and Validate Data
    print("\n[2/4] Ingesting and cleaning dataset (missing value imputation, duplicate removal, outlier IQR check)...")
    df_clean, clean_report = clean_request_logs(df_raw)
    print(f"      Cleaned dataset rows: {len(df_clean):,} (Duplicates removed: {clean_report['duplicates_removed']}).")

    # Step 3: Run Full Analytics Suite
    print("\n[3/4] Executing 8 Syllabus Statistical Modules...")
    analytics_start = time.time()
    results = run_full_analytics_suite(df_clean, incidents=incidents)
    analytics_duration = time.time() - analytics_start
    print(f"      Completed all 8 syllabus analyses in {analytics_duration:.2f} seconds.")

    # Print summary highlights
    r2 = results["syllabus_2_simple_linear_regression"]["metrics"]["r_squared"]
    mult_r2 = results["syllabus_3_multiple_linear_regression"]["metrics"]["adjusted_r_squared"]
    bp_p = results["syllabus_3_multiple_linear_regression"]["diagnostics"]["heteroscedasticity_breusch_pagan"]["p_value"]
    auc_v = results["syllabus_4_logistic_regression"]["binary"]["roc_analysis"]["auc"]
    gini_v = results["syllabus_4_logistic_regression"]["binary"]["roc_analysis"]["gini_coefficient"]
    best_fc = results["syllabus_6_forecasting"]["forecast_evaluation"]["chosen_method"]
    arima_order = results["syllabus_7_timeseries_analysis"]["arima_model"]["order"]

    print("\n  Statistical Verification Highlights:")
    print(f"  • Simple OLS (latency ~ comp_tokens):    R² = {r2:.4f}")
    print(f"  • Multiple Regression:                  Adj R² = {mult_r2:.4f}, Breusch-Pagan p = {bp_p:.4e}")
    print(f"  • Logistic Regression (Failure):        ROC-AUC = {auc_v:.4f}, Gini = {gini_v:.4f}")
    print(f"  • Best Forecasting Method:             {best_fc}")
    print(f"  • Time Series ARIMA Order:              ARIMA{tuple(arima_order)}")

    # Step 4: Write Consolidated Results JSON
    results_path = os.path.join(args.output_dir, "results.json")
    print(f"\n[4/4] Writing consolidated results JSON to: {results_path}...")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Step 5: Render Analytical Figures
    if not args.no_figures:
        print(f"      Generating analytical figures into: {figures_dir}...")
        fig_paths = generate_analytical_figures(results, output_dir=figures_dir)
        print(f"      Generated {len(fig_paths)} analytical figures.")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  LLMetrics Run Complete! Total elapsed time: {total_elapsed:.2f} seconds.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

