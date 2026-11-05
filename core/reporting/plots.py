"""LLMetrics PNG Figure Generation Module for CLI & Visual Artifacts.

Renders high-resolution analytical figures for the 8 syllabus items using matplotlib's Agg backend.
"""

import os
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

def generate_analytical_figures(
    results: Dict[str, Any],
    output_dir: str = "data/figures",
) -> List[str]:
    """Generates analytical figures and saves them as PNGs.
    
    Args:
        results: Full analysis dictionary from core.
        output_dir: Directory where figures should be saved.
        
    Returns:
        List of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. EDA Distributions
    p1 = os.path.join(output_dir, "01_eda_distributions.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if "syllabus_1_eda" in results and "univariate" in results["syllabus_1_eda"]:
        num_dict = results["syllabus_1_eda"]["univariate"]["numeric"]
        
        # Latency
        if "latency_ms" in num_dict:
            bins = num_dict["latency_ms"]["histogram"]
            mids = [(b["bin_start"] + b["bin_end"]) / 2 for b in bins]
            counts = [b["count"] for b in bins]
            axes[0].bar(mids, counts, width=(mids[1] - mids[0]) if len(mids) > 1 else 10, color="#2563eb", alpha=0.7)
            axes[0].set_title("Latency Distribution (ms)")
            axes[0].set_xlabel("Latency (ms)")
            axes[0].set_ylabel("Requests")

        # Completion Tokens
        if "completion_tokens" in num_dict:
            bins = num_dict["completion_tokens"]["histogram"]
            mids = [(b["bin_start"] + b["bin_end"]) / 2 for b in bins]
            counts = [b["count"] for b in bins]
            axes[1].bar(mids, counts, width=(mids[1] - mids[0]) if len(mids) > 1 else 10, color="#059669", alpha=0.7)
            axes[1].set_title("Completion Tokens Distribution")
            axes[1].set_xlabel("Tokens")

        # Cost
        if "cost_usd" in num_dict:
            bins = num_dict["cost_usd"]["histogram"]
            mids = [(b["bin_start"] + b["bin_end"]) / 2 for b in bins]
            counts = [b["count"] for b in bins]
            axes[2].bar(mids, counts, width=(mids[1] - mids[0]) if len(mids) > 1 else 0.001, color="#d97706", alpha=0.7)
            axes[2].set_title("Cost USD Distribution")
            axes[2].set_xlabel("Cost (USD)")

    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    saved_files.append(p1)

    # 2. Simple Linear Regression & Prediction Intervals
    p2 = os.path.join(output_dir, "02_simple_regression_intervals.png")
    if "syllabus_2_simple_linear_regression" in results:
        reg_simple = results["syllabus_2_simple_linear_regression"]
        pts = reg_simple["residual_analysis"]["scatter_points"]
        intervals = reg_simple["intervals"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter + Fit + Prediction Intervals
        tokens_x = [p["completion_tokens"] for p in pts]
        latency_y = [p["actual_latency_ms"] for p in pts]
        ax1.scatter(tokens_x, latency_y, alpha=0.3, s=15, color="#6b7280", label="Observed Requests")

        grid_x = [iv["completion_tokens"] for iv in intervals]
        pred_y = [iv["mean_prediction"] for iv in intervals]
        ci_low = [iv["ci_lower"] for iv in intervals]
        ci_up = [iv["ci_upper"] for iv in intervals]
        pi_low = [iv["pi_lower"] for iv in intervals]
        pi_up = [iv["pi_upper"] for iv in intervals]

        ax1.plot(grid_x, pred_y, color="#dc2626", lw=2, label="OLS Fitted Line")
        ax1.fill_between(grid_x, ci_low, ci_up, color="#dc2626", alpha=0.25, label="95% Confidence Interval (Mean)")
        ax1.fill_between(grid_x, pi_low, pi_up, color="#fca5a5", alpha=0.20, label="95% Prediction Interval (Obs)")
        ax1.set_title("Simple Linear Regression: Latency ~ Completion Tokens")
        ax1.set_xlabel("Completion Tokens")
        ax1.set_ylabel("Latency (ms)")
        ax1.legend(loc="upper left")

        # Residuals vs Fitted
        fitted_x = [p["fitted_latency_ms"] for p in pts]
        res_y = [p["residual"] for p in pts]
        ax2.scatter(fitted_x, res_y, alpha=0.35, s=15, color="#2563eb")
        ax2.axhline(0, color="black", linestyle="--", lw=1)
        ax2.set_title("Residuals vs Fitted Values (Heteroscedasticity Check)")
        ax2.set_xlabel("Fitted Latency (ms)")
        ax2.set_ylabel("Residuals")

        plt.tight_layout()
        plt.savefig(p2, dpi=150)
        plt.close()
        saved_files.append(p2)

    # 3. Residual Q-Q Plot
    p3 = os.path.join(output_dir, "03_residual_qq_plot.png")
    if "syllabus_2_simple_linear_regression" in results:
        qq_pts = results["syllabus_2_simple_linear_regression"]["residual_analysis"]["qq_plot"]
        t_q = [p["theoretical"] for p in qq_pts]
        s_q = [p["sample"] for p in qq_pts]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(t_q, s_q, alpha=0.5, s=20, color="#7c3aed")
        min_v = min(min(t_q), min(s_q))
        max_v = max(max(t_q), max(s_q))
        ax.plot([min_v, max_v], [min_v, max_v], color="#dc2626", linestyle="--", lw=1.5, label="Normal Line")
        ax.set_title("Normal Q-Q Plot of Standardized Residuals")
        ax.set_xlabel("Theoretical Normal Quantiles")
        ax.set_ylabel("Standardized Sample Quantiles")
        ax.legend()
        plt.tight_layout()
        plt.savefig(p3, dpi=150)
        plt.close()
        saved_files.append(p3)

    # 4. Logistic Regression ROC Curve
    p4 = os.path.join(output_dir, "04_logistic_roc_curve.png")
    if "syllabus_4_logistic_regression" in results:
        roc_pts = results["syllabus_4_logistic_regression"]["binary"]["roc_analysis"]["curve_points"]
        auc_v = results["syllabus_4_logistic_regression"]["binary"]["roc_analysis"]["auc"]
        fpr_pts = [p["fpr"] for p in roc_pts]
        tpr_pts = [p["tpr"] for p in roc_pts]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr_pts, tpr_pts, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {auc_v:.3f})")
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1, label="Chance Baseline")
        ax.set_title("Receiver Operating Characteristic (ROC) - Failure Prediction")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(p4, dpi=150)
        plt.close()
        saved_files.append(p4)

    # 5. Forecasting Comparison & Horizon Plot
    p5 = os.path.join(output_dir, "05_forecast_spend_intervals.png")
    if "syllabus_6_forecasting" in results:
        f_data = results["syllabus_6_forecasting"]["forecast_evaluation"]
        hist_pts = f_data["history"][-30:]  # last 30 days
        fc_pts = f_data["forecast"]

        hist_x = [p["date"] for p in hist_pts]
        hist_y = [p["actual"] for p in hist_pts]

        fc_x = [p["date"] for p in fc_pts]
        fc_y = [p["forecast"] for p in fc_pts]
        fc_low80 = [p["lower_80"] for p in fc_pts]
        fc_up80 = [p["upper_80"] for p in fc_pts]
        fc_low95 = [p["lower_95"] for p in fc_pts]
        fc_up95 = [p["upper_95"] for p in fc_pts]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(hist_x)), hist_y, color="#1e293b", marker="o", markersize=3, label="Actual Daily Spend ($)")
        
        fc_indices = range(len(hist_x), len(hist_x) + len(fc_x))
        ax.plot(fc_indices, fc_y, color="#2563eb", marker="s", markersize=3, lw=2, label=f"Forecast ({f_data['chosen_method']})")
        ax.fill_between(fc_indices, fc_low80, fc_up80, color="#93c5fd", alpha=0.35, label="80% Prediction Interval")
        ax.fill_between(fc_indices, fc_low95, fc_up95, color="#bfdbfe", alpha=0.25, label="95% Prediction Interval")
        ax.set_title(f"Daily Spend History & {f_data['horizon_days']}-Day Out-of-Sample Forecast")
        ax.set_ylabel("Daily Spend (USD)")
        ax.set_xlabel("Timeline (Days)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(p5, dpi=150)
        plt.close()
        saved_files.append(p5)

    # 6. Quality Control Charts & Incidents
    p6 = os.path.join(output_dir, "06_qc_control_charts.png")
    if "syllabus_8_quality_control" in results and "control_charts" in results["syllabus_8_quality_control"]:
        charts = results["syllabus_8_quality_control"]["control_charts"]
        shewhart_pts = charts["shewhart"]["points"]
        cusum_pts = charts["cusum"]["points"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        days_idx = range(len(shewhart_pts))
        shew_vals = [p["value"] for p in shewhart_pts]
        ucl_vals = [p["ucl"] for p in shewhart_pts]
        cl_vals = [p["center_line"] for p in shewhart_pts]

        ax1.plot(days_idx, shew_vals, color="#3b82f6", label="Daily P95 Latency")
        ax1.plot(days_idx, ucl_vals, color="#dc2626", linestyle="--", label="UCL (3-Sigma)")
        ax1.plot(days_idx, cl_vals, color="#6b7280", linestyle=":", label="Center Line")

        # Highlight alarms
        for i, p in enumerate(shewhart_pts):
            if p["alarm"]:
                ax1.scatter(i, p["value"], color="#dc2626", s=40, zorder=5)

        ax1.set_title("Shewhart Control Chart (Individual P95 Latency)")
        ax1.set_ylabel("Latency (ms)")
        ax1.legend(loc="upper left")

        # CUSUM Chart
        c_plus = [p["cusum_plus"] for p in cusum_pts]
        h_val = cusum_pts[0]["decision_interval_h"] if cusum_pts else 100
        ax2.plot(days_idx, c_plus, color="#059669", lw=1.5, label="CUSUM C+")
        ax2.axhline(h_val, color="#dc2626", linestyle="--", label=f"Decision Interval h={h_val:.1f}")
        for i, p in enumerate(cusum_pts):
            if p["alarm"]:
                ax2.scatter(i, p["cusum_plus"], color="#dc2626", s=40, zorder=5)

        ax2.set_title("Tabular CUSUM Chart (Positive Drift Accumulation)")
        ax2.set_ylabel("Cumulative Sum")
        ax2.set_xlabel("Day Index")
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(p6, dpi=150)
        plt.close()
        saved_files.append(p6)

    return saved_files
