"""LLMetrics Exploratory Data Analysis (EDA) Module.

Provides comprehensive univariate, bivariate, and multivariate statistical analyses
tailored to LLM application metrics (latency, cost, tokens, and reliability).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats

def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes univariate, bivariate, and multivariate analysis on the dataset.
    
    Args:
        df: Cleaned request logs DataFrame.
        
    Returns:
        Structured dictionary containing statistical distributions, correlations,
        cross-tabulations, and categorical breakdowns.
    """
    numeric_cols = ["latency_ms", "cost_usd", "prompt_tokens", "completion_tokens", "retry_count"]
    categorical_cols = ["model", "feature", "status", "cache_hit"]

    # 1. Univariate Analysis
    univariate: Dict[str, Any] = {"numeric": {}, "categorical": {}}

    for col in numeric_cols:
        if col in df.columns:
            s = df[col].dropna()
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            mean_v = float(s.mean())
            std_v = float(s.std())
            median_v = float(s.median())
            skew_v = float(stats.skew(s))
            kurt_v = float(stats.kurtosis(s))

            # Generate histogram bins (e.g. 15 bins)
            counts, bin_edges = np.histogram(s, bins=15)
            bins_data = [
                {
                    "bin_start": round(float(bin_edges[i]), 2),
                    "bin_end": round(float(bin_edges[i + 1]), 2),
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ]

            univariate["numeric"][col] = {
                "count": int(len(s)),
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "median": round(median_v, 4),
                "p5": round(float(s.quantile(0.05)), 4),
                "p25": round(q25, 4),
                "p75": round(q75, 4),
                "p95": round(float(s.quantile(0.95)), 4),
                "p99": round(float(s.quantile(0.99)), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "iqr": round(q75 - q25, 4),
                "skewness": round(skew_v, 4),
                "kurtosis": round(kurt_v, 4),
                "histogram": bins_data,
            }

    for col in categorical_cols:
        if col in df.columns:
            freq = df[col].value_counts()
            pct = df[col].value_counts(normalize=True) * 100.0
            univariate["categorical"][col] = [
                {
                    "category": str(k),
                    "count": int(freq[k]),
                    "percentage": round(float(pct[k]), 2),
                }
                for k in freq.index
            ]

    # 2. Bivariate Analysis
    bivariate: Dict[str, Any] = {}

    # Correlation analysis (Pearson and Spearman)
    avail_num = [c for c in numeric_cols if c in df.columns]
    pearson_corr = df[avail_num].corr(method="pearson").round(4).to_dict()
    spearman_corr = df[avail_num].corr(method="spearman").round(4).to_dict()

    bivariate["correlations"] = {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }

    # Latency & Cost grouped by Model
    if "model" in df.columns:
        model_stats = (
            df.groupby("model")
            .agg(
                count=("latency_ms", "count"),
                latency_mean=("latency_ms", "mean"),
                latency_median=("latency_ms", "median"),
                latency_p95=("latency_ms", lambda x: np.percentile(x, 95)),
                cost_mean=("cost_usd", "mean"),
                cost_total=("cost_usd", "sum"),
                error_rate=("status", lambda x: np.mean(x != "ok") * 100.0),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        )
        bivariate["by_model"] = model_stats

    # Latency & Cost grouped by Feature
    if "feature" in df.columns:
        feature_stats = (
            df.groupby("feature")
            .agg(
                count=("latency_ms", "count"),
                latency_mean=("latency_ms", "mean"),
                cost_mean=("cost_usd", "mean"),
                error_rate=("status", lambda x: np.mean(x != "ok") * 100.0),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        )
        bivariate["by_feature"] = feature_stats

    # Latency grouped by Status
    if "status" in df.columns:
        status_stats = (
            df.groupby("status")
            .agg(
                count=("latency_ms", "count"),
                latency_mean=("latency_ms", "mean"),
                latency_p95=("latency_ms", lambda x: np.percentile(x, 95)),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        )
        bivariate["by_status"] = status_stats

    # 3. Multivariate Analysis
    multivariate: Dict[str, Any] = {}

    # Cross-tabulation: Model vs Status
    if "model" in df.columns and "status" in df.columns:
        crosstab = pd.crosstab(df["model"], df["status"], normalize="index") * 100.0
        multivariate["model_status_crosstab_pct"] = crosstab.round(2).to_dict(orient="index")

    # Grouped by Model and Peak Hour
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        is_peak = ((ts.dt.hour >= 9) & (ts.dt.hour <= 18)).astype(int)
        df_temp = df.copy()
        df_temp["is_peak_hour"] = is_peak
        
        peak_model_stats = (
            df_temp.groupby(["model", "is_peak_hour"])
            .agg(
                latency_mean=("latency_ms", "mean"),
                latency_p95=("latency_ms", lambda x: np.percentile(x, 95)),
                error_rate=("status", lambda x: np.mean(x != "ok") * 100.0),
                count=("latency_ms", "count"),
            )
            .round(3)
            .reset_index()
            .to_dict(orient="records")
        )
        multivariate["by_model_and_peak"] = peak_model_stats

    return {
        "dataset_summary": {
            "total_requests": len(df),
            "date_range_start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "date_range_end": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
            "total_cost_usd": round(float(df["cost_usd"].sum()), 4) if "cost_usd" in df.columns else 0.0,
            "overall_error_rate_pct": round(float((df["status"] != "ok").mean() * 100.0), 2) if "status" in df.columns else 0.0,
        },
        "univariate": univariate,
        "bivariate": bivariate,
        "multivariate": multivariate,
    }
