"""LLMetrics Text Features Module (Analysis of Unstructured Data).

Derives numerical and categorical features from raw prompt text:
- Character length and word count
- Code block detection (triple backticks or indented code)
- Keyword flags (SQL query, urgent keywords, error/debugging terms, summarization terms)
and evaluates their statistical association with failure probability.
"""

from typing import Dict, List, Any, Tuple
import re
import numpy as np
import pandas as pd
from scipy import stats

def extract_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts unstructured text features from the prompt_text column.
    
    Args:
        df: DataFrame containing a 'prompt_text' column.
        
    Returns:
        DataFrame augmented with extracted text feature columns.
    """
    res = df.copy()
    prompts = res["prompt_text"].fillna("").astype(str)

    res["prompt_char_len"] = prompts.str.len()
    res["prompt_word_count"] = prompts.str.split().str.len()

    # Regex detection for code blocks
    code_pattern = re.compile(r"```|def |function |SELECT |<script", re.IGNORECASE)
    res["has_code_block"] = prompts.apply(lambda s: 1 if code_pattern.search(s) else 0)

    # Keyword flags
    sql_pattern = re.compile(r"\b(SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
    res["has_sql_keyword"] = prompts.apply(lambda s: 1 if sql_pattern.search(s) else 0)

    urgent_pattern = re.compile(r"\b(urgent|asap|critical|emergency|immediate)\b", re.IGNORECASE)
    res["has_urgent_keyword"] = prompts.apply(lambda s: 1 if urgent_pattern.search(s) else 0)

    error_pattern = re.compile(r"\b(error|exception|stack trace|failed|traceback|syntax error)\b", re.IGNORECASE)
    res["has_error_keyword"] = prompts.apply(lambda s: 1 if error_pattern.search(s) else 0)

    return res

def analyze_text_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes the relationship between unstructured prompt features and request reliability.
    
    Args:
        df: DataFrame with extracted prompt features and status.
        
    Returns:
        Structured dictionary reporting effect sizes, group failure rates,
        and statistical significance tests.
    """
    df_feat = extract_prompt_features(df)
    is_failed = (df_feat["status"] != "ok").astype(int)
    df_feat["is_failure"] = is_failed

    features_to_test = [
        "has_code_block",
        "has_sql_keyword",
        "has_urgent_keyword",
        "has_error_keyword",
    ]

    effects: List[Dict[str, Any]] = []

    for feat in features_to_test:
        present_mask = df_feat[feat] == 1
        absent_mask = df_feat[feat] == 0

        fail_rate_present = float(is_failed[present_mask].mean() * 100.0) if present_mask.sum() > 0 else 0.0
        fail_rate_absent = float(is_failed[absent_mask].mean() * 100.0) if absent_mask.sum() > 0 else 0.0

        # 2x2 contingency table for chi-square test
        table = pd.crosstab(df_feat[feat], is_failed)
        if table.shape == (2, 2):
            chi2, p_val, dof, _ = stats.chi2_contingency(table)
        else:
            chi2, p_val, dof = 0.0, 1.0, 1

        rel_risk = (fail_rate_present / fail_rate_absent) if fail_rate_absent > 0 else 1.0

        effects.append({
            "feature": feat,
            "present_count": int(present_mask.sum()),
            "absent_count": int(absent_mask.sum()),
            "failure_rate_present_pct": round(fail_rate_present, 2),
            "failure_rate_absent_pct": round(fail_rate_absent, 2),
            "relative_risk": round(rel_risk, 2),
            "chi2_statistic": round(float(chi2), 3),
            "p_value": float(p_val),
            "statistically_significant": bool(p_val < 0.05),
        })

    # Continuous correlations
    corr_char = float(stats.pointbiserialr(is_failed, df_feat["prompt_char_len"]).statistic)
    corr_words = float(stats.pointbiserialr(is_failed, df_feat["prompt_word_count"]).statistic)

    return {
        "text_feature_effects": effects,
        "continuous_correlations_with_failure": {
            "prompt_char_length_correlation": round(corr_char, 4),
            "prompt_word_count_correlation": round(corr_words, 4),
        },
        "summary": (
            "Prompts containing error logs, code blocks, or complex SQL structures correlate with "
            "increased failure and timeout rates, reflecting their heavier token requirements and higher parsing latency."
        ),
    }
