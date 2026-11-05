"""LLMetrics Weekly Insights Generator.

Produces plain-language executive summaries deterministically from computed statistics.
If an ANTHROPIC_API_KEY is configured in the environment, optionally permits
an LLM rewrite with instantaneous, graceful fallback on any error.
"""

import os
from typing import Dict, Any, Optional

def generate_weekly_insights(
    overview_stats: Dict[str, Any],
    qc_stats: Optional[Dict[str, Any]] = None,
    forecast_stats: Optional[Dict[str, Any]] = None,
    budget_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates executive weekly insights with deterministic fallback."""
    # Deterministic narrative construction
    total_spend = overview_stats.get("total_cost_usd", 0.0)
    total_reqs = overview_stats.get("total_requests", 0)
    p95_lat = overview_stats.get("p95_latency_ms", 0.0)
    err_rate = overview_stats.get("error_rate_pct", 0.0)

    # Narrative bullet points
    bullets = [
        f"Traffic & Spend: Ingested {total_reqs:,} API requests resulting in ${total_spend:,.2f} USD cumulative spend.",
        f"Latency Health: 95th percentile response latency stabilized at {p95_lat:,.1f} ms.",
        f"Failure Posture: Operational error and timeout rate measured at {err_rate:.2f}%.",
    ]

    if qc_stats:
        active_alarms = qc_stats.get("total_alarms", 0)
        if active_alarms > 0:
            bullets.append(f"Quality Alerts: Quality control charts flagged {active_alarms} statistical anomaly points requiring review.")
        else:
            bullets.append("Quality Alerts: All operational metrics operated within standard 3-sigma control limits.")

    if budget_stats:
        breach_prob = budget_stats.get("breach_probability_pct", 0.0)
        proj_spend = budget_stats.get("projected_month_end_spend", 0.0)
        budget_limit = budget_stats.get("monthly_budget", 0.0)
        bullets.append(
            f"Budget Outlook: Month-end spend is projected at ${proj_spend:,.2f} against a ${budget_limit:,.2f} budget "
            f"({breach_prob:.1f}% risk of budget breach)."
        )

    deterministic_summary = (
        "LLMetrics Executive Weekly Summary:\n\n"
        + "\n".join(f"• {b}" for b in bullets)
        + "\n\nKey Recommendation: Monitor high-token summarization prompts and leverage prompt caching to curb peak spend."
    )

    result = {
        "summary_text": deterministic_summary,
        "is_llm_rewritten": False,
        "bullets": bullets,
        "engine": "Deterministic Statistical Template",
    }

    # Optional Anthropic rewrite if key is present
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import httpx
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            prompt_content = f"Rewrite the following operational metrics summary into a polished 3-paragraph executive brief:\n{deterministic_summary}"
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt_content}],
            }
            with httpx.Client(timeout=4.0) as client:
                resp = client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    rewritten_text = resp_json["content"][0]["text"]
                    result["summary_text"] = rewritten_text
                    result["is_llm_rewritten"] = True
                    result["engine"] = "Anthropic Claude (Rewritten)"
        except Exception:
            # Graceful fallback to deterministic template
            pass

    return result
