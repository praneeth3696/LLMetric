"""LLMetrics Data Configuration & Model Pricing Placeholders.

IMPORTANT NOTICE ON MODEL PRICING:
The pricing values configured below are ILLUSTRATIVE PLACEHOLDERS intended for
academic modeling and demonstration within this coursework project.
Actual provider pricing varies across tiers, contract volumes, and model updates.
Users deploying this in production MUST substitute their provider's current official rates.
"""

from typing import Dict, Any

# Model pricing in USD per 1,000 tokens (ILLUSTRATIVE PLACEHOLDERS)
# format: {'prompt_price_per_1k': float, 'completion_price_per_1k': float, 'cache_discount': float}
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {
        "prompt_price_per_1k": 0.00015,
        "completion_price_per_1k": 0.00060,
        "cache_discount": 0.50,  # 50% discount on cached prompt tokens
    },
    "claude-3-haiku": {
        "prompt_price_per_1k": 0.00025,
        "completion_price_per_1k": 0.00125,
        "cache_discount": 0.90,  # 90% discount on prompt cache hits
    },
    "llama-3-70b": {
        "prompt_price_per_1k": 0.00059,
        "completion_price_per_1k": 0.00079,
        "cache_discount": 0.0,
    },
}

# Features available in LLM applications
APP_FEATURES = ["chat", "summarize", "search", "code_gen"]

# Status categories
STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_RATE_LIMITED = "rate_limited"
ALL_STATUSES = [STATUS_OK, STATUS_TIMEOUT, STATUS_ERROR, STATUS_RATE_LIMITED]
FAILURE_STATUSES = [STATUS_TIMEOUT, STATUS_ERROR, STATUS_RATE_LIMITED]

# Ground Truth Injected Incidents Definition
# Used for evaluating quality control charts, CUSUM, EWMA, and model swap detectors.
GROUND_TRUTH_INCIDENTS = [
    {
        "id": "incident_model_swap",
        "name": "Silent Model Swap (Distribution Shift)",
        "type": "model_swap",
        "start_day": 35,
        "end_day": 42,
        "target_model": "gpt-4o-mini",
        "description": "Upstream provider silently swapped the model backend, causing an increase in completion token distribution and baseline latency.",
    },
    {
        "id": "incident_latency_regression",
        "name": "Upstream Latency Regression",
        "type": "latency_regression",
        "start_day": 55,
        "end_day": 57,
        "target_model": "all",
        "description": "Provider networking routing degradation introduced an abrupt +400ms server overhead across all requests.",
    },
    {
        "id": "incident_cost_drift",
        "name": "Prompt Bloat Cost Drift",
        "type": "cost_drift",
        "start_day": 70,
        "end_day": 75,
        "target_model": "claude-3-haiku",
        "description": "A developer prompt update appended large system prompt boilerplate, drifting prompt token usage and spend upwards.",
    },
    {
        "id": "incident_failure_spike",
        "name": "Transient Rate Limit & Failure Spike",
        "type": "failure_spike",
        "start_day": 78,
        "end_day": 79,
        "target_model": "all",
        "description": "A sudden concurrency burst triggered upstream rate limits and socket timeouts.",
    },
]
