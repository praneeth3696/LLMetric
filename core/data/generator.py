"""LLMetrics Synthetic Data Generator.

Generates realistic LLM API access logs with diurnal and weekly seasonality,
token-dependent latency with genuine heteroscedastic variance, logistic failure
dynamics, and labelled injected anomalies.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any, Optional
import uuid
import numpy as np
import pandas as pd

from core.data.config import (
    MODEL_PRICING,
    APP_FEATURES,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_ERROR,
    STATUS_RATE_LIMITED,
    GROUND_TRUTH_INCIDENTS,
)

SAMPLE_PROMPTS = [
    "Hello! Can you help me write a Python function: ```python\ndef process_data(records):\n    return [r for r in records if r.valid]\n``` to filter records?",
    "Please summarize this customer support transcript and highlight unresolved complaints.",
    "SELECT users.id, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.status = 'urgent' AND orders.cost > 500;",
    "What is the difference between synchronous and asynchronous execution in Node.js?",
    "Summarize the executive summary of the quarterly financial report into 3 bullet points.",
    "Debug this error in my SQL query: syntax error near unexpected token 'WHERE'. Urgent fix needed.",
    "Write a unit test for my authentication system module.",
    "Can you provide a simple explanation of transformer attention mechanisms?",
    "Generate a structured JSON schema for user registration payload with validation constraints.",
    "Quick chat: recommend three great books on distributed systems and reliability engineering.",
    "Analyze the following logs for critical error messages and stack traces: [CRITICAL] Connection reset by peer.",
    "Draft a concise follow-up email to the engineering leadership regarding sprint deliverables.",
]

def generate_synthetic_logs(
    days: int = 90,
    daily_volume: int = 1200,
    seed: int = 42,
    start_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generates a synthetic DataFrame of LLM API logs alongside ground-truth incidents.
    
    Args:
        days: Number of days to simulate (default: 90).
        daily_volume: Average number of requests per day (default: 1200).
        seed: Random seed for exact reproducibility (default: 42).
        start_date: Starting datetime (default: 90 days before now).
        
    Returns:
        Tuple of (pd.DataFrame of requests, list of ground truth incidents).
    """
    rng = np.random.default_rng(seed)

    if start_date is None:
        # Fixed base start date to ensure strict determinism across runs
        start_date = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

    # 1. Generate hourly request counts following diurnal & weekly cycles
    total_hours = days * 24
    records = []

    models = list(MODEL_PRICING.keys())
    # Model popularity weights: gpt-4o-mini (55%), claude-3-haiku (30%), llama-3-70b (15%)
    model_probs = [0.55, 0.30, 0.15]

    # Pre-calculate incident day mappings
    incidents = list(GROUND_TRUTH_INCIDENTS)

    for h_idx in range(total_hours):
        current_hour_dt = start_date + timedelta(hours=h_idx)
        current_day = h_idx // 24
        hour_of_day = current_hour_dt.hour
        day_of_week = current_hour_dt.weekday()  # 0=Monday, 6=Sunday

        # Seasonality factors:
        # Peak business hours: 9am - 18pm (peak factor up to 1.8x, night trough ~0.3x)
        diurnal_factor = 0.35 + 1.15 * np.exp(-0.5 * ((hour_of_day - 14) / 4.5) ** 2)
        # Weekday vs Weekend (Saturday/Sunday traffic drops to ~55%)
        weekly_factor = 0.55 if day_of_week >= 5 else 1.05

        base_hourly_lambda = (daily_volume / 24.0) * diurnal_factor * weekly_factor

        # Check for Incident D (Failure and traffic spike around Day 78 14:00 - Day 79 02:00)
        is_incident_d = (current_day == 78 and hour_of_day >= 14) or (current_day == 79 and hour_of_day <= 2)
        if is_incident_d:
            base_hourly_lambda *= 2.2  # 2.2x traffic surge

        n_requests = rng.poisson(max(1, base_hourly_lambda))

        if n_requests == 0:
            continue

        # Distribute requests uniformly or jittered across the hour
        seconds_offsets = rng.uniform(0, 3600, size=n_requests)
        seconds_offsets.sort()

        # Generate batch parameters
        req_models = rng.choice(models, size=n_requests, p=model_probs)
        req_features = rng.choice(APP_FEATURES, size=n_requests, p=[0.40, 0.25, 0.20, 0.15])
        is_peak = 1 if 9 <= hour_of_day <= 18 else 0

        # Incident flags for current day
        is_incident_a = (35 <= current_day <= 42)  # Model swap on gpt-4o-mini
        is_incident_b = (55 <= current_day <= 57)  # Latency regression
        is_incident_c = (70 <= current_day <= 75)  # Cost/prompt drift on claude-3-haiku

        for i in range(n_requests):
            req_time = current_hour_dt + timedelta(seconds=float(seconds_offsets[i]))
            m = req_models[i]
            feat = req_features[i]

            # Cache hit probability: 25% base for chat/search, 0% for code_gen
            cache_p = 0.30 if feat in ["chat", "search"] else 0.05
            cache_hit = 1 if rng.uniform() < cache_p else 0

            # Token generation: Prompt tokens depend on feature
            if feat == "summarize":
                base_prompt = rng.normal(1600, 300)
                base_comp = rng.normal(320, 70)
            elif feat == "code_gen":
                base_prompt = rng.normal(850, 200)
                base_comp = rng.normal(450, 110)
            elif feat == "search":
                base_prompt = rng.normal(400, 100)
                base_comp = rng.normal(180, 40)
            else:  # chat
                base_prompt = rng.normal(350, 120)
                base_comp = rng.normal(220, 60)

            # Apply Incident C: Prompt bloat on claude-3-haiku
            if is_incident_c and m == "claude-3-haiku":
                base_prompt *= 1.45

            # Apply Incident A: Silent model swap on gpt-4o-mini (outputs get longer and more verbose)
            if is_incident_a and m == "gpt-4o-mini":
                base_comp *= 1.65

            prompt_tokens = int(max(15, base_prompt))
            completion_tokens = int(max(5, base_comp))

            # Latency Physical Generating Process:
            # Base intercept per model:
            # gpt-4o-mini: 160ms, claude-3-haiku: 130ms, llama-3-70b: 280ms
            model_base = 160.0 if m == "gpt-4o-mini" else (130.0 if m == "claude-3-haiku" else 280.0)
            
            # Latency rate per completion token ~1.15 ms/token, prompt prefill ~0.04 ms/token
            beta_comp = 1.15
            beta_prompt = 0.04
            peak_penalty = 110.0 if is_peak else 0.0

            # Genuine Heteroscedastic noise: standard deviation scales with sqrt(completion_tokens)
            # This directly provides the physical property tested by the Breusch-Pagan test!
            noise_sigma = 25.0 + 3.8 * np.sqrt(completion_tokens)
            noise = rng.normal(0, noise_sigma)

            raw_latency = model_base + beta_comp * completion_tokens + beta_prompt * prompt_tokens + peak_penalty + noise

            # Apply Incident B: Latency regression (+400ms server network degradation)
            if is_incident_b:
                raw_latency += 420.0

            # Apply Incident A: Silent model swap adds +180ms backend routing overhead
            if is_incident_a and m == "gpt-4o-mini":
                raw_latency += 190.0

            if cache_hit == 1:
                # Cache hits avoid model generation overhead
                raw_latency = raw_latency * 0.12 + rng.uniform(15, 35)

            latency_ms = round(max(20.0, float(raw_latency)), 2)

            # Cost calculation from pricing config
            pricing = MODEL_PRICING[m]
            effective_prompt_price = pricing["prompt_price_per_1k"] * (
                (1.0 - pricing["cache_discount"]) if cache_hit else 1.0
            )
            cost_usd = (
                (prompt_tokens / 1000.0) * effective_prompt_price
                + (completion_tokens / 1000.0) * pricing["completion_price_per_1k"]
            )
            cost_usd = round(float(cost_usd), 6)

            # Failure Probability Modeling (Logistic Data Generating Process):
            # logit(p) = -3.8 + 0.0012 * latency_ms + 0.0004 * prompt_tokens + 0.45 * is_peak
            logit_p = -4.0 + 0.0014 * latency_ms + 0.00035 * prompt_tokens + 0.50 * is_peak
            
            # If Incident D (failure spike):
            if is_incident_d:
                logit_p += 2.8  # Major failure surge

            p_fail = 1.0 / (1.0 + np.exp(-logit_p))
            is_failed = rng.uniform() < p_fail

            retry_count = 0
            if not is_failed:
                status = STATUS_OK
            else:
                retry_count = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
                # Assign failure type based on triggers
                if is_incident_d or (is_peak and rng.uniform() < 0.65):
                    status = STATUS_RATE_LIMITED
                elif latency_ms > 2200 or rng.uniform() < 0.45:
                    status = STATUS_TIMEOUT
                else:
                    status = STATUS_ERROR

            # Prompt text selection
            prompt_text = rng.choice(SAMPLE_PROMPTS)

            # Deterministic UUID from 16 random bytes
            req_uuid = uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8)))

            records.append({
                "timestamp": req_time.isoformat(),
                "request_id": str(req_uuid),
                "feature": feat,
                "model": m,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "status": status,
                "retry_count": retry_count,
                "cache_hit": cache_hit,
                "prompt_text": prompt_text,
                "cost_usd": cost_usd,
            })

    df = pd.DataFrame(records)
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, incidents
