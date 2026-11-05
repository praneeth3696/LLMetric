# LLMetrics: Engineering & Statistical Decisions Log

This document records architectural, statistical, and design decisions made during the development of LLMetrics.

---

## 1. Project Root & Structure
- **Decision**: The workspace root (`/Users/praneeth/Sandbox/PAproject/`) serves as the root directory for `llmetrics`, containing `core/`, `backend/`, `frontend/`, `data/`, and `docs/`.
- **Reason**: Allows seamless integration with root Makefile scripts, testing frameworks, and tooling without unnecessary nested path prefixes while fulfilling all required repo conventions.

## 2. Python Environment & Dependency Management
- **Decision**: Use Python 3.11.15 via a dedicated `.venv` managed with `uv`.
- **Reason**: Ensures high reproducibility, sub-second dependency installation, and clean isolation from the system Python runtime.

## 3. Data Generation Strategy & Incident Ground Truth
- **Decision**: Synthesize realistic 90-day hourly traffic (~1000-1500 req/day, totaling ~100k-120k requests) with deterministic seed (default: 42).
- **Physical Realism**:
  - Seasonality: Diurnal (peak hours 09:00 - 18:00 UTC) + weekly (lower weekend volume).
  - Latency: Base intercept + rate per completion token + prompt token coefficient + model offset + peak hour inflation - 85% cache speedup.
  - Heteroscedastic noise: $\sigma_\epsilon$ scales with $\sqrt{completion\_tokens}$, guaranteeing true heteroscedasticity for the Breusch-Pagan test.
  - Failure probability: Modeled using a standard logistic sigmoid function of latency, prompt length, and peak-hour concurrency.
  - Labelled Injected Incidents:
    1. Incident A (Day 35–42): Silent model swap (distribution shift on token length and response latency).
    2. Incident B (Day 55–57): 2-day latency regression (+400ms server baseline).
    3. Incident C (Day 70–75): Gradual cost drift (+25% token inflation).
    4. Incident D (Day 78 14:00 – Day 79 02:00): Failure/timeout spike (transient provider degradation).
- **Reason**: Matches real-world LLM production behavior and provides unambiguous ground-truth labels for evaluating control charts and detector performance.

## 4. Model Pricing Placeholders
- **Decision**: Token pricing is stored in `core/data/config.py` with explicit docstrings and UI labels marked as `ILLUSTRATIVE PLACEHOLDERS`.
- **Reason**: LLM provider API pricing fluctuates regularly; transparent placeholder configuration prevents false claims of real-time billing accuracy.

## 5. Statistical Tree Implementation (CHAID)
- **Decision**: Implement a native, vectorized Chi-square Automatic Interaction Detection (CHAID) engine using `scipy.stats.chi2_contingency` on binned continuous and categorical predictors, supplemented with comprehensive unit tests.
- **Reason**: Standard scikit-learn does not provide CHAID (only CART). A pure-Python implementation using scipy guarantees zero external untrusted dependencies and transparent, auditable chi-square splitting logic.

## 6. Time Series & Forecasting Architecture
- **Decision**: Daily aggregation for primary time-series forecasting (spend and token usage). Implement rolling-origin backtesting across Naive, Seasonal Naive, Moving Average, Simple Exponential Smoothing, Holt's linear, Holt-Winters additive/multiplicative, and ARIMAX causal modeling.
- **Reason**: Production LLM billing cycles and capacity planning operate primarily on daily horizons; rolling-origin cross-validation avoids lookahead bias.

## 7. Quality Control Charts & Change Point Detection
- **Decision**: Implement Shewhart ($I-MR$), EWMA ($\lambda=0.2, L=3.0$), and CUSUM ($k=0.5\sigma, h=5\sigma$). For distribution shift detection, use a rolling two-sample Kolmogorov-Smirnov (K-S) test comparing recent windows to a stable baseline window.
- **Reason**: Catches sudden spikes (Shewhart), subtle drift (CUSUM/EWMA), and unannounced model swaps (K-S test on output token distributions) with quantified detection delays and false alarm rates.

## 8. Backend Architecture & SQLite Invalidation Caching
- **Decision**: FastAPI service backed by SQLite with an `analysis_cache` table keyed by dataset hash and analysis section name. When new data is generated or uploaded, cache entries are invalidated and background status is tracked in `task_status`.
- **Reason**: Enables sub-100ms response times for all analytical GET endpoints once computed, while preventing UI freeze on initial cold runs.

## 9. Frontend Architecture & Design Language
- **Decision**: Vite + React 18 + TypeScript + Recharts. Interface adheres to strict data-dense, calm aesthetics: zero decorative gradients, zero box-shadows, zero floating animations, zero emoji, and a system monospace/sans-serif typographic hierarchy.
- **Reason**: Provides an uncluttered, authoritative observability experience suitable for rigorous academic evaluation and production operations.

## 10. Traceability Matrix & Methodology View
- **Decision**: Build an explicit "Methodology" tab directly into the UI mapping each of the 8 syllabus topics to its mathematical formulation, underlying assumptions, code files, and anticipated viva examination questions.
- **Reason**: Guarantees complete transparency during viva demonstrations, enabling immediate navigation between empirical plots and statistical theory.

## 11. Offline Reliability & Deterministic Extensions
- **Decision**: The system has zero mandatory external API dependencies. Synthetic generation is deterministic (seed=42). Weekly insights are generated via a rule-based deterministic summary engine with an optional Anthropic API rewrite fallback. Budget breach risk is computed via normal-approximation cumulative forecast integrals.
- **Reason**: Prevents demo failures due to network disruption, quota exhaustion, or external provider outages.


