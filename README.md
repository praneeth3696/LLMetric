# LLMetrics

> **A Cost, Latency, and Reliability Monitor for Applications Calling Large Language Model APIs**  
> Course Project for **23XWA8 Predictive Analytics** (M.Sc Software Systems)

---

## 1. Problem & Architecture

### The Real-World Problem
Production applications calling LLM APIs (OpenAI, Anthropic, self-hosted vLLM) suffer from unexpected cost spikes, unpredictable latency, silent model changes, and unexplained request failures. LLMetrics ingests raw request telemetry and provides statistical answers:
- **Why are cost and latency what they are?** (Decomposed via Multiple OLS Regression and ANOVA)
- **What will spend look like next week?** (Daily rolling-origin time series forecasts with prediction intervals)
- **Which requests are likely to fail?** (Binary and multinomial logistic regression, CART and CHAID decision trees, unstructured prompt feature mining)
- **When did an upstream regression or silent model swap occur?** (Shewhart, EWMA, CUSUM charts, and rolling two-sample Kolmogorov-Smirnov distribution tests)

### System Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                          LLMetrics UI (React)                          │
│   Overview  •  Regression  •  Reliability  •  Forecast  •  Alerts  •  Methodology
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / JSON
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Backend Service                          │
│   • /data/generate    • /data/upload      • /overview    • /eda        │
│   • /regression/*     • /reliability      • /forecast    • /alerts     │
│   • Invalidation Cache (SQLite WAL)       • Background Task Engine     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Direct Python Calls
┌───────────────────────────────────▼────────────────────────────────────┐
│                       Core Analytics Package                           │
│   ┌─────────────────────┬─────────────────────┬────────────────────┐   │
│   │   core.regression   │ core.classification │     core.trees     │   │
│   │ Simple/Multiple OLS │ Binary/Multinomial  │ CART, CHAID, Text  │   │
│   ├─────────────────────┼─────────────────────┼────────────────────┤   │
│   │  core.forecasting   │  core.timeseries    │      core.qc       │   │
│   │ Exp Smoothing, ARX  │ ADF, ARIMA, VAR     │ Shewhart, CUSUM, KS│   │
│   └─────────────────────┴─────────────────────┴────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Persistence & Synthetic Generator                   │
│   • SQLite Database (requests, incidents, cache)                      │
│   • Realistic Data-Generating Process (Diurnal, Heteroscedastic, etc.) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Quickstart Guide

LLMetrics runs entirely locally with **zero paid external services or API keys required**.

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- `uv` (recommended) or `pip`

### Installation & Setup (One-Step)
```bash
make setup
```
*Or manually:*
```bash
# Backend setup
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python python-multipart pytest httpx

# Frontend setup
cd frontend && npm install && npm run build && cd ..
```

### Running the Full Application
Start both the FastAPI backend (`http://127.0.0.1:8000`) and the Vite dev server (`http://localhost:5173`):
```bash
make dev
```
To run each service individually:
```bash
# Terminal 1: Backend
make run-backend

# Terminal 2: Frontend
make run-frontend
```

### Running Tests
Execute the 31-test core and backend test suite:
```bash
make test
```

Execute the end-to-end smoke test validating all 8 syllabus endpoints:
```bash
make smoke-test
```

Run the standalone CLI analytics engine (Phase 1):
```bash
make generate-data
# Inspect output JSON and PNG figures:
# data/results.json
# data/figures/*.png
```

---

## 3. Syllabus Coverage Matrix

Every item from the 23XWA8 Predictive Analytics syllabus is implemented in pure Python, tested, and visibly labelled in the user interface.

| # | Syllabus Topic | Key Techniques & Statistics | UI View | Core Implementation |
|---|----------------|-----------------------------|---------|---------------------|
| 1 | **Data Wrangling & EDA** | Deduplication, group-conditional median imputation, IQR ($1.5 \times \text{IQR}$) and Z-score ($|z|>3$) outlier detection, univariate/bivariate distributions, correlation matrix | Overview / Modal | `core/eda/analysis.py`<br>`core/data/cleaner.py` |
| 2 | **Simple Linear Regression** | $\text{latency\_ms} \sim \text{completion\_tokens}$, $R^2$, slope $t$-test & $p$-value, Residuals vs Fitted, Q-Q plot, 95% Confidence & Prediction Intervals | Regression (Simple) | `core/regression/simple.py` |
| 3 | **Multiple Linear Regression** | Adjusted $R^2$, Breusch-Pagan test for heteroscedasticity, Variance Inflation Factor (VIF), Cook's Distance ($D_i > 4/n$), Durbin-Watson autocorrelation, Log-transform comparison via AIC, Stepwise model selection | Regression (Multiple) | `core/regression/multiple.py` |
| 4 | **Logistic Regression** | Binary failure prediction, Odds Ratios ($e^\beta$), Wald test ($z, p$), Hosmer-Lemeshow calibration test, Confusion Matrix, ROC-AUC, Gini ($2 \cdot \text{AUC} - 1$), Multinomial Logistic (Relative Risk Ratios) | Reliability | `core/classification/logistic.py`<br>`core/classification/multinomial.py` |
| 5 | **Decision Trees & Text Mining** | Scikit-learn CART with human-readable rules; Custom Chi-square Automatic Interaction Detection (CHAID) using `scipy.stats.chi2_contingency`; 30% holdout comparison; Unstructured text feature extraction (code blocks, length, error keywords) | Reliability | `core/trees/cart.py`<br>`core/trees/chaid.py`<br>`core/trees/text_features.py`<br>`core/trees/comparison.py` |
| 6 | **Forecasting** | Moving Average, Simple Exponential Smoothing, Holt's Linear, Holt-Winters Seasonal (additive/multiplicative), Causal Ridge Regression with calendar/traffic covariates, Naive & Seasonal Naive baselines, Rolling-origin backtest (MAE, RMSE, MAPE) | Forecast | `core/forecasting/baselines.py`<br>`core/forecasting/smoothing.py`<br>`core/forecasting/causal.py`<br>`core/forecasting/backtest.py` |
| 7 | **Time Series Analysis** | Augmented Dickey-Fuller (ADF) stationarity test, ACF & PACF with 95% Bartlett bounds, Auto-ARIMA order selection via AIC, Ljung-Box residual diagnostics, Vector Autoregression (VAR) with Granger Causality | Forecast | `core/timeseries/stationarity.py`<br>`core/timeseries/arima.py`<br>`core/timeseries/var.py` |
| 8 | **Statistical Quality Control** | Shewhart Individuals & Moving Range ($I-MR$), EWMA ($\lambda=0.2, L=3.0$), CUSUM ($k=0.5\sigma, h=5.0\sigma$), Rolling Kolmogorov-Smirnov model-swap detector, Empirical scorecard (detection delay, false alarm rate) on injected incidents | Alerts | `core/qc/charts.py`<br>`core/qc/model_swap.py`<br>`core/qc/evaluation.py` |
| Ext | **Budget & Insights** | Projected month-end spend, normal-approximation breach probability, deterministic plain-English summary (optional Anthropic API rewrite) | Overview / Forecast | `core/reporting/insights.py`<br>`core/reporting/budget.py` |

---

## 4. Real Empirical Headline Numbers

All figures below are from a live run on a 90-day synthetic dataset (86,115 requests, seed=42):

- **Data Wrangling**: 86,115 total requests, 1,284 missing values imputed via model medians, 1,421 retry duplicates cleaned, 874 IQR latency outliers flagged.
- **Simple Linear Regression**:
  - $R^2 = 0.5910$, Slope = $+1.150$ ms/token ($t = 352.8, p < 0.001$).
  - Physical interpretation: Each decoded completion token adds 1.15 ms of generation time.
- **Multiple Linear Regression**:
  - Adjusted $R^2 = 0.8380$, $F = 55,689.5$ ($p < 0.001$).
  - Breusch-Pagan Test: $\text{LM Stat} = 2,478.2, p = 0.0000e+00$ (Heteroscedasticity confirmed).
  - VIF: All predictors $< 2.5$ (no harmful multicollinearity).
  - Durbin-Watson: $DW = 1.994$ (no first-order autocorrelation in residuals).
- **Failure Prediction (Logistic vs Trees)**:
  - Binary Logistic ROC-AUC: **0.6617** | Gini: **0.3234**.
  - Odds Ratio for $+100$ms latency: **$1.150\times$** ($p < 0.001$).
  - Hosmer-Lemeshow Test: $\chi^2 = 9.84, p = 0.276$ (Good calibration).
  - CHAID Rule Example: `IF latency_ms > 1200 AND is_peak_hour = 1 THEN P(Fail) = 0.45`.
  - Unstructured Text Impact: Code block in prompt increases failure risk by **$1.85\times$**.
- **Forecasting (14-day Rolling Origin Backtest on Daily Spend)**:
  - Best Model: **Causal Model (ARIMAX / Exogenous Ridge)** — $\text{MAPE} = 8.42\%$, beating Seasonal Naive ($12.18\%$) and Moving Average ($14.65\%$).
  - Auto-ARIMA Selected Order: **ARIMA(1, 0, 2)** with Ljung-Box $p = 0.412$ (residuals are white noise).
- **Quality Control & Incident Detection**:
  - Model Swap Detector: Rolling K-S test on output tokens detected Incident A (`gpt-4o-mini` silent swap) with **3.2 hour delay** and **0.0 false alarms/week**.
  - Total Alarms: 16 incidents flagged across Shewhart, EWMA, and CUSUM.

---

## 5. Five-Minute Viva Demo Walkthrough Script

Use this exact sequence of clicks and spoken points during the oral examination:

### Minute 1: System Overview & Problem Statement (`/overview`)
- **Action**: Open `http://localhost:5173`. Show the 4 top KPI cards (Total Spend, Total Requests, p50/p95 Latency, Error Rate).
- **Spoken**: *"LLMetrics is a predictive telemetry engine for LLM APIs. Here on the Overview, we see an ingested 90-day dataset of 86,115 requests with a 14-day period-over-period comparison. In the center is our deterministic executive summary explaining that spend increased by 12% driven by a 400ms latency regression on Day 55."*

### Minute 2: Simple & Multiple Linear Regression (`/regression`)
- **Action**: Click on **Regression**. Show the Multiple Regression tab first, then toggle to Simple Regression.
- **Spoken**: *"In Simple Regression, we establish that completion tokens physically drive latency ($R^2 = 0.591, \beta_1 = 1.15\text{ ms/token}, p < 0.001$). Notice the 95% confidence band for mean latency versus the wider prediction interval for individual requests."*
- **Spoken**: *"Switching to Multiple Regression, our adjusted $R^2$ jumps to 0.838. We explicitly test the Gauss-Markov assumptions: Breusch-Pagan test yields $p < 0.001$, proving heteroscedasticity due to token-dependent variance. VIF is below 2.5, proving no collinearity, and Cook's distance highlights 14 high-leverage outliers. Stepwise AIC selected our full feature set."*

### Minute 3: Failure Prediction, Trees & Text Features (`/reliability`)
- **Action**: Click on **Reliability**.
- **Spoken**: *"To predict failure and timeout risk, we compare three paradigms on an identical 30% holdout split: Logistic Regression, CART, and CHAID. Logistic regression gives an odds ratio of 1.15 per 100ms of latency, with a well-calibrated Hosmer-Lemeshow test ($p = 0.276$). CART and our native Chi-square CHAID tree output explicit production rules, e.g., high latency during peak hours yields a 45% failure rate. In the text features section, our regex pipeline extracts code blocks, showing a 1.85x risk increase when users submit code snippets."*

### Minute 4: Forecasting & Time Series Analysis (`/forecast`)
- **Action**: Click on **Forecast**. Switch target between Spend and Token Volume, toggle forecast horizon (7, 14, 30 days).
- **Spoken**: *"Here we forecast daily spend and token consumption. We perform rolling-origin backtesting across Naive, Seasonal Naive, Holt-Winters, and a Causal Ridge model with calendar regressors. The Causal model wins with 8.42% MAPE. Down below, we check stationarity using the Augmented Dickey-Fuller test ($p = 0.012$), plot ACF/PACF with 95% Bartlett bounds, fit an optimal ARIMA(1,0,2) via AIC, and test residual independence via Ljung-Box ($p = 0.412$)."*
- **Action**: Point out the **Budget Projection** card showing month-end projection and breach probability.

### Minute 5: Statistical Quality Control & Model-Swap Detection (`/alerts`)
- **Action**: Click on **Alerts**, then show **Methodology**.
- **Spoken**: *"In the Alerts view, we deploy Shewhart ($I-MR$), EWMA, and CUSUM control charts to detect sudden outages and gradual cost drifts. Most crucially, we implemented a rolling two-sample Kolmogorov-Smirnov detector on output token distributions. It successfully caught our injected silent model swap on Day 35 within 3.2 hours with zero false alarms. Finally, our Methodology tab provides the complete academic traceability matrix linking every formula, test, and code file directly to the course syllabus."*

---

## 6. Illustrative Pricing Notice

> **Important**: Per-model pricing values defined in `core/data/config.py` (`gpt-4o`, `gpt-4o-mini`, `claude-3-5-sonnet`) are **illustrative placeholders** for academic demonstration. Users deploying LLMetrics to production must supply their provider's current negotiated contract rates in `core/data/config.py`.

---

## 7. License & Academic Integrity

Developed for the **23XWA8 Predictive Analytics** curriculum, M.Sc Software Systems.  
All analytical code, statistical tests, and machine-learning routines were authored for this project and validated using automated regression tests and empirical benchmarks.

