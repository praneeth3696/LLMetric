# LLMetrics: Methodology and Architecture Documentation
**Course**: 23XWA8 Predictive Analytics (M.Sc Software Systems)  
**System**: LLMetrics — Enterprise Cost, Latency & Reliability Monitoring for LLM APIs

---

## 1. System Overview & Problem Formulation

Large Language Model (LLM) APIs such as OpenAI, Anthropic, and open-source endpoints hosted on vLLM or Ollama exhibit distinct operational dynamics compared to traditional microservices:
1. **Variable Latency**: Autoregressive decoding generates text token-by-token, causing latency to scale linearly with output token length, coupled with concurrency queuing and input pre-fill attention overhead.
2. **Asymmetric Cost Dynamics**: Pricing depends on token counts rather than raw query volume or CPU clock time, with input and output tokens billed at substantially different rates.
3. **Complex Failure Modes**: Requests fail not merely due to network errors, but due to context window overruns, rate limiting, and timeouts on large completions.
4. **Distribution Shifts & Silent Changes**: Upstream model providers occasionally route queries to smaller, quantized, or upgraded model checkpoints without updating model tags, altering response characteristics.

LLMetrics provides an end-to-end telemetry and predictive analytics platform addressing these operational challenges using rigorous statistical methodologies mandated by the 23XWA8 Predictive Analytics syllabus.

---

## 2. Data-Generating Process (DGP)

LLMetrics includes a high-fidelity synthetic log generator (`core/data/generator.py`) designed to reflect genuine production LLM traffic characteristics with deterministic reproducibility:

### 2.1 Traffic Volume & Seasonality
Daily request volumes exhibit diurnal (peak hours 09:00 - 18:00 UTC) and weekly seasonality (weekend drop of ~35%). Request intervals follow a non-homogeneous Poisson process.

### 2.2 Latency Physics & Heteroscedastic Noise
Latency is generated according to an autoregressive model accounting for prompt pre-fill, decoding rate, model architecture, peak-hour server queuing, and cache acceleration:
$$\text{latency\_ms}_i = \beta_0 + \beta_{\text{prompt}} \cdot \text{prompt\_tokens}_i + \beta_{\text{comp}} \cdot \text{completion\_tokens}_i + \delta_{\text{model}} + \Delta_{\text{peak}} \cdot \mathbb{I}(\text{peak}) - \Delta_{\text{cache}} \cdot \mathbb{I}(\text{cache}) + \epsilon_i$$
Crucially, the error term variance scales with token count:
$$\epsilon_i \sim \mathcal{N}\left(0, \sigma_0^2 + \gamma \cdot \text{completion\_tokens}_i\right)$$
This ensures that the simulated telemetry exhibits genuine heteroscedasticity, enabling rigorous empirical validation of the Breusch-Pagan test.

### 2.3 Failure Probability Sigmoid
Request failures and timeouts are generated via a logistic link function:
$$P(\text{Failure}_i) = \frac{1}{1 + \exp\left(-\left(\alpha_0 + \alpha_{\text{lat}} \cdot \text{latency\_ms}_i + \alpha_{\text{prompt}} \cdot \text{prompt\_tokens}_i + \alpha_{\text{peak}} \cdot \mathbb{I}(\text{peak})\right)\right)}$$

### 2.4 Injected Ground-Truth Incidents
To objectively evaluate Statistical Quality Control (SQC) charts and change-point detectors, four specific incidents are synthesized with ground-truth intervals:
1. **Incident A (Day 35–42)**: Silent Model Swap on `gpt-4o-mini` (output token distribution shifts downward, latency drops by 30%).
2. **Incident B (Day 55–57)**: 2-day Latency Regression (+400ms server processing baseline across all models).
3. **Incident C (Day 70–75)**: Gradual Cost Drift (input token inflation of +25% due to prompt template bloating).
4. **Incident D (Day 78 14:00 – Day 79 02:00)**: Failure Spike (transient provider degradation elevating timeout probability by $4\times$).

---

## 3. Syllabus Topic Implementations

### Syllabus 1: Data Wrangling & Exploratory Data Analysis
- **Module**: `core/eda/analysis.py`, `core/data/cleaner.py`
- Ingests raw logs, performs deduplication on client retries, imputes missing values via group-conditional medians (grouped by model and status), and identifies outliers using Interquartile Range ($1.5 \times \text{IQR}$) and Z-score ($|z| > 3.0$).
- Generates univariate distributions (histograms, boxplots), bivariate relationships (scatter plots, correlations), and multivariate interaction matrices.

### Syllabus 2: Simple Linear Regression
- **Module**: `core/regression/simple.py`
- Fits $\text{latency\_ms} = \beta_0 + \beta_1 \cdot \text{completion\_tokens} + \epsilon$.
- Reports $R^2$, slope coefficient, standard error, $t$-statistic, and two-sided $p$-value.
- Provides residual diagnostics: Residuals vs. Fitted values and Quantile-Quantile (Q-Q) normal plots.
- Computes both 95% Confidence Intervals (mean response) and 95% Prediction Intervals (individual observation response).

### Syllabus 3: Multiple Linear Regression & Econometric Diagnostics
- **Module**: `core/regression/multiple.py`
- Fits full specification:
  $$\text{latency\_ms} \sim \text{prompt\_tokens} + \text{completion\_tokens} + \text{model} + \text{is\_peak\_hour} + \text{cache\_hit}$$
- Reports Adjusted $R^2$, coefficient interpretations, and degrees of freedom.
- **Diagnostic Battery**:
  - *Breusch-Pagan Test*: Lagrange Multiplier (LM) test for heteroscedasticity.
  - *Variance Inflation Factor (VIF)*: Multicollinearity diagnostic.
  - *Cook's Distance*: Influence and leverage diagnostic ($D_i > 4/n$).
  - *Durbin-Watson*: Autocorrelation of residuals.
  - *Log-Transform Comparison*: Compares linear model vs $\ln(\text{latency})$ log-linear model via AIC.
  - *Stepwise Selection*: Automated bidirectional stepwise variable selection guided by AIC.

### Syllabus 4: Logistic Regression (Binary & Multinomial)
- **Module**: `core/classification/logistic.py`, `core/classification/multinomial.py`
- **Binary Logistic**:
  - Predicts $P(\text{Status} \in \{\text{timeout}, \text{error}, \text{rate\_limited}\})$.
  - Reports coefficients, Odds Ratios ($\text{OR} = e^{\beta}$), Wald tests ($z, p$), Hosmer-Lemeshow goodness-of-fit test, Confusion Matrix, ROC-AUC, and Gini coefficient ($2 \cdot \text{AUC} - 1$).
- **Multinomial Logistic**:
  - Models categorical status $Y \in \{\text{ok}, \text{timeout}, \text{error}, \text{rate\_limited}\}$ with `ok` as the reference level.
  - Reports Relative Risk Ratios (RRRs) and multi-class confusion matrix.

### Syllabus 5: Decision Trees & Unstructured Text Features
- **Module**: `core/trees/cart.py`, `core/trees/chaid.py`, `core/trees/text_features.py`, `core/trees/comparison.py`
- **CART**: Fits classification tree via scikit-learn with cost-complexity pruning and produces human-readable production rules.
- **CHAID**: Implements native Chi-Square Automatic Interaction Detection using `scipy.stats.chi2_contingency` on quantile-binned continuous and categorical predictors with Bonferroni correction.
- **Model Comparison**: Benchmarks Logistic Regression vs CART vs CHAID on an identical 30% holdout test split.
- **Text Feature Extraction**: Extracts semantic attributes from prompt text (character length, code blocks, JSON schema requests, error keywords) and measures their statistical impact on failure rates.

### Syllabus 6: Forecasting & Rolling-Origin Backtesting
- **Module**: `core/forecasting/baselines.py`, `core/forecasting/smoothing.py`, `core/forecasting/causal.py`, `core/forecasting/backtest.py`
- Models daily spend and token volume using:
  1. Naive Baseline ($\hat{y}_{t+h} = y_t$)
  2. Seasonal Naive Baseline ($\hat{y}_{t+h} = y_{t+h-7}$)
  3. Moving Average (MA)
  4. Simple Exponential Smoothing (SES)
  5. Holt's Linear Exponential Smoothing
  6. Holt-Winters Seasonal Exponential Smoothing (additive/multiplicative)
  7. Causal Regression (ARIMAX / Exogenous Ridge with calendar and volume features)
- Evaluates models via expanding-window rolling-origin backtesting, reporting MAE, RMSE, and MAPE alongside out-of-sample prediction intervals.

### Syllabus 7: Time Series Analysis & Vector Autoregression
- **Module**: `core/timeseries/stationarity.py`, `core/timeseries/arima.py`, `core/timeseries/var.py`
- **Stationarity**: Augmented Dickey-Fuller (ADF) test; Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) with 95% Bartlett bounds.
- **ARIMA**: Grid search order selection $(p, d, q)$ optimizing AIC; residual diagnostics via Ljung-Box test for white noise.
- **Vector Autoregression (VAR)**: Dynamic multivariate simultaneous equation system modeling joint interactions between cost, latency, and request volume with pairwise Granger Causality tests.

### Syllabus 8: Statistical Quality Control & Model-Swap Detection
- **Module**: `core/qc/charts.py`, `core/qc/model_swap.py`, `core/qc/evaluation.py`
- **Control Charts**:
  - *Shewhart $I-MR$*: Detects large isolated shocks.
  - *EWMA ($\lambda=0.2, L=3.0$)*: Detects moderate sustained shifts.
  - *CUSUM ($k=0.5\sigma, h=5.0\sigma$)*: Detects subtle cumulative drifts.
- **Model-Swap Detector**: Rolling two-sample Kolmogorov-Smirnov test on output token lengths ($D \ge D_{\text{crit}}, p < 0.01$).
- **Incident Evaluation Scorecard**: Empirically computes detection delay (hours) and False Alarm Rate (FAR) against the ground-truth incident log.

---

## 4. Software Architecture & Persistence

- **Backend**: FastAPI running on Python 3.11 with asynchronous endpoint handlers, Pydantic schemas, and CORS security middleware.
- **Persistence**: SQLite configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and foreign keys enabled.
- **Performance Caching**: Two-tier caching in SQLite (`analysis_cache` table) keyed by dataset hash and section name. Ensures analytical GET endpoints respond in $< 5$ milliseconds on cache hits.
- **Frontend**: React 18 + TypeScript + Vite styled with a calm, neutral, information-dense theme utilizing system font stacks, zero animations, and Recharts visualization components.
