# LLMetrics: Viva Examination Cheat-Sheet
**Course**: 23XWA8 Predictive Analytics (M.Sc Software Systems)  
**Project**: LLMetrics — Cost, Latency, and Reliability Monitor for LLM Applications  

This document serves as an exhaustive reference for the oral viva examination. It details each syllabus topic, the theoretical justification for its selection, underlying assumptions, interpretation of diagnostic tests, and how each is implemented and verified in the codebase.

---

## 1. Data Wrangling & Exploratory Data Analysis (EDA)

### What is it?
The ingestion, validation, cleaning, and empirical characterization of raw LLM API telemetry data across univariate, bivariate, and multivariate distributions.

### Why in LLM Telemetry?
Raw LLM logs contain missing latency values (timeouts), duplicate requests due to client-side retries, and extreme outliers caused by network delays or provider throttling. EDA establishes baseline operational parameters before fitting parametric models.

### Key Techniques & Concepts
- **Missing Value Handling**: Distinguish between structural missingness (e.g. `completion_tokens` missing when status is `timeout`) versus random drops. In LLMetrics, missing latencies are imputed via group-conditional medians (grouped by model and status) to avoid distorting downstream regressions.
- **Deduplication**: Identifies retry duplicates on `(timestamp_minute, prompt_tokens, model)` tuples.
- **Outlier Detection**:
  - *Interquartile Range (IQR)*: Outliers defined as $x < Q_1 - 1.5 \times \text{IQR}$ or $x > Q_3 + 1.5 \times \text{IQR}$.
  - *Z-score*: Identifies extreme latency observations ($|z| > 3.0$).
- **Distributions**: Right-skewed token distributions and bimodal latency distributions (cache hits vs cache misses).
- **Correlation Matrix**: Pearson correlation between numerical features (`prompt_tokens`, `completion_tokens`, `latency_ms`, `cost_usd`).

### Typical Viva Questions & Answers
- **Q**: *Why use median imputation rather than mean imputation for latency?*  
  **A**: Latency distributions in distributed systems follow heavy-tailed log-normal distributions. The sample mean is heavily biased upward by catastrophic timeouts; the median preserves the central tendency without inflating normal operating estimates.
- **Q**: *How do you prevent data leakage during wrangling?*  
  **A**: Cleaning transformations (such as imputation medians and outlier cutoffs) are computed strictly on the training partition and applied downstream to test sets.

---

## 2. Simple Linear Regression

### Model Specification
$$\text{latency\_ms}_i = \beta_0 + \beta_1 \cdot \text{completion\_tokens}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

### Why in LLM Telemetry?
Autoregressive Large Language Models generate text token by token. Each token requires a forward pass through the transformer decoder. Hence, response time has a direct physical linear dependency on completion token length.

### Assumptions & Diagnostics
1. **Linearity**: The expected value of latency is a linear function of output tokens. Verified via Residuals vs Fitted plot (residuals should scatter symmetrically around zero with no curved trend).
2. **Homoscedasticity**: Error variance $\text{Var}(\epsilon_i) = \sigma^2$ is constant. (In practice, autoregressive generation exhibits increasing variance, which is explicitly detected by Breusch-Pagan).
3. **Normality of Residuals**: Assessed via Q-Q (Quantile-Quantile) plot against standard normal quantiles.
4. **Independence**: Residuals are uncorrelated.

### Statistics Reported
- **Slope ($\beta_1$)**: Marginal latency per token (typically ~1.0 to 1.5 ms/token in baseline models).
- **t-statistic & p-value**: Tests $H_0: \beta_1 = 0$ vs $H_1: \beta_1 \neq 0$. $p < 0.001$ confirms token count is a statistically significant predictor of latency.
- **$R^2$**: Proportion of variance in latency explained by completion tokens alone (empirically ~0.55 to 0.65 in un-cached calls).
- **Confidence Interval (95%)**: Uncertainty around the estimated mean response line:
  $$\hat{y} \pm t_{n-2, 0.025} \cdot \text{SE}(\hat{y})$$
- **Prediction Interval (95%)**: Uncertainty for an individual future request, including irreducible error $\sigma$:
  $$\hat{y} \pm t_{n-2, 0.025} \cdot \sqrt{\text{SE}(\hat{y})^2 + s^2}$$

### Typical Viva Questions & Answers
- **Q**: *Why is the prediction interval always wider than the confidence interval?*  
  **A**: The confidence interval only accounts for estimation uncertainty of the parameters $\beta_0, \beta_1$. The prediction interval accounts for both parameter estimation uncertainty AND the intrinsic random variance $\sigma^2$ of individual future observations.
- **Q**: *What does the intercept $\beta_0$ represent physically?*  
  **A**: The fixed baseline overhead: TLS handshake, network round-trip time, prompt ingestion/pre-fill time, and initial scheduling latency before the first token is generated.

---

## 3. Multiple Linear Regression

### Model Specification
$$\text{latency\_ms} = \beta_0 + \beta_1 \text{prompt\_tokens} + \beta_2 \text{completion\_tokens} + \sum \beta_k \text{model}_k + \beta_{\text{peak}} \text{is\_peak\_hour} + \beta_{\text{cache}} \text{cache\_hit} + \epsilon$$

### Why in LLM Telemetry?
Single regression ignores confounding variables: prompt pre-fill time, model architecture variations, server queuing during peak traffic hours, and CDN/semantic cache hits.

### Diagnostic Battery & Statistical Tests
1. **Interpretation of Coefficients**:
   - `completion_tokens`: Decoding generation cost per token.
   - `prompt_tokens`: Attention prefill cost (typically an order of magnitude faster than generation due to parallel processing).
   - `model_*`: Categorical dummy coefficients showing relative latency shifts against reference model (`gpt-4o-mini`).
   - `is_peak_hour`: Concurrency penalty (positive coefficient).
   - `cache_hit`: Cache bypass benefit (large negative coefficient, cutting latency by ~80-90%).
2. **Adjusted $R^2$**:
   $$R^2_{\text{adj}} = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$
   Penalizes the addition of non-informative regressors.
3. **Breusch-Pagan Test**:
   - $H_0$: Error variance is homoscedastic ($\sigma_i^2 = \sigma^2$).
   - In our generator, variance grows with $\sqrt{\text{tokens}}$. Breusch-Pagan yields $p < 0.001$, decisively rejecting homoscedasticity and confirming the presence of heteroscedasticity.
4. **Variance Inflation Factor (VIF)**:
   $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
   Measures multicollinearity. Values $< 5.0$ indicate absence of harmful collinearity.
5. **Cook's Distance**:
   $$D_i = \frac{\sum (\hat{y}_j - \hat{y}_{j(i)})^2}{(p + 1) s^2}$$
   Flags influential leverage points ($D_i > \frac{4}{n}$).
6. **Durbin-Watson Test**:
   $$DW = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2}$$
   Detects first-order autocorrelation in residuals. $DW \approx 2.0$ confirms uncorrelated errors.
7. **Log-Transform Comparison**:
   Fits $\ln(\text{latency}) \sim \dots$ to stabilize variance and linearize multiplicative effects. Compared via AIC.
8. **Stepwise Model Selection by AIC**:
   Iteratively evaluates Akaike Information Criterion ($\text{AIC} = 2k - 2\ln(\hat{L})$) using forward-backward elimination to identify the most parsimonious feature subset.

---

## 4. Logistic Regression (Binary & Multinomial)

### 4.1 Binary Logistic Regression
Predicts failure probability: $Y \in \{0 = \text{Success}, 1 = \text{Failure/Timeout}\}$.
$$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 \text{latency\_ms} + \beta_2 \text{prompt\_tokens} + \beta_3 \text{is\_peak\_hour}$$

- **Odds Ratio ($\text{OR} = e^{\beta}$)**: A unit increase in predictor multiplies the odds of failure by $e^{\beta}$.
- **Wald Test**: $z = \frac{\hat{\beta}}{\text{SE}(\hat{\beta})}$, tests significance of individual log-odds coefficients.
- **Hosmer-Lemeshow Test**: Goodness-of-fit test partitioning predictions into 10 deciles of risk and comparing observed vs expected frequencies via $\chi^2$. Non-significant $p > 0.05$ indicates good calibration.
- **ROC-AUC & Gini**: Area Under the ROC Curve measures discrimination capacity across thresholds. Gini coefficient is defined as:
  $$\text{Gini} = 2 \times \text{AUC} - 1$$

### 4.2 Multinomial Logistic Regression
Predicts categorical outcomes with $K > 2$ discrete categories:
$$Y \in \{\text{ok}, \text{timeout}, \text{error}, \text{rate\_limited}\}$$
Takes `ok` as the reference class, fitting $K - 1$ log-odds equations to output Relative Risk Ratios ($\text{RRR} = e^\beta$).

---

## 5. Decision Trees (CART & CHAID) & Text Features

### 5.1 CART (Classification and Regression Trees)
- Uses Gini impurity for classification:
  $$I_G(t) = 1 - \sum_{i=1}^C p(i|t)^2$$
- Pruned via cost-complexity pruning ($\alpha$) or max depth to generate human-readable production decision rules:
  `IF latency_ms > 1200 AND is_peak_hour = 1 THEN P(Fail) = 0.42`

### 5.2 CHAID (Chi-square Automatic Interaction Detection)
- Non-parametric recursive partitioning using Pearson's Chi-square test ($\chi^2 = \sum \frac{(O - E)^2}{E}$) on binned predictors.
- Continuous predictors are discretized into quantile bins.
- Splits only if the minimum p-value across candidate predictors is below the significance threshold $\alpha = 0.05$, with Bonferroni correction for multi-category testing.
- Self-stopping: does not require post-pruning.

### 5.3 Comparative Hold-Out Evaluation
Logistic Regression, CART, and CHAID are evaluated on the exact same 30% held-out test split, reporting:
- Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Trade-off analysis: Logistic provides smooth probability gradients; CART offers explicit binary orthogonal rules; CHAID reveals significant categorical interactions without distributional assumptions.

### 5.4 Unstructured Text Feature Analysis
Derives explicit semantic and structural indicators from the unstructured `prompt_text`:
- `prompt_char_length`: Character length.
- `has_code_block`: Boolean indicator of markdown triple backticks (```).
- `has_json_schema`: Presence of JSON structure instructions.
- `has_error_keywords`: Presence of terms like `traceback`, `fail`, `debug`.
Demonstrates empirical correlation between code-heavy/complex inputs and elevated timeout rates.

---

## 6. Forecasting & Rolling-Origin Backtesting

### 6.1 Forecasting Models
Evaluated on daily aggregated token volume and spend:
1. **Naive Baseline**: $\hat{y}_{t+h} = y_t$
2. **Seasonal Naive**: $\hat{y}_{t+h} = y_{t+h - 7}$ (repeating last week's corresponding day)
3. **Moving Average (MA)**: Rolling $k$-day window mean.
4. **Simple Exponential Smoothing (SES)**: Level smoothing parameter $\alpha$.
5. **Holt's Linear Trend**: Level ($\alpha$) + Trend ($\beta$) equations.
6. **Holt-Winters Exponential Smoothing**: Level ($\alpha$) + Trend ($\beta$) + Seasonal ($\gamma$) with 7-day periodicity.
7. **Causal Model (ARIMAX / Exogenous Ridge)**: Autoregressive lags supplemented with calendar regressors (`is_weekend`, `day_of_week`) and volume covariates.

### 6.2 Rolling-Origin Backtesting
Evaluates forecast accuracy without lookahead bias across expanding cross-validation splits:
- **MAE**: $\frac{1}{H} \sum |y_t - \hat{y}_t|$ (Scale-dependent mean absolute error)
- **RMSE**: $\sqrt{\frac{1}{H} \sum (y_t - \hat{y}_t)^2}$ (Penalizes large errors heavily)
- **MAPE**: $\frac{100\%}{H} \sum \left|\frac{y_t - \hat{y}_t}{y_t}\right|$ (Scale-free relative percentage error)

---

## 7. Time Series Analysis & Vector Autoregression (VAR)

### 7.1 Stationarity & Order Selection
- **Augmented Dickey-Fuller (ADF) Test**:
  - $H_0$: Series possesses a unit root (non-stationary).
  - If $p > 0.05$, the series is differenced ($d = 1$) until stationarity is achieved.
- **Autocorrelation (ACF) & Partial Autocorrelation (PACF)**:
  - Computed with Bartlett's 95% confidence bands ($\pm 1.96 / \sqrt{N}$).
  - ACF cutoff guides MA order $q$; PACF cutoff guides AR order $p$.
- **Auto-ARIMA by AIC**:
  - Grid search over $(p, d, q) \in [0..3] \times [0..1] \times [0..3]$ minimizing AIC.
- **Ljung-Box Test on Residuals**:
  - $H_0$: Residuals are independently distributed (white noise).
  - $p > 0.05$ indicates adequate model specification with no remaining autocorrelation.

### 7.2 Vector Autoregression (VAR)
Models the dynamic joint interaction between three endogenous variables:
$$\mathbf{Y}_t = \begin{bmatrix} \text{cost}_t \\ \text{latency}_t \\ \text{request\_count}_t \end{bmatrix} = \mathbf{c} + \mathbf{A}_1 \mathbf{Y}_{t-1} + \dots + \mathbf{A}_p \mathbf{Y}_{t-p} + \boldsymbol{\epsilon}_t$$
- **Granger Causality**: Tests whether past values of variable $X$ contain information that helps predict variable $Y$ beyond past values of $Y$ alone.

---

## 8. Statistical Quality Control (SQC) & Model-Swap Detection

### 8.1 Control Charts
- **Shewhart Individuals & Moving Range ($I-MR$)**:
  - Upper Control Limit $\text{UCL} = \bar{X} + 3 \frac{\bar{MR}}{d_2}$
  - Flags sudden, high-magnitude isolated shocks (e.g. outage spikes).
- **EWMA (Exponentially Weighted Moving Average)**:
  - $Z_i = \lambda X_i + (1 - \lambda) Z_{i-1}$ with $\lambda = 0.2, L = 3.0$.
  - Rapidly detects moderate persistent shifts (~$1\sigma$ to $2\sigma$).
- **CUSUM (Cumulative Sum Control Chart)**:
  - $C_i^+ = \max(0, X_i - (\mu_0 + K) + C_{i-1}^+)$
  - Tuned with reference value $k = 0.5\sigma$ and decision interval $h = 5.0\sigma$.
  - Optimal detector for subtle, sustained mean drifts.

### 8.2 Rolling Kolmogorov-Smirnov (K-S) Model-Swap Detector
- **Problem**: Upstream model providers occasionally route queries to smaller, quantized, or different model checkpoints without updating model tags.
- **Mechanism**: Maintains a sliding evaluation window (e.g. last 100 requests) and compares its empirical cumulative distribution function (ECDF) of `completion_tokens` or `latency_ms` against a golden baseline window using the two-sample K-S statistic:
  $$D = \sup_x |F_{\text{baseline}}(x) - F_{\text{recent}}(x)|$$
- Flags a silent swap when $D \ge D_{\text{crit}}$ and $p < 0.01$.

### 8.3 Injected Incident Evaluation Scorecard
All detectors are evaluated against ground-truth incident records:
1. **Detection Delay**: Hours elapsed between incident start and first detector alarm.
2. **False Alarm Rate (FAR)**: Number of false positive alarms raised outside declared incident windows.

---

## 9. System Architecture & Extensions

- **FastAPI + SQLite WAL**: Invalidation cache keyed on dataset hash ensures all mathematical calculations execute once and respond in under 5ms on subsequent requests.
- **Budget Alerting**: Calculates projected month-end cumulative spend using forecast trajectory, computes breach probability via normal approximation of cumulative forecast variance:
  $$P(\text{Spend} > \text{Budget}) = 1 - \Phi\left(\frac{\text{Budget} - \hat{\mu}_{\text{end}}}{\hat{\sigma}_{\text{end}}}\right)$$
- **Weekly Insights Generator**: Fully deterministic template generating structured English summaries from computed stats, with optional Anthropic Claude API enrichment when `ANTHROPIC_API_KEY` is present.
