import React from 'react';

export const MethodologyView: React.FC = () => {
  const syllabusMap = [
    {
      unit: '1. Data Wrangling & EDA',
      topics: 'Ingest, clean (missing values, duplicates, outliers), univariate, bivariate, multivariate analysis.',
      uiLocation: 'Overview View & EDA Modal',
      codeFiles: 'core/data/generator.py, core/data/cleaner.py, core/eda/analysis.py',
      apiEndpoint: 'POST /data/generate, POST /data/upload, GET /eda',
      theoryFormula: 'IQR = Q₃ - Q₁; Z = (x - μ) / σ; Pearson r = Cov(X,Y) / (σx σy); Spearman ρ.',
      vivaTip: 'Demonstrates data hygiene, handling missing values via median/mode imputation, and detecting extreme outliers.',
    },
    {
      unit: '2. Simple Linear Regression',
      topics: 'latency_ms ~ completion_tokens. R², t-test, p-value, residual analysis (residuals vs fitted, Q-Q plot), confidence & prediction intervals.',
      uiLocation: 'Why is it slow/expensive -> Simple Regression Tab',
      codeFiles: 'core/regression/simple.py',
      apiEndpoint: 'GET /regression/simple',
      theoryFormula: 'ŷ = β₀ + β₁X; t = (b₁ - β₁) / SE(b₁); Prediction Interval = ŷ ± t_{crit} * s_e * √(1 + 1/n + (x-x̄)²/SS_x).',
      vivaTip: 'Explains why individual request prediction intervals are wider than mean confidence intervals due to random latency variance.',
    },
    {
      unit: '3. Multiple Linear Regression',
      topics: 'latency_ms ~ prompt_tok + comp_tok + model + peak + cache. Adjusted R², Breusch-Pagan, VIF, Cook\'s D, log-transform, Durbin-Watson, Stepwise AIC.',
      uiLocation: 'Why is it slow/expensive -> Multiple Regression Tab',
      codeFiles: 'core/regression/multiple.py',
      apiEndpoint: 'GET /regression/multiple',
      theoryFormula: 'LM = n * R²_aux ~ χ²(p); VIF_j = 1 / (1 - R²_j); Cook\'s D = e_i² / (p * s²) * [h_ii / (1 - h_ii)²]; DW = ∑(e_t - e_{t-1})² / ∑e_t².',
      vivaTip: 'Breusch-Pagan rejects homoscedasticity because token noise physically expands with payload size; log-transform mitigates skewness.',
    },
    {
      unit: '4. Logistic Regression',
      topics: 'Binary P(fails/timeouts): Odds ratios, Wald test, Hosmer-Lemeshow, confusion matrix, ROC-AUC, Gini (2*AUC - 1). Multinomial logistic regression.',
      uiLocation: 'Reliability & Failure Prediction View',
      codeFiles: 'core/classification/logistic.py, core/classification/multinomial.py',
      apiEndpoint: 'GET /reliability',
      theoryFormula: 'logit(p) = ln(p/(1-p)) = Xβ; OR = e^β; Wald = (β/SE)²; HL = ∑ (O_k - E_k)² / [E_k(1 - E_k/n_k)]; Gini = 2*AUC - 1.',
      vivaTip: 'Odds ratio shows multiplicative change in failure risk per 100ms latency or peak hour concurrency.',
    },
    {
      unit: '5. Decision Trees & Unstructured Text',
      topics: 'CART (scikit-learn) with human-readable rules. CHAID implementation using scipy chi-square contingency splits. CART vs CHAID vs Logistic benchmark. Text feature derivation.',
      uiLocation: 'Reliability & Failure Prediction View (Rules & Comparison Cards)',
      codeFiles: 'core/trees/cart.py, core/trees/chaid.py, core/trees/text_features.py, core/trees/comparison.py',
      apiEndpoint: 'GET /reliability',
      theoryFormula: 'CHAID: χ² = ∑ (O_ij - E_ij)² / E_ij; splits selected by minimum p-value; Relative Risk = P(Fail | feat=1) / P(Fail | feat=0).',
      vivaTip: 'CHAID produces multiway splits based on statistical independence, whereas CART uses binary Gini impurity minimization.',
    },
    {
      unit: '6. Time-Series Forecasting',
      topics: 'Moving average, simple/Holt/Holt-Winters exponential smoothing, causal model with exogenous regressors (is_weekend, traffic volume). Rolling-origin backtest (MAE, RMSE, MAPE).',
      uiLocation: 'Forecast & Budget View',
      codeFiles: 'core/forecasting/baselines.py, core/forecasting/smoothing.py, core/forecasting/causal.py, core/forecasting/backtest.py',
      apiEndpoint: 'GET /forecast',
      theoryFormula: 'SES: ŷ_{t+1} = α y_t + (1-α)ŷ_t; Holt-Winters: L_t + h b_t + s_{t+h-m}; Rolling Origin CV avoids lookahead bias.',
      vivaTip: 'Rolling-origin backtesting evaluates models strictly on unseen out-of-sample steps across historical time slices.',
    },
    {
      unit: '7. Time Series Analysis',
      topics: 'Stationarity (ADF test, ACF/PACF with Bartlett bounds), ARIMA order selection by AIC with Ljung-Box white noise test, and multivariate VAR on cost, latency, and volume.',
      uiLocation: 'Forecast View & CLI figures',
      codeFiles: 'core/timeseries/stationarity.py, core/timeseries/arima.py, core/timeseries/var.py',
      apiEndpoint: 'GET /forecast, core.run_all figures',
      theoryFormula: 'ADF: Δy_t = α + βt + γy_{t-1} + ∑ δ_i Δy_{t-i} + e_t; Ljung-Box: Q = n(n+2) ∑ r_k²/(n-k); VAR(p): Y_t = c + A₁Y_{t-1} + ... + e_t.',
      vivaTip: 'Granger causality tests whether past request volumes contain statistically significant information to predict latency and spend spikes.',
    },
    {
      unit: '8. Statistical Quality Control',
      topics: 'Shewhart (3-sigma), EWMA (λ=0.2, L=3), and CUSUM (k=0.5, h=5) charts on latency and cost. Rolling two-sample Kolmogorov-Smirnov model-swap detector scored on ground-truth incidents.',
      uiLocation: 'Alerts & Quality Control View',
      codeFiles: 'core/qc/charts.py, core/qc/model_swap.py, core/qc/evaluation.py',
      apiEndpoint: 'GET /alerts',
      theoryFormula: 'Shewhart: μ ± 3σ; EWMA: z_t = λx_t + (1-λ)z_{t-1}; CUSUM: C⁺_t = max(0, x_t - μ - k + C⁺_{t-1}); K-S: D = sup_x |F₁(x) - F₂(x)|.',
      vivaTip: 'CUSUM detects subtle persistent regressions faster than Shewhart; K-S test catches distribution shifts without assuming normality.',
    },
    {
      unit: 'Extension: Budget & Insights',
      topics: 'Projected month-end spend with 95% prediction interval and analytical breach probability; plain-language template summary with optional Anthropic Claude rewrite.',
      uiLocation: 'Overview & Forecast Views',
      codeFiles: 'core/reporting/budget.py, core/reporting/insights.py',
      apiEndpoint: 'POST /budget, GET /budget, GET /insights',
      theoryFormula: 'Z = (Budget - Projected_Spend) / σ_{cumulative}; P(Breach) = 1 - Φ(Z).',
      vivaTip: 'Deterministic template ensures the dashboard never fails even without external API credentials.',
    },
  ];

  return (
    <div>
      <div className="technique-tag">23XWA8 Predictive Analytics Course Syllabus Traceability</div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Syllabus-to-Code Traceability Matrix</h3>
            <p className="card-subtitle">
              Every syllabus topic is implemented in pure-Python core analytics, exposed via FastAPI REST endpoints, and visualized in the UI.
            </p>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '180px' }}>Syllabus Unit</th>
                <th style={{ width: '220px' }}>Key Analytical Topics</th>
                <th>UI Location</th>
                <th>Core Python Implementation</th>
                <th>Mathematical Formulation</th>
                <th>Viva Examination Relevance</th>
              </tr>
            </thead>
            <tbody>
              {syllabusMap.map((item) => (
                <tr key={item.unit}>
                  <td style={{ fontWeight: 700, color: 'var(--accent)' }}>{item.unit}</td>
                  <td style={{ fontSize: '0.8rem' }}>{item.topics}</td>
                  <td style={{ fontWeight: 600 }}>{item.uiLocation}</td>
                  <td className="mono" style={{ fontSize: '0.72rem', color: '#0369a1' }}>
                    {item.codeFiles}
                    <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>{item.apiEndpoint}</div>
                  </td>
                  <td className="mono" style={{ fontSize: '0.72rem' }}>{item.theoryFormula}</td>
                  <td style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{item.vivaTip}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

