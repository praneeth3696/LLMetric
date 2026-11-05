import React from 'react';
import { ReliabilityData } from '../api';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface ReliabilityViewProps {
  reliability: ReliabilityData | null;
  isLoading: boolean;
}

export const ReliabilityView: React.FC<ReliabilityViewProps> = ({
  reliability,
  isLoading,
}) => {
  if (isLoading && !reliability) {
    return (
      <div className="loading-panel">
        <p className="empty-title">Evaluating Failure Prediction Models...</p>
        <p className="empty-desc">Computing Logistic Regression, CART, CHAID, and prompt text features.</p>
      </div>
    );
  }

  if (!reliability) {
    return (
      <div className="empty-panel">
        <h3 className="empty-title">No Reliability Data Available</h3>
        <p className="empty-desc">Generate or upload data to analyze request failures and timeouts.</p>
      </div>
    );
  }

  const binLog = reliability.binary_logistic;
  const multiLog = reliability.multinomial_logistic;
  const compTable = reliability.model_comparison.comparison_table;
  const textFeats = reliability.unstructured_text_features;

  return (
    <div>
      <div className="technique-tag">Syllabus Items 4 & 5: Failure Classification & Decision Trees</div>

      {/* Model Benchmark Comparison Table (Logistic vs CART vs CHAID) */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Hold-Out Set Model Benchmark</div>
            <h3 className="card-title">Comparative Failure Prediction: Logistic vs CART vs CHAID</h3>
            <p className="card-subtitle">
              Evaluated on the exact same hold-out test set (n = {reliability.model_comparison.test_samples.toLocaleString()})
            </p>
          </div>
          <span className="badge badge-green" style={{ fontSize: '0.8rem', padding: '4px 8px' }}>
            Top Model: {reliability.model_comparison.best_model_by_auc}
          </span>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall (Sensitivity)</th>
                <th>Specificity</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>Gini Index (2*AUC - 1)</th>
              </tr>
            </thead>
            <tbody>
              {compTable.map((row) => {
                const isBest = row.model === reliability.model_comparison.best_model_by_auc;
                return (
                  <tr key={row.model} style={{ backgroundColor: isBest ? '#f0fdf4' : 'inherit' }}>
                    <td style={{ fontWeight: isBest ? 700 : 500 }}>
                      {row.model}
                    </td>
                    <td className="mono">{(row.accuracy * 100).toFixed(1)}%</td>
                    <td className="mono">{(row.precision * 100).toFixed(1)}%</td>
                    <td className="mono">{(row.recall * 100).toFixed(1)}%</td>
                    <td className="mono">{(row.specificity * 100).toFixed(1)}%</td>
                    <td className="mono">{row.f1_score.toFixed(3)}</td>
                    <td className="mono" style={{ fontWeight: 600 }}>{row.roc_auc.toFixed(3)}</td>
                    <td className="mono" style={{ fontWeight: 600 }}>{row.gini_coefficient.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '8px' }}>
          {reliability.model_comparison.recommendation}
        </p>
      </div>

      {/* Binary Logistic Regression Details & ROC Curve */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Syllabus Item 4: Binary Logistic Regression: P(Fail) ~ X</div>
            <h3 className="card-title">Odds Ratios, Wald Tests & Goodness-of-Fit</h3>
            <p className="card-subtitle">
              AUC = {binLog.roc_analysis.auc.toFixed(3)}, Gini = {binLog.roc_analysis.gini_coefficient.toFixed(3)}, Hosmer-Lemeshow p = {binLog.hosmer_lemeshow_test.p_value.toFixed(4)}
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px', marginBottom: '20px' }}>
          {/* ROC Curve Chart */}
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
              Receiver Operating Characteristic (ROC Curve)
            </div>
            <div style={{ height: '260px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={binLog.roc_analysis.curve_points} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="fpr" tick={{ fontSize: 10 }} label={{ value: 'False Positive Rate (1 - Spec)', position: 'insideBottom', offset: -10 }} />
                  <YAxis tick={{ fontSize: 10 }} label={{ value: 'True Positive Rate (Sens)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="tpr" stroke="#2563eb" strokeWidth={2} dot={false} name="Model ROC" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
              Classification Matrix (Cutoff = {binLog.threshold})
            </div>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Actual \ Pred</th>
                    <th>Predicted Success</th>
                    <th>Predicted Failure</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Actual Success</td>
                    <td className="mono">{binLog.classification_table.confusion_matrix.true_negative.toLocaleString()} (TN)</td>
                    <td className="mono">{binLog.classification_table.confusion_matrix.false_positive.toLocaleString()} (FP)</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 600 }}>Actual Failure</td>
                    <td className="mono">{binLog.classification_table.confusion_matrix.false_negative.toLocaleString()} (FN)</td>
                    <td className="mono">{binLog.classification_table.confusion_matrix.true_positive.toLocaleString()} (TP)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <div>Sensitivity / Recall: <strong>{(binLog.classification_table.metrics.sensitivity_recall * 100).toFixed(1)}%</strong></div>
              <div>Specificity: <strong>{(binLog.classification_table.metrics.specificity * 100).toFixed(1)}%</strong></div>
              <div>Hosmer-Lemeshow Calibration: <strong>{binLog.hosmer_lemeshow_test.well_calibrated ? 'Well Calibrated (p > 0.05)' : 'Deviates'}</strong></div>
            </div>
          </div>
        </div>

        {/* Odds Ratios Table */}
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Predictor</th>
                <th>Coefficient (β)</th>
                <th>Wald z</th>
                <th>p-value</th>
                <th>Odds Ratio (e^β)</th>
                <th>95% CI of OR</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(binLog.odds_ratios).map(([feat, o]) => {
                const c = binLog.coefficients[feat];
                return (
                  <tr key={feat}>
                    <td style={{ fontWeight: 600 }} className="mono">{feat}</td>
                    <td className="mono">{c?.coefficient.toFixed(4)}</td>
                    <td className="mono">{c?.wald_z.toFixed(2)}</td>
                    <td className="mono">{c?.p_value < 0.001 ? '< 0.001' : c?.p_value.toFixed(4)}</td>
                    <td className="mono" style={{ fontWeight: 600 }}>{o.odds_ratio.toFixed(3)}x</td>
                    <td className="mono">[{o.ci_lower_95.toFixed(3)}, {o.ci_upper_95.toFixed(3)}]</td>
                    <td style={{ fontSize: '0.78rem' }}>{o.interpretation}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Human-Readable Decision Rules (CART & CHAID) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px', marginBottom: '24px' }}>
        {/* CART Rules */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="technique-tag">Syllabus Item 5: CART (Classification & Regression Trees)</div>
              <h3 className="card-title">CART Pruned Decision Rules</h3>
              <p className="card-subtitle">Extracted from scikit-learn DecisionTreeClassifier</p>
            </div>
          </div>
          <div>
            {reliability.cart_classification.human_readable_rules.map((rule, idx) => (
              <div key={idx} className="rule-item">{rule}</div>
            ))}
          </div>
        </div>

        {/* CHAID Rules */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="technique-tag">Syllabus Item 5: CHAID (Chi-Square Automatic Interaction)</div>
              <h3 className="card-title">CHAID Multiway Decision Rules</h3>
              <p className="card-subtitle">Scipy-powered chi-square contingency splitting</p>
            </div>
          </div>
          <div>
            {reliability.chaid_classification.human_readable_rules.map((rule, idx) => (
              <div key={idx} className="rule-item" style={{ borderLeftColor: '#059669' }}>{rule}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Analysis of Unstructured Data: Text Features */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Analysis of Unstructured Data: Prompt Text Features</div>
            <h3 className="card-title">Statistical Association of Prompt Content on Failures</h3>
            <p className="card-subtitle">{textFeats.summary}</p>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Extracted Text Feature</th>
                <th>Present Count</th>
                <th>Failure Rate (When Present)</th>
                <th>Failure Rate (When Absent)</th>
                <th>Relative Risk</th>
                <th>Chi-Square p-value</th>
                <th>Significance</th>
              </tr>
            </thead>
            <tbody>
              {textFeats.text_feature_effects.map((feat) => (
                <tr key={feat.feature}>
                  <td style={{ fontWeight: 600 }} className="mono">{feat.feature}</td>
                  <td>{feat.present_count.toLocaleString()}</td>
                  <td className="mono" style={{ color: feat.relative_risk > 1 ? 'var(--status-red-text)' : 'inherit' }}>
                    {feat.failure_rate_present_pct.toFixed(1)}%
                  </td>
                  <td className="mono">{feat.failure_rate_absent_pct.toFixed(1)}%</td>
                  <td className="mono" style={{ fontWeight: 600 }}>{feat.relative_risk.toFixed(2)}x</td>
                  <td className="mono">{feat.p_value < 0.001 ? '< 0.001' : feat.p_value.toFixed(4)}</td>
                  <td>
                    {feat.statistically_significant ? (
                      <span className="badge badge-amber">Significant (p &lt; 0.05)</span>
                    ) : (
                      <span className="badge badge-neutral">Not Significant</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Multinomial Logistic Regression: Multi-class Status */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Syllabus Item 4: Multinomial Logistic Regression</div>
            <h3 className="card-title">Multi-Class Status Outcome (Reference: 'ok')</h3>
            <p className="card-subtitle">
              Relative Risk Ratios (RRR = e^β) comparing failure modes against standard success
            </p>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Outcome Class</th>
                <th>Latency (ms) RRR</th>
                <th>Prompt Tokens RRR</th>
                <th>Completion Tokens RRR</th>
                <th>Peak Hour RRR</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(multiLog.relative_risk_ratios_vs_ok).map(([cls, rrr]) => {
                const m = multiLog.metrics_by_class[cls];
                return (
                  <tr key={cls}>
                    <td style={{ fontWeight: 600 }} className="mono">{cls}</td>
                    <td className="mono">{rrr.latency_ms?.toFixed(3)}x</td>
                    <td className="mono">{rrr.prompt_tokens?.toFixed(3)}x</td>
                    <td className="mono">{rrr.completion_tokens?.toFixed(3)}x</td>
                    <td className="mono">{rrr.is_peak_hour?.toFixed(3)}x</td>
                    <td className="mono">{((m?.precision || 0) * 100).toFixed(1)}%</td>
                    <td className="mono">{((m?.recall || 0) * 100).toFixed(1)}%</td>
                    <td className="mono">{m?.f1_score.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
