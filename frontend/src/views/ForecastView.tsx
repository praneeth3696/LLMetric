import React, { useState } from 'react';
import { ForecastData, BudgetData } from '../api';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';

interface ForecastViewProps {
  forecast: ForecastData | null;
  budget: BudgetData | null;
  target: 'spend' | 'tokens';
  horizon: number;
  method: string;
  onParamsChange: (target: 'spend' | 'tokens', horizon: number, method: string) => void;
  onUpdateBudget: (amount: number) => Promise<void>;
  isLoading: boolean;
}

export const ForecastView: React.FC<ForecastViewProps> = ({
  forecast,
  budget,
  target,
  horizon,
  method,
  onParamsChange,
  onUpdateBudget,
  isLoading,
}) => {
  const [budgetInput, setBudgetInput] = useState(budget ? budget.monthly_budget.toString() : '1200');
  const [budgetUpdating, setBudgetUpdating] = useState(false);

  const handleBudgetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const val = parseFloat(budgetInput);
    if (!isNaN(val) && val > 0) {
      setBudgetUpdating(true);
      await onUpdateBudget(val);
      setBudgetUpdating(false);
    }
  };

  if (isLoading && !forecast) {
    return (
      <div className="loading-panel">
        <p className="empty-title">Computing rolling-origin forecasts...</p>
        <p className="empty-desc">Evaluating cross-validation errors across time-series models.</p>
      </div>
    );
  }

  if (!forecast) {
    return (
      <div className="empty-panel">
        <h3 className="empty-title">No Forecast Data Available</h3>
        <p className="empty-desc">Generate or upload logs to compute time series projections.</p>
      </div>
    );
  }

  // Combine history (last 30 points) and forecast for chart
  const recentHistory = forecast.history.slice(-30).map((h) => ({
    date: h.date,
    actual: h.actual,
    forecast: null,
    lower_95: null,
    upper_95: null,
    lower_80: null,
    upper_80: null,
  }));

  const futurePoints = forecast.forecast.map((f) => ({
    date: f.date,
    actual: null,
    forecast: f.forecast,
    lower_95: f.lower_95,
    upper_95: f.upper_95,
    lower_80: f.lower_80,
    upper_80: f.upper_80,
  }));

  const chartData = [...recentHistory, ...futurePoints];

  return (
    <div>
      <div className="technique-tag">Syllabus Items 6 & 7: Time-Series Forecasting & Rolling-Origin Backtesting</div>

      {/* Budget & Alerting Extension Card */}
      {budget && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <div className="card-header">
            <div>
              <div className="technique-tag">Extension: Budget Alerting & Month-End Risk</div>
              <h3 className="card-title">Budget Projection & Breach Risk</h3>
              <p className="card-subtitle">Analytical probability of exceeding monthly financial ceiling</p>
            </div>
            <span
              className={`badge ${
                budget.budget_status === 'HEALTHY'
                  ? 'badge-green'
                  : budget.budget_status === 'WARNING'
                  ? 'badge-amber'
                  : 'badge-red'
              }`}
              style={{ fontSize: '0.8rem', padding: '4px 10px' }}
            >
              Status: {budget.budget_status} ({budget.breach_probability_pct}% Risk)
            </span>
          </div>

          <div className="kpi-grid" style={{ marginBottom: '16px' }}>
            <div className="kpi-card">
              <div className="kpi-label">Target Monthly Budget</div>
              <div className="kpi-value">${budget.monthly_budget.toLocaleString()}</div>
              <div className="kpi-subtext">Day {budget.days_elapsed} of 30 elapsed</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Current Cycle Spend</div>
              <div className="kpi-value">${budget.current_spend_usd.toFixed(2)}</div>
              <div className="kpi-subtext">{budget.days_remaining} days remaining</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Projected Month-End Spend</div>
              <div className="kpi-value">${budget.projected_month_end_spend.toFixed(2)}</div>
              <div className="kpi-subtext">
                95% PI: [${budget.prediction_interval_95[0].toFixed(0)}, ${budget.prediction_interval_95[1].toFixed(0)}]
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Breach Risk Probability</div>
              <div
                className="kpi-value"
                style={{
                  color:
                    budget.breach_probability_pct > 70
                      ? 'var(--status-red-text)'
                      : budget.breach_probability_pct > 30
                      ? 'var(--status-amber-text)'
                      : 'var(--status-green-text)',
                }}
              >
                {budget.breach_probability_pct}%
              </div>
              <div className="kpi-subtext">Analytical normal CDF</div>
            </div>
          </div>

          {/* Budget Setting Form */}
          <form onSubmit={handleBudgetSubmit} style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
              Adjust Monthly Budget Ceiling ($):
            </span>
            <input
              type="number"
              min="100"
              max="100000"
              step="50"
              value={budgetInput}
              onChange={(e) => setBudgetInput(e.target.value)}
              className="form-input"
              style={{ width: '140px' }}
            />
            <button type="submit" className="btn btn-primary btn-sm" disabled={budgetUpdating}>
              {budgetUpdating ? 'Updating...' : 'Update Budget'}
            </button>
          </form>
        </div>
      )}

      {/* Forecast Controls Bar */}
      <div className="card" style={{ padding: '16px 20px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'center' }}>
          <div>
            <label className="form-label">Projection Target</label>
            <select
              className="form-select"
              value={target}
              onChange={(e) => onParamsChange(e.target.value as any, horizon, method)}
              style={{ minWidth: '150px' }}
            >
              <option value="spend">Daily Spend (USD)</option>
              <option value="tokens">Daily Token Usage</option>
            </select>
          </div>

          <div>
            <label className="form-label">Forecast Horizon</label>
            <select
              className="form-select"
              value={horizon}
              onChange={(e) => onParamsChange(target, parseInt(e.target.value), method)}
              style={{ minWidth: '130px' }}
            >
              <option value={7}>7 Days Ahead</option>
              <option value={14}>14 Days Ahead</option>
              <option value={30}>30 Days Ahead</option>
            </select>
          </div>

          <div>
            <label className="form-label">Forecasting Method</label>
            <select
              className="form-select"
              value={method}
              onChange={(e) => onParamsChange(target, horizon, e.target.value)}
              style={{ minWidth: '220px' }}
            >
              <option value="auto">Auto (Best Backtest Method)</option>
              <option value="causal">Causal Regression (Exogenous)</option>
              <option value="arima">ARIMA (Order Selection by AIC)</option>
              <option value="ets">Holt-Winters Exponential Smoothing</option>
              <option value="ma">Moving Average (7-Day)</option>
            </select>
          </div>

          <div style={{ marginLeft: 'auto', alignSelf: 'flex-end', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Active Algorithm: <strong>{forecast.chosen_method}</strong>
          </div>
        </div>
      </div>

      {/* Main Forecast Chart */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Out-of-Sample Prediction Intervals</div>
            <h3 className="card-title">
              {target === 'spend' ? 'Daily Spend' : 'Daily Token Usage'} Forecast ({horizon} Days)
            </h3>
            <p className="card-subtitle">
              History + point forecast with 80% and 95% prediction intervals (Model: {forecast.chosen_method})
            </p>
          </div>
        </div>

        <div style={{ height: '340px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              {/* Actuals Line */}
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#0f172a"
                strokeWidth={2}
                dot={{ r: 2 }}
                name="Historical Actual"
                connectNulls={false}
              />
              {/* Forecast Line */}
              <Line
                type="monotone"
                dataKey="forecast"
                stroke="#2563eb"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 3 }}
                name="Forecast"
                connectNulls={false}
              />
              {/* 95% Confidence Band */}
              <Line type="monotone" dataKey="upper_95" stroke="#93c5fd" strokeWidth={1} dot={false} name="95% Upper PI" />
              <Line type="monotone" dataKey="lower_95" stroke="#93c5fd" strokeWidth={1} dot={false} name="95% Lower PI" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Backtest Benchmark Table */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Syllabus Item 6: Rolling-Origin Time-Series Cross Validation</div>
            <h3 className="card-title">Model Backtest Error Evaluation</h3>
            <p className="card-subtitle">
              {forecast.backtest_evaluation.justification}
            </p>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Forecasting Model</th>
                <th>Mean Absolute Error (MAE)</th>
                <th>Root Mean Squared Error (RMSE)</th>
                <th>Mean Abs % Error (MAPE)</th>
                <th>Origins Evaluated</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {forecast.backtest_evaluation.comparison_table.map((row) => {
                const isBest = row.method === forecast.backtest_evaluation.best_method;
                return (
                  <tr key={row.method} style={{ backgroundColor: isBest ? '#f0fdf4' : 'inherit' }}>
                    <td style={{ fontWeight: isBest ? 700 : 500 }}>
                      {row.method}
                    </td>
                    <td className="mono">{row.mae.toFixed(3)}</td>
                    <td className="mono">{row.rmse.toFixed(3)}</td>
                    <td className="mono">{row.mape.toFixed(2)}%</td>
                    <td>{row.num_origins_evaluated} rolling origins</td>
                    <td>
                      {isBest ? (
                        <span className="badge badge-green">Top Performer (Lowest Error)</span>
                      ) : (
                        <span className="badge badge-neutral">Baseline</span>
                      )}
                    </td>
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
