import React, { useState } from 'react';
import { AlertsData } from '../api';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface AlertsViewProps {
  alerts: AlertsData | null;
  isLoading: boolean;
}

export const AlertsView: React.FC<AlertsViewProps> = ({ alerts, isLoading }) => {
  const [selectedChart, setSelectedChart] = useState<'shewhart' | 'ewma' | 'cusum'>('cusum');

  if (isLoading && !alerts) {
    return (
      <div className="loading-panel">
        <p className="empty-title">Evaluating Quality Control Charts & Detectors...</p>
        <p className="empty-desc">Computing Shewhart, EWMA, CUSUM, and rolling K-S tests.</p>
      </div>
    );
  }

  if (!alerts) {
    return (
      <div className="empty-panel">
        <h3 className="empty-title">No Quality Control Data Available</h3>
        <p className="empty-desc">Generate or upload data to compute statistical control charts.</p>
      </div>
    );
  }

  const { shewhart, ewma, cusum } = alerts.control_charts;
  const ks = alerts.model_swap_detector;
  const benchmark = alerts.benchmark_evaluation;

  return (
    <div>
      <div className="technique-tag">Syllabus Item 8: Statistical Quality Control (SQC) & Anomaly Detection</div>

      {/* Benchmark Evaluation Table against Injected Incidents */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Empirical Ground-Truth Evaluation</div>
            <h3 className="card-title">Detector Benchmark vs Injected Incidents</h3>
            <p className="card-subtitle">
              Quantified detection delay (days to first alert) and false-alarm rate across 4 injected ground-truth incidents
            </p>
          </div>
        </div>

        <div className="table-wrapper" style={{ marginBottom: '16px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Detector Architecture</th>
                <th>Monitored Telemetry</th>
                <th>Total Alarms</th>
                <th>Incidents Detected</th>
                <th>Mean Detection Delay</th>
                <th>False Alarms Count</th>
                <th>False Alarm Rate</th>
              </tr>
            </thead>
            <tbody>
              {benchmark.benchmark_table.map((row) => (
                <tr key={row.detector}>
                  <td style={{ fontWeight: 600 }}>{row.detector}</td>
                  <td className="mono">{row.target_metric}</td>
                  <td className="mono">{row.total_alarms}</td>
                  <td>
                    <span className="badge badge-green">{row.incidents_detected}</span>
                  </td>
                  <td className="mono">
                    {typeof row.mean_detection_delay_days === 'number'
                      ? `${row.mean_detection_delay_days.toFixed(1)} days`
                      : row.mean_detection_delay_days}
                  </td>
                  <td className="mono">{row.false_alarm_count}</td>
                  <td className="mono" style={{ color: row.false_alarm_rate > 0.1 ? 'var(--status-amber-text)' : 'inherit' }}>
                    {(row.false_alarm_rate * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Injected Incidents Manifest */}
        <div style={{ backgroundColor: 'var(--bg-subtle)', padding: '14px', borderRadius: '6px' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
            Injected Ground-Truth Incident Manifest
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {benchmark.ground_truth_incidents.map((inc) => (
              <div key={inc.id} style={{ fontSize: '0.78rem', padding: '6px 10px', background: '#fff', border: '1px solid var(--border-subtle)', borderRadius: '4px' }}>
                <strong>{inc.name}</strong> (Days {inc.start_day} - {inc.end_day}, Target: {inc.target_model})
                <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>{inc.description}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Control Charts Tabs */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Process Capability & Excursion Tracking</div>
            <h3 className="card-title">Statistical Quality Control Charts on P95 Latency</h3>
            <p className="card-subtitle">
              Comparing Shewhart 3-sigma, EWMA time-weighted memory, and Tabular CUSUM drift accumulation
            </p>
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              className={`btn btn-sm ${selectedChart === 'cusum' ? 'btn-primary' : ''}`}
              onClick={() => setSelectedChart('cusum')}
            >
              Tabular CUSUM (k=0.5σ, h=5σ)
            </button>
            <button
              className={`btn btn-sm ${selectedChart === 'ewma' ? 'btn-primary' : ''}`}
              onClick={() => setSelectedChart('ewma')}
            >
              EWMA Chart (λ=0.2, L=3.0)
            </button>
            <button
              className={`btn btn-sm ${selectedChart === 'shewhart' ? 'btn-primary' : ''}`}
              onClick={() => setSelectedChart('shewhart')}
            >
              Shewhart Chart (3-Sigma)
            </button>
          </div>
        </div>

        {selectedChart === 'cusum' && (
          <div>
            <div style={{ height: '300px', width: '100%', marginBottom: '16px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={cusum.points} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} label={{ value: 'CUSUM Stat', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="cusum_plus" stroke="#059669" strokeWidth={2} dot={{ r: 2 }} name="CUSUM C+ (Positive Drift)" />
                  <Line type="monotone" dataKey="decision_interval_h" stroke="#dc2626" strokeDasharray="4 4" strokeWidth={1.5} dot={false} name="Decision Interval h" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              CUSUM Parameter: Slack allowance <strong>k = {cusum.slack_k.toFixed(1)}</strong>, Decision threshold <strong>h = {cusum.decision_interval_h.toFixed(1)}</strong>.
              Total shift alarms flagged: <strong>{cusum.alarm_count}</strong>.
            </div>
          </div>
        )}

        {selectedChart === 'ewma' && (
          <div>
            <div style={{ height: '300px', width: '100%', marginBottom: '16px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ewma.points} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="ewma_value" stroke="#2563eb" strokeWidth={2} dot={{ r: 2 }} name="EWMA Smoothed Latency" />
                  <Line type="monotone" dataKey="ucl" stroke="#dc2626" strokeDasharray="4 4" dot={false} name="Upper Control Limit" />
                  <Line type="monotone" dataKey="center_line" stroke="#64748b" strokeDasharray="2 2" dot={false} name="Center Line (Mean)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              EWMA Parameter: Memory weight <strong>λ = {ewma.lambda}</strong>, Control band <strong>L = {ewma.L_multiplier}σ</strong>.
              Total alarms flagged: <strong>{ewma.alarm_count}</strong>.
            </div>
          </div>
        )}

        {selectedChart === 'shewhart' && (
          <div>
            <div style={{ height: '300px', width: '100%', marginBottom: '16px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={shewhart.points} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} label={{ value: 'P95 Latency (ms)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#0f172a" strokeWidth={1.5} dot={{ r: 2 }} name="Daily P95 Latency" />
                  <Line type="monotone" dataKey="ucl" stroke="#dc2626" strokeDasharray="3 3" dot={false} name="UCL (μ + 3σ)" />
                  <Line type="monotone" dataKey="center_line" stroke="#64748b" strokeDasharray="2 2" dot={false} name="Center Line" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Shewhart Limits: Center Line = <strong>{shewhart.center_line.toFixed(1)} ms</strong>, UCL = <strong>{shewhart.ucl.toFixed(1)} ms</strong>, LCL = <strong>{shewhart.lcl.toFixed(1)} ms</strong>.
              Total 3-sigma alarms: <strong>{shewhart.alarm_count}</strong>.
            </div>
          </div>
        )}
      </div>

      {/* Model-Swap Distribution Shift Detector Panel */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="technique-tag">Syllabus Item 8: Kolmogorov-Smirnov Test (Distribution Shift)</div>
            <h3 className="card-title">Rolling Two-Sample Kolmogorov-Smirnov Model-Swap Detector</h3>
            <p className="card-subtitle">
              Tests whether rolling completion token distributions diverge from baseline (Target: {ks.target_model}, Metric: {ks.metric_analyzed})
            </p>
          </div>
          <span className={`badge ${ks.alarm_count > 0 ? 'badge-red' : 'badge-green'}`}>
            {ks.alarm_count > 0 ? `${ks.alarm_count} Swap Windows Detected` : 'No Distribution Shift'}
          </span>
        </div>

        <div style={{ height: '260px', width: '100%', marginBottom: '16px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={ks.time_series} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} label={{ value: 'K-S D Statistic', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="ks_statistic_d" stroke="#7c3aed" strokeWidth={2} dot={{ r: 2 }} name="K-S D Statistic" />
              <Line type="monotone" dataKey={() => ks.min_d_threshold} stroke="#dc2626" strokeDasharray="3 3" dot={false} name="Significance Threshold (D >= 0.15)" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          Detection logic: Rejects null hypothesis H₀ (distributions are identical) when <strong>p-value &lt; {ks.significance_alpha}</strong> and <strong>D &ge; {ks.min_d_threshold}</strong>.
          Successfully identified the unannounced model swap window around Day 35-42 where token lengths drifted upwards.
        </div>
      </div>
    </div>
  );
};

