import React, { useState } from 'react';
import { X, Upload, RefreshCw } from 'lucide-react';
import { api } from '../api';

interface DataModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const DataModal: React.FC<DataModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [activeTab, setActiveTab] = useState<'generate' | 'upload'>('generate');
  const [days, setDays] = useState(90);
  const [volume, setVolume] = useState(1200);
  const [seed, setSeed] = useState(42);
  const [file, setFile] = useState<File | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setErrorMsg(null);
    try {
      await api.generateData(days, volume, seed);
      onSuccess();
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Data generation failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setErrorMsg('Please select a valid CSV or JSON file.');
      return;
    }
    setIsSubmitting(true);
    setErrorMsg(null);
    try {
      await api.uploadData(file);
      onSuccess();
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'File upload failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-card">
        <div className="modal-header">
          <h3 className="modal-title">Dataset Management</h3>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '10px' }}>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'generate' ? 'btn-primary' : ''}`}
            onClick={() => { setActiveTab('generate'); setErrorMsg(null); }}
          >
            <RefreshCw size={14} /> Generate Synthetic Data
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'upload' ? 'btn-primary' : ''}`}
            onClick={() => { setActiveTab('upload'); setErrorMsg(null); }}
          >
            <Upload size={14} /> Upload Custom Logs
          </button>
        </div>

        {errorMsg && (
          <div style={{ padding: '10px 14px', background: 'var(--status-red-bg)', color: 'var(--status-red-text)', border: '1px solid var(--status-red-border)', borderRadius: '6px', fontSize: '0.8rem', marginBottom: '16px' }}>
            {errorMsg}
          </div>
        )}

        {activeTab === 'generate' ? (
          <form onSubmit={handleGenerate}>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
              Synthesizes realistic LLM access logs with diurnal/weekly seasonality, heteroscedastic noise, logistic failures, and 4 labelled injected incidents.
            </p>

            <div className="form-group">
              <label className="form-label">Simulation Timeframe (Days)</label>
              <input
                type="number"
                min="15"
                max="180"
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value) || 90)}
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Average Daily Request Volume</label>
              <input
                type="number"
                min="100"
                max="5000"
                step="100"
                value={volume}
                onChange={(e) => setVolume(parseInt(e.target.value) || 1200)}
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Random Seed (Exact Reproducibility)</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                className="form-input"
              />
            </div>

            <div className="modal-actions">
              <button type="button" className="btn" onClick={onClose} disabled={isSubmitting}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                {isSubmitting ? 'Generating...' : 'Generate & Ingest'}
              </button>
            </div>
          </form>
        ) : (
          <form onSubmit={handleUpload}>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
              Upload your own production logs in CSV or JSON format. Schema is strictly validated.
            </p>

            <div className="form-group">
              <label className="form-label">Select File (.csv or .json)</label>
              <input
                type="file"
                accept=".csv,.json"
                onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                className="form-input"
              />
            </div>

            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', background: 'var(--bg-subtle)', padding: '10px', borderRadius: '4px', marginBottom: '16px' }}>
              <strong>Required Schema Columns:</strong> timestamp, request_id, feature, model, prompt_tokens, completion_tokens, latency_ms, status, retry_count, cache_hit, prompt_text, cost_usd.
            </div>

            <div className="modal-actions">
              <button type="button" className="btn" onClick={onClose} disabled={isSubmitting}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={isSubmitting || !file}>
                {isSubmitting ? 'Uploading...' : 'Upload & Clean'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

