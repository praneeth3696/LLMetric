import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  TrendingUp,
  HelpCircle,
  ShieldAlert,
  BellRing,
  BookOpen,
  Database,
  RefreshCw,
} from 'lucide-react';

import {
  api,
  HealthData,
  OverviewData,
  ForecastData,
  SimpleRegressionData,
  MultipleRegressionData,
  ReliabilityData,
  AlertsData,
  InsightsData,
  BudgetData,
} from './api';

import { OverviewView } from './views/OverviewView';
import { ForecastView } from './views/ForecastView';
import { RegressionView } from './views/RegressionView';
import { ReliabilityView } from './views/ReliabilityView';
import { AlertsView } from './views/AlertsView';
import { MethodologyView } from './views/MethodologyView';
import { DataModal } from './components/DataModal';

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<
    'overview' | 'forecast' | 'regression' | 'reliability' | 'alerts' | 'methodology'
  >('overview');

  // API Data States
  const [health, setHealth] = useState<HealthData | null>(null);
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [insights, setInsights] = useState<InsightsData | null>(null);
  const [budget, setBudget] = useState<BudgetData | null>(null);
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [simpleReg, setSimpleReg] = useState<SimpleRegressionData | null>(null);
  const [multipleReg, setMultipleReg] = useState<MultipleRegressionData | null>(null);
  const [reliability, setReliability] = useState<ReliabilityData | null>(null);
  const [alerts, setAlerts] = useState<AlertsData | null>(null);

  // Forecast view controls
  const [fcTarget, setFcTarget] = useState<'spend' | 'tokens'>('spend');
  const [fcHorizon, setFcHorizon] = useState<number>(14);
  const [fcMethod, setFcMethod] = useState<string>('auto');

  // Loading & Modal States
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isDataModalOpen, setIsDataModalOpen] = useState<boolean>(false);

  // Initial load
  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    setIsLoading(true);
    try {
      const h = await api.getHealth();
      setHealth(h);

      if (h.dataset_loaded) {
        // Fetch all endpoints concurrently
        const [ov, ins, bg, fc, sReg, mReg, rel, al] = await Promise.all([
          api.getOverview().catch(() => null),
          api.getInsights().catch(() => null),
          api.getBudget().catch(() => null),
          api.getForecast(fcTarget, fcHorizon, fcMethod).catch(() => null),
          api.getSimpleRegression().catch(() => null),
          api.getMultipleRegression().catch(() => null),
          api.getReliability().catch(() => null),
          api.getAlerts().catch(() => null),
        ]);

        setOverview(ov);
        setInsights(ins);
        setBudget(bg);
        setForecast(fc);
        setSimpleReg(sReg);
        setMultipleReg(mReg);
        setReliability(rel);
        setAlerts(al);
      }
    } catch (err) {
      console.error('Error fetching API data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleForecastParamsChange = async (target: 'spend' | 'tokens', horizon: number, method: string) => {
    setFcTarget(target);
    setFcHorizon(horizon);
    setFcMethod(method);
    try {
      const fc = await api.getForecast(target, horizon, method);
      setForecast(fc);
    } catch (err) {
      console.error('Error updating forecast:', err);
    }
  };

  const handleUpdateBudget = async (amount: number) => {
    try {
      const updated = await api.updateBudget(amount);
      setBudget(updated);
    } catch (err) {
      console.error('Error updating budget:', err);
    }
  };

  const handleDataSuccess = () => {
    loadAllData();
  };

  return (
    <div className="app-container">
      {/* Left Narrow Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-title">
            <span>LLMetrics</span>
            <span className="brand-badge">23XWA8</span>
          </div>
          <div className="brand-subtitle">LLM Cost & Reliability Monitor</div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-link ${activeView === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveView('overview')}
          >
            <LayoutDashboard size={17} /> Overview
          </button>

          <button
            className={`nav-link ${activeView === 'forecast' ? 'active' : ''}`}
            onClick={() => setActiveView('forecast')}
          >
            <TrendingUp size={17} /> Forecast & Budget
          </button>

          <button
            className={`nav-link ${activeView === 'regression' ? 'active' : ''}`}
            onClick={() => setActiveView('regression')}
          >
            <HelpCircle size={17} /> Why is it Slow / Costly
          </button>

          <button
            className={`nav-link ${activeView === 'reliability' ? 'active' : ''}`}
            onClick={() => setActiveView('reliability')}
          >
            <ShieldAlert size={17} /> Reliability & Failures
          </button>

          <button
            className={`nav-link ${activeView === 'alerts' ? 'active' : ''}`}
            onClick={() => setActiveView('alerts')}
          >
            <BellRing size={17} /> Alerts & Quality Control
          </button>

          <button
            className={`nav-link ${activeView === 'methodology' ? 'active' : ''}`}
            onClick={() => setActiveView('methodology')}
          >
            <BookOpen size={17} /> Syllabus Methodology
          </button>
        </nav>

        <div className="sidebar-footer">
          <div>Dataset Version:</div>
          <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-primary)', marginTop: '2px' }}>
            {health?.current_dataset_version || 'None'}
          </div>
          <div style={{ marginTop: '8px', fontSize: '0.7rem' }}>
            Total Ingested: <strong>{health?.total_requests.toLocaleString() || 0}</strong>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="top-navbar">
          <h2 className="top-title">
            {activeView === 'overview' && 'System Overview'}
            {activeView === 'forecast' && 'Time-Series Forecasting & Budget'}
            {activeView === 'regression' && 'Regression Analytics (Latency & Cost Drivers)'}
            {activeView === 'reliability' && 'Reliability & Failure Classification'}
            {activeView === 'alerts' && 'Statistical Quality Control & Incident Detection'}
            {activeView === 'methodology' && 'Syllabus Traceability & Examination Notes'}
          </h2>

          <div className="top-actions">
            <button className="btn btn-sm" onClick={loadAllData} title="Refresh telemetry">
              <RefreshCw size={14} className={isLoading ? 'spin' : ''} /> Refresh
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => setIsDataModalOpen(true)}>
              <Database size={14} /> Dataset Controls
            </button>
          </div>
        </header>

        <div className="page-body">
          {activeView === 'overview' && (
            <OverviewView
              overview={overview}
              insights={insights}
              onOpenDataModal={() => setIsDataModalOpen(true)}
              isLoading={isLoading}
            />
          )}

          {activeView === 'forecast' && (
            <ForecastView
              forecast={forecast}
              budget={budget}
              target={fcTarget}
              horizon={fcHorizon}
              method={fcMethod}
              onParamsChange={handleForecastParamsChange}
              onUpdateBudget={handleUpdateBudget}
              isLoading={isLoading}
            />
          )}

          {activeView === 'regression' && (
            <RegressionView
              simple={simpleReg}
              multiple={multipleReg}
              isLoading={isLoading}
            />
          )}

          {activeView === 'reliability' && (
            <ReliabilityView
              reliability={reliability}
              isLoading={isLoading}
            />
          )}

          {activeView === 'alerts' && (
            <AlertsView
              alerts={alerts}
              isLoading={isLoading}
            />
          )}

          {activeView === 'methodology' && (
            <MethodologyView />
          )}
        </div>
      </main>

      {/* Dataset Generation & Upload Modal */}
      <DataModal
        isOpen={isDataModalOpen}
        onClose={() => setIsDataModalOpen(false)}
        onSuccess={handleDataSuccess}
      />
    </div>
  );
};
