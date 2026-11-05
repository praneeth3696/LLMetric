/**
 * LLMetrics Frontend API Client
 */

export interface HealthData {
  status: string;
  version: string;
  database_connected: boolean;
  dataset_loaded: boolean;
  total_requests: number;
  current_dataset_version: string;
}

export interface OverviewData {
  total_requests: number;
  total_cost_usd: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  error_rate_pct: number;
  active_alerts_count: number;
  period_comparison: {
    prior_period_14d: { spend: number; latency_p95: number; error_rate: number; requests: number };
    recent_period_14d: { spend: number; latency_p95: number; error_rate: number; requests: number };
    spend_change_pct: number;
    latency_p95_change_ms: number;
  };
  dataset_version: string;
}

export interface EdaData {
  syllabus_mapping: string;
  dataset_summary: {
    total_requests: number;
    date_range_start: string;
    date_range_end: string;
    total_cost_usd: number;
    overall_error_rate_pct: number;
  };
  univariate: {
    numeric: Record<string, {
      count: number;
      mean: number;
      std: number;
      median: number;
      p5: number;
      p25: number;
      p75: number;
      p95: number;
      p99: number;
      min: number;
      max: number;
      iqr: number;
      skewness: number;
      kurtosis: number;
      histogram: Array<{ bin_start: number; bin_end: number; count: number }>;
    }>;
    categorical: Record<string, Array<{ category: string; count: number; percentage: number }>>;
  };
  bivariate: {
    correlations: {
      pearson: Record<string, Record<string, number>>;
      spearman: Record<string, Record<string, number>>;
    };
    by_model: Array<{
      model: string;
      count: number;
      latency_mean: number;
      latency_median: number;
      latency_p95: number;
      cost_mean: number;
      cost_total: number;
      error_rate: number;
    }>;
    by_feature: Array<{
      feature: string;
      count: number;
      latency_mean: number;
      cost_mean: number;
      error_rate: number;
    }>;
    by_status: Array<{
      status: string;
      count: number;
      latency_mean: number;
      latency_p95: number;
    }>;
  };
  multivariate: {
    model_status_crosstab_pct: Record<string, Record<string, number>>;
    by_model_and_peak?: Array<{
      model: string;
      is_peak_hour: number;
      latency_mean: number;
      latency_p95: number;
      error_rate: number;
      count: number;
    }>;
  };
}

export interface SimpleRegressionData {
  syllabus_mapping: string;
  model_name: string;
  sample_size: number;
  metrics: {
    r_squared: number;
    adjusted_r_squared: number;
    f_statistic: number;
    f_pvalue: number;
    rmse: number;
  };
  coefficients: {
    intercept: { value: number; std_error: number; t_statistic: number; p_value: number; conf_interval_95: number[] };
    completion_tokens: { value: number; std_error: number; t_statistic: number; p_value: number; conf_interval_95: number[] };
  };
  residual_analysis: {
    mean: number;
    std: number;
    jarque_bera: { statistic: number; p_value: number; is_normal: boolean };
    shapiro_wilk: { statistic: number; p_value: number; is_normal: boolean };
    scatter_points: Array<{
      completion_tokens: number;
      actual_latency_ms: number;
      fitted_latency_ms: number;
      residual: number;
      std_residual: number;
    }>;
    qq_plot: Array<{ theoretical: number; sample: number }>;
  };
  intervals: Array<{
    completion_tokens: number;
    mean_prediction: number;
    ci_lower: number;
    ci_upper: number;
    pi_lower: number;
    pi_upper: number;
  }>;
  interpretations: string[];
}

export interface MultipleRegressionData {
  syllabus_mapping: string;
  model_name: string;
  target_variable: string;
  sample_size: number;
  metrics: {
    r_squared: number;
    adjusted_r_squared: number;
    f_statistic: number;
    f_pvalue: number;
    aic: number;
    bic: number;
    rmse: number;
  };
  coefficients: Record<string, {
    coefficient: number;
    std_error: number;
    t_statistic: number;
    p_value: number;
    conf_int_95: number[];
    statistically_significant: boolean;
  }>;
  interpretations: string[];
  diagnostics: {
    heteroscedasticity_breusch_pagan: {
      lagrange_multiplier_stat: number;
      p_value: number;
      f_statistic: number;
      has_heteroscedasticity: boolean;
      interpretation: string;
    };
    multicollinearity_vif: Array<{
      feature: string;
      vif: number;
      severity: string;
      multicollinear: boolean;
    }>;
    outliers_and_influence: {
      cooks_distance_threshold: number;
      influential_points_count: number;
      influential_points_percentage: number;
      top_influential_cases: Array<{
        index: number;
        cooks_distance: number;
        leverage: number;
        studentized_resid: number;
        completion_tokens: number;
        latency_ms: number;
      }>;
    };
    autocorrelation_durbin_watson: {
      statistic: number;
      interpretation: string;
    };
  };
  log_transform_comparison: {
    linear_model: { r_squared: number; aic: number; breusch_pagan_pvalue: number; residual_skewness: number };
    log_linear_model: { formula: string; r_squared: number; aic: number; breusch_pagan_pvalue: number; residual_skewness: number; heteroscedasticity?: boolean };
    takeaway: string;
  };
  model_building_procedure: {
    method: string;
    selected_features: string[];
    step_history: Array<{
      step: number;
      action: string;
      feature_considered: string;
      current_features: string[];
      aic: number;
      delta_aic: number;
    }>;
  };
}

export interface ReliabilityData {
  syllabus_mapping: string;
  binary_logistic: {
    sample_size: number;
    failure_rate_pct: number;
    threshold: number;
    coefficients: Record<string, { coefficient: number; std_error: number; wald_z: number; wald_chi2: number; p_value: number; conf_int_95: number[] }>;
    odds_ratios: Record<string, { odds_ratio: number; ci_lower_95: number; ci_upper_95: number; interpretation: string }>;
    hosmer_lemeshow_test: { statistic: number; degrees_of_freedom: number; p_value: number; well_calibrated: boolean; interpretation: string; decile_groups: any[] };
    classification_table: {
      confusion_matrix: { true_negative: number; false_positive: number; false_negative: number; true_positive: number };
      metrics: { accuracy: number; precision: number; sensitivity_recall: number; specificity: number; f1_score: number };
    };
    roc_analysis: { auc: number; gini_coefficient: number; interpretation: string; curve_points: Array<{ fpr: number; tpr: number }> };
  };
  multinomial_logistic: {
    reference_class: string;
    classes: string[];
    accuracy: number;
    relative_risk_ratios_vs_ok: Record<string, Record<string, number>>;
    confusion_matrix: Record<string, Record<string, number>>;
    metrics_by_class: Record<string, { precision: number; recall: number; f1_score: number; support: number }>;
  };
  cart_classification: {
    metrics: { accuracy: number; precision: number; recall: number; f1_score: number; roc_auc: number };
    feature_importances: Record<string, number>;
    human_readable_rules: string[];
  };
  chaid_classification: {
    metrics: { accuracy: number; precision: number; recall: number; f1_score: number };
    human_readable_rules: string[];
  };
  unstructured_text_features: {
    text_feature_effects: Array<{
      feature: string;
      present_count: number;
      absent_count: number;
      failure_rate_present_pct: number;
      failure_rate_absent_pct: number;
      relative_risk: number;
      p_value: number;
      statistically_significant: boolean;
    }>;
    continuous_correlations_with_failure: {
      prompt_char_length_correlation: number;
      prompt_word_count_correlation: number;
    };
    summary: string;
  };
  model_comparison: {
    task: string;
    train_samples: number;
    test_samples: number;
    best_model_by_auc: string;
    comparison_table: Array<{
      model: string;
      accuracy: number;
      precision: number;
      recall: number;
      specificity: number;
      f1_score: number;
      roc_auc: number;
      gini_coefficient: number;
    }>;
    recommendation: string;
  };
}

export interface ForecastData {
  syllabus_mapping: string;
  target: string;
  horizon_days: number;
  chosen_method: string;
  backtest_evaluation: {
    comparison_table: Array<{
      method: string;
      mae: number;
      rmse: number;
      mape: number;
      num_origins_evaluated: number;
    }>;
    best_method: string;
    justification: string;
  };
  history: Array<{ date: string; actual: number }>;
  forecast: Array<{
    date: string;
    forecast: number;
    lower_80: number;
    upper_80: number;
    lower_95: number;
    upper_95: number;
  }>;
}

export interface AlertsData {
  syllabus_mapping: string;
  control_charts: {
    shewhart: {
      center_line: number;
      ucl: number;
      lcl: number;
      sigma: number;
      alarm_count: number;
      points: Array<{ timestamp: string; value: number; center_line: number; ucl: number; lcl: number; alarm: boolean }>;
    };
    ewma: {
      lambda: number;
      L_multiplier: number;
      center_line: number;
      alarm_count: number;
      points: Array<{ timestamp: string; raw_value: number; ewma_value: number; ucl: number; lcl: number; alarm: boolean }>;
    };
    cusum: {
      slack_k: number;
      decision_interval_h: number;
      center_line: number;
      alarm_count: number;
      points: Array<{ timestamp: string; raw_value: number; cusum_plus: number; cusum_minus: number; decision_interval_h: number; alarm: boolean }>;
    };
  };
  model_swap_detector: {
    detector: string;
    target_model: string;
    metric_analyzed: string;
    significance_alpha: number;
    min_d_threshold: number;
    alarm_count: number;
    time_series: Array<{
      date: string;
      ks_statistic_d: number;
      p_value: number;
      distribution_shift_alarm: boolean;
    }>;
  };
  benchmark_evaluation: {
    benchmark_table: Array<{
      detector: string;
      target_metric: string;
      total_alarms: number;
      incidents_detected: string;
      mean_detection_delay_days: string | number;
      false_alarm_count: number;
      false_alarm_rate: number;
    }>;
    ground_truth_incidents: Array<{
      id: string;
      name: string;
      type: string;
      start_day: number;
      end_day: number;
      target_model: string;
      description: string;
    }>;
    summary: string;
  };
}

export interface InsightsData {
  summary_text: string;
  is_llm_rewritten: boolean;
  engine: string;
  bullets: string[];
}

export interface BudgetData {
  monthly_budget: number;
  days_elapsed: number;
  days_remaining: number;
  current_spend_usd: number;
  projected_month_end_spend: number;
  prediction_interval_95: [number, number];
  breach_probability_pct: number;
  budget_status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  interpretation: string;
}

const API_BASE = '';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(API_BASE + url, options);
  if (!resp.ok) {
    const errorBody = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(errorBody.detail || `Request failed with status ${resp.status}`);
  }
  return resp.json();
}

export const api = {
  getHealth: () => fetchJson<HealthData>('/health'),
  getOverview: () => fetchJson<OverviewData>('/overview'),
  getEda: () => fetchJson<EdaData>('/eda'),
  getSimpleRegression: () => fetchJson<SimpleRegressionData>('/regression/simple'),
  getMultipleRegression: () => fetchJson<MultipleRegressionData>('/regression/multiple'),
  getReliability: () => fetchJson<ReliabilityData>('/reliability'),
  getForecast: (target = 'spend', horizon = 14, method = 'auto') =>
    fetchJson<ForecastData>(`/forecast?target=${target}&horizon=${horizon}&method=${method}`),
  getAlerts: () => fetchJson<AlertsData>('/alerts'),
  getInsights: () => fetchJson<InsightsData>('/insights'),
  getBudget: () => fetchJson<BudgetData>('/budget'),
  updateBudget: (amount: number) =>
    fetchJson<BudgetData>('/budget', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ monthly_budget: amount }),
    }),
  generateData: (days = 90, daily_volume = 1200, seed = 42) =>
    fetchJson<{ status: string; message: string; total_records: number; dataset_version: string }>('/data/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ days, daily_volume, seed }),
    }),
  uploadData: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJson<{ status: string; message: string; total_records: number; dataset_version: string }>('/data/upload', {
      method: 'POST',
      body: formData,
    });
  },
  getTaskStatus: () =>
    fetchJson<{ status: string; progress_pct: number; current_step: string; error?: string }>('/task/status'),
};
