// Mirrors the FastAPI response shapes (snake_case preserved end-to-end).

export type Direction = "up" | "flat" | "down";

export interface HorizonPrediction {
  direction: Direction;
  confidence: number;
  probabilities: { up: number; flat: number; down: number };
  regime: string | null;
}

export interface TickerPrediction {
  timestamp: string | null;
  horizons: Record<string, HorizonPrediction>;
}

export type PredictionsResponse = Record<string, TickerPrediction>;

export interface AppConfig {
  tickers: string[];
  horizons: number[];
  deadbands: Record<string, number>;
  model_settings: { setting: string; value: string }[];
}

export type QuotesResponse = Record<
  string,
  { price: number | null; as_of: string | null }
>;

export interface SeriesPoint {
  time: string;
  value: number;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PricesResponse {
  ticker: string;
  period: string;
  last_close: number;
  last_rsi: number | null;
  candles: Candle[];
  sma20: SeriesPoint[];
  sma50: SeriesPoint[];
  rsi14: SeriesPoint[];
}

export interface FeaturesResponse {
  ticker: string;
  features: { name: string; score: number }[];
}

export interface SplitMetrics {
  accuracy?: number;
  f1?: number;
  n_folds?: number;
}

export interface HorizonMetrics {
  val?: SplitMetrics;
  test?: SplitMetrics;
  loocv?: SplitMetrics;
  walk_forward?: {
    n_folds?: number;
    mean_accuracy?: number;
    std_accuracy?: number;
    mean_f1?: number;
  };
}

export interface MetricsResponse {
  ticker: string;
  horizons: Record<string, HorizonMetrics>;
}

export interface TradeLog {
  ticker: string;
  action: "long" | "short" | "skip" | null;
  reason: string | null;
  direction: Direction | null;
  horizon: number | null;
  ml_confidence: number | null;
  tech_conf_score: number | null;
  blended_confidence: number | null;
  confirming_signals: string | null;
  tech_signals: Record<string, boolean>;
  entry_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  sl_distance: number | null;
  tp_distance: number | null;
  risk_reward_ratio: number | null;
  position_size: number | null;
  dollar_risk: number | null;
  dollar_reward: number | null;
  atr_used: number | null;
  account_size: number | null;
  timestamp: string | null;
}

export interface TradesResponse {
  last_run: string | null;
  summary: {
    evaluated: number;
    active: number;
    skipped: number;
    total_risk: number;
    total_reward: number;
  };
  active: TradeLog[];
  skipped: TradeLog[];
}

export interface OutcomeRecord {
  id: string;
  ticker: string;
  horizon_days: number;
  predicted_at: string | null;
  predicted_direction: Direction;
  predicted_confidence: number;
  entry_price: number | null;
  outcome_date: string | null;
  actual_price: number | null;
  actual_direction: Direction | null;
  actual_pct_change: number | null;
  correct: boolean | null;
  status: "pending" | "correct" | "incorrect";
}

export interface TrackRecordResponse {
  summary: {
    overall_accuracy: number | null;
    correct: number;
    incorrect: number;
    pending: number;
    random_baseline: number;
  };
  by_horizon: {
    horizon_days: number;
    evaluated: number;
    correct: number;
    accuracy: number;
  }[];
  by_ticker: {
    ticker: string;
    evaluated: number;
    correct: number;
    accuracy: number;
  }[];
  recent: OutcomeRecord[];
  pending: OutcomeRecord[];
  daily_accuracy: { date: string; pct_correct: number; n_resolved: number }[];
}
