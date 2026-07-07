// Labels, colors, and explainer copy — ported verbatim from ui/app.py so the
// two frontends describe the model identically.

import type { Direction } from "../api/types";

export const HORIZON_LABELS: Record<string, string> = {
  "1": "Tomorrow",
  "3": "3 Days",
  "5": "5 Days",
  "10": "10 Days",
};

export const DIRECTION_COLORS: Record<Direction, string> = {
  up: "#30d158",
  flat: "#ff9f0a",
  down: "#ff453a",
};

export const DIRECTION_ARROWS: Record<Direction, string> = {
  up: "↑",
  flat: "→",
  down: "↓",
};

export const CHART_COLORS = {
  accent: "#0a84ff",
  sma20: "#ff9f0a",
  sma50: "#bf5af2",
  grid: "rgba(255,255,255,0.08)",
  surface: "#1c1c1e",
  text: "#98989d",
};

export const FEATURE_LABELS: Record<string, string> = {
  RSI: "RSI — Momentum (overbought / oversold)",
  MACD: "MACD Line (trend momentum)",
  MACD_signal: "MACD Signal Line (trigger line)",
  MACD_hist: "MACD Histogram (momentum shift)",
  BB_upper: "Bollinger Upper Band (resistance level)",
  BB_middle: "Bollinger Middle Band (20-day avg price)",
  BB_lower: "Bollinger Lower Band (support level)",
  BB_position: "Price Position in Bollinger Bands (0=bottom, 1=top)",
  SMA_20: "20-Day Average Price",
  SMA_50: "50-Day Average Price",
  SMA_200: "200-Day Average Price (long-term trend)",
  price_vs_SMA200_pct: "% Deviation from 200-Day Avg (bull/bear context)",
  ROC: "Rate of Change — 14-day price momentum %",
  Volume_SMA: "20-Day Average Trading Volume",
  Volume_ratio: "Today's Volume vs 20-Day Average",
  ATR_14: "ATR-14 — Daily Volatility (true range)",
  Stoch_K: "Stochastic %K (momentum oscillator)",
  Stoch_D: "Stochastic %D (signal line)",
  High_Low_ratio: "High-Low Ratio (intraday price range / close)",
  price_vs_52w_high: "Price vs 52-Week High (1.0 = at the high)",
  price_vs_52w_low: "Price vs 52-Week Low (1.0 = at the low)",
  return_1d: "1-Day Return %",
  return_5d: "5-Day Return %",
  return_20d: "20-Day Return %",
  price_vs_SMA50_pct: "% Deviation from 50-Day Avg",
  VIX_close: "VIX — Market Fear Index",
  yield_spread: "10Y-3M Treasury Yield Spread (recession signal)",
  sector_mom_5d: "Sector ETF 5-Day Momentum",
  sector_mom_20d: "Sector ETF 20-Day Momentum",
  rel_strength_5d: "Stock vs Sector 5-Day Relative Strength",
  days_to_earnings: "Days Until Next Earnings Report",
  earnings_imminent: "Earnings Within 5 Days (binary flag)",
};

export const FEATURE_EXPLANATIONS: Record<string, string> = {
  RSI: "RSI — above 70 = overbought (potential dip); below 30 = oversold (potential bounce); 30–70 = neutral.",
  MACD: "MACD — gap between the 12-day and 26-day moving averages. Positive = short-term above long-term trend.",
  MACD_signal:
    "MACD Signal — 9-day smoothed average of MACD. Crossovers with MACD generate buy/sell signals.",
  MACD_hist:
    "MACD Histogram — gap between MACD and its signal line. Growing bars = building momentum; shrinking = fading.",
  BB_upper:
    "Bollinger Upper Band — 20-day avg + 2 std deviations. Price near here can signal overbought conditions.",
  BB_middle:
    "Bollinger Middle — the 20-day moving average; a baseline for 'fair value'.",
  BB_lower:
    "Bollinger Lower Band — 20-day avg − 2 std deviations. Price near here can signal oversold conditions.",
  BB_position:
    "Bollinger Position — 0 = price at the bottom band (cheap vs. recent history), 1 = at the top (expensive).",
  SMA_20: "20-Day Moving Average — average close price over 20 days. Rising = uptrend.",
  SMA_50:
    "50-Day Moving Average — longer-term trend signal. Price crossing above SMA50 is often bullish.",
  SMA_200:
    "200-Day Moving Average — the primary bull/bear dividing line. Above = bull regime; below = bear.",
  price_vs_SMA200_pct:
    "% Above/Below 200-Day Avg — the model uses this to detect market regimes (bull/bear/sideways).",
  ROC: "Rate of Change — % price change over 14 days. Positive = upward momentum; negative = downward.",
  Volume_SMA: "Avg Volume (20-day) — typical daily trading activity baseline.",
  Volume_ratio:
    "Volume Ratio > 1 means heavier-than-normal trading, which tends to confirm the current price move.",
  ATR_14:
    "ATR-14 (Average True Range) — how much the price actually moves each day on average, including gaps. High ATR = volatile; low ATR = calm.",
  Stoch_K:
    "Stochastic %K — compares today's close to the High-Low range over 14 days. Above 80 = overbought; below 20 = oversold.",
  Stoch_D:
    "Stochastic %D — 3-day smoothed version of %K. Crossovers between %K and %D generate signals.",
  High_Low_ratio:
    "High-Low Ratio — today's range (High − Low) as a fraction of Close. Large ratio = high intraday volatility.",
  price_vs_52w_high:
    "52-Week High Ratio — how close price is to its yearly high. 1.0 = at the high; 0.9 = 10% below.",
  price_vs_52w_low:
    "52-Week Low Ratio — how close price is to its yearly low. Useful for spotting turnaround potential.",
  return_1d: "1-Day Return % — how much the stock moved yesterday. Very short-term momentum.",
  return_5d: "5-Day Return % — one-week momentum signal.",
  return_20d: "20-Day Return % — one-month momentum. Captures intermediate trends.",
  price_vs_SMA50_pct:
    "% Above/Below 50-Day Avg — positive = price is above its 50-day average (uptrend); negative = below it.",
  VIX_close:
    "VIX — the 'fear index'. High VIX = stressed market; low VIX = calm. Useful context for all directional models.",
  yield_spread:
    "Yield Spread (10Y-3M) — negative spread often precedes recessions. Model uses this for macro regime context.",
  sector_mom_5d:
    "Sector 5-Day Momentum — how the stock's sector ETF moved over the past week. Strong sector = tailwind.",
  sector_mom_20d: "Sector 20-Day Momentum — monthly trend of the broader sector.",
  rel_strength_5d:
    "Relative Strength vs Sector — stock outperforming its sector = positive; underperforming = negative.",
  days_to_earnings:
    "Days to Earnings — markets are typically more volatile near earnings. Model accounts for proximity.",
  earnings_imminent:
    "Earnings Imminent — binary flag when earnings are ≤ 5 days away. Predicts elevated uncertainty.",
};

export const SIGNAL_LABELS: Record<string, [string, string]> = {
  ichimoku_tk: [
    "Ichimoku T/K Cross",
    "Tenkan-sen vs Kijun-sen crossover aligns with direction",
  ],
  ichimoku_cloud: [
    "Ichimoku Cloud",
    "Price is above (long) or below (short) both Senkou A & B",
  ],
  adx_di: [
    "ADX Trend Strength",
    "ADX > 20 confirms an active trend; +DI/-DI confirm direction",
  ],
  bbw_regime: [
    "BBW Volatility Gate",
    "Bollinger Band Width 0.01–0.12: not too quiet, not in panic mode",
  ],
};

export const REASON_LABELS: Record<string, string> = {
  direction_flat: "Model direction uncertain (FLAT) — ML couldn't pick UP or DOWN",
};

export const PERIOD_OPTIONS = [
  { label: "1M", value: "1mo" },
  { label: "3M", value: "3mo" },
  { label: "6M", value: "6mo" },
  { label: "1Y", value: "1y" },
  { label: "2Y", value: "2y" },
];

export const RANDOM_BASELINE = 1 / 3;
