"""
Streamlit dashboard — Stock Forecaster.
Displays classification predictions (up/flat/down) with confidence and
probability bars, directional markers on price chart, and accuracy/F1/LOOCV
model metrics from the v2 ensemble ML engine.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Feature labels & explanations ────────────────────────────────────────────
FEATURE_LABELS = {
    "RSI":                  "RSI — Momentum (overbought / oversold)",
    "MACD":                 "MACD Line (trend momentum)",
    "MACD_signal":          "MACD Signal Line (trigger line)",
    "MACD_hist":            "MACD Histogram (momentum shift)",
    "BB_upper":             "Bollinger Upper Band (resistance level)",
    "BB_middle":            "Bollinger Middle Band (20-day avg price)",
    "BB_lower":             "Bollinger Lower Band (support level)",
    "BB_position":          "Price Position in Bollinger Bands (0=bottom, 1=top)",
    "SMA_20":               "20-Day Average Price",
    "SMA_50":               "50-Day Average Price",
    "SMA_200":              "200-Day Average Price (long-term trend)",
    "price_vs_SMA200_pct":  "% Deviation from 200-Day Avg (bull/bear context)",
    "ROC":                  "Rate of Change — 14-day price momentum %",
    "Volume_SMA":           "20-Day Average Trading Volume",
    "Volume_ratio":         "Today's Volume vs 20-Day Average",
    "ATR_14":               "ATR-14 — Daily Volatility (true range)",
    "Stoch_K":              "Stochastic %K (momentum oscillator)",
    "Stoch_D":              "Stochastic %D (signal line)",
    "High_Low_ratio":       "High-Low Ratio (intraday price range / close)",
    "price_vs_52w_high":    "Price vs 52-Week High (1.0 = at the high)",
    "price_vs_52w_low":     "Price vs 52-Week Low (1.0 = at the low)",
    "return_1d":            "1-Day Return %",
    "return_5d":            "5-Day Return %",
    "return_20d":           "20-Day Return %",
    "price_vs_SMA50_pct":   "% Deviation from 50-Day Avg",
    "VIX_close":            "VIX — Market Fear Index",
    "yield_spread":         "10Y-3M Treasury Yield Spread (recession signal)",
    "sector_mom_5d":        "Sector ETF 5-Day Momentum",
    "sector_mom_20d":       "Sector ETF 20-Day Momentum",
    "rel_strength_5d":      "Stock vs Sector 5-Day Relative Strength",
    "days_to_earnings":     "Days Until Next Earnings Report",
    "earnings_imminent":    "Earnings Within 5 Days (binary flag)",
}

FEATURE_EXPLANATIONS = {
    "RSI":                  "**RSI** — above 70 = overbought (potential dip); below 30 = oversold (potential bounce); 30–70 = neutral.",
    "MACD":                 "**MACD** — gap between the 12-day and 26-day moving averages. Positive = short-term above long-term trend.",
    "MACD_signal":          "**MACD Signal** — 9-day smoothed average of MACD. Crossovers with MACD generate buy/sell signals.",
    "MACD_hist":            "**MACD Histogram** — gap between MACD and its signal line. Growing bars = building momentum; shrinking = fading.",
    "BB_upper":             "**Bollinger Upper Band** — 20-day avg + 2 std deviations. Price near here can signal overbought conditions.",
    "BB_middle":            "**Bollinger Middle** — the 20-day moving average; a baseline for 'fair value'.",
    "BB_lower":             "**Bollinger Lower Band** — 20-day avg − 2 std deviations. Price near here can signal oversold conditions.",
    "BB_position":          "**Bollinger Position** — 0 = price at the bottom band (cheap vs. recent history), 1 = at the top (expensive).",
    "SMA_20":               "**20-Day Moving Average** — average close price over 20 days. Rising = uptrend.",
    "SMA_50":               "**50-Day Moving Average** — longer-term trend signal. Price crossing above SMA50 is often bullish.",
    "SMA_200":              "**200-Day Moving Average** — the primary bull/bear dividing line. Above = bull regime; below = bear.",
    "price_vs_SMA200_pct":  "**% Above/Below 200-Day Avg** — the model uses this to detect market regimes (bull/bear/sideways).",
    "ROC":                  "**Rate of Change** — % price change over 14 days. Positive = upward momentum; negative = downward.",
    "Volume_SMA":           "**Avg Volume (20-day)** — typical daily trading activity baseline.",
    "Volume_ratio":         "**Volume Ratio** > 1 means heavier-than-normal trading, which tends to confirm the current price move.",
    "ATR_14":               "**ATR-14 (Average True Range)** — how much the price actually moves each day on average, including gaps. High ATR = volatile; low ATR = calm.",
    "Stoch_K":              "**Stochastic %K** — compares today's close to the High-Low range over 14 days. Above 80 = overbought; below 20 = oversold.",
    "Stoch_D":              "**Stochastic %D** — 3-day smoothed version of %K. Crossovers between %K and %D generate signals.",
    "High_Low_ratio":       "**High-Low Ratio** — today's range (High − Low) as a fraction of Close. Large ratio = high intraday volatility.",
    "price_vs_52w_high":    "**52-Week High Ratio** — how close price is to its yearly high. 1.0 = at the high; 0.9 = 10% below.",
    "price_vs_52w_low":     "**52-Week Low Ratio** — how close price is to its yearly low. Useful for spotting turnaround potential.",
    "return_1d":            "**1-Day Return %** — how much the stock moved yesterday. Very short-term momentum.",
    "return_5d":            "**5-Day Return %** — one-week momentum signal.",
    "return_20d":           "**20-Day Return %** — one-month momentum. Captures intermediate trends.",
    "price_vs_SMA50_pct":   "**% Above/Below 50-Day Avg** — positive = price is above its 50-day average (uptrend); negative = below it.",
    "VIX_close":            "**VIX** — the 'fear index'. High VIX = stressed market; low VIX = calm. Useful context for all directional models.",
    "yield_spread":         "**Yield Spread (10Y-3M)** — negative spread often precedes recessions. Model uses this for macro regime context.",
    "sector_mom_5d":        "**Sector 5-Day Momentum** — how the stock's sector ETF moved over the past week. Strong sector = tailwind.",
    "sector_mom_20d":       "**Sector 20-Day Momentum** — monthly trend of the broader sector.",
    "rel_strength_5d":      "**Relative Strength vs Sector** — stock outperforming its sector = positive; underperforming = negative.",
    "days_to_earnings":     "**Days to Earnings** — markets are typically more volatile near earnings. Model accounts for proximity.",
    "earnings_imminent":    "**Earnings Imminent** — binary flag when earnings are ≤ 5 days away. Predicts elevated uncertainty.",
}

HORIZON_LABELS = {"1": "Tomorrow", "3": "3 Days", "5": "5 Days", "10": "10 Days"}
HORIZON_COLORS = {"1": "#2196F3", "3": "#FF9800", "5": "#9C27B0", "10": "#F44336"}
DIRECTION_COLORS = {"up": "#22c55e", "flat": "#f59e0b", "down": "#ef4444"}
DIRECTION_ARROWS = {"up": "↑", "flat": "→", "down": "↓"}


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return {}


@st.cache_data(ttl=300)
def load_results():
    return (
        _load_json("predictions.json"),
        _load_json("feature_importance.json"),
        _load_json("model_metrics.json"),
        _load_json("trade_logs.json"),
        _load_prediction_history(),
    )


def _load_prediction_history() -> list:
    path = RESULTS_DIR / "prediction_history.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        data = yf.download(ticker, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"])
        return data
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def bday(date: pd.Timestamp, n: int) -> pd.Timestamp:
    return date + pd.tseries.offsets.BDay(n)


# ── CSS & page setup ──────────────────────────────────────────────────────────

def setup_page():
    st.set_page_config(
        page_title=settings.STREAMLIT_PAGE_TITLE,
        page_icon=settings.STREAMLIT_PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    .fc-card {
        border-radius: 14px;
        padding: 22px 26px 18px;
        margin-bottom: 4px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.10);
        position: relative;
        overflow: hidden;
    }
    .fc-card::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 6px;
        border-radius: 14px 0 0 14px;
    }
    .fc-up   { background: #f0fdf4; }
    .fc-down { background: #fff5f5; }
    .fc-flat { background: #fffbeb; }
    .fc-up::before   { background: #22c55e; }
    .fc-down::before { background: #ef4444; }
    .fc-flat::before { background: #f59e0b; }

    .fc-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 10px;
    }
    .fc-direction {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .fc-direction-up   { color: #15803d !important; }
    .fc-direction-down { color: #b91c1c !important; }
    .fc-direction-flat { color: #92400e !important; }
    .fc-confidence {
        font-size: 14px;
        color: #555 !important;
        margin-bottom: 10px;
    }
    .fc-probbar-wrap {
        margin-top: 10px;
    }
    .fc-probbar-label {
        font-size: 11px;
        color: #666 !important;
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }
    .fc-probbar-track {
        background: #e5e7eb;
        border-radius: 4px;
        height: 7px;
        margin-bottom: 5px;
        position: relative;
        overflow: hidden;
    }
    .fc-probbar-fill {
        height: 100%;
        border-radius: 4px;
        position: absolute;
        left: 0; top: 0;
    }
    .fc-regime {
        font-size: 11px;
        color: #888 !important;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar(tickers: list) -> dict:
    st.sidebar.title("⚙️ Settings")
    ticker       = st.sidebar.selectbox("Stock to analyse", tickers, index=0)
    period_map   = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year":  "1y",  "2 Years":  "2y",
    }
    period_label = st.sidebar.selectbox("Chart history", list(period_map.keys()), index=3)
    top_n        = st.sidebar.slider("Indicators shown in chart", 3, 32, 10)
    st.sidebar.divider()
    st.sidebar.caption("Re-run `python main.py` to refresh forecasts.")
    return {"ticker": ticker, "period": period_map[period_label], "top_n": top_n}


# ── Forecast cards ────────────────────────────────────────────────────────────

def _prob_bar_html(label: str, pct: float, color: str) -> str:
    width = max(0, min(100, pct * 100))
    return f"""
<div class="fc-probbar-wrap">
  <div class="fc-probbar-label"><span>{label}</span><span>{pct:.0%}</span></div>
  <div class="fc-probbar-track">
    <div class="fc-probbar-fill" style="width:{width:.1f}%;background:{color};"></div>
  </div>
</div>"""


def _card_html(label: str, direction: str, confidence: float,
               probabilities: dict, regime: str) -> str:
    d         = direction.lower()
    card_cls  = f"fc-{d}"
    dir_cls   = f"fc-direction-{d}"
    arrow     = DIRECTION_ARROWS.get(d, "→")
    color     = DIRECTION_COLORS.get(d, "#888")

    prob_bars = (
        _prob_bar_html("↑ Up",   probabilities.get("up",   0), "#22c55e") +
        _prob_bar_html("→ Flat", probabilities.get("flat", 0), "#f59e0b") +
        _prob_bar_html("↓ Down", probabilities.get("down", 0), "#ef4444")
    )

    regime_html = (
        f'<div class="fc-regime">Market regime: <b>{regime}</b></div>'
        if regime else ""
    )

    return f"""
<div class="fc-card {card_cls}">
    <div class="fc-label">{label}</div>
    <div class="fc-direction {dir_cls}">{arrow}  {d.upper()}</div>
    <div class="fc-confidence">Confidence: <b>{confidence:.0%}</b></div>
    {prob_bars}
    {regime_html}
</div>
"""


def render_forecast_cards(horizons_data: dict):
    """2 × 2 grid of forecast cards — one per horizon."""
    if not horizons_data:
        st.info("No forecast data — run `python main.py` first.")
        return

    horizons    = sorted(horizons_data.keys(), key=int)
    left, right = st.columns(2, gap="large")

    for idx, h_str in enumerate(horizons):
        pred      = horizons_data[h_str]
        direction = pred.get("direction", "flat")
        confidence = pred.get("confidence", 0.0)
        probs     = pred.get("probabilities", {})
        regime    = pred.get("regime", "")
        label     = HORIZON_LABELS.get(h_str, f"{h_str} days")
        col       = left if idx % 2 == 0 else right
        with col:
            st.markdown(
                _card_html(label, direction, confidence, probs, regime),
                unsafe_allow_html=True,
            )


# ── Price chart ───────────────────────────────────────────────────────────────

def render_price_chart(stock_df: pd.DataFrame, ticker: str,
                       horizons_data: dict, last_close: float | None):
    if stock_df.empty:
        st.warning(f"No price data for {ticker}")
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.04,
        subplot_titles=(f"{ticker} — Price History + Directional Forecast",
                        "Trading Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=stock_df["Date"],
        open=stock_df["Open"], high=stock_df["High"],
        low=stock_df["Low"],   close=stock_df["Close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    sma20 = stock_df["Close"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=stock_df["Date"], y=sma20,
        name="20-Day Avg", line=dict(color="#f97316", width=1.5, dash="dot"),
    ), row=1, col=1)

    sma50 = stock_df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=stock_df["Date"], y=sma50,
        name="50-Day Avg", line=dict(color="#8b5cf6", width=1.5, dash="dot"),
    ), row=1, col=1)

    # Directional forecast markers — diamonds at future dates at last_close level,
    # color-coded by direction, labeled with horizon + confidence.
    if horizons_data and last_close:
        last_date = stock_df["Date"].iloc[-1]
        for h_str in sorted(horizons_data.keys(), key=int):
            pred       = horizons_data[h_str]
            direction  = pred.get("direction", "flat").lower()
            confidence = pred.get("confidence", 0.0)
            color      = DIRECTION_COLORS.get(direction, "#888")
            arrow      = DIRECTION_ARROWS.get(direction, "→")
            fut_date   = bday(last_date, int(h_str))
            label      = HORIZON_LABELS.get(h_str, f"{h_str}d")

            # Dashed line from last close to forecast point
            fig.add_trace(go.Scatter(
                x=[last_date, fut_date],
                y=[last_close, last_close],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False,
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[fut_date],
                y=[last_close],
                mode="markers+text",
                marker=dict(symbol="diamond", size=14, color=color,
                            line=dict(color="white", width=2)),
                text=[f"  {label}  {arrow} {direction.upper()} ({confidence:.0%})"],
                textposition="middle right",
                textfont=dict(size=11, color=color),
                name=label,
            ), row=1, col=1)

    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(stock_df["Close"], stock_df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=stock_df["Date"], y=stock_df["Volume"],
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        margin=dict(t=80, b=20, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Volume",   row=2, col=1, gridcolor="#f0f0f0")
    fig.update_xaxes(gridcolor="#f0f0f0")

    st.plotly_chart(fig, use_container_width=True)


# ── RSI chart ─────────────────────────────────────────────────────────────────

def render_rsi_chart(stock_df: pd.DataFrame):
    if stock_df.empty or len(stock_df) < 15:
        return
    close = stock_df["Close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    last_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
    if last_rsi is None:
        return

    status = ("Overbought — could pull back" if last_rsi > 70
              else "Oversold — could bounce" if last_rsi < 30
              else "Neutral")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_df["Date"], y=rsi,
        line=dict(color="#3b82f6", width=2), name="RSI",
        fill="tozeroy", fillcolor="rgba(59,130,246,0.05)",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                  annotation_text="Overbought (70)", annotation_position="top left")
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                  annotation_text="Oversold (30)", annotation_position="bottom left")
    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.04, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#22c55e", opacity=0.04, line_width=0)
    fig.update_layout(
        title=dict(text=f"RSI Momentum — currently <b>{last_rsi:.1f}</b>  ·  {status}",
                   font=dict(size=14)),
        yaxis=dict(title="RSI", range=[0, 100], gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="#f0f0f0"),
        height=230,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=50, b=20, l=10, r=10),
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("RSI above 70 = overbought · below 30 = oversold · 30–70 = neutral")


# ── Feature importance ────────────────────────────────────────────────────────

def render_feature_importance(importance: dict, ticker: str, top_n: int):
    if not importance:
        st.info("No feature data — run `python main.py` first.")
        return

    items  = list(importance.items())[:top_n]
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    scores = [v for _, v in items]
    max_s  = max(scores) or 1
    colors = [
        f"rgba(59,130,246,{0.25 + 0.75*(s/max_s):.2f})" for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(
            text=f"Which indicators matter most for {ticker}?",
            font=dict(size=15),
        ),
        xaxis=dict(title="Influence on prediction  (higher = more weight)",
                   gridcolor="#f0f0f0"),
        yaxis=dict(automargin=True),
        height=max(320, top_n * 46),
        template="plotly_white",
        margin=dict(l=10, r=80, t=60, b=40),
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 What do these indicators mean?"):
        for k, _ in items:
            if k in FEATURE_EXPLANATIONS:
                st.markdown(f"- {FEATURE_EXPLANATIONS[k]}")


# ── Model metrics ──────────────────────────────────────────────────────────────

def render_metrics(metrics: dict, ticker: str):
    if not metrics or ticker not in metrics:
        st.info("No model metrics — run `python main.py` first.")
        return

    ticker_m = metrics[ticker]
    horizons = sorted(ticker_m.keys(), key=int)

    st.info(
        "**How to read these numbers:**  \n"
        "**Accuracy** — what % of the time the model correctly called UP / FLAT / DOWN "
        "(3 classes; random guessing = 33%).  \n"
        "**F1 Score** — balanced accuracy across all three classes, "
        "accounting for class imbalance. 1.0 = perfect; 0.33 = random.  \n"
        "**LOOCV** — temporal Leave-One-Out Cross-Validation across rolling "
        "expanding windows; the most honest estimate of real-world performance."
    )

    st.write("")

    for h_str in horizons:
        label = HORIZON_LABELS.get(h_str, f"{h_str} days")
        m     = ticker_m[h_str]
        val   = m.get("val",   {})
        test  = m.get("test",  {})
        loocv = m.get("loocv", {})

        with st.expander(f"📅  {label} model", expanded=(h_str == "1")):
            st.write("")

            c1, gap_col, c2 = st.columns([5, 1, 5])

            with c1:
                st.markdown("#### 🔧 Validation")
                st.caption("Used only for early stopping — not for evaluation.")
                st.write("")
                st.metric("Accuracy",  f"{val.get('accuracy', 0)*100:.1f} %",
                          help="% of val days called correctly (up/flat/down).")
                st.write("")
                st.metric("F1 Score",  f"{val.get('f1', 0):.3f}",
                          help="Macro-averaged F1 across 3 classes.")

            with c2:
                st.markdown("#### 🧪 Test — unseen data")
                st.caption("Recent data the model never saw. This is what matters.")
                st.write("")
                st.metric("Accuracy",  f"{test.get('accuracy', 0)*100:.1f} %",
                          help="% of test days called correctly.")
                st.write("")
                st.metric("F1 Score",  f"{test.get('f1', 0):.3f}",
                          help="Macro-averaged F1 on test set.")

            if loocv:
                st.divider()
                st.markdown("#### 🔄 Temporal LOOCV")
                st.caption(
                    "Each fold trains on all data up to a point and tests on "
                    "the single next sample — no look-ahead bias."
                )
                st.write("")
                lc1, gap2, lc2 = st.columns([5, 1, 5])
                with lc1:
                    st.metric(
                        "LOOCV Accuracy",
                        f"{loocv.get('accuracy', 0)*100:.1f} %",
                    )
                with lc2:
                    st.metric(
                        "LOOCV F1",
                        f"{loocv.get('f1', 0):.3f}",
                    )
                n_folds = loocv.get("n_folds")
                if n_folds:
                    st.caption(f"Evaluated over {n_folds} folds.")

            st.write("")

    st.divider()
    st.markdown("#### ⚙️ Model Settings")
    st.write("")
    p = settings.XGBOOST_CLASSIFIER_PARAMS.copy()
    p.pop("n_jobs", None)
    setting_rows = [
        {"Setting": k, "Value": str(v)}
        for k, v in {
            "Number of trees":       p.get("n_estimators"),
            "Max tree depth":        p.get("max_depth"),
            "Learning rate":         p.get("learning_rate"),
            "L1 regularization":     p.get("reg_alpha"),
            "L2 regularization":     p.get("reg_lambda"),
            "Row sampling per tree": p.get("subsample"),
            "Feature sampling":      p.get("colsample_bytree"),
            "Min leaf weight":       p.get("min_child_weight"),
            "Early stopping":        f"{p.get('early_stopping_rounds')} trees without improvement",
            "Total features":        "32 (technical + macro + earnings + regime)",
            "Ensemble":              "XGBClassifier + LogisticRegression (val-accuracy weighted)",
            "Target":                "3-class: up / flat / down (per-horizon deadband)",
            "Forecast horizons":     ", ".join(
                HORIZON_LABELS.get(str(h), f"{h}d")
                for h in settings.FORECAST_HORIZONS
            ),
        }.items()
    ]
    st.dataframe(pd.DataFrame(setting_rows), use_container_width=True, hide_index=True)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    setup_page()

    st.title("📈 Stock Forecaster")
    st.caption(
        "XGBoost + LogisticRegression ensemble · 32 features · "
        "Ternary classification (UP / FLAT / DOWN) · "
        "Regime-aware models (bull / bear / sideways) · "
        "Diamonds (◆) on chart show each forecast horizon"
    )
    st.divider()

    predictions, feature_importance, metrics, trade_logs, pred_history = load_results()
    tickers = settings.STOCK_TICKERS.split(",")
    cfg     = render_sidebar(tickers)
    ticker  = cfg["ticker"]

    ticker_pred   = predictions.get(ticker, {})
    horizons_data = ticker_pred.get("horizons", {})

    if not predictions:
        st.warning(
            "No forecast data. Run `OMP_NUM_THREADS=1 python main.py` "
            "from the `stock_forecaster/` directory."
        )

    # ── Ticker + current price ────────────────────────────────────────────────
    snap       = fetch_stock_data(ticker, "5d")
    last_close = float(snap["Close"].iloc[-1]) if not snap.empty else None
    ts         = ticker_pred.get("timestamp", "")

    info_col, _ = st.columns([3, 7])
    with info_col:
        st.metric(
            label=ticker,
            value=f"${last_close:,.2f}" if last_close else "Loading…",
        )
    if ts:
        st.caption(f"Forecast generated: {ts[:19].replace('T', ' ')}")

    st.write("")

    # ── Forecast cards ────────────────────────────────────────────────────────
    render_forecast_cards(horizons_data)

    st.write("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊  Price Chart", "🤖  Model Performance", "🔍  All Tickers", "⚡  Trade Execution", "📈  Track Record"]
    )

    with tab1:
        stock_df = fetch_stock_data(ticker, cfg["period"])
        st.write("")
        render_price_chart(stock_df, ticker, horizons_data, last_close)

        st.write("")
        st.divider()
        st.markdown("#### RSI Momentum")
        render_rsi_chart(stock_df)

        st.divider()
        st.markdown("#### Which indicators drove the forecast?")
        st.caption(
            "Scores from the 1-day model. "
            "Longer bar = the model leaned on this indicator more heavily."
        )
        st.write("")
        render_feature_importance(
            feature_importance.get(ticker, {}), ticker, cfg["top_n"]
        )

    with tab2:
        st.write("")
        st.markdown(f"### Model Performance — {ticker}")
        st.write("")
        render_metrics(metrics, ticker)

    with tab3:
        st.write("")
        st.markdown("### Multi-Day Forecasts — All Tickers")
        st.write("")
        if predictions:
            snap_cache = {}
            rows = []
            for t, t_pred in predictions.items():
                h_data = t_pred.get("horizons", {})
                if t not in snap_cache:
                    tmp = fetch_stock_data(t, "5d")
                    snap_cache[t] = float(tmp["Close"].iloc[-1]) if not tmp.empty else None
                lc  = snap_cache[t]
                row = {"Ticker": t, "Current Price": f"${lc:,.2f}" if lc else "—"}
                for h_str in ["1", "3", "5", "10"]:
                    lbl = HORIZON_LABELS.get(h_str, f"{h_str}d")
                    if h_str in h_data:
                        p     = h_data[h_str]
                        d     = p.get("direction", "?").upper()
                        conf  = p.get("confidence", 0.0)
                        arrow = DIRECTION_ARROWS.get(p.get("direction", "").lower(), "?")
                        row[lbl] = f"{arrow} {d}  ({conf:.0%})"
                    else:
                        row[lbl] = "—"
                row["Regime"]  = h_data.get("1", {}).get("regime", "—")
                row["Updated"] = t_pred.get("timestamp", "")[:16].replace("T", " ")
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions yet — run the pipeline first.")

        st.write("")
        with st.expander("📖 Full Glossary — plain English explanations"):
            st.markdown("""
**Tomorrow / 3 Days / 5 Days / 10 Days**
Each is a separate per-regime ensemble model (XGBoost + Logistic Regression)
trained to predict whether the stock will be UP, FLAT, or DOWN after *N* trading days.

**Direction** — UP / FLAT / DOWN. Classified using a per-horizon deadband:
±0.3% for 1d, ±0.7% for 3d, ±1.0% for 5d, ±1.5% for 10d.

**Confidence** — the winning class probability from the ensemble.

**Probability Bars** — full 3-class probability distribution (up / flat / down).
All three bars sum to 100%.

**Regime** — bull (close > SMA50 > SMA200), bear (reverse), or sideways.
The model selects a regime-specific sub-model when enough training data exists.

**Accuracy** — % of time the model called the correct direction (3 classes;
random = 33%); above 40%+ consistently is a meaningful edge.

**F1 Score** — macro-averaged F1 across all three classes, accounting for class
imbalance. 1.0 = perfect; 0.33 = random.

**Temporal LOOCV** — expanding-window Leave-One-Out CV: each fold trains on
all past data and tests on a single future sample. No look-ahead bias.

---
**RSI** — 0–100. Above 70 = overbought; below 30 = oversold; 30–70 = neutral.

**MACD** — gap between 12-day and 26-day averages. Positive = short-term above long-term.

**Bollinger Bands** — price channel ±2 std deviations around the 20-day average.
Near upper = potentially expensive; near lower = potentially cheap.

**ATR-14** — average daily price range including gaps. High = volatile; low = calm.

**Stochastic %K / %D** — close vs. High-Low range over 14 days.
Above 80 = overbought; below 20 = oversold.

**52-Week Ratios** — where today's price sits within its yearly range.

**1 / 5 / 20-Day Return %** — recent momentum signals.

**VIX** — the 'fear index'. High = stressed market; model uses it for macro context.

**Yield Spread (10Y-3M)** — negative spread often precedes recessions.

**Sector ETF Momentum** — how the stock's sector (e.g., QQQ for tech) is trending.

**Days to Earnings / Earnings Imminent** — captures pre-earnings volatility.

**Volume Ratio** — today's volume vs. 20-day average. Above 1.5 = unusually heavy.
""")

    with tab4:
        st.write("")
        st.markdown("### ⚡ Stage 4 — ATR-Based Trade Execution")
        st.caption(
            "Each ML prediction is cross-checked against 4 technical indicators before a trade fires. "
            "The final decision uses a blended score: 65% ML model confidence + 35% technical confirmation. "
            "A trade only executes when the blended score clears 50% and the model has a clear direction (UP or DOWN)."
        )
        st.write("")

        if not trade_logs:
            st.info("No trade log data yet — run the pipeline first (`python main.py`).")
        else:
            active  = [v for v in trade_logs.values() if v.get("action") not in ("skip", None)]
            skipped = [v for v in trade_logs.values() if v.get("action") == "skip"]

            # ── Run timestamp ────────────────────────────────────────────────
            any_ts = next((v.get("timestamp") for v in trade_logs.values() if v.get("timestamp")), None)
            if any_ts:
                ts_str = any_ts[:19].replace("T", " ")
                st.caption(f"Last pipeline run: {ts_str} UTC")

            # ── Summary metrics ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tickers Evaluated", len(trade_logs))
            m2.metric("Active Trades", len(active),
                      help="Trades that passed both confidence threshold and direction check")
            m3.metric("Skipped", len(skipped),
                      help="Insufficient confidence, flat direction, or no clear ML signal")
            total_risk   = sum(v.get("dollar_risk",   0) for v in active)
            total_reward = sum(v.get("dollar_reward", 0) for v in active)
            m4.metric("Total Exposure", f"${total_risk:,.0f} risked",
                      help=f"Max reward if all TPs hit: ${total_reward:,.0f}")

            st.divider()

            # ── Active trade cards ───────────────────────────────────────────
            if active:
                st.markdown("#### Active Trade Signals")
                st.caption(
                    "These tickers passed all filters. Position size is calculated so that "
                    "hitting the Stop-Loss costs exactly 1% of the $100,000 account ($1,000)."
                )
                st.write("")

                SIGNAL_LABELS = {
                    "ichimoku_tk":    ("Ichimoku T/K Cross",
                                       "Tenkan-sen vs Kijun-sen crossover aligns with direction"),
                    "ichimoku_cloud": ("Ichimoku Cloud",
                                       "Price is above (long) or below (short) both Senkou A & B"),
                    "adx_di":         ("ADX Trend Strength",
                                       "ADX > 20 confirms an active trend; +DI/-DI confirm direction"),
                    "bbw_regime":     ("BBW Volatility Gate",
                                       "Bollinger Band Width 0.01–0.12: not too quiet, not in panic mode"),
                }

                for log in active:
                    is_long      = log["action"] == "long"
                    action_color = "#22c55e" if is_long else "#ef4444"
                    action_label = "LONG  ↑" if is_long else "SHORT  ↓"
                    direction_word = "long (buy)" if is_long else "short (sell)"

                    st.markdown(
                        f"<div style='border-left:4px solid {action_color};"
                        f"padding:10px 16px;border-radius:4px;"
                        f"background:rgba(0,0,0,0.03);margin-bottom:8px'>"
                        f"<span style='background:{action_color};color:white;"
                        f"padding:3px 12px;border-radius:4px;font-weight:bold;font-size:0.95em'>"
                        f"{action_label}</span>"
                        f"&nbsp;&nbsp;<span style='font-size:1.2em;font-weight:bold'>"
                        f"{log['ticker']}</span>"
                        f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.85em'>"
                        f"1-day horizon · ATR ${log.get('atr_used', 0):.2f}/day</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Price levels
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Entry Price",
                        f"${log['entry_price']:,.2f}",
                        help="Last closing price — this is where the trade would open",
                    )
                    # SL distance label is direction-aware: SHORT has SL above entry
                    sl_dist = log["sl_distance"]
                    sl_dir  = f"+${sl_dist:.2f} above entry" if not is_long else f"-${sl_dist:.2f} below entry"
                    c2.metric(
                        "Stop-Loss",
                        f"${log['stop_loss']:,.2f}",
                        delta=sl_dir,
                        delta_color="off",
                        help=f"Exit with a loss if price reaches this level ({sl_dir}). "
                             f"Distance = 2 × ATR (${log.get('atr_used',0):.2f})",
                    )
                    tp_dist = log["tp_distance"]
                    tp_dir  = f"-${tp_dist:.2f} below entry" if not is_long else f"+${tp_dist:.2f} above entry"
                    c3.metric(
                        "Take-Profit",
                        f"${log['take_profit']:,.2f}",
                        delta=tp_dir,
                        delta_color="off",
                        help=f"Exit with a profit if price reaches this level ({tp_dir}). "
                             f"Distance = 3 × ATR (${log.get('atr_used',0):.2f})",
                    )
                    c4.metric(
                        "Position Size",
                        f"{log['position_size']} shares",
                        help=f"Sized so a stop-out costs exactly ${log['dollar_risk']:,.0f} (1% of account). "
                             f"Risk/Reward = 1:{log['risk_reward_ratio']}",
                    )

                    # Risk / reward row
                    rr1, rr2, rr3 = st.columns(3)
                    rr1.metric("$ at Risk",       f"${log['dollar_risk']:,.0f}",
                               help="Max loss if Stop-Loss is hit (1% of $100k account)")
                    rr2.metric("$ Potential Gain", f"${log['dollar_reward']:,.0f}",
                               help="Profit if Take-Profit is reached")
                    rr3.metric("Risk / Reward",    f"1 : {log['risk_reward_ratio']}",
                               help="For every $1 risked, the target profit is $1.50")

                    st.write("")

                    # Confidence blend
                    ml_c  = log.get("ml_confidence", 0)
                    tc    = log.get("tech_conf_score", 0)
                    bl_c  = log.get("blended_confidence", 0)
                    sigs  = log.get("confirming_signals", "?")
                    st.markdown(
                        f"**Confidence breakdown** &nbsp;—&nbsp; "
                        f"ML model: **{ml_c:.1%}** &nbsp;×&nbsp; 65% &nbsp;+&nbsp; "
                        f"Technical confirmation: **{tc:.1%}** &nbsp;×&nbsp; 35% "
                        f"&nbsp;=&nbsp; Blended: **{bl_c:.1%}** &nbsp;|&nbsp; "
                        f"Indicators aligned: **{sigs}**"
                    )
                    st.progress(min(bl_c, 1.0), text=f"Blended confidence {bl_c:.1%} (threshold 50%)")

                    # Signal breakdown
                    detail = log.get("tech_signals", {})
                    if detail:
                        st.write("")
                        st.markdown("**Technical signal breakdown:**")
                        sig_cols = st.columns(len(detail))
                        for col, (key, confirmed) in zip(sig_cols, detail.items()):
                            label, tooltip = SIGNAL_LABELS.get(key, (key, ""))
                            icon  = "✅" if confirmed else "❌"
                            color = "#22c55e" if confirmed else "#ef4444"
                            result_word = "Confirmed" if confirmed else "Not confirmed"
                            col.markdown(
                                f"<div style='text-align:center;padding:8px;border-radius:6px;"
                                f"border:1px solid {color}20;background:{color}10'>"
                                f"<div style='font-size:1.4em'>{icon}</div>"
                                f"<div style='font-weight:bold;font-size:0.8em;margin-top:4px'>{label}</div>"
                                f"<div style='color:{color};font-size:0.75em'>{result_word}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                                help=tooltip,
                            )

                    st.write("")
                    st.divider()

            else:
                st.info(
                    "No active trades this run — all tickers were either flat, "
                    "or the blended confidence was below the 50% threshold."
                )
                st.divider()

            # ── Skipped trades table ─────────────────────────────────────────
            st.markdown("#### Why Each Ticker Was Skipped")
            st.caption(
                "Tickers skipped for 'FLAT direction' never reached the blending stage — "
                "the ML model itself couldn't pick a clear direction, so no trade is warranted."
            )

            REASON_LABELS = {
                "direction_flat": "Model direction uncertain (FLAT) — ML couldn't pick UP or DOWN",
            }

            skip_rows = []
            for log in skipped:
                raw_reason = log.get("reason", "—")
                direction  = log.get("direction", "flat").upper()
                is_flat    = raw_reason == "direction_flat"

                # Prettify reason string
                if raw_reason in REASON_LABELS:
                    pretty_reason = REASON_LABELS[raw_reason]
                elif raw_reason.startswith("low_blended_confidence"):
                    bl = log.get("blended_confidence", 0)
                    pretty_reason = f"Blended confidence too low ({bl:.1%} < 50% threshold)"
                else:
                    pretty_reason = raw_reason

                skip_rows.append({
                    "Ticker":        log["ticker"],
                    "ML Direction":  direction,
                    "ML Confidence": f"{log.get('ml_confidence', log.get('confidence', 0)):.1%}",
                    "Tech Score":    "N/A (skipped early)" if is_flat else f"{log.get('tech_conf_score', 0):.1%}",
                    "Blended":       "N/A (skipped early)" if is_flat else f"{log.get('blended_confidence', 0):.1%}",
                    "Signals":       "N/A" if is_flat else log.get("confirming_signals", "—"),
                    "Skip Reason":   pretty_reason,
                })
            st.dataframe(pd.DataFrame(skip_rows), use_container_width=True, hide_index=True)

            st.write("")
            with st.expander("📖 How the execution engine works — plain English"):
                st.markdown("""
#### Step 1 — ML Direction check
The model must predict either **UP** or **DOWN** with any confidence. If it outputs **FLAT**
(too uncertain to call), the ticker is skipped immediately — no point blending indecision.

#### Step 2 — Technical indicator confirmation (4 signals)
Four independent chart signals are checked. Each one either agrees or disagrees with the
ML direction. The fraction that agree becomes the **technical score** (0% = none agree, 100% = all agree).

| Signal | What it checks |
|---|---|
| **Ichimoku T/K Cross** | Tenkan-sen (9-period midline) crossed above/below Kijun-sen (26-period), and price confirmed the move |
| **Ichimoku Cloud** | Price is fully above both Senkou A & B (bullish) or fully below (bearish) — cloud acts as a support/resistance zone |
| **ADX Trend Strength** | ADX > 20 means an actual trend exists (not just noise); +DI > −DI confirms it's an uptrend, and vice versa |
| **BBW Volatility Gate** | Bollinger Band Width between 1%–12%: too narrow = price coiling with no breakout yet; too wide = panic/spike, unreliable signals |

#### Step 3 — Confidence blending
> **Blended = ML confidence × 0.65 + Technical score × 0.35**

A strong ML signal with weak chart support gets modestly boosted or held back.
A 52% ML signal with 3/4 indicators confirming → ~62% blended. Below 50% → skip.

#### Step 4 — Position sizing (ATR-based)
> **Shares = floor($1,000 risk ÷ Stop-Loss distance)**

Stop-Loss distance = **2 × ATR** (the average daily price range).
Take-Profit = **3 × ATR** above/below entry → always a **1.5:1 reward/risk ratio**.
$1,000 = 1% of the $100,000 simulated account. Maximum loss per trade is always $1,000.
""")


    with tab5:
        st.write("")
        st.markdown("### 📈 Real-World Prediction Track Record")
        st.caption(
            "Every time the pipeline runs, predictions are saved with an expected outcome date. "
            "When that date arrives, the next run fetches the actual closing price, applies the same "
            "deadband thresholds used during training, and records whether the call was right or wrong. "
            "This is the only accuracy that matters — not backtest accuracy, but live predictions."
        )
        st.write("")

        if not pred_history:
            st.info(
                "No prediction history yet. Run the pipeline once to start recording. "
                "Outcomes are resolved automatically on subsequent runs once each horizon passes."
            )
        else:
            resolved  = [r for r in pred_history if r["status"] in ("correct", "incorrect")]
            pending   = [r for r in pred_history if r["status"] == "pending"]

            # ── Top-level accuracy metrics ────────────────────────────────
            if resolved:
                n_correct = sum(1 for r in resolved if r["status"] == "correct")
                overall_acc = n_correct / len(resolved)

                a1, a2, a3, a4 = st.columns(4)
                a1.metric(
                    "Overall Accuracy",
                    f"{overall_acc:.1%}",
                    help="Across all tickers and all horizons. Random baseline = 33%.",
                )
                a2.metric("Correct",   n_correct,    help="Predictions that matched the actual direction")
                a3.metric("Incorrect", len(resolved) - n_correct)
                a4.metric("Pending",   len(pending),  help="Horizon not yet reached — outcome unknown")

                st.write("")

                # ── Accuracy by horizon ───────────────────────────────────
                st.markdown("#### Accuracy by Forecast Horizon")
                st.caption("A random 3-class model scores 33%. Anything consistently above that has edge.")
                horizon_rows = []
                for h in [1, 3, 5, 10]:
                    h_res = [r for r in resolved if r["horizon_days"] == h]
                    if not h_res:
                        continue
                    h_correct = sum(1 for r in h_res if r["status"] == "correct")
                    h_acc = h_correct / len(h_res)
                    horizon_rows.append({
                        "Horizon":   f"{h} day{'s' if h > 1 else ''}",
                        "Evaluated": len(h_res),
                        "Correct":   h_correct,
                        "Accuracy":  f"{h_acc:.1%}",
                        "vs Random": f"{(h_acc - 0.333):+.1%}",
                    })
                if horizon_rows:
                    st.dataframe(
                        pd.DataFrame(horizon_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.write("")

                # ── Accuracy by ticker ────────────────────────────────────
                st.markdown("#### Accuracy by Ticker")
                ticker_rows = []
                for t in sorted({r["ticker"] for r in resolved}):
                    t_res     = [r for r in resolved if r["ticker"] == t]
                    t_correct = sum(1 for r in t_res if r["status"] == "correct")
                    t_acc     = t_correct / len(t_res)
                    ticker_rows.append({
                        "Ticker":    t,
                        "Evaluated": len(t_res),
                        "Correct":   t_correct,
                        "Accuracy":  f"{t_acc:.1%}",
                        "vs Random": f"{(t_acc - 0.333):+.1%}",
                    })
                st.dataframe(
                    pd.DataFrame(ticker_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.write("")

                # ── Recent resolved predictions ───────────────────────────
                st.markdown("#### Recent Resolved Predictions")
                recent = sorted(resolved, key=lambda r: r.get("predicted_at", ""), reverse=True)[:30]
                rec_rows = []
                for r in recent:
                    pct = r.get("actual_pct_change")
                    rec_rows.append({
                        "Date":      r.get("predicted_at", "")[:10],
                        "Ticker":    r["ticker"],
                        "Horizon":   f"{r['horizon_days']}d",
                        "Predicted": r["predicted_direction"].upper(),
                        "Conf":      f"{r['predicted_confidence']:.1%}",
                        "Actual":    r.get("actual_direction", "—").upper(),
                        "Move %":    f"{pct:+.2f}%" if pct is not None else "—",
                        "Result":    "✅ Correct" if r["status"] == "correct" else "❌ Wrong",
                    })
                st.dataframe(
                    pd.DataFrame(rec_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    f"**{len(pending)} prediction(s) pending.** "
                    "No outcomes have resolved yet — the model needs at least one horizon "
                    "to pass before accuracy can be measured. "
                    "For 1-day predictions, run the pipeline again tomorrow."
                )

            # ── Pending predictions ───────────────────────────────────────
            if pending:
                st.write("")
                with st.expander(f"⏳ {len(pending)} pending prediction(s) — awaiting outcome"):
                    pend_rows = []
                    for r in sorted(pending, key=lambda x: x.get("outcome_date", "")):
                        pend_rows.append({
                            "Ticker":         r["ticker"],
                            "Horizon":        f"{r['horizon_days']}d",
                            "Predicted":      r["predicted_direction"].upper(),
                            "Confidence":     f"{r['predicted_confidence']:.1%}",
                            "Entry Price":    f"${r['entry_price']:,.2f}" if r.get("entry_price") else "—",
                            "Outcome Date":   r.get("outcome_date", "—"),
                        })
                    st.dataframe(
                        pd.DataFrame(pend_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.write("")
            with st.expander("📖 How outcomes are resolved"):
                st.markdown(f"""
The pipeline uses the **same deadband thresholds** during outcome resolution as it does when
creating training labels. This ensures the accuracy score is apples-to-apples with what the
model was actually trained to predict.

| Horizon | Deadband | Meaning |
|---|---|---|
| 1 day | ±0.3% | Move < 0.3% either way = FLAT |
| 3 days | ±0.7% | Move < 0.7% = FLAT |
| 5 days | ±1.0% | Move < 1.0% = FLAT |
| 10 days | ±1.5% | Move < 1.5% = FLAT |

**Outcome date** is calculated as the N-th business day after the prediction date,
using pandas `bdate_range` (weekends excluded; note: no holiday calendar applied).

**Why 33% is the random baseline** — the model predicts one of three classes (UP, FLAT, DOWN).
A coin-flip model scores 33%. Any consistent reading above ~38–40% over many predictions
suggests the model is finding a real signal.
""")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Stock Forecaster  ·  32-feature XGBoost+LR ensemble  ·  "
        f"Ternary classification · Regime-aware models  ·  "
        f"Refreshed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
