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
    )


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

    predictions, feature_importance, metrics = load_results()
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
    tab1, tab2, tab3 = st.tabs(
        ["📊  Price Chart", "🤖  Model Performance", "🔍  All Tickers"]
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

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Stock Forecaster  ·  32-feature XGBoost+LR ensemble  ·  "
        f"Ternary classification · Regime-aware models  ·  "
        f"Refreshed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
