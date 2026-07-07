"""yfinance price service with a 300s TTL cache and server-side indicators.

SMA-20/50 and RSI-14 are computed here with the exact formulas the Streamlit
UI used (ui/app.py), so numbers match between the two frontends.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from config import settings

logger = logging.getLogger(__name__)

TTL_SECONDS = 300
ALLOWED_PERIODS = ("5d", "1mo", "3mo", "6mo", "1y", "2y")

_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_lock = threading.Lock()


def allowed_tickers() -> list[str]:
    return [t.strip() for t in settings.STOCK_TICKERS.split(",") if t.strip()]


def _cache_get(key: tuple[str, str], allow_stale: bool = False):
    with _lock:
        entry = _cache.get(key)
    if entry is None:
        return None
    expires_at, payload = entry
    if allow_stale or time.monotonic() < expires_at:
        return payload
    return None


def _cache_put(key: tuple[str, str], payload: Any) -> None:
    with _lock:
        _cache[key] = (time.monotonic() + TTL_SECONDS, payload)


def _download(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def _series_points(dates: pd.Series, values: pd.Series) -> list[dict]:
    """[{time: 'YYYY-MM-DD', value: float}], NaN rows dropped —
    the shape lightweight-charts consumes directly."""
    out = []
    for d, v in zip(dates, values):
        if pd.isna(v):
            continue
        out.append({"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)})
    return out


def compute_rsi(close: pd.Series) -> pd.Series:
    """RSI-14, formula identical to ui/app.py render_rsi_chart."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def get_prices(ticker: str, period: str) -> Optional[dict]:
    """Candles + SMA20/50 + RSI14 for one ticker. Returns cached payload on
    upstream failure if one exists, else None."""
    key = (ticker, period)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        df = _download(ticker, period)
    except Exception as e:
        logger.error(f"yfinance failed for {ticker} {period}: {e}")
        return _cache_get(key, allow_stale=True)

    if df.empty:
        return _cache_get(key, allow_stale=True)

    candles = [
        {
            "time": d.strftime("%Y-%m-%d"),
            "open": round(float(o), 4),
            "high": round(float(h), 4),
            "low": round(float(lo), 4),
            "close": round(float(c), 4),
            "volume": int(v),
        }
        for d, o, h, lo, c, v in zip(
            df["Date"], df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
        )
        if not any(pd.isna(x) for x in (o, h, lo, c))
    ]

    rsi = compute_rsi(df["Close"])
    last_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None

    payload = {
        "ticker": ticker,
        "period": period,
        "last_close": round(float(df["Close"].iloc[-1]), 4),
        "last_rsi": round(last_rsi, 2) if last_rsi is not None else None,
        "candles": candles,
        "sma20": _series_points(df["Date"], df["Close"].rolling(20).mean()),
        "sma50": _series_points(df["Date"], df["Close"].rolling(50).mean()),
        "rsi14": _series_points(df["Date"], rsi),
    }
    _cache_put(key, payload)
    return payload


def get_quotes() -> dict:
    """Latest close for every configured ticker in one batched download."""
    tickers = allowed_tickers()
    key = ("__quotes__", ",".join(tickers))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    quotes: dict[str, dict] = {t: {"price": None, "as_of": None} for t in tickers}
    try:
        data = yf.download(tickers, period="5d", progress=False, group_by="ticker")
        for t in tickers:
            try:
                closes = (data[t]["Close"] if len(tickers) > 1 else data["Close"]).dropna()
                if not closes.empty:
                    quotes[t] = {
                        "price": round(float(closes.iloc[-1]), 4),
                        "as_of": closes.index[-1].strftime("%Y-%m-%d"),
                    }
            except Exception:
                continue
    except Exception as e:
        logger.error(f"yfinance batch quotes failed: {e}")
        stale = _cache_get(key, allow_stale=True)
        if stale is not None:
            return stale

    _cache_put(key, quotes)
    return quotes
