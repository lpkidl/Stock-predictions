"""
Shared pytest fixtures for the Stock Forecaster test suite.
All fixtures that produce DataFrames use purely synthetic data so tests
never make real network calls.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Synthetic OHLCV data
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, ticker: str = "AAPL", seed: int = 42) -> pd.DataFrame:
    """
    Generate n rows of synthetic daily OHLCV with realistic price movement.
    The series is a geometric random walk so technical indicators are
    non-trivial but reproducible.
    """
    rng = np.random.default_rng(seed)
    base = 150.0
    returns = rng.normal(0.0003, 0.015, n)
    close = base * np.exp(np.cumsum(returns))

    high  = close * (1 + rng.uniform(0.001, 0.02, n))
    low   = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    vol   = rng.integers(5_000_000, 50_000_000, n).astype(float)

    dates = pd.bdate_range(end=date.today(), periods=n)
    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": vol,
        "ticker": ticker,
        "date":   pd.to_datetime(dates),
    })
    return df.reset_index(drop=True)


@pytest.fixture
def stock_df() -> pd.DataFrame:
    """300-row synthetic AAPL OHLCV DataFrame."""
    return _make_ohlcv(300, "AAPL")


@pytest.fixture
def short_stock_df() -> pd.DataFrame:
    """60-row synthetic OHLCV — too short for full indicator warm-up."""
    return _make_ohlcv(60, "AAPL")


@pytest.fixture
def multi_ticker_dfs() -> dict:
    """Synthetic OHLCV for three tickers."""
    return {
        "AAPL": _make_ohlcv(300, "AAPL", seed=1),
        "NVDA": _make_ohlcv(300, "NVDA", seed=2),
        "TSLA": _make_ohlcv(300, "TSLA", seed=3),
    }


# ---------------------------------------------------------------------------
# Synthetic macro data
# ---------------------------------------------------------------------------

@pytest.fixture
def macro_data(stock_df) -> dict:
    """
    Minimal macro DataFrames (VIX, TNX, IRX, QQQ) aligned to stock_df dates.
    All columns match what ingestion/pipeline.py produces.
    """
    dates = stock_df["date"]
    rng   = np.random.default_rng(99)

    def _series(base, std):
        vals = base + rng.normal(0, std, len(dates))
        return pd.DataFrame({"date": dates, "Close": vals, "ticker": "macro"})

    return {
        "^VIX": _series(20.0, 3.0),
        "^TNX": _series(4.5, 0.3),
        "^IRX": _series(5.0, 0.2),
        "QQQ":  _series(400.0, 5.0),
        "XLK":  _series(200.0, 3.0),
        "XLY":  _series(150.0, 2.0),
    }


# ---------------------------------------------------------------------------
# Synthetic news scores
# ---------------------------------------------------------------------------

@pytest.fixture
def news_df() -> pd.DataFrame:
    """Single-row news score DataFrame as produced by NewsAnalyzer."""
    today = pd.Timestamp(date.today())
    return pd.DataFrame([{
        "date":                today,
        "ticker":              "AAPL",
        "news_sentiment":      0.65,
        "news_confidence":     0.80,
        "news_risk_score":     0.20,
        "news_catalyst_score": 0.75,
        "news_themes":         "earnings beat, strong guidance",
        "news_article_count":  8,
    }])


@pytest.fixture
def multi_news_df() -> pd.DataFrame:
    """News scores for two tickers."""
    today = pd.Timestamp(date.today())
    return pd.DataFrame([
        {
            "date": today, "ticker": "AAPL",
            "news_sentiment": 0.65, "news_confidence": 0.80,
            "news_risk_score": 0.20, "news_catalyst_score": 0.75,
            "news_themes": "earnings beat", "news_article_count": 8,
        },
        {
            "date": today, "ticker": "NVDA",
            "news_sentiment": -0.30, "news_confidence": 0.70,
            "news_risk_score": 0.60, "news_catalyst_score": 0.25,
            "news_themes": "regulation concern", "news_article_count": 5,
        },
    ])


# ---------------------------------------------------------------------------
# Sample news articles (as returned by NewsAPI)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_articles() -> list:
    return [
        {
            "title": "Apple beats Q2 earnings estimates",
            "description": "iPhone demand remains strong in China and Europe.",
            "publishedAt": "2026-07-01T14:00:00Z",
            "url": "https://example.com/1",
        },
        {
            "title": "Apple raises full-year guidance",
            "description": "Services revenue hit an all-time high.",
            "publishedAt": "2026-07-01T10:00:00Z",
            "url": "https://example.com/2",
        },
    ]


# ---------------------------------------------------------------------------
# Groq LLM response stub
# ---------------------------------------------------------------------------

@pytest.fixture
def groq_scores() -> dict:
    return {
        "sentiment":      0.72,
        "confidence":     0.85,
        "risk_score":     0.15,
        "catalyst_score": 0.80,
        "themes":         ["earnings beat", "guidance raise", "strong demand"],
    }
