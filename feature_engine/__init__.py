"""
feature_engine — supplementary data and feature layer.

Provides four production-ready modules that expand the base pipeline:
  1. AlphaVantageClient   — async REST client for NEWS_SENTIMENT & TIME_SERIES_INTRADAY
  2. RedditSocialStream   — PRAW-based social volume and keyword-polarity stream
  3. MarketHoursAligner   — NYSE market-hours resampling and overnight sentiment alignment
  4. TechnicalIndicatorEngine — pandas-ta indicators (Ichimoku, ATR, ADX, BBW)
  5. ExecutionEngine      — ATR-based stop-loss / take-profit trade simulation
"""

from feature_engine.alpha_vantage import AlphaVantageClient
from feature_engine.reddit_stream import RedditSocialStream
from feature_engine.market_hours import MarketHoursAligner
from feature_engine.tech_indicators import TechnicalIndicatorEngine
from feature_engine.execution import ExecutionEngine

__all__ = [
    "AlphaVantageClient",
    "RedditSocialStream",
    "MarketHoursAligner",
    "TechnicalIndicatorEngine",
    "ExecutionEngine",
]
