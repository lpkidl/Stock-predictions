"""
feature_engine — supplementary data and feature layer.

Provides four production-ready modules that expand the base pipeline:
  1. AlphaVantageClient   — async REST client for NEWS_SENTIMENT & TIME_SERIES_INTRADAY
  2. RedditSocialStream   — PRAW-based social volume and keyword-polarity stream
  3. MarketHoursAligner   — NYSE market-hours resampling and overnight sentiment alignment
  4. TechnicalIndicatorEngine — pandas-ta indicators (Ichimoku, ATR, ADX, BBW)
  5. ExecutionEngine      — ATR-based stop-loss / take-profit trade simulation
"""

import logging as _logging
_logger = _logging.getLogger(__name__)

# Core modules — always available
from feature_engine.tech_indicators import TechnicalIndicatorEngine
from feature_engine.market_hours import MarketHoursAligner
from feature_engine.execution import ExecutionEngine

# Optional modules — depend on third-party libraries that may not be installed
try:
    from feature_engine.alpha_vantage import AlphaVantageClient
except ImportError:
    AlphaVantageClient = None  # type: ignore
    _logger.debug("aiohttp not installed — AlphaVantageClient unavailable")

try:
    from feature_engine.reddit_stream import RedditSocialStream
except ImportError:
    RedditSocialStream = None  # type: ignore
    _logger.debug("praw not installed — RedditSocialStream unavailable")

__all__ = [
    "AlphaVantageClient",
    "RedditSocialStream",
    "MarketHoursAligner",
    "TechnicalIndicatorEngine",
    "ExecutionEngine",
]
