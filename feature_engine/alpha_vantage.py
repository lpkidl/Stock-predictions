"""
AlphaVantageClient — async REST client for Alpha Vantage API.

Covers two endpoints used to expand the feature matrix:
  • NEWS_SENTIMENT        → per-article sentiment scores with ticker relevance
  • TIME_SERIES_INTRADAY  → 60-minute OHLCV bars for intraday alignment

Rate-limit strategy (free tier):
  - Hard cap: 75 calls/minute, 500 calls/day
  - Between each request the client sleeps for (60 / calls_per_minute) seconds,
    ensuring we never exceed the per-minute quota.
  - A monotonic call counter resets at midnight UTC to enforce the daily cap.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# Schema contracts — these column names are the canonical output of this module.
NEWS_COLUMNS   = ["timestamp", "title", "source", "sentiment_score", "relevance_score"]
OHLCV_COLUMNS  = ["timestamp", "open", "high", "low", "close", "volume"]


class AlphaVantageClient:
    """
    Async REST client for Alpha Vantage market data.

    Usage (always via async context manager to manage the aiohttp session):

        async with AlphaVantageClient(api_key="YOUR_KEY") as client:
            news_df  = await client.fetch_news_sentiment("AAPL")
            ohlcv_df = await client.fetch_intraday_ohlcv("AAPL")
    """

    BASE_URL            = "https://www.alphavantage.co/query"
    DAILY_CALL_CAP      = 500

    def __init__(
        self,
        api_key: str,
        calls_per_minute: int = 75,
    ) -> None:
        self._api_key          = api_key
        self._calls_per_minute = calls_per_minute
        # Minimum wall-clock gap between successive HTTP calls (seconds)
        self._min_interval: float = 60.0 / calls_per_minute

        self._last_call_ts: float = 0.0          # monotonic timestamp of last call
        self._daily_call_count: int = 0
        self._reset_date: str = ""               # "YYYY-MM-DD" in UTC
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Async context manager — owns the aiohttp session lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AlphaVantageClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_and_reset_daily_counter(self) -> None:
        """Reset the daily counter at UTC midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._daily_call_count = 0
            self._reset_date = today

    async def _rate_limited_get(self, params: dict) -> dict:
        """
        Enforce per-minute and per-day rate limits, then execute a GET request.

        Sleep strategy:
          elapsed = time since last call (monotonic)
          if elapsed < min_interval → sleep the remainder
          This guarantees ≤ calls_per_minute requests per 60-second rolling window.

        Raises:
          RuntimeError  if the 500-call daily cap has been reached.
          RuntimeError  if the session is not open (client used outside context manager).
        """
        if self._session is None or self._session.closed:
            raise RuntimeError(
                "AlphaVantageClient must be used as an async context manager."
            )

        # Daily cap check
        self._check_and_reset_daily_counter()
        if self._daily_call_count >= self.DAILY_CALL_CAP:
            raise RuntimeError(
                f"Alpha Vantage daily call cap ({self.DAILY_CALL_CAP}) reached. "
                "Resets at midnight UTC."
            )

        # Per-minute throttle
        elapsed = time.monotonic() - self._last_call_ts
        gap     = self._min_interval - elapsed
        if gap > 0:
            logger.debug(f"AV rate limiter: sleeping {gap:.2f}s")
            await asyncio.sleep(gap)

        params["apikey"] = self._api_key
        self._last_call_ts = time.monotonic()
        self._daily_call_count += 1

        async with self._session.get(self.BASE_URL, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    @staticmethod
    def _parse_av_timestamp(raw: str) -> datetime:
        """
        Parse Alpha Vantage's compact timestamp format: '20240105T130000'
        Returns a tz-naive datetime (timestamps from AV are in US Eastern time).
        """
        return datetime.strptime(raw, "%Y%m%dT%H%M%S")

    # ------------------------------------------------------------------
    # Public API: news sentiment
    # ------------------------------------------------------------------

    async def fetch_news_sentiment(
        self,
        ticker: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch financial news articles with per-ticker sentiment scores.

        Alpha Vantage endpoint: NEWS_SENTIMENT
        Response path: response["feed"] → list of article dicts
          Each article has a "ticker_sentiment" list; we extract the entry
          whose ticker_sentiment[i]["ticker"] matches our target ticker.

        Args:
            ticker: Equity ticker (e.g., "AAPL").
            limit:  Max articles to retrieve (capped by AV at 1000; free tier ~200).

        Returns:
            DataFrame with columns: timestamp, title, source,
            sentiment_score (float, -1 to 1), relevance_score (float, 0 to 1).
            Returns an empty but correctly-schema'd DataFrame on any failure.
        """
        empty = pd.DataFrame(columns=NEWS_COLUMNS)

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers":  ticker,
            "limit":    limit,
            "sort":     "LATEST",
        }

        try:
            data = await self._rate_limited_get(params)
        except RuntimeError:
            raise   # propagate cap errors upward
        except Exception as exc:
            logger.error(f"AV news request failed for {ticker}: {exc}")
            return empty

        # AV returns {"Information": "..."} when rate-limited (no HTTP error code)
        if "Information" in data or "Note" in data:
            msg = data.get("Information") or data.get("Note")
            logger.warning(f"AV rate-limit message for {ticker}: {msg}")
            return empty

        feed = data.get("feed")
        if not feed:
            logger.info(f"No news feed returned for {ticker}")
            return empty

        rows = []
        for article in feed:
            # Find the sentiment entry for our specific ticker
            ts_list = article.get("ticker_sentiment", [])
            ts_entry = next(
                (t for t in ts_list if t.get("ticker", "").upper() == ticker.upper()),
                None,
            )
            if ts_entry is None:
                continue

            try:
                rows.append({
                    "timestamp":       self._parse_av_timestamp(article["time_published"]),
                    "title":           article.get("title", ""),
                    "source":          article.get("source", ""),
                    "sentiment_score": float(ts_entry.get("ticker_sentiment_score", 0.0)),
                    "relevance_score": float(ts_entry.get("relevance_score", 0.0)),
                })
            except (KeyError, ValueError) as exc:
                logger.debug(f"Skipping malformed article: {exc}")
                continue

        logger.info(f"AV news: fetched {len(rows)} articles for {ticker}")
        return pd.DataFrame(rows, columns=NEWS_COLUMNS) if rows else empty

    # ------------------------------------------------------------------
    # Public API: intraday OHLCV
    # ------------------------------------------------------------------

    async def fetch_intraday_ohlcv(
        self,
        ticker: str,
        interval: str = "60min",
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV bars from Alpha Vantage.

        Alpha Vantage endpoint: TIME_SERIES_INTRADAY
        Response path: response[f"Time Series ({interval})"] → dict keyed by
          timestamp strings (e.g., "2024-01-05 09:30:00") in US Eastern time.

        Notes:
          • outputsize="full" returns up to 30 days of intraday data (free tier).
          • outputsize="compact" returns the last 100 bars only.
          • The returned DataFrame index is a tz-aware pd.DatetimeIndex in UTC
            (localized from US/Eastern then converted) for internal consistency.

        Args:
            ticker:     Equity ticker symbol.
            interval:   Bar width — "1min" | "5min" | "15min" | "30min" | "60min".
            outputsize: "compact" (last 100 bars) or "full" (up to 30 days).

        Returns:
            DataFrame indexed by UTC timestamps with columns:
            open, high, low, close (float) and volume (int).
            Returns an empty but correctly-schema'd DataFrame on failure.
        """
        import pytz
        eastern = pytz.timezone("America/New_York")
        empty   = pd.DataFrame(columns=OHLCV_COLUMNS[1:])   # no timestamp col — it's the index

        params = {
            "function":   "TIME_SERIES_INTRADAY",
            "symbol":     ticker,
            "interval":   interval,
            "outputsize": outputsize,
            "adjusted":   "true",
            "datatype":   "json",
        }

        try:
            data = await self._rate_limited_get(params)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error(f"AV intraday request failed for {ticker}: {exc}")
            return empty

        if "Information" in data or "Note" in data:
            msg = data.get("Information") or data.get("Note")
            logger.warning(f"AV rate-limit message for {ticker}: {msg}")
            return empty

        series_key = f"Time Series ({interval})"
        ts_data    = data.get(series_key)
        if not ts_data:
            logger.warning(f"No intraday data returned for {ticker} [{interval}]")
            return empty

        rows = {}
        for ts_str, bar in ts_data.items():
            try:
                # AV timestamps are in US/Eastern with no timezone suffix
                naive_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                est_dt   = eastern.localize(naive_dt)
                utc_dt   = est_dt.astimezone(pytz.utc)
                rows[utc_dt] = {
                    "open":   float(bar["1. open"]),
                    "high":   float(bar["2. high"]),
                    "low":    float(bar["3. low"]),
                    "close":  float(bar["4. close"]),
                    "volume": int(bar["5. volume"]),
                }
            except (KeyError, ValueError) as exc:
                logger.debug(f"Skipping malformed bar '{ts_str}': {exc}")
                continue

        if not rows:
            return empty

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.DatetimeIndex(df.index, name="timestamp")
        df = df.sort_index()

        logger.info(f"AV intraday: fetched {len(df)} {interval} bars for {ticker}")
        return df
