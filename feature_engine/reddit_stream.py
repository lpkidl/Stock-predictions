"""
RedditSocialStream — PRAW-based social volume and keyword-polarity stream.

Design choices vs. the existing HTTP approach in ingestion/pipeline.py:
  • Uses the official praw library in read-only mode (no OAuth login required
    for public subreddit reads; only a Reddit API "script" app is needed).
  • Polarity is computed via lightweight keyword matching rather than FinBERT,
    keeping submission processing fast and avoiding a second heavy model load.
  • Output is resampled into 1-hour buckets so it aligns cleanly with the
    MarketHoursAligner downstream.

PRAW rate-limiting:
  Reddit's API allows ~60 requests/minute for read-only script apps.
  Setting ratelimit_seconds=60 tells praw to sleep and retry automatically
  when it receives a 429 response — no manual throttle needed here.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import praw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries for fast, dependency-free sentiment polarity
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS: frozenset = frozenset({
    "buy", "bull", "bullish", "calls", "call", "long", "moon", "mooning",
    "breakout", "rally", "rip", "squeeze", "pumping", "upside", "gains",
    "tendies", "ath", "yolo", "green", "rocket", "uptrend", "support",
    "accumulate", "oversold", "undervalued", "cheap",
})

BEARISH_KEYWORDS: frozenset = frozenset({
    "sell", "bear", "bearish", "puts", "put", "short", "shorting", "crash",
    "dump", "dumping", "downside", "loss", "losses", "tank", "tanking",
    "overbought", "overvalued", "expensive", "drop", "dropping", "falling",
    "broke", "breakdown", "resistance", "top", "bubble", "correction",
    "recession", "bankrupt", "bankruptcy",
})

# Output column contract
SOCIAL_COLUMNS = ["mention_count", "sentiment_polarity", "submission_count"]


class RedditSocialStream:
    """
    Fetches recent submissions from a given subreddit, filters for a specified
    ticker, and aggregates the stream into hourly social-volume / polarity buckets.

    Usage:
        stream = RedditSocialStream(
            client_id="...",
            client_secret="...",
            user_agent="StockForecaster/1.0 by u/your_username",
        )
        df = stream.fetch_submissions("AAPL", subreddit="wallstreetbets")
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
    ) -> None:
        """
        Args:
            client_id:     Reddit API script app client ID.
            client_secret: Reddit API script app client secret.
            user_agent:    Descriptive string per Reddit API rules:
                           "AppName/Version by u/username"
        """
        # read_only=True — no user login needed for public subreddits
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            ratelimit_seconds=60,   # praw auto-sleeps on 429
        )
        self._reddit.read_only = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_submissions(
        self,
        ticker: str,
        subreddit: str = "wallstreetbets",
        limit: int = 100,
        sort: str = "new",
    ) -> pd.DataFrame:
        """
        Search a subreddit for a ticker symbol, compute rolling social volume
        and keyword polarity, and return hourly-aggregated results.

        Args:
            ticker:    Stock ticker to search for (e.g., "AAPL", "GME").
            subreddit: Target subreddit (default: "wallstreetbets").
            limit:     Max submissions to retrieve (Reddit caps at 1000).
            sort:      Listing sort — "new" | "hot" | "top" | "relevance".

        Returns:
            DataFrame indexed by hourly UTC timestamps with columns:
              mention_count      (int)   — number of ticker mentions in that hour
              sentiment_polarity (float) — mean keyword polarity in [-1.0, 1.0]
              submission_count   (int)   — raw number of submissions processed
            Returns empty DataFrame with correct columns on failure.
        """
        empty = pd.DataFrame(columns=SOCIAL_COLUMNS)

        try:
            sub    = self._reddit.subreddit(subreddit)
            search = sub.search(query=ticker, sort=sort, limit=limit)
            raw_rows = self._parse_submissions(search, ticker)
        except Exception as exc:
            logger.error(f"PRAW fetch failed for {ticker} in r/{subreddit}: {exc}")
            return empty

        if not raw_rows:
            logger.info(f"No submissions found for {ticker} in r/{subreddit}")
            return empty

        hourly = self._aggregate_to_hourly(raw_rows)
        logger.info(
            f"PRAW: {len(raw_rows)} submissions → {len(hourly)} hourly buckets "
            f"for {ticker} in r/{subreddit}"
        )
        return hourly

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_submissions(
        self,
        submissions,
        ticker: str,
    ) -> list[dict]:
        """
        Iterate over a PRAW submission generator, extract relevant fields,
        and count per-submission ticker mentions.

        A submission is included only when the ticker string appears in the
        combined title + selftext (case-insensitive), ensuring we don't count
        off-topic posts that merely appeared in the search index.
        """
        raw_rows = []
        ticker_upper = ticker.upper()

        for submission in submissions:
            try:
                title    = getattr(submission, "title",    "") or ""
                selftext = getattr(submission, "selftext", "") or ""
                combined = f"{title} {selftext}"

                # Count how many times the ticker appears as a whole word
                # e.g., "AAPL" should not match "AAPLA"
                words = combined.upper().split()
                mentions = words.count(ticker_upper)

                # Also count cashtag format: $AAPL
                cashtag  = words.count(f"${ticker_upper}")
                total_mentions = mentions + cashtag

                if total_mentions == 0:
                    continue

                # created_utc is a Unix epoch float from PRAW
                ts = datetime.fromtimestamp(
                    float(submission.created_utc),
                    tz=timezone.utc,
                )

                raw_rows.append({
                    "timestamp":     ts,
                    "polarity":      self._keyword_polarity(combined),
                    "mention_count": total_mentions,
                })

            except Exception as exc:
                logger.debug(f"Skipping malformed submission: {exc}")
                continue

        return raw_rows

    @staticmethod
    def _keyword_polarity(text: str) -> float:
        """
        Assign a directional polarity score in [-1.0, 1.0] using pre-defined
        keyword sets.  Designed for speed — no model inference, no tokenisation.

        Formula:
          polarity = (bull_hits - bear_hits) / max(bull_hits + bear_hits, 1)

        Properties:
          • Pure bullish text  → +1.0
          • Pure bearish text  → -1.0
          • Balanced / neutral → 0.0
          • Empty text         → 0.0

        This is intentionally coarse.  The FinBERT stage in nlp/sentiment.py
        handles deeper semantic analysis; this is a lightweight social-volume
        signal used for feature alignment, not final sentiment classification.
        """
        words       = text.lower().split()
        bull_hits   = sum(1 for w in words if w in BULLISH_KEYWORDS)
        bear_hits   = sum(1 for w in words if w in BEARISH_KEYWORDS)
        denominator = max(bull_hits + bear_hits, 1)
        return (bull_hits - bear_hits) / denominator

    def _aggregate_to_hourly(self, raw_rows: list[dict]) -> pd.DataFrame:
        """
        Bucket raw submission rows into non-overlapping 1-hour windows.

        Aggregation rules:
          sentiment_polarity → mean   (central tendency of hourly mood)
          mention_count      → sum    (total ticker mentions in the hour)
          submission_count   → count  (number of distinct submissions)

        The index is a tz-aware DatetimeIndex in UTC, with the label at the
        start of each hour (resample label="left", closed="left" default).

        Returns an empty DataFrame with correct columns if input is empty.
        """
        if not raw_rows:
            return pd.DataFrame(columns=SOCIAL_COLUMNS)

        df = pd.DataFrame(raw_rows)                                  # timestamp, polarity, mention_count
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        hourly = df.resample("1h").agg(
            sentiment_polarity=("polarity",      "mean"),
            mention_count      =("mention_count", "sum"),
            submission_count   =("mention_count", "count"),
        )

        # Drop completely empty hours (no submissions at all)
        hourly = hourly[hourly["submission_count"] > 0]
        return hourly
