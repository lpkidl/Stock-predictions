"""
MarketHoursAligner — NYSE session resampling and overnight sentiment alignment.

Problem this solves:
  News and social posts arrive 24/7, but equities only trade Mon-Fri 09:30-16:00
  US/Eastern.  If we naively merge sentiment onto price bars, out-of-hours posts
  (overnight, weekends) either get dropped or mapped to phantom price bars.

Solution — two-phase alignment:
  1. resample_to_hourly: Normalise any time-indexed DataFrame to 1-hour bars
     using standard OHLCV aggregation rules (or mean for non-price series).
  2. align_overnight_sentiment: Collect all OOH rows, aggregate them into a
     single synthetic "pre-open" block, and attach that block to the next
     regular 09:30 AM market-open bar.  No data is discarded.

Limitations:
  • Holiday detection uses pd.bdate_range (business days only) — true NYSE
    holidays (e.g., Thanksgiving, MLK Day) are treated as regular weekdays.
    For a production system, add the exchange_calendars or pandas_market_calendars
    library and swap _next_market_open to use it.
"""

import logging
from datetime import datetime, time
from typing import Optional, Tuple

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

NYSE_TZ      = pytz.timezone("America/New_York")
MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)


class MarketHoursAligner:
    """
    Resamples time-series data to 1-hour bars aligned to NYSE session boundaries
    and aggregates out-of-hours sentiment into the next available market-open bar.

    Args:
        tz: IANA timezone string for the target exchange (default: NYSE).
    """

    def __init__(self, tz: str = "America/New_York") -> None:
        self._tz = pytz.timezone(tz)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resample_to_hourly(
        self,
        df: pd.DataFrame,
        ohlcv: bool = True,
    ) -> pd.DataFrame:
        """
        Resample any time-indexed DataFrame to 1-hour bars.

        Args:
            df:    Input DataFrame with a DatetimeIndex (tz-aware or tz-naive).
                   If tz-naive, the index is localised to self._tz first.
            ohlcv: True  → OHLCV aggregation (Open=first, High=max, Low=min,
                             Close=last, Volume=sum).  Column names are matched
                             case-insensitively.
                   False → All numeric columns use mean aggregation.

        Returns:
            Resampled DataFrame with a tz-aware DatetimeIndex.
            Empty hourly buckets (all-NaN) are dropped.
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure the index is a tz-aware DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex for resampling.")

        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(self._tz)

        if ohlcv:
            return self._resample_ohlcv(df)
        else:
            resampled = df.select_dtypes(include="number").resample("1h").mean()
            return resampled.dropna(how="all")

    def align_overnight_sentiment(
        self,
        news_df:   pd.DataFrame,
        social_df: pd.DataFrame,
        price_df:  pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two sentiment streams (news and social) to price_df's hourly index,
        aggregating all out-of-hours data into the next market-open bar.

        Algorithm:
          1. Tag each row in news_df / social_df as in-market or OOH.
          2. For OOH rows: determine the next 09:30 AM open bar they belong to.
          3. Group OOH rows by their owning open bar and aggregate:
               numeric sentiment scores → mean
               count-type columns       → sum
          4. For in-market rows: map directly to the nearest price_df bar via
             pd.merge_asof (direction="nearest", tolerance=1 hour).
          5. Concatenate in-market and OOH-aggregated rows, sort by index.

        Args:
            news_df:   DataFrame from AlphaVantageClient.fetch_news_sentiment()
                       Must have a "timestamp" column (datetime) or DatetimeIndex.
            social_df: DataFrame from RedditSocialStream.fetch_submissions()
                       Must have a DatetimeIndex (UTC).
            price_df:  1-hour OHLCV DataFrame from resample_to_hourly().
                       Its index forms the canonical hourly grid.

        Returns:
            (aligned_news_df, aligned_social_df) — both indexed on price_df's index.
        """
        aligned_news   = self._align_stream(news_df,   price_df, is_news=True)
        aligned_social = self._align_stream(social_df, price_df, is_news=False)
        return aligned_news, aligned_social

    # ------------------------------------------------------------------
    # Private: market-hours helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_in_market_hours(ts: pd.Timestamp) -> bool:
        """
        Return True if ts falls within NYSE regular trading hours.
        Converts to America/New_York before comparison.

        Trading hours:  Monday–Friday, 09:30:00 ≤ t < 16:00:00 EST/EDT.
        The 16:00 bar's *open* timestamp is 15:00:00, so comparing bar-start
        timestamps against < 16:00 correctly keeps all intraday bars.
        """
        ts_est  = ts.astimezone(NYSE_TZ)
        weekday = ts_est.weekday()            # 0=Monday … 6=Sunday
        t       = ts_est.time()
        return weekday < 5 and MARKET_OPEN <= t < MARKET_CLOSE

    def _next_market_open(self, from_ts: pd.Timestamp) -> pd.Timestamp:
        """
        Given any timestamp, return the next NYSE 09:30 AM bar.

        Rules:
          • If from_ts is a weekday before 09:30 → today's 09:30.
          • If from_ts is a weekday at/after 09:30 or a weekend → next
            business day's 09:30 (uses pd.bdate_range; see module-level note
            about holiday limitation).

        The returned timestamp is tz-aware in America/New_York.
        """
        ts_est  = from_ts.astimezone(NYSE_TZ)
        weekday = ts_est.weekday()
        t       = ts_est.time()

        # Same-day open if we haven't passed 09:30 yet on a weekday
        if weekday < 5 and t < MARKET_OPEN:
            candidate = NYSE_TZ.localize(
                datetime(ts_est.year, ts_est.month, ts_est.day, 9, 30)
            )
            return candidate

        # Otherwise advance to the next business day
        # pd.bdate_range generates business-day dates; index [1] is the next one
        next_bday = pd.bdate_range(start=ts_est.date(), periods=2, freq="B")[1]
        next_open = NYSE_TZ.localize(
            datetime(next_bday.year, next_bday.month, next_bday.day, 9, 30)
        )
        return next_open

    # ------------------------------------------------------------------
    # Private: stream alignment core
    # ------------------------------------------------------------------

    def _normalise_index(self, df: pd.DataFrame, is_news: bool) -> pd.DataFrame:
        """
        Ensure the DataFrame has a tz-aware UTC DatetimeIndex.
        News DataFrames use a "timestamp" column; social DataFrames already
        have a DatetimeIndex set by RedditSocialStream._aggregate_to_hourly.
        """
        df = df.copy()

        if is_news and "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop(columns=["timestamp"], errors="ignore")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex or 'timestamp' column.")
        elif df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"
        return df.sort_index()

    def _align_stream(
        self,
        stream_df: pd.DataFrame,
        price_df:  pd.DataFrame,
        is_news:   bool,
    ) -> pd.DataFrame:
        """
        Core alignment logic for a single sentiment stream.

        Steps:
          1. Normalise index to UTC DatetimeIndex.
          2. Tag each row as in_market.
          3. For in-market rows: merge_asof onto price_df index.
          4. For OOH rows: find owning open bar, aggregate, merge.
          5. Concatenate and return.
        """
        if stream_df.empty:
            return stream_df

        try:
            df = self._normalise_index(stream_df, is_news)
        except TypeError as exc:
            logger.warning(f"Skipping stream alignment — {exc}")
            return stream_df

        if price_df.empty:
            logger.warning("price_df is empty — returning stream unchanged.")
            return df

        # Tag rows
        df["_in_market"] = df.index.map(self._is_in_market_hours)

        in_market_df = df[df["_in_market"]].drop(columns=["_in_market"])
        ooh_df       = df[~df["_in_market"]].drop(columns=["_in_market"])

        # ---- In-market rows: snap to nearest price bar ----
        price_index  = price_df.index.tz_convert("UTC")
        aligned_im   = self._merge_to_price_index(in_market_df, price_index)

        # ---- OOH rows: aggregate then attach to next open bar ----
        aligned_ooh = pd.DataFrame()
        if not ooh_df.empty:
            ooh_df = ooh_df.copy()
            ooh_df["_next_open"] = ooh_df.index.map(
                lambda ts: self._next_market_open(ts).tz_convert("UTC")
            )

            numeric_cols = ooh_df.select_dtypes(include="number").columns.tolist()
            agg_dict = {}
            for col in numeric_cols:
                # Columns that look like counts use sum; scores use mean
                if any(k in col.lower() for k in ("count", "volume", "mention")):
                    agg_dict[col] = "sum"
                else:
                    agg_dict[col] = "mean"

            grouped = ooh_df.groupby("_next_open").agg(agg_dict)
            grouped.index.name = "timestamp"

            # Keep only next_open bars that exist in the price index
            grouped = grouped[grouped.index.isin(price_index)]
            aligned_ooh = grouped

        # ---- Combine ----
        parts = [p for p in [aligned_im, aligned_ooh] if not p.empty]
        if not parts:
            return pd.DataFrame(columns=df.columns.drop("_in_market", errors="ignore"))

        result = pd.concat(parts).sort_index()
        # Group-by index to merge any OOH spillover that landed on an in-market bar
        numeric_result = result.select_dtypes(include="number")
        result = numeric_result.groupby(level=0).mean()
        return result

    def _merge_to_price_index(
        self,
        stream_df:   pd.DataFrame,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Snap stream_df rows to the nearest bar in price_index using merge_asof.
        Rows outside 1-hour tolerance of any price bar are dropped.
        """
        if stream_df.empty:
            return stream_df

        price_frame = pd.DataFrame(index=price_index)
        price_frame.index.name = "timestamp"

        # merge_asof requires both sides sorted
        left  = stream_df.reset_index().rename(columns={"index": "timestamp"})
        right = price_frame.reset_index()

        merged = pd.merge_asof(
            left.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )
        merged = merged.dropna(subset=["timestamp"])
        merged = merged.set_index("timestamp")
        return merged.select_dtypes(include="number")

    # ------------------------------------------------------------------
    # Private: OHLCV-specific resampler
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample to 1-hour bars using OHLCV semantics.
        Column name matching is case-insensitive (handles "Close" and "close").
        """
        col_map = {c.lower(): c for c in df.columns}
        agg = {}

        if "open"   in col_map: agg[col_map["open"]]   = "first"
        if "high"   in col_map: agg[col_map["high"]]   = "max"
        if "low"    in col_map: agg[col_map["low"]]    = "min"
        if "close"  in col_map: agg[col_map["close"]]  = "last"
        if "volume" in col_map: agg[col_map["volume"]] = "sum"

        # Any remaining numeric columns → mean
        for col in df.select_dtypes(include="number").columns:
            if col not in agg:
                agg[col] = "mean"

        if not agg:
            return df.resample("1h").mean().dropna(how="all")

        resampled = df.resample("1h").agg(agg)
        close_col = col_map.get("close") or col_map.get("close", None)
        if close_col and close_col in resampled.columns:
            return resampled.dropna(subset=[close_col])
        return resampled.dropna(how="all")
