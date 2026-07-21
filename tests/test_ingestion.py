"""
Tests for ingestion/pipeline.py — DataIngestionPipeline.

Real yfinance calls are used for a tiny 5-day window to verify the
integration works; all Reddit and X calls are mocked.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from ingestion.pipeline import DataIngestionPipeline


def _pipeline() -> DataIngestionPipeline:
    return DataIngestionPipeline(executor=ThreadPoolExecutor(max_workers=2))


# ---------------------------------------------------------------------------
# yfinance integration (real network, minimal data)
# ---------------------------------------------------------------------------

class TestFetchYfinanceData:
    @pytest.mark.asyncio
    async def test_returns_dataframe_for_valid_ticker(self):
        async with _pipeline() as p:
            df = await p.fetch_yfinance_data("AAPL", period="5d")
        assert df is not None
        assert not df.empty
        assert "Close" in df.columns

    @pytest.mark.asyncio
    async def test_adds_ticker_column(self):
        async with _pipeline() as p:
            df = await p.fetch_yfinance_data("MSFT", period="5d")
        assert "ticker" in df.columns
        assert (df["ticker"] == "MSFT").all()

    @pytest.mark.asyncio
    async def test_returns_none_for_invalid_ticker(self):
        async with _pipeline() as p:
            df = await p.fetch_yfinance_data("ZZZZNOTREAL", period="5d")
        assert df is None

    @pytest.mark.asyncio
    async def test_flattens_multiindex_columns(self):
        async with _pipeline() as p:
            df = await p.fetch_yfinance_data("AAPL", period="5d")
        # After flattening, no column should be a tuple
        assert all(isinstance(c, str) for c in df.columns)


# ---------------------------------------------------------------------------
# fetch_reddit_posts — PRAW path
# ---------------------------------------------------------------------------

class TestFetchRedditPostsPraw:
    @pytest.mark.asyncio
    async def test_uses_praw_when_credentials_set(self):
        from config import settings
        settings.PRAW_CLIENT_ID     = "fake_id"
        settings.PRAW_CLIENT_SECRET = "fake_secret"

        async with _pipeline() as p:
            with patch.object(p, "_fetch_reddit_via_praw", new_callable=AsyncMock) as mock_praw:
                mock_praw.return_value = [{"source": "reddit", "ticker": "AAPL", "title": "t"}]
                posts = await p.fetch_reddit_posts("AAPL", limit=5)

        mock_praw.assert_called_once_with("AAPL", 5)
        assert len(posts) == 1

        # Restore
        settings.PRAW_CLIENT_ID     = None
        settings.PRAW_CLIENT_SECRET = None

    @pytest.mark.asyncio
    async def test_praw_posts_have_required_keys(self):
        from config import settings
        settings.PRAW_CLIENT_ID     = "id"
        settings.PRAW_CLIENT_SECRET = "secret"

        fake_posts = [{
            "source": "reddit", "subreddit": "stocks", "ticker": "AAPL",
            "title": "AAPL hits ATH", "text": "Great quarter",
            "score": 100, "timestamp": datetime.now(),
            "url": "https://reddit.com/r/stocks/xyz",
        }]

        async with _pipeline() as p:
            with patch.object(p, "_fetch_reddit_via_praw", new_callable=AsyncMock,
                               return_value=fake_posts):
                posts = await p.fetch_reddit_posts("AAPL", limit=5)

        assert all(k in posts[0] for k in ("source", "ticker", "title", "timestamp"))

        settings.PRAW_CLIENT_ID     = None
        settings.PRAW_CLIENT_SECRET = None


# ---------------------------------------------------------------------------
# fetch_reddit_posts — raw HTTP fallback (403 handling)
# ---------------------------------------------------------------------------

class TestFetchRedditPostsHttp:
    @pytest.mark.asyncio
    async def test_returns_empty_list_on_403(self):
        from config import settings
        settings.PRAW_CLIENT_ID     = None
        settings.PRAW_CLIENT_SECRET = None

        import httpx

        async with _pipeline() as p:
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "403", request=MagicMock(), response=mock_resp
            )
            with patch.object(p.session, "get", new_callable=AsyncMock,
                               return_value=mock_resp):
                posts = await p.fetch_reddit_posts("AAPL", limit=5)

        assert posts == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_network_error(self):
        from config import settings
        settings.PRAW_CLIENT_ID     = None
        settings.PRAW_CLIENT_SECRET = None

        async with _pipeline() as p:
            with patch.object(p.session, "get", new_callable=AsyncMock,
                               side_effect=Exception("DNS failure")):
                posts = await p.fetch_reddit_posts("AAPL", limit=5)

        assert posts == []


# ---------------------------------------------------------------------------
# _fetch_reddit_via_praw
# ---------------------------------------------------------------------------

class TestFetchRedditViaPraw:
    @pytest.mark.asyncio
    async def test_returns_posts_from_both_subreddits(self):
        from config import settings
        settings.PRAW_CLIENT_ID     = "id"
        settings.PRAW_CLIENT_SECRET = "secret"

        fake_submission = MagicMock()
        fake_submission.title       = "AAPL to the moon"
        fake_submission.selftext    = "Buy calls"
        fake_submission.score       = 500
        fake_submission.created_utc = datetime.now().timestamp()
        fake_submission.permalink   = "/r/wallstreetbets/comments/abc"

        fake_sub = MagicMock()
        fake_sub.search.return_value = [fake_submission]

        fake_reddit = MagicMock()
        fake_reddit.subreddit.return_value = fake_sub

        async with _pipeline() as p:
            with patch("ingestion.pipeline.praw" if False else "__main__.praw",
                       create=True), \
                 patch("praw.Reddit", return_value=fake_reddit, create=True):
                # Patch inside the module's local import
                import ingestion.pipeline as ip_mod
                with patch.object(ip_mod, "__builtins__", ip_mod.__builtins__):
                    with patch("builtins.__import__", wraps=__import__) as mock_import:
                        def _import(name, *args, **kwargs):
                            if name == "praw":
                                m = MagicMock()
                                m.Reddit.return_value = fake_reddit
                                return m
                            return __import__(name, *args, **kwargs)
                        mock_import.side_effect = _import
                        posts = await p._fetch_reddit_via_praw("AAPL", 10)

        # Each of 2 subreddits yields 1 post → 2 total
        assert len(posts) == 2
        assert all(post["ticker"] == "AAPL" for post in posts)

        settings.PRAW_CLIENT_ID     = None
        settings.PRAW_CLIENT_SECRET = None

    @pytest.mark.asyncio
    async def test_returns_empty_on_praw_exception(self):
        from config import settings
        settings.PRAW_CLIENT_ID     = "id"
        settings.PRAW_CLIENT_SECRET = "secret"

        async with _pipeline() as p:
            with patch("builtins.__import__", side_effect=ImportError("praw")):
                posts = await p._fetch_reddit_via_praw("AAPL", 10)

        assert posts == []

        settings.PRAW_CLIENT_ID     = None
        settings.PRAW_CLIENT_SECRET = None


# ---------------------------------------------------------------------------
# fetch_x_posts — credential gate
# ---------------------------------------------------------------------------

class TestFetchXPosts:
    @pytest.mark.asyncio
    async def test_returns_empty_without_credentials(self):
        from config import settings
        settings.X_USERNAME = None
        settings.X_PASSWORD = None
        settings.X_EMAIL    = None

        async with _pipeline() as p:
            posts = await p.fetch_x_posts("AAPL", limit=5)

        assert posts == []

    @pytest.mark.asyncio
    async def test_initialize_twikit_returns_false_without_creds(self):
        from config import settings
        settings.X_USERNAME = None
        settings.X_PASSWORD = None
        settings.X_EMAIL    = None

        async with _pipeline() as p:
            result = await p.initialize_twikit_client()

        assert result is False


# ---------------------------------------------------------------------------
# fetch_macro_data
# ---------------------------------------------------------------------------

class TestFetchMacroData:
    @pytest.mark.asyncio
    async def test_returns_dict_for_all_macro_tickers(self):
        from config import settings
        async with _pipeline() as p:
            macro = await p.fetch_macro_data()

        # At least some macro tickers must succeed
        assert isinstance(macro, dict)
        assert len(macro) > 0

    @pytest.mark.asyncio
    async def test_macro_values_are_dataframes(self):
        async with _pipeline() as p:
            macro = await p.fetch_macro_data()

        for ticker, df in macro.items():
            assert isinstance(df, pd.DataFrame), f"{ticker} is not a DataFrame"
            assert "Close" in df.columns


# ---------------------------------------------------------------------------
# aggregate_all_data structure
# ---------------------------------------------------------------------------

class TestAggregateAllData:
    @pytest.mark.asyncio
    async def test_returns_required_keys(self):
        from config import settings
        settings.STOCK_TICKERS  = "AAPL"
        settings.PRAW_CLIENT_ID = None
        settings.X_USERNAME     = None

        async with _pipeline() as p:
            data = await p.aggregate_all_data()

        for key in ("stocks", "reddit", "x", "macro", "earnings", "timestamp"):
            assert key in data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_stocks_contains_dataframes(self):
        from config import settings
        settings.STOCK_TICKERS = "AAPL"

        async with _pipeline() as p:
            data = await p.aggregate_all_data()

        for ticker, df in data["stocks"].items():
            assert isinstance(df, pd.DataFrame)
            assert "Close" in df.columns

    @pytest.mark.asyncio
    async def test_reddit_and_x_are_lists(self):
        from config import settings
        settings.STOCK_TICKERS  = "AAPL"
        settings.PRAW_CLIENT_ID = None
        settings.X_USERNAME     = None

        async with _pipeline() as p:
            data = await p.aggregate_all_data()

        assert isinstance(data["reddit"], list)
        assert isinstance(data["x"], list)
