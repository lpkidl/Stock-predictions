"""
Tests for the news pipeline stage in main.py (StockForecasterPipeline.run_news_stage).

Verifies skip behaviour when keys are absent, correct population of
self.news_scores when keys are present, non-fatal behaviour on errors,
and end-to-end integration of news features into the ML feature matrix.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from main import StockForecasterPipeline


def _pipeline(**config_overrides):
    """Return a bare StockForecasterPipeline without calling initialize()."""
    p = StockForecasterPipeline()
    # Patch settings for this instance's lifetime
    from config import settings
    for k, v in config_overrides.items():
        setattr(settings, k, v)
    return p


# ---------------------------------------------------------------------------
# run_news_stage — key-absent path
# ---------------------------------------------------------------------------

class TestRunNewsStageSkip:
    @pytest.mark.asyncio
    async def test_returns_true_when_newsapi_key_missing(self, caplog):
        p = _pipeline()
        from config import settings
        settings.NEWSAPI_KEY  = None
        settings.GROQ_API_KEY = "some-key"
        result = await p.run_news_stage()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_groq_key_missing(self, caplog):
        p = _pipeline()
        from config import settings
        settings.NEWSAPI_KEY  = "some-key"
        settings.GROQ_API_KEY = None
        result = await p.run_news_stage()
        assert result is True

    @pytest.mark.asyncio
    async def test_news_scores_remain_none_when_skipped(self):
        p = _pipeline()
        from config import settings
        settings.NEWSAPI_KEY  = None
        settings.GROQ_API_KEY = None
        await p.run_news_stage()
        assert p.news_scores is None

    @pytest.mark.asyncio
    async def test_logs_skip_message_when_keys_absent(self, caplog):
        import logging
        p = _pipeline()
        from config import settings
        settings.NEWSAPI_KEY  = None
        settings.GROQ_API_KEY = None
        with caplog.at_level(logging.INFO):
            await p.run_news_stage()
        assert any("NEWSAPI_KEY" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_news_stage — success path
# ---------------------------------------------------------------------------

class TestRunNewsStageSuccess:
    @pytest.mark.asyncio
    async def test_populates_news_scores(self, news_df):
        from config import settings
        settings.NEWSAPI_KEY  = "key"
        settings.GROQ_API_KEY = "key"
        settings.STOCK_TICKERS = "AAPL"

        p = _pipeline()

        with patch("main.NewsAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_tickers.return_value = news_df
            result = await p.run_news_stage()

        assert result is True
        assert p.news_scores is not None
        assert len(p.news_scores) == 1
        assert p.news_scores.iloc[0]["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_passes_tickers_to_analyzer(self, news_df):
        from config import settings
        settings.NEWSAPI_KEY   = "key"
        settings.GROQ_API_KEY  = "key"
        settings.STOCK_TICKERS = "AAPL,NVDA"

        p = _pipeline()

        with patch("main.NewsAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_tickers.return_value = news_df
            await p.run_news_stage()

        instance.analyze_tickers.assert_called_once_with(["AAPL", "NVDA"])

    @pytest.mark.asyncio
    async def test_passes_config_to_analyzer(self, news_df):
        from config import settings
        settings.NEWSAPI_KEY        = "newskey"
        settings.GROQ_API_KEY       = "groqkey"
        settings.NEWS_LOOKBACK_DAYS = 5
        settings.NEWS_MAX_ARTICLES  = 7
        settings.STOCK_TICKERS      = "AAPL"

        p = _pipeline()

        with patch("main.NewsAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_tickers.return_value = news_df
            await p.run_news_stage()

        MockAnalyzer.assert_called_once_with(
            newsapi_key="newskey",
            groq_api_key="groqkey",
            lookback_days=5,
            max_articles=7,
        )

    @pytest.mark.asyncio
    async def test_handles_empty_news_scores_gracefully(self):
        from config import settings
        settings.NEWSAPI_KEY  = "key"
        settings.GROQ_API_KEY = "key"
        settings.STOCK_TICKERS = "AAPL"

        empty_df = pd.DataFrame(columns=[
            "date", "ticker", "news_sentiment", "news_confidence",
            "news_risk_score", "news_catalyst_score",
        ])
        p = _pipeline()

        with patch("main.NewsAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_tickers.return_value = empty_df
            result = await p.run_news_stage()

        assert result is True
        assert p.news_scores is not None
        assert p.news_scores.empty


# ---------------------------------------------------------------------------
# run_news_stage — error resilience
# ---------------------------------------------------------------------------

class TestRunNewsStageErrors:
    @pytest.mark.asyncio
    async def test_returns_true_on_unexpected_exception(self):
        """News stage must never crash the pipeline."""
        from config import settings
        settings.NEWSAPI_KEY  = "key"
        settings.GROQ_API_KEY = "key"
        settings.STOCK_TICKERS = "AAPL"

        p = _pipeline()

        with patch("main.NewsAnalyzer", side_effect=RuntimeError("boom")):
            result = await p.run_news_stage()

        assert result is True  # non-fatal

    @pytest.mark.asyncio
    async def test_returns_true_when_analyze_tickers_raises(self):
        from config import settings
        settings.NEWSAPI_KEY  = "key"
        settings.GROQ_API_KEY = "key"
        settings.STOCK_TICKERS = "AAPL"

        p = _pipeline()

        with patch("main.NewsAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_tickers.side_effect = Exception("api timeout")
            result = await p.run_news_stage()

        assert result is True


# ---------------------------------------------------------------------------
# Integration: news features reach merged_data in run_ml_stage
# ---------------------------------------------------------------------------

class TestNewsFeaturesIntegration:
    """
    Smoke-test that news_scores populated by run_news_stage are subsequently
    passed to MLPredictor.merge_news_features inside run_ml_stage.
    We stub every expensive call (yfinance already downloaded; ML skipped).
    """

    @pytest.mark.asyncio
    async def test_merge_news_features_called_when_news_scores_set(
        self, stock_df, news_df
    ):
        from config import settings
        from ml_engine.predictor import MLPredictor

        settings.STOCK_TICKERS     = "AAPL"
        settings.FORECAST_HORIZONS = [1]

        p = _pipeline()
        # Manually initialise components so we don't need the full async init
        p.ml_predictor = MLPredictor()
        # Pre-populate pipeline state as if stages 1–2 ran
        p.aggregated_data = {
            "stocks":   {"AAPL": stock_df},
            "macro":    {},
            "earnings": {"AAPL": []},
        }
        p.sentiment_index = None
        p.news_scores     = news_df  # inject news scores directly

        merge_calls = []
        original_fn = MLPredictor.merge_news_features

        def _spy(self, s_df, n_df, ticker):
            merge_calls.append(ticker)
            return original_fn(self, s_df, n_df, ticker)

        with patch.object(MLPredictor, "merge_news_features", _spy):
            # Patch prepare_training_data to return None so ML exits fast
            with patch.object(MLPredictor, "prepare_training_data", return_value=None):
                await p.run_ml_stage()

        assert "AAPL" in merge_calls, (
            "merge_news_features was never called for AAPL"
        )
