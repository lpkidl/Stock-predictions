"""
Tests for feature_engine/news_sentiment.py — NewsAnalyzer.

All external calls (NewsAPI, Groq) are mocked so tests run offline
and don't consume API quota.
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_engine.news_sentiment import NewsAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(**kwargs):
    defaults = {"newsapi_key": "test-news-key", "groq_api_key": "test-groq-key"}
    defaults.update(kwargs)
    return NewsAnalyzer(**defaults)


def _mock_httpx_response(articles: list, status_code: int = 200):
    """Build a fake httpx response for NewsAPI."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"status": "ok", "articles": articles}
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_groq_response(content: dict):
    """Build a fake Groq chat completion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# fetch_articles
# ---------------------------------------------------------------------------

class TestFetchArticles:
    def test_returns_articles_on_success(self, sample_articles):
        analyzer = _make_analyzer()
        with patch("httpx.get", return_value=_mock_httpx_response(sample_articles)):
            result = analyzer.fetch_articles("AAPL")
        assert len(result) == len(sample_articles)
        assert result[0]["title"] == sample_articles[0]["title"]

    def test_returns_empty_list_on_401(self):
        analyzer = _make_analyzer()
        with patch("httpx.get", return_value=_mock_httpx_response([], status_code=401)):
            result = analyzer.fetch_articles("AAPL")
        assert result == []

    def test_returns_empty_list_on_426(self):
        analyzer = _make_analyzer()
        with patch("httpx.get", return_value=_mock_httpx_response([], status_code=426)):
            result = analyzer.fetch_articles("AAPL")
        assert result == []

    def test_returns_empty_list_on_network_error(self):
        analyzer = _make_analyzer()
        with patch("httpx.get", side_effect=Exception("connection refused")):
            result = analyzer.fetch_articles("AAPL")
        assert result == []

    def test_passes_ticker_in_query(self, sample_articles):
        analyzer = _make_analyzer()
        with patch("httpx.get", return_value=_mock_httpx_response(sample_articles)) as mock_get:
            analyzer.fetch_articles("NVDA")
        called_params = mock_get.call_args.kwargs["params"]
        assert called_params["q"] == "NVDA"

    def test_respects_lookback_days(self, sample_articles):
        analyzer = _make_analyzer(lookback_days=3)
        with patch("httpx.get", return_value=_mock_httpx_response(sample_articles)) as mock_get:
            analyzer.fetch_articles("AAPL")
        called_params = mock_get.call_args.kwargs["params"]
        from datetime import datetime, timedelta, timezone
        expected_from = (
            datetime.now(timezone.utc) - timedelta(days=3)
        ).strftime("%Y-%m-%d")
        assert called_params["from"] == expected_from

    def test_respects_max_articles(self, sample_articles):
        analyzer = _make_analyzer(max_articles=5)
        with patch("httpx.get", return_value=_mock_httpx_response(sample_articles)) as mock_get:
            analyzer.fetch_articles("AAPL")
        assert mock_get.call_args.kwargs["params"]["pageSize"] == 5

    def test_empty_articles_list_in_response(self):
        analyzer = _make_analyzer()
        with patch("httpx.get", return_value=_mock_httpx_response([])):
            result = analyzer.fetch_articles("AAPL")
        assert result == []


# ---------------------------------------------------------------------------
# score_with_llm
# ---------------------------------------------------------------------------

class TestScoreWithLlm:
    def test_returns_dict_with_required_keys(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(groq_scores)
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            result = analyzer.score_with_llm("AAPL", sample_articles)
        assert result is not None
        assert set(result.keys()) == {"sentiment", "confidence", "risk_score", "catalyst_score", "themes"}

    def test_clamps_sentiment_to_minus_one_plus_one(self, sample_articles):
        analyzer = _make_analyzer()
        bad_scores = {"sentiment": 5.0, "confidence": -2.0,
                      "risk_score": 99.0, "catalyst_score": -10.0, "themes": []}
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(bad_scores)
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            result = analyzer.score_with_llm("AAPL", sample_articles)
        assert result["sentiment"]      == 1.0
        assert result["confidence"]     == 0.0
        assert result["risk_score"]     == 1.0
        assert result["catalyst_score"] == 0.0

    def test_returns_none_on_empty_articles(self):
        analyzer = _make_analyzer()
        result = analyzer.score_with_llm("AAPL", [])
        assert result is None

    def test_returns_none_on_articles_with_no_titles(self):
        analyzer = _make_analyzer()
        blank_articles = [{"title": "", "description": ""}, {"title": None}]
        result = analyzer.score_with_llm("AAPL", blank_articles)
        assert result is None

    def test_returns_none_on_groq_exception(self, sample_articles):
        analyzer = _make_analyzer()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("rate limit")
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            result = analyzer.score_with_llm("AAPL", sample_articles)
        assert result is None

    def test_returns_none_on_invalid_json(self, sample_articles):
        analyzer = _make_analyzer()
        choice = MagicMock()
        choice.message.content = "not valid json {"
        completion = MagicMock()
        completion.choices = [choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = completion
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            result = analyzer.score_with_llm("AAPL", sample_articles)
        assert result is None

    def test_themes_list_included(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(groq_scores)
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            result = analyzer.score_with_llm("AAPL", sample_articles)
        assert isinstance(result["themes"], list)
        assert "earnings beat" in result["themes"]

    def test_uses_correct_model(self, sample_articles, groq_scores):
        from feature_engine.news_sentiment import GROQ_MODEL
        analyzer = _make_analyzer()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(groq_scores)
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            analyzer.score_with_llm("AAPL", sample_articles)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == GROQ_MODEL

    def test_respects_max_articles_in_prompt(self, groq_scores):
        """Only max_articles articles should be sent, even if more are provided."""
        analyzer = _make_analyzer(max_articles=2)
        many_articles = [
            {"title": f"Article {i}", "description": f"Desc {i}"} for i in range(10)
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_groq_response(groq_scores)
        with patch("feature_engine.news_sentiment.Groq", return_value=mock_client):
            analyzer.score_with_llm("AAPL", many_articles)
        user_msg = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        # Only first 2 articles should appear in the prompt
        assert "Article 0" in user_msg
        assert "Article 1" in user_msg
        assert "Article 2" not in user_msg


# ---------------------------------------------------------------------------
# analyze_ticker
# ---------------------------------------------------------------------------

class TestAnalyzeTicker:
    def test_returns_dataframe_with_correct_columns(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_ticker("AAPL")
        assert result is not None
        expected_cols = {
            "date", "ticker", "news_sentiment", "news_confidence",
            "news_risk_score", "news_catalyst_score",
            "news_themes", "news_article_count",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_scores_stored_correctly(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_ticker("AAPL")
        row = result.iloc[0]
        assert row["news_sentiment"]      == groq_scores["sentiment"]
        assert row["news_confidence"]     == groq_scores["confidence"]
        assert row["news_risk_score"]     == groq_scores["risk_score"]
        assert row["news_catalyst_score"] == groq_scores["catalyst_score"]

    def test_ticker_column_matches_input(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_ticker("NVDA")
        assert result.iloc[0]["ticker"] == "NVDA"

    def test_date_is_today(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_ticker("AAPL")
        today = pd.Timestamp(date.today())
        assert result.iloc[0]["date"] == today

    def test_returns_none_when_no_articles(self):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=[]):
            result = analyzer.analyze_ticker("AAPL")
        assert result is None

    def test_returns_none_when_llm_fails(self, sample_articles):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=None):
            result = analyzer.analyze_ticker("AAPL")
        assert result is None

    def test_article_count_recorded(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_ticker("AAPL")
        assert result.iloc[0]["news_article_count"] == len(sample_articles)


# ---------------------------------------------------------------------------
# analyze_tickers (batch)
# ---------------------------------------------------------------------------

class TestAnalyzeTickers:
    def test_returns_dataframe_for_multiple_tickers(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=sample_articles), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_tickers(["AAPL", "NVDA", "TSLA"])
        assert len(result) == 3
        assert set(result["ticker"]) == {"AAPL", "NVDA", "TSLA"}

    def test_returns_empty_df_when_all_fail(self):
        analyzer = _make_analyzer()
        with patch.object(analyzer, "fetch_articles", return_value=[]):
            result = analyzer.analyze_tickers(["AAPL", "NVDA"])
        assert result.empty
        assert "news_sentiment" in result.columns

    def test_partial_success_returns_successful_tickers(self, sample_articles, groq_scores):
        analyzer = _make_analyzer()

        def _fetch(ticker):
            return sample_articles if ticker == "AAPL" else []

        with patch.object(analyzer, "fetch_articles", side_effect=_fetch), \
             patch.object(analyzer, "score_with_llm", return_value=groq_scores):
            result = analyzer.analyze_tickers(["AAPL", "NVDA"])
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"
