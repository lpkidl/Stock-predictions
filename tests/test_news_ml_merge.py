"""
Tests for MLPredictor.merge_news_features().

Verifies that LLM-scored news features are correctly joined into the stock
feature matrix, that historical rows are 0-filled, and that the columns
are picked up by the feature selector used during training.
"""

import numpy as np
import pandas as pd
import pytest

from ml_engine.predictor import MLPredictor


# ---------------------------------------------------------------------------
# merge_news_features
# ---------------------------------------------------------------------------

class TestMergeNewsFeatures:
    NEWS_COLS = [
        "news_sentiment", "news_confidence",
        "news_risk_score", "news_catalyst_score",
    ]

    def test_adds_four_news_columns(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        for col in self.NEWS_COLS:
            assert col in merged.columns, f"Missing column: {col}"

    def test_today_row_has_correct_values(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        # The last row should be today (latest date in stock_df)
        last = merged.iloc[-1]
        assert abs(last["news_sentiment"]      - 0.65) < 1e-6
        assert abs(last["news_confidence"]     - 0.80) < 1e-6
        assert abs(last["news_risk_score"]     - 0.20) < 1e-6
        assert abs(last["news_catalyst_score"] - 0.75) < 1e-6

    def test_historical_rows_zero_filled(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        # All rows except the most recent should have 0 or forward-filled values
        # before news coverage starts; at minimum the first row must be 0
        assert merged.iloc[0]["news_sentiment"] == 0.0

    def test_returns_stock_df_when_news_df_is_none(self, stock_df):
        pred   = MLPredictor()
        result = pred.merge_news_features(stock_df, None, "AAPL")
        assert list(result.columns) == list(stock_df.columns)

    def test_returns_stock_df_when_news_df_is_empty(self, stock_df):
        pred     = MLPredictor()
        empty_df = pd.DataFrame(columns=[
            "date", "ticker", "news_sentiment", "news_confidence",
            "news_risk_score", "news_catalyst_score",
        ])
        result   = pred.merge_news_features(stock_df, empty_df, "AAPL")
        assert list(result.columns) == list(stock_df.columns)

    def test_wrong_ticker_produces_zero_filled_columns(self, stock_df, news_df):
        """News scored for NVDA should not bleed into AAPL."""
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "NVDA")
        # news_df only has AAPL; filtering to NVDA returns nothing → all zeros
        assert (merged["news_sentiment"] == 0.0).all()

    def test_no_rows_dropped(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        assert len(merged) == len(stock_df)

    def test_original_columns_preserved(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        for col in stock_df.columns:
            assert col in merged.columns

    def test_news_columns_numeric(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        for col in self.NEWS_COLS:
            assert pd.api.types.is_numeric_dtype(merged[col]), f"{col} not numeric"

    def test_multi_ticker_news_df_filters_correctly(self, stock_df, multi_news_df):
        pred = MLPredictor()
        merged_aapl = pred.merge_news_features(stock_df, multi_news_df, "AAPL")
        merged_nvda = pred.merge_news_features(stock_df, multi_news_df, "NVDA")
        # AAPL should have sentiment 0.65, NVDA should have -0.30
        assert merged_aapl.iloc[-1]["news_sentiment"] == pytest.approx(0.65)
        assert merged_nvda.iloc[-1]["news_sentiment"] == pytest.approx(-0.30)


# ---------------------------------------------------------------------------
# _get_feature_cols picks up news columns
# ---------------------------------------------------------------------------

class TestFeatureColsIncludeNews:
    def test_news_cols_in_feature_selector(self, stock_df, news_df):
        pred   = MLPredictor()
        merged = pred.merge_news_features(stock_df, news_df, "AAPL")
        feature_cols = pred._get_feature_cols(merged)
        news_in_features = [c for c in feature_cols if c.startswith("news_")]
        assert len(news_in_features) == 4, (
            f"Expected 4 news feature columns, got {news_in_features}"
        )

    def test_news_themes_not_in_feature_cols(self, stock_df, news_df):
        """news_themes is a string column and must be excluded from ML features."""
        pred   = MLPredictor()
        # Manually add news_themes to mimic the full news DataFrame merge
        stock_with_themes = stock_df.copy()
        stock_with_themes["news_themes"] = "earnings beat"
        feature_cols = pred._get_feature_cols(stock_with_themes)
        assert "news_themes" not in feature_cols
