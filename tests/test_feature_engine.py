"""
Tests for feature_engine modules:
  - TechnicalIndicatorEngine  (tech_indicators.py)
  - ExecutionEngine            (execution.py)
  - NewsAnalyzer import guard  (__init__.py safe-import)
"""

import math

import numpy as np
import pandas as pd
import pytest

from feature_engine.tech_indicators import TechnicalIndicatorEngine
from feature_engine.execution import ExecutionEngine
from tests.conftest import _make_ohlcv


# ---------------------------------------------------------------------------
# TechnicalIndicatorEngine
# ---------------------------------------------------------------------------

class TestTechnicalIndicatorEngine:
    ICHIMOKU_COLS = ["Tenkan_9", "Kijun_26", "Senkou_A", "Senkou_B"]
    ADX_COLS      = ["ADX_14", "DMP_14", "DMN_14"]
    OTHER_COLS    = ["BBW"]

    @pytest.fixture
    def df(self):
        return _make_ohlcv(300)

    def test_apply_all_adds_ichimoku_columns(self, df):
        eng    = TechnicalIndicatorEngine()
        result = eng.apply_all(df)
        for col in self.ICHIMOKU_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_apply_all_adds_adx_columns(self, df):
        eng    = TechnicalIndicatorEngine()
        result = eng.apply_all(df)
        for col in self.ADX_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_apply_all_adds_bbw(self, df):
        eng    = TechnicalIndicatorEngine()
        result = eng.apply_all(df)
        assert "BBW" in result.columns

    def test_row_count_preserved(self, df):
        eng    = TechnicalIndicatorEngine()
        result = eng.apply_all(df)
        assert len(result) == len(df)

    def test_original_df_not_mutated(self, df):
        cols_before = list(df.columns)
        eng         = TechnicalIndicatorEngine()
        eng.apply_all(df)
        assert list(df.columns) == cols_before

    # Ichimoku look-ahead bias check
    def test_ichimoku_senkou_no_forward_shift(self, df):
        """
        Senkou A/B must be computed from data at or before row t.
        We verify this by checking that Senkou_A at row t equals
        (Tenkan_9[t] + Kijun_26[t]) / 2 — no look-ahead shift.
        """
        eng    = TechnicalIndicatorEngine()
        result = eng.apply_all(df)
        # Pick a row well past the warm-up (index 100)
        row = result.iloc[100]
        expected_senkou_a = (row["Tenkan_9"] + row["Kijun_26"]) / 2.0
        assert abs(row["Senkou_A"] - expected_senkou_a) < 1e-6

    def test_bbw_is_non_negative(self, df):
        from ml_engine.predictor import MLPredictor
        pred   = MLPredictor()
        with_bb = pred.calculate_technical_indicators(df, "AAPL")
        eng     = TechnicalIndicatorEngine()
        result  = eng.apply_all(with_bb)
        bbw     = result["BBW"].dropna()
        assert (bbw >= 0).all()

    def test_adx_bounded_0_to_100(self, df):
        eng    = TechnicalIndicatorEngine()
        result = eng.apply_all(df)
        adx    = result["ADX_14"].dropna()
        assert (adx >= 0).all() and (adx <= 100).all()

    def test_atr_skipped_when_already_present(self, df):
        """calculate_atr is a no-op if ATR_14 already exists."""
        df_with_atr = df.copy()
        df_with_atr["ATR_14"] = 1.5  # pre-populate
        eng    = TechnicalIndicatorEngine()
        result = eng.calculate_atr(df_with_atr)
        # Column should still be there, values unchanged
        assert (result["ATR_14"] == 1.5).all()

    def test_compute_signal_confirmation_returns_expected_keys(self, df):
        from ml_engine.predictor import MLPredictor
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(df, "AAPL")
        eng     = TechnicalIndicatorEngine()
        full_df = eng.apply_all(with_ti)
        result  = eng.compute_signal_confirmation(full_df, "up")
        for key in ("confirming_count", "total_signals", "confirmation_score", "detail"):
            assert key in result

    def test_compute_signal_confirmation_score_in_0_1(self, df):
        from ml_engine.predictor import MLPredictor
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(df, "AAPL")
        eng     = TechnicalIndicatorEngine()
        full_df = eng.apply_all(with_ti)
        result  = eng.compute_signal_confirmation(full_df, "down")
        assert 0.0 <= result["confirmation_score"] <= 1.0

    def test_compute_signal_confirmation_empty_df(self):
        eng    = TechnicalIndicatorEngine()
        result = eng.compute_signal_confirmation(pd.DataFrame(), "up")
        assert result["confirming_count"]   == 0
        assert result["total_signals"]      == 0
        assert result["confirmation_score"] == 0.0


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    def _engine(self, **kwargs):
        defaults = dict(
            account_size=100_000,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=3.0,
            position_risk_pct=0.01,
            min_confidence=0.50,
        )
        defaults.update(kwargs)
        return ExecutionEngine(**defaults)

    def _stock_df_with_atr(self, n=50, atr=2.0, close=100.0):
        df           = _make_ohlcv(n)
        df["Close"]  = close
        df["ATR_14"] = atr
        return df

    def _pred(self, direction="up", confidence=0.70, ticker="AAPL"):
        return {"direction": direction, "confidence": confidence,
                "ticker": ticker, "horizon": 1}

    # Trade log structure
    def test_simulate_trade_returns_dict(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(), self._stock_df_with_atr())
        assert isinstance(result, dict)

    def test_trade_log_has_required_keys(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(), self._stock_df_with_atr())
        for key in ("action", "entry_price", "stop_loss", "take_profit",
                    "position_size", "risk_reward_ratio"):
            assert key in result, f"Missing: {key}"

    def test_long_trade_on_up_prediction(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70),
                                    self._stock_df_with_atr())
        assert result["action"] == "long"

    def test_short_trade_on_down_prediction(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="down", confidence=0.70),
                                    self._stock_df_with_atr())
        assert result["action"] == "short"

    def test_skip_on_low_confidence(self):
        eng    = self._engine(min_confidence=0.60)
        result = eng.simulate_trade(self._pred(confidence=0.45),
                                    self._stock_df_with_atr())
        assert result["action"] == "skip"

    def test_skip_on_flat_prediction(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="flat", confidence=0.80),
                                    self._stock_df_with_atr())
        assert result["action"] == "skip"

    def test_stop_loss_below_entry_for_long(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70),
                                    self._stock_df_with_atr(close=100.0, atr=2.0))
        assert result["stop_loss"] < result["entry_price"]

    def test_take_profit_above_entry_for_long(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70),
                                    self._stock_df_with_atr(close=100.0, atr=2.0))
        assert result["take_profit"] > result["entry_price"]

    def test_risk_reward_ratio_correct(self):
        eng    = self._engine(atr_multiplier_sl=2.0, atr_multiplier_tp=3.0)
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70),
                                    self._stock_df_with_atr(close=100.0, atr=2.0))
        if result["action"] != "skip":
            assert result["risk_reward_ratio"] == pytest.approx(1.5, rel=0.05)

    def test_position_size_positive_integer(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70),
                                    self._stock_df_with_atr())
        if result["action"] != "skip":
            assert isinstance(result["position_size"], int)
            assert result["position_size"] > 0

    def test_skip_when_all_atr_sources_are_zero(self):
        """
        All three ATR fallbacks (ATR_14 col, H-L rolling, price-vol proxy) must
        be defeated to trigger a skip.  Set ATR_14=0, High=Low=Close so H-L=0,
        and use a constant price so pct_change is also 0.
        """
        eng = self._engine()
        n   = 50
        df  = pd.DataFrame({
            "Close":  [100.0] * n,
            "High":   [100.0] * n,
            "Low":    [100.0] * n,
            "ATR_14": [0.0]   * n,
        })
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70), df)
        assert result["action"] == "skip"

    def test_blended_confidence_in_result(self):
        eng    = self._engine()
        result = eng.simulate_trade(self._pred(direction="up", confidence=0.70),
                                    self._stock_df_with_atr())
        assert "blended_confidence" in result
        assert 0.0 <= result["blended_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# feature_engine package import safety
# ---------------------------------------------------------------------------

class TestFeatureEngineImports:
    def test_core_modules_always_importable(self):
        from feature_engine import TechnicalIndicatorEngine, MarketHoursAligner, ExecutionEngine
        assert TechnicalIndicatorEngine is not None
        assert MarketHoursAligner       is not None
        assert ExecutionEngine          is not None

    def test_news_analyzer_importable(self):
        from feature_engine import NewsAnalyzer
        assert NewsAnalyzer is not None

    def test_optional_modules_are_none_or_class(self):
        from feature_engine import AlphaVantageClient, RedditSocialStream
        # They may be None if the optional dep is absent, or a class if present.
        # Either is valid — the key requirement is that importing __init__ does NOT raise.
        assert AlphaVantageClient is None or callable(AlphaVantageClient)
        assert RedditSocialStream is None or callable(RedditSocialStream)
