"""
Tests for ml_engine/predictor.py — MLPredictor.

Uses the synthetic stock_df fixture (300 rows) so all tests are
deterministic and run offline.  A small subset deliberately uses 500 rows
so that the SMA-200 warmup period completes.
"""

import numpy as np
import pandas as pd
import pytest

from ml_engine.predictor import (
    MLPredictor,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_SIDEWAYS,
    CLASS_LABELS,
    N_CLASSES,
    _align_proba,
)
from tests.conftest import _make_ohlcv


TICKER = "AAPL"


# ---------------------------------------------------------------------------
# calculate_technical_indicators
# ---------------------------------------------------------------------------

class TestCalculateTechnicalIndicators:
    EXPECTED_COLS = [
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_middle", "BB_lower", "BB_position",
        "SMA_20", "SMA_50", "SMA_200",
        "ROC", "Volume_SMA", "Volume_ratio",
        "ATR_14", "Stoch_K", "Stoch_D", "High_Low_ratio",
        "price_vs_52w_high", "price_vs_52w_low",
        "return_1d", "return_5d", "return_20d",
        "price_vs_SMA50_pct", "price_vs_SMA200_pct",
    ]

    def test_adds_core_indicator_columns(self, stock_df):
        pred   = MLPredictor()
        result = pred.calculate_technical_indicators(stock_df, TICKER)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_adds_advanced_indicator_columns(self, stock_df):
        pred   = MLPredictor()
        result = pred.calculate_technical_indicators(stock_df, TICKER)
        for col in ("Tenkan_9", "Kijun_26", "ADX_14", "BBW"):
            assert col in result.columns, f"Missing advanced indicator: {col}"

    def test_returns_same_row_count(self, stock_df):
        pred   = MLPredictor()
        result = pred.calculate_technical_indicators(stock_df, TICKER)
        assert len(result) == len(stock_df)

    def test_returns_original_on_empty_df(self):
        pred   = MLPredictor()
        empty  = pd.DataFrame()
        result = pred.calculate_technical_indicators(empty, TICKER)
        assert result.empty

    def test_returns_original_when_close_missing(self, stock_df):
        pred    = MLPredictor()
        no_close = stock_df.drop(columns=["Close"])
        result  = pred.calculate_technical_indicators(no_close, TICKER)
        assert "RSI" not in result.columns

    def test_rsi_bounded_0_to_100(self, stock_df):
        pred   = MLPredictor()
        result = pred.calculate_technical_indicators(stock_df, TICKER)
        rsi    = result["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bb_position_mostly_0_to_1(self, stock_df):
        pred     = MLPredictor()
        result   = pred.calculate_technical_indicators(stock_df, TICKER)
        bb_pos   = result["BB_position"].dropna()
        # Some extreme moves can exceed [0,1]; at least 90% should be in range
        in_range = ((bb_pos >= -0.1) & (bb_pos <= 1.1)).mean()
        assert in_range > 0.9

    def test_date_column_added_when_index_is_datetime(self):
        pred = MLPredictor()
        df   = _make_ohlcv(300)
        df   = df.set_index("date")
        result = pred.calculate_technical_indicators(df, TICKER)
        assert "date" in result.columns


# ---------------------------------------------------------------------------
# detect_regime
# ---------------------------------------------------------------------------

class TestDetectRegime:
    def test_regime_column_added(self, stock_df):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.detect_regime(with_ti)
        assert "regime" in result.columns

    def test_regime_values_valid(self, stock_df):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.detect_regime(with_ti)
        valid   = {REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS}
        assert set(result["regime"].unique()).issubset(valid)

    def test_sideways_when_sma200_nan(self, stock_df):
        """Rows without SMA_200 warmup must be labelled sideways."""
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.detect_regime(with_ti)
        nan_sma = result[result["SMA_200"].isna()]
        assert (nan_sma["regime"] == REGIME_SIDEWAYS).all()

    def test_returns_sideways_without_sma_columns(self, stock_df):
        pred   = MLPredictor()
        result = pred.detect_regime(stock_df)
        assert "regime" in result.columns
        assert (result["regime"] == REGIME_SIDEWAYS).all()

    def test_get_current_regime_returns_string(self, stock_df):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.detect_regime(with_ti)
        regime  = pred.get_current_regime(result)
        assert regime in {REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS}

    def test_get_current_regime_fallback_without_column(self, stock_df):
        pred   = MLPredictor()
        regime = pred.get_current_regime(stock_df)
        assert regime == REGIME_SIDEWAYS


# ---------------------------------------------------------------------------
# merge_macro_features
# ---------------------------------------------------------------------------

class TestMergeMacroFeatures:
    def test_adds_vix_column(self, stock_df, macro_data):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.merge_macro_features(with_ti, macro_data, TICKER)
        assert "VIX_close" in result.columns

    def test_adds_yield_spread_column(self, stock_df, macro_data):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.merge_macro_features(with_ti, macro_data, TICKER)
        assert "yield_spread" in result.columns

    def test_adds_sector_momentum_columns(self, stock_df, macro_data):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.merge_macro_features(with_ti, macro_data, TICKER)
        assert "sector_mom_5d"  in result.columns
        assert "sector_mom_20d" in result.columns

    def test_no_nan_in_vix_after_merge(self, stock_df, macro_data):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.merge_macro_features(with_ti, macro_data, TICKER)
        assert result["VIX_close"].isna().sum() == 0

    def test_returns_original_on_empty_macro_data(self, stock_df):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.merge_macro_features(with_ti, {}, TICKER)
        assert list(result.columns) == list(with_ti.columns)

    def test_row_count_unchanged(self, stock_df, macro_data):
        pred    = MLPredictor()
        with_ti = pred.calculate_technical_indicators(stock_df, TICKER)
        result  = pred.merge_macro_features(with_ti, macro_data, TICKER)
        assert len(result) == len(with_ti)


# ---------------------------------------------------------------------------
# add_earnings_features
# ---------------------------------------------------------------------------

class TestAddEarningsFeatures:
    def test_adds_days_to_earnings_and_imminent(self, stock_df):
        pred   = MLPredictor()
        result = pred.add_earnings_features(stock_df, [], TICKER)
        assert "days_to_earnings"  in result.columns
        assert "earnings_imminent" in result.columns

    def test_defaults_to_60_when_no_earnings_dates(self, stock_df):
        pred   = MLPredictor()
        result = pred.add_earnings_features(stock_df, [], TICKER)
        assert (result["days_to_earnings"] == 60).all()
        assert (result["earnings_imminent"] == 0).all()

    def test_imminent_flag_set_when_earnings_near(self, stock_df):
        from datetime import date, timedelta
        import pandas as pd
        pred = MLPredictor()
        # Set an earnings date 3 days after the last row
        last_date    = stock_df["date"].max()
        near_earning = last_date + timedelta(days=3)
        result       = pred.add_earnings_features(stock_df, [near_earning], TICKER)
        assert result.iloc[-1]["earnings_imminent"] == 1

    def test_days_to_earnings_capped_at_60(self, stock_df):
        from datetime import date, timedelta
        far_future = pd.Timestamp(date.today()) + pd.Timedelta(days=200)
        pred   = MLPredictor()
        result = pred.add_earnings_features(stock_df, [far_future], TICKER)
        assert (result["days_to_earnings"] <= 60).all()


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    def _enriched(self, n=300):
        """Return a fully enriched DataFrame ready for training."""
        df   = _make_ohlcv(n, TICKER)
        pred = MLPredictor()
        df   = pred.calculate_technical_indicators(df, TICKER)
        df   = pred.detect_regime(df)
        df   = pred.add_earnings_features(df, [], TICKER)
        return df, pred

    def test_returns_tuple_of_correct_length(self):
        df, pred = self._enriched()
        result   = pred.prepare_training_data(df, forecast_horizon=5)
        assert result is not None
        assert len(result) == 9  # X_train, X_val, X_test, y_train, y_val, y_test, reg_train, feature_cols, date_ranges

    def test_train_val_test_sizes_sum_to_total(self):
        df, pred = self._enriched()
        result   = pred.prepare_training_data(df, forecast_horizon=5)
        X_train, X_val, X_test = result[0], result[1], result[2]
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        # Total should equal len(df) - forecast_horizon
        assert total == len(df) - 5

    def test_feature_cols_excludes_non_numeric(self):
        df, pred = self._enriched()
        result   = pred.prepare_training_data(df, forecast_horizon=5)
        feat_cols = result[7]
        assert "ticker" not in feat_cols
        assert "date"   not in feat_cols
        assert "regime" not in feat_cols

    def test_returns_none_on_insufficient_data(self):
        df, pred = self._enriched(n=50)  # too short
        result   = pred.prepare_training_data(df, forecast_horizon=5)
        assert result is None

    def test_y_labels_are_0_1_2(self):
        df, pred = self._enriched()
        result   = pred.prepare_training_data(df, forecast_horizon=5)
        y_train  = result[3]
        assert set(np.unique(y_train)).issubset({0, 1, 2})

    def test_date_ranges_dict_has_required_keys(self):
        df, pred = self._enriched()
        result   = pred.prepare_training_data(df, forecast_horizon=5)
        date_ranges = result[8]
        for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end"):
            assert key in date_ranges


# ---------------------------------------------------------------------------
# train_model + predict
# ---------------------------------------------------------------------------

class TestTrainAndPredict:
    def _trained_predictor(self, horizon=5):
        df   = _make_ohlcv(300, TICKER)
        pred = MLPredictor()
        df   = pred.calculate_technical_indicators(df, TICKER)
        df   = pred.detect_regime(df)
        df   = pred.add_earnings_features(df, [], TICKER)

        result = pred.prepare_training_data(df, forecast_horizon=horizon)
        X_train, X_val, X_test, y_train, y_val, y_test, reg_train, feat_cols, _ = result
        pred.train_model(X_train, X_val, X_test, y_train, y_val, y_test, reg_train, horizon=horizon)
        return pred, df, X_test

    def test_predict_returns_dict_with_required_keys(self):
        pred, df, X_test = self._trained_predictor()
        current_regime   = pred.get_current_regime(df)
        result = pred.predict(X_test[-1:], regime=current_regime, horizon=5)
        for key in ("direction", "confidence", "probabilities"):
            assert key in result, f"Missing key: {key}"

    def test_direction_is_valid_string(self):
        pred, df, X_test = self._trained_predictor()
        result = pred.predict(X_test[-1:], regime=pred.get_current_regime(df), horizon=5)
        assert result["direction"] in ("up", "flat", "down")

    def test_confidence_between_0_and_1(self):
        pred, df, X_test = self._trained_predictor()
        result = pred.predict(X_test[-1:], regime=pred.get_current_regime(df), horizon=5)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_1(self):
        pred, df, X_test = self._trained_predictor()
        result = pred.predict(X_test[-1:], regime=pred.get_current_regime(df), horizon=5)
        probs  = result["probabilities"]
        total  = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_models_stored_after_training(self):
        pred, _, _ = self._trained_predictor(horizon=5)
        assert 5 in pred.models
        assert len(pred.models[5]) >= 1

    def test_metrics_report_has_val_and_test(self):
        pred, _, _ = self._trained_predictor(horizon=5)
        report = pred.get_metrics_report(horizon=5)
        assert "val_metrics"  in report
        assert "test_metrics" in report


# ---------------------------------------------------------------------------
# _align_proba helper
# ---------------------------------------------------------------------------

class TestAlignProba:
    def test_passthrough_when_already_n_classes(self):
        proba   = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        classes = np.array([0, 1, 2])
        result  = _align_proba(proba, classes)
        np.testing.assert_array_equal(result, proba)

    def test_fills_missing_class_with_zero(self):
        # Only classes 0 and 2 present
        proba   = np.array([[0.3, 0.7]])
        classes = np.array([0, 2])
        result  = _align_proba(proba, classes)
        assert result.shape == (1, N_CLASSES)
        assert result[0, 0] == pytest.approx(0.3)
        assert result[0, 1] == pytest.approx(0.0)   # missing class → 0
        assert result[0, 2] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _make_ternary_labels
# ---------------------------------------------------------------------------

class TestMakeTernaryLabels:
    def test_positive_return_above_deadband_is_up(self):
        pred = MLPredictor()
        y    = pred._make_ternary_labels(np.array([2.0]), horizon=5)
        assert y[0] == 2  # up

    def test_negative_return_below_deadband_is_down(self):
        pred = MLPredictor()
        y    = pred._make_ternary_labels(np.array([-2.0]), horizon=5)
        assert y[0] == 0  # down

    def test_return_within_deadband_is_flat(self):
        pred = MLPredictor()
        y    = pred._make_ternary_labels(np.array([0.5]), horizon=5)  # deadband=1.0 for h5
        assert y[0] == 1  # flat

    def test_horizon_deadband_scales(self):
        pred = MLPredictor()
        # h1 deadband=0.3; a 0.5% move should be "up" for h1 but "flat" for h5
        y1 = pred._make_ternary_labels(np.array([0.5]), horizon=1)
        y5 = pred._make_ternary_labels(np.array([0.5]), horizon=5)
        assert y1[0] == 2  # up (0.5 > 0.3)
        assert y5[0] == 1  # flat (0.5 < 1.0)
