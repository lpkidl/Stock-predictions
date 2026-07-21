"""
ML Prediction Engine — multi-horizon ensemble classifier with regime detection.

Pipeline per ticker per horizon:
  • Technical indicators (original 23 + SMA_200 for regime labels)
  • Macro features: VIX level, 10Y-3M yield spread, sector ETF momentum
  • Earnings proximity: days_to_earnings, earnings_imminent binary flag
  • Regime detection: bull / bear / sideways via dual SMA crossover
  • Ternary classification target with horizon-scaled deadband (up / flat / down)
  • Ensemble: XGBClassifier + LogisticRegression, weighted by validation accuracy
  • Per-regime models with fallback to full-data model when regime has < REGIME_MIN_ROWS
  • Temporal (expanding-window) LOOCV on the training portion only
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from config import settings

REGIME_CODE = {"bear": -1.0, "sideways": 0.0, "bull": 1.0}

logger = logging.getLogger(__name__)

REGIME_BULL     = "bull"
REGIME_BEAR     = "bear"
REGIME_SIDEWAYS = "sideways"
CLASS_LABELS    = {0: "down", 1: "flat", 2: "up"}
N_CLASSES       = 3


def _align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Ensure probability matrix has exactly N_CLASSES columns.

    Needed when a regime slice misses a class entirely — XGB/LR will return
    fewer columns than N_CLASSES in that case.
    """
    if proba.shape[1] == N_CLASSES:
        return proba
    full = np.zeros((len(proba), N_CLASSES))
    for col_idx, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, col_idx]
    return full


class MLPredictor:
    """
    Multi-horizon ensemble classifier: XGBClassifier + LogisticRegression.
    One set of regime-specific ensembles per forecast horizon.
    """

    def __init__(self):
        # models[horizon][regime_or_"_fallback"] = {"xgb": ..., "lr": ..., "w_xgb": ..., "w_lr": ...}
        self.models:          Dict[int, Dict[str, dict]] = {}
        self.scalers:         Dict[int, StandardScaler]  = {}
        self.feature_columns: Optional[List[str]]        = None
        self.loocv_metrics:   Dict[int, Dict]            = {}
        self.val_metrics:     Dict[int, Dict]            = {}
        self.test_metrics:    Dict[int, Dict]            = {}

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def calculate_technical_indicators(
        self, df: pd.DataFrame, ticker: str = "UNKNOWN"
    ) -> pd.DataFrame:
        """
        Compute all technical features.  Adds SMA_200 on top of the original
        23 indicators so that regime labels can be derived downstream.
        """
        if df.empty or "Close" not in df.columns:
            logger.warning(f"Invalid data for {ticker}: missing Close column")
            return df
        try:
            df = df.copy()
            if not isinstance(df.index, pd.RangeIndex):
                df["date"] = pd.to_datetime(df.index)
                df = df.reset_index(drop=True)

            close = df["Close"]

            df["RSI"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
            macd              = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            df["MACD"]        = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_hist"]   = macd.macd_diff()

            bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            df["BB_upper"]    = bb.bollinger_hband()
            df["BB_middle"]   = bb.bollinger_mavg()
            df["BB_lower"]    = bb.bollinger_lband()
            df["BB_position"] = (
                (close - bb.bollinger_lband())
                / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-6)
            )

            df["SMA_20"]  = ta.trend.SMAIndicator(close, window=20).sma_indicator()
            df["SMA_50"]  = ta.trend.SMAIndicator(close, window=50).sma_indicator()
            df["SMA_200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
            df["ROC"]     = ta.momentum.ROCIndicator(close, window=14).roc()

            if "Volume" in df.columns:
                vol = df["Volume"].astype(float)
                df["Volume_SMA"]   = ta.trend.SMAIndicator(vol, window=20).sma_indicator()
                df["Volume_ratio"] = vol / (df["Volume_SMA"] + 1e-6)

            if "High" in df.columns and "Low" in df.columns:
                high, low = df["High"], df["Low"]
                df["ATR_14"]        = ta.volatility.AverageTrueRange(
                    high, low, close, window=14
                ).average_true_range()
                stoch               = ta.momentum.StochasticOscillator(
                    high, low, close, window=14, smooth_window=3
                )
                df["Stoch_K"]       = stoch.stoch()
                df["Stoch_D"]       = stoch.stoch_signal()
                df["High_Low_ratio"]= (high - low) / (close + 1e-6)

            df["price_vs_52w_high"] = close / (close.rolling(252, min_periods=50).max() + 1e-6)
            df["price_vs_52w_low"]  = close / (close.rolling(252, min_periods=50).min() + 1e-6)
            df["return_1d"]         = close.pct_change(1)  * 100
            df["return_5d"]         = close.pct_change(5)  * 100
            df["return_20d"]        = close.pct_change(20) * 100

            if "SMA_50" in df.columns:
                df["price_vs_SMA50_pct"] = (
                    (close - df["SMA_50"]) / (df["SMA_50"] + 1e-6) * 100
                )
            if "SMA_200" in df.columns:
                df["price_vs_SMA200_pct"] = (
                    (close - df["SMA_200"]) / (df["SMA_200"] + 1e-6) * 100
                )

            # ----------------------------------------------------------------
            # Layer 3: pandas-ta indicators (Ichimoku, ADX, BBW)
            # Runs after `ta` has already computed BB_upper/middle/lower and
            # ATR_14, so TechnicalIndicatorEngine can reuse those columns.
            # Wrapped in try/except so a missing pandas-ta install does not
            # abort the existing 23-indicator pipeline.
            # ----------------------------------------------------------------
            try:
                from feature_engine.tech_indicators import TechnicalIndicatorEngine
                df = TechnicalIndicatorEngine().apply_all(df)
            except Exception as _tic_exc:
                logger.warning(
                    f"TechnicalIndicatorEngine failed for {ticker} — "
                    f"continuing without Ichimoku/ADX/BBW. Error: {_tic_exc}"
                )

            n_ind = len([
                c for c in df.columns
                if c not in {"Open", "High", "Low", "Close", "Volume",
                             "Adj Close", "ticker", "date"}
            ])
            logger.info(f"Calculated {n_ind} indicators for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {e}")
            return df

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each row bull / bear / sideways using a dual SMA crossover.

        Bull:     close > SMA_50 > SMA_200
        Bear:     close < SMA_50 < SMA_200
        Sideways: anything else (including rows where SMA_200 is still NaN)
        """
        df = df.copy()
        if "SMA_50" not in df.columns or "SMA_200" not in df.columns:
            df["regime"] = REGIME_SIDEWAYS
            df["regime_code"] = REGIME_CODE[REGIME_SIDEWAYS]
            return df

        close = df["Close"]
        bull  = (close > df["SMA_50"])  & (df["SMA_50"]  > df["SMA_200"])
        bear  = (close < df["SMA_50"])  & (df["SMA_50"]  < df["SMA_200"])
        df["regime"] = np.select(
            [bull, bear], [REGIME_BULL, REGIME_BEAR], default=REGIME_SIDEWAYS
        )
        df.loc[df["SMA_200"].isna(), "regime"] = REGIME_SIDEWAYS
        # Numeric encoding so the single all-data model can still use the regime
        # as a feature (replaces the old per-regime model split).
        df["regime_code"] = df["regime"].map(REGIME_CODE).astype(float)
        return df

    def get_current_regime(self, df: pd.DataFrame) -> str:
        """Return the regime label of the most recent row."""
        if "regime" not in df.columns:
            return REGIME_SIDEWAYS
        last = df["regime"].dropna()
        return str(last.iloc[-1]) if len(last) > 0 else REGIME_SIDEWAYS

    # ------------------------------------------------------------------
    # Macro feature merging
    # ------------------------------------------------------------------

    def merge_macro_features(
        self,
        stock_df: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame],
        ticker: str,
    ) -> pd.DataFrame:
        """
        Join macro series into stock_df by date:
          • VIX close level
          • 10Y - 3M yield spread (^TNX - ^IRX)
          • Sector ETF 5d and 20d momentum, and stock relative strength vs sector
        """
        if not macro_data:
            return stock_df
        try:
            df = stock_df.copy()
            if "date" not in df.columns:
                df["date"] = pd.to_datetime(df.index)
            df["date"] = pd.to_datetime(df["date"])

            # VIX
            if "^VIX" in macro_data and macro_data["^VIX"] is not None:
                vix = macro_data["^VIX"].copy()
                if "date" not in vix.columns:
                    vix["date"] = pd.to_datetime(vix.index)
                vix = vix[["date", "Close"]].rename(columns={"Close": "VIX_close"})
                vix["date"] = pd.to_datetime(vix["date"])
                df = df.merge(vix, on="date", how="left")
                df["VIX_close"] = df["VIX_close"].ffill().fillna(20.0)

            # Yield curve spread
            has_tnx = "^TNX" in macro_data and macro_data["^TNX"] is not None
            has_irx = "^IRX" in macro_data and macro_data["^IRX"] is not None
            if has_tnx and has_irx:
                def _yield_df(key, col):
                    d = macro_data[key].copy()
                    if "date" not in d.columns:
                        d["date"] = pd.to_datetime(d.index)
                    d["date"] = pd.to_datetime(d["date"])
                    return d[["date", "Close"]].rename(columns={"Close": col})

                tnx = _yield_df("^TNX", "yield_10y")
                irx = _yield_df("^IRX", "yield_3m")
                yields = tnx.merge(irx, on="date", how="outer").sort_values("date")
                yields["yield_spread"] = yields["yield_10y"] - yields["yield_3m"]
                df = df.merge(yields[["date", "yield_spread"]], on="date", how="left")
                df["yield_spread"] = df["yield_spread"].ffill().fillna(0.0)

            # Sector ETF momentum & relative strength
            sector_ticker = settings.SECTOR_ETF_MAP.get(ticker)
            if sector_ticker and sector_ticker in macro_data and macro_data[sector_ticker] is not None:
                etf = macro_data[sector_ticker].copy()
                if "date" not in etf.columns:
                    etf["date"] = pd.to_datetime(etf.index)
                etf["date"] = pd.to_datetime(etf["date"])
                etf = etf[["date", "Close"]].rename(columns={"Close": "etf_close"}).sort_values("date")
                etf["sector_mom_5d"]  = etf["etf_close"].pct_change(5)  * 100
                etf["sector_mom_20d"] = etf["etf_close"].pct_change(20) * 100

                df = df.merge(
                    etf[["date", "sector_mom_5d", "sector_mom_20d"]], on="date", how="left"
                )
                df["sector_mom_5d"]  = df["sector_mom_5d"].ffill().fillna(0.0)
                df["sector_mom_20d"] = df["sector_mom_20d"].ffill().fillna(0.0)

                # Stock return minus sector return → relative strength
                df["rel_strength_5d"] = df["return_5d"] - df["sector_mom_5d"]

            return df

        except Exception as e:
            logger.error(f"Error merging macro features for {ticker}: {e}")
            return stock_df

    # ------------------------------------------------------------------
    # Earnings proximity features
    # ------------------------------------------------------------------

    def add_earnings_features(
        self,
        stock_df: pd.DataFrame,
        earnings_dates: List,
        ticker: str,
    ) -> pd.DataFrame:
        """
        For each row, compute:
          days_to_earnings  — calendar days until the next earnings event (capped at 60)
          earnings_imminent — 1 if earnings_within <= EARNINGS_WINDOW_DAYS, else 0
        Only future earnings dates (relative to each row) are considered.
        """
        df = stock_df.copy()
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.index)
        df["date"] = pd.to_datetime(df["date"])

        if not earnings_dates:
            df["days_to_earnings"]  = 60
            df["earnings_imminent"] = 0
            return df

        try:
            def _to_naive(d):
                ts = pd.Timestamp(d)
                return ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert(None)

            ts_list = sorted(_to_naive(d) for d in earnings_dates)

            def _days_to_next(row_date):
                future = [d for d in ts_list if d > row_date]
                return min((future[0] - row_date).days, 60) if future else 60

            df["days_to_earnings"]  = df["date"].apply(_days_to_next)
            df["earnings_imminent"] = (
                df["days_to_earnings"] <= settings.EARNINGS_WINDOW_DAYS
            ).astype(int)
            return df

        except Exception as e:
            logger.error(f"Error adding earnings features for {ticker}: {e}")
            df["days_to_earnings"]  = 60
            df["earnings_imminent"] = 0
            return df

    # ------------------------------------------------------------------
    # Sentiment merging (unchanged logic, kept for compatibility)
    # ------------------------------------------------------------------

    def merge_feature_data(
        self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        try:
            if "date" in stock_df.columns:
                stock_df["date"] = pd.to_datetime(stock_df["date"])
            elif stock_df.index.name == "Date":
                stock_df = stock_df.reset_index()
                stock_df["date"] = pd.to_datetime(stock_df["Date"])
                stock_df = stock_df.drop("Date", axis=1)
            else:
                stock_df["date"] = pd.to_datetime(stock_df.index)

            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            filtered = sentiment_df[sentiment_df["ticker"] == ticker].copy()

            merged = stock_df.merge(
                filtered[["date", "sentiment_score", "sentiment_std", "post_count"]],
                on="date", how="left",
            )
            merged["sentiment_score"] = merged["sentiment_score"].ffill()
            merged["sentiment_std"]   = merged["sentiment_std"].fillna(0.0)
            merged["post_count"]      = merged["post_count"].fillna(0.0)
            merged = merged.fillna(0.0)
            logger.info(f"Merged features for {ticker}: {merged.shape}")
            return merged

        except Exception as e:
            logger.error(f"Error merging feature data for {ticker}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Target & feature helpers
    # ------------------------------------------------------------------

    def _make_ternary_labels(self, y_pct: np.ndarray, horizon: int) -> np.ndarray:
        """Convert % return array → ternary class labels with a horizon-scaled deadband."""
        db = settings.TARGET_DEADBAND.get(horizon, 0.5)
        return np.where(y_pct > db, 2,        # up
               np.where(y_pct < -db, 0, 1))   # down / flat

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            "date", "ticker", "Close", "Open", "High", "Low",
            "Volume", "Adj Close", "Unnamed: 0", "Index", "regime",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and not c.startswith("target")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    def select_features(
        self, df: pd.DataFrame, forecast_horizon: int, top_k: Optional[int] = None
    ) -> List[str]:
        """Rank features by XGBoost gain on the TRAINING slice only (first
        1 - val - test of the data) and return the top-k names. Fitting on the
        training slice keeps val/test out of the selection to avoid leakage.

        Returns all features when top_k is None or there aren't enough rows to
        rank reliably — callers fall back to the full set gracefully."""
        all_cols = self._get_feature_cols(df)
        if top_k is None or len(all_cols) <= top_k:
            return all_cols
        try:
            df_clean = df.dropna(subset=["Close"])
            X_raw, y, _, _ = self._build_xy(df_clean, all_cols, forecast_horizon)
            train_end = int(len(X_raw) * (1.0 - 0.15 - 0.15))
            X_tr, y_tr = X_raw[:train_end], y[:train_end]
            if len(np.unique(y_tr)) < 2 or len(X_tr) < 50:
                return all_cols
            ranker = xgb.XGBClassifier(**self._xgb_params(remove_early_stop=True))
            ranker.fit(X_tr, y_tr, verbose=False)
            order = np.argsort(ranker.feature_importances_)[::-1][:top_k]
            selected = [all_cols[i] for i in sorted(order)]
            logger.info(
                f"Feature selection h{forecast_horizon}d: kept {len(selected)}/"
                f"{len(all_cols)} → {selected}"
            )
            return selected
        except Exception as e:
            logger.warning(f"Feature selection failed (h{forecast_horizon}d): {e} — keeping all")
            return all_cols

    def _build_xy(
        self,
        df_clean: pd.DataFrame,
        feature_cols: List[str],
        forecast_horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build X, ternary y, regime array, and date array."""
        # Fill NaN (early indicator warmup rows) with 0 before scaling
        X_raw = df_clean[feature_cols].fillna(0.0).values
        y_pct = np.array([
            (df_clean.iloc[i + forecast_horizon]["Close"]
             - df_clean.iloc[i]["Close"])
            / df_clean.iloc[i]["Close"] * 100
            for i in range(len(df_clean) - forecast_horizon)
        ])
        X_raw   = X_raw[:-forecast_horizon]
        y       = self._make_ternary_labels(y_pct, forecast_horizon)
        regimes = (
            df_clean["regime"].values[:-forecast_horizon]
            if "regime" in df_clean.columns
            else np.full(len(X_raw), REGIME_SIDEWAYS)
        )
        dates = (
            df_clean["date"].values[:-forecast_horizon]
            if "date" in df_clean.columns
            else np.arange(len(X_raw))
        )
        return X_raw, y, regimes, dates

    # ------------------------------------------------------------------
    # Training data preparation — 3-way split
    # ------------------------------------------------------------------

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        forecast_horizon: int = 1,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        feature_cols: Optional[List[str]] = None,
    ) -> Optional[Tuple]:
        """
        70% train / 15% val / 15% test chronological split.
        Val is used exclusively for early stopping and ensemble weighting.
        Returns scaled arrays and ternary class labels.

        Pass `feature_cols` to use a pre-selected subset (kept identical to the
        one handed to the walk-forward backtester); otherwise all numeric
        features are used.
        """
        try:
            df_clean = df.dropna(subset=["Close"])
            if len(df_clean) < 100 + forecast_horizon:
                logger.error(
                    f"Insufficient data for horizon {forecast_horizon}: {len(df_clean)} rows"
                )
                return None

            if feature_cols is None:
                feature_cols = self._get_feature_cols(df_clean)
            X_raw, y, regimes, dates = self._build_xy(df_clean, feature_cols, forecast_horizon)

            n         = len(X_raw)
            train_end = int(n * (1.0 - val_ratio - test_ratio))
            val_end   = int(n * (1.0 - test_ratio))

            X_train, y_train = X_raw[:train_end],        y[:train_end]
            X_val,   y_val   = X_raw[train_end:val_end], y[train_end:val_end]
            X_test,  y_test  = X_raw[val_end:],          y[val_end:]
            reg_train        = regimes[:train_end]
            train_dates      = dates[:train_end]
            val_dates        = dates[train_end:val_end]
            test_dates       = dates[val_end:]

            scaler    = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X_val)
            X_test_s  = scaler.transform(X_test)
            self.scalers[forecast_horizon] = scaler
            self.feature_columns = feature_cols

            dist = {CLASS_LABELS[c]: int((y_train == c).sum()) for c in range(N_CLASSES)}
            logger.info(
                f"Horizon {forecast_horizon}d — "
                f"Train {X_train_s.shape[0]}, Val {X_val_s.shape[0]}, "
                f"Test {X_test_s.shape[0]} | features: {len(feature_cols)} | "
                f"class dist: {dist}"
            )
            return (
                X_train_s, X_val_s, X_test_s,
                y_train, y_val, y_test,
                reg_train, feature_cols,
                {
                    "train_start": str(train_dates[0]),
                    "train_end":   str(train_dates[-1]),
                    "val_start":   str(val_dates[0]),
                    "val_end":     str(val_dates[-1]),
                    "test_start":  str(test_dates[0]),
                    "test_end":    str(test_dates[-1]),
                },
            )

        except Exception as e:
            logger.error(f"Error preparing training data (horizon {forecast_horizon}): {e}")
            return None

    # ------------------------------------------------------------------
    # Ensemble training helpers
    # ------------------------------------------------------------------

    def _xgb_params(self, remove_early_stop: bool = False) -> dict:
        p = dict(settings.XGBOOST_CLASSIFIER_PARAMS)
        if remove_early_stop:
            p.pop("early_stopping_rounds", None)
        return p

    def _train_ensemble(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label: str = "",
        use_early_stop: bool = True,
    ) -> Optional[dict]:
        """
        Train XGBClassifier + LogisticRegression.
        Returns a dict with both models and their val-accuracy-based weights.
        Returns None if training is impossible (e.g. fewer than 2 classes in y_tr).
        """
        if len(np.unique(y_tr)) < 2:
            logger.warning(f"{label}: only 1 class in training data, skipping")
            return None
        try:
            # Balance class weights so the model doesn't collapse toward the
            # majority ("flat"/"up") class.
            sample_weight = (
                compute_sample_weight("balanced", y_tr)
                if settings.BALANCE_CLASS_WEIGHTS else None
            )

            calibrating = settings.CALIBRATE_PROBABILITIES and len(np.unique(y_val)) >= 2
            # Early stopping and prefit-calibration don't compose (calibration
            # refits nothing but reads a fixed model); skip ES when calibrating.
            use_es  = use_early_stop and not calibrating
            params  = self._xgb_params(remove_early_stop=not use_es)
            xgb_clf = xgb.XGBClassifier(**params)
            if use_es:
                xgb_clf.fit(X_tr, y_tr, sample_weight=sample_weight,
                            eval_set=[(X_val, y_val)], verbose=False)
            else:
                xgb_clf.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)

            lr = LogisticRegression(
                max_iter=1000, C=0.5, class_weight="balanced",
                solver="lbfgs", random_state=42,
            )
            lr.fit(X_tr, y_tr)

            xgb_model, lr_model = xgb_clf, lr
            if calibrating:
                # Prefit calibration on the held-out validation slice so the
                # `confidence` the execution engine trades on is trustworthy.
                try:
                    xgb_model = CalibratedClassifierCV(
                        FrozenEstimator(xgb_clf), method="sigmoid"
                    ).fit(X_val, y_val)
                    lr_model = CalibratedClassifierCV(
                        FrozenEstimator(lr), method="sigmoid"
                    ).fit(X_val, y_val)
                except Exception as ce:
                    logger.warning(f"{label}: calibration failed ({ce}) — using raw probabilities")
                    xgb_model, lr_model = xgb_clf, lr

            xgb_acc = accuracy_score(y_val, xgb_model.predict(X_val))
            lr_acc  = accuracy_score(y_val, lr_model.predict(X_val))
            total   = xgb_acc + lr_acc
            w_xgb   = xgb_acc / total if total > 0 else 0.5
            w_lr    = lr_acc  / total if total > 0 else 0.5

            return {
                "xgb": xgb_model, "lr": lr_model,
                "w_xgb": w_xgb, "w_lr": w_lr,
                "xgb_raw": xgb_clf,  # uncalibrated tree model, for feature importance
            }

        except Exception as e:
            logger.error(f"Error training ensemble {label}: {e}")
            return None

    def _proba(self, ensemble: dict, X: np.ndarray) -> np.ndarray:
        """Weighted-average softmax probabilities from XGB + LR."""
        p_xgb = _align_proba(ensemble["xgb"].predict_proba(X), ensemble["xgb"].classes_)
        p_lr  = _align_proba(ensemble["lr"].predict_proba(X),  ensemble["lr"].classes_)
        return ensemble["w_xgb"] * p_xgb + ensemble["w_lr"] * p_lr

    def _predict_cls(self, ensemble: dict, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._proba(ensemble, X), axis=1)

    # ------------------------------------------------------------------
    # Model training — regime-specific ensembles
    # ------------------------------------------------------------------

    def train_model(
        self,
        X_train:      np.ndarray,
        X_val:        np.ndarray,
        X_test:       np.ndarray,
        y_train:      np.ndarray,
        y_val:        np.ndarray,
        y_test:       np.ndarray,
        regimes_train: np.ndarray,
        horizon:      int = 1,
    ):
        """Train regime-specific ensembles plus a full-data fallback."""
        logger.info(f"Training ensemble for horizon {horizon}d ...")
        self.models[horizon] = {}

        # Full-data fallback (always trained)
        fallback = self._train_ensemble(X_train, y_train, X_val, y_val, label=f"h{horizon}:all")
        if fallback is None:
            logger.error(f"Horizon {horizon}d: fallback training failed")
            return
        self.models[horizon]["_fallback"] = fallback

        # Per-regime models (optional). Splitting the training set into
        # bull/bear/sideways slices starved every sub-model of data — most fell
        # below REGIME_MIN_ROWS and fell back anyway. Off by default: the single
        # all-data model uses `regime_code` as a feature instead.
        regime_counts = {}
        if settings.USE_REGIME_MODELS:
            for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS]:
                mask = regimes_train == regime
                n    = int(mask.sum())
                regime_counts[regime] = n
                if n < settings.REGIME_MIN_ROWS:
                    logger.info(
                        f"  Regime {regime}: {n} rows < {settings.REGIME_MIN_ROWS} min, using fallback"
                    )
                    continue
                ens = self._train_ensemble(
                    X_train[mask], y_train[mask], X_val, y_val,
                    label=f"h{horizon}:{regime}",
                )
                if ens is not None:
                    self.models[horizon][regime] = ens
                    acc = accuracy_score(y_val, self._predict_cls(ens, X_val))
                    logger.info(f"  Regime {regime} ({n} rows): val acc {acc:.3f}")

        # Metrics on val and test using the fallback ensemble
        y_val_pred  = self._predict_cls(fallback, X_val)
        y_test_pred = self._predict_cls(fallback, X_test)

        def _cls_metrics(y_true, y_pred):
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }

        self.val_metrics[horizon]  = _cls_metrics(y_val, y_val_pred)
        self.test_metrics[horizon] = _cls_metrics(y_test, y_test_pred)

        trained_regimes = [r for r in self.models[horizon] if r != "_fallback"]
        logger.info(
            f"Horizon {horizon}d — "
            f"Val acc: {self.val_metrics[horizon]['accuracy']:.3f} | "
            f"Test acc: {self.test_metrics[horizon]['accuracy']:.3f} | "
            f"Regime rows: {regime_counts} | "
            f"Regime models: {trained_regimes or ['none (fallback only)']}"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, X: np.ndarray, regime: str = REGIME_SIDEWAYS, horizon: int = 1
    ) -> Dict:
        """
        Predict direction and confidence for one sample.
        X must already be scaled (output of prepare_training_data / the stored scaler).
        """
        if horizon not in self.models:
            return {"error": f"No model trained for horizon {horizon}"}
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            ens   = self.models[horizon].get(regime) or self.models[horizon]["_fallback"]
            proba = self._proba(ens, X)
            cls   = int(np.argmax(proba, axis=1)[0])
            conf  = float(proba[0][cls])
            return {
                "class":         cls,
                "direction":     CLASS_LABELS[cls],
                "confidence":    conf,
                "probabilities": {CLASS_LABELS[i]: float(proba[0][i]) for i in range(N_CLASSES)},
                "timestamp":     datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error predicting (horizon {horizon}): {e}")
            return {"error": str(e)}

    def predict_latest(
        self,
        merged_df: pd.DataFrame,
        feature_cols: List[str],
        horizon: int = 1,
        regime: str = REGIME_SIDEWAYS,
    ) -> Dict:
        """Forecast the FUTURE `horizon`-day move from the most recent bar.

        `train_model`/`predict` operate on `X_test[-1]`, whose features are
        `horizon` days stale (the label needs `i + horizon`, so `_build_xy`
        drops the final `horizon` rows). That row's move is already realized —
        not a forecast. This scores the true latest feature row (today) with the
        horizon's fitted scaler so the recorded prediction is genuinely forward
        looking and aligned with today's entry price.
        """
        if horizon not in self.scalers:
            return {"error": f"No scaler for horizon {horizon}"}
        try:
            df_clean = merged_df.dropna(subset=["Close"])
            latest_raw = df_clean[feature_cols].iloc[[-1]].fillna(0.0).values
            latest_scaled = self.scalers[horizon].transform(latest_raw)
            return self.predict(latest_scaled, regime=regime, horizon=horizon)
        except Exception as e:
            logger.error(f"Error predicting latest (horizon {horizon}): {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, horizon: int = 1, top_n: int = 10) -> Dict:
        if horizon not in self.models or self.feature_columns is None:
            return {}
        try:
            fallback   = self.models[horizon].get("_fallback")
            if fallback is None:
                return {}
            xgb_for_imp = fallback.get("xgb_raw", fallback["xgb"])
            if not hasattr(xgb_for_imp, "feature_importances_"):
                return {}
            scores     = xgb_for_imp.feature_importances_
            sorted_idx = np.argsort(scores)[::-1][:top_n]
            return {
                self.feature_columns[i]: float(scores[i])
                for i in sorted_idx
                if i < len(self.feature_columns)
            }
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    # ------------------------------------------------------------------
    # Metrics report
    # ------------------------------------------------------------------

    def get_metrics_report(self, horizon: int = 1) -> Dict:
        return {
            "val_metrics":   self.val_metrics.get(horizon, {}),
            "test_metrics":  self.test_metrics.get(horizon, {}),
            "loocv_metrics": self.loocv_metrics.get(horizon, {}),
            "model_params":  settings.XGBOOST_CLASSIFIER_PARAMS,
            "timestamp":     datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Temporal LOOCV (classification)
    # ------------------------------------------------------------------

    def loocv_validate(
        self,
        df: pd.DataFrame,
        forecast_horizon: int = 1,
        min_train: int = 60,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict:
        """
        Expanding-window temporal LOOCV over the training portion only.

        For each index i from min_train to train_end:
          • trains an equal-weight XGB + LR ensemble on rows [0 : i]
          • predicts row i
        This guarantees no future data leaks into any fold's model.

        Uses equal weights (0.5 / 0.5) to avoid a nested pseudo-val loop.
        """
        try:
            df_clean     = df.dropna(subset=["Close"])
            feature_cols = self._get_feature_cols(df_clean)
            X_raw, y, _, _ = self._build_xy(df_clean, feature_cols, forecast_horizon)

            n         = len(X_raw)
            train_end = int(n * (1.0 - val_ratio - test_ratio))

            if train_end <= min_train:
                logger.warning(
                    f"Not enough training data for LOOCV h{forecast_horizon}d "
                    f"(train_end={train_end})"
                )
                return {}

            params = self._xgb_params(remove_early_stop=True)

            loo_preds   = []
            loo_actuals = []

            # Stride so we evaluate at most LOOCV_MAX_FOLDS evenly-spaced folds —
            # LOOCV retrains per fold, so an unbounded loop scales badly on the
            # 10y window (~1.7k fits/horizon).
            stride = max(1, (train_end - min_train) // max(1, settings.LOOCV_MAX_FOLDS))

            for i in range(min_train, train_end, stride):
                X_tr = X_raw[:i]
                y_tr = y[:i]
                if len(np.unique(y_tr)) < 2:
                    continue

                scaler  = StandardScaler()
                X_tr_s  = scaler.fit_transform(X_tr)
                X_loo_s = scaler.transform(X_raw[i : i + 1])

                xgb_clf = xgb.XGBClassifier(**params)
                xgb_clf.fit(X_tr_s, y_tr, verbose=False)

                lr = LogisticRegression(
                    max_iter=500, C=0.5, class_weight="balanced",
                    solver="lbfgs", random_state=42,
                )
                lr.fit(X_tr_s, y_tr)

                p_xgb = _align_proba(xgb_clf.predict_proba(X_loo_s), xgb_clf.classes_)
                p_lr  = _align_proba(lr.predict_proba(X_loo_s),       lr.classes_)
                proba = 0.5 * p_xgb + 0.5 * p_lr
                loo_preds.append(int(np.argmax(proba)))
                loo_actuals.append(int(y[i]))

            if not loo_preds:
                return {}

            loo_preds   = np.array(loo_preds)
            loo_actuals = np.array(loo_actuals)
            n_folds     = len(loo_preds)
            acc         = float(accuracy_score(loo_actuals, loo_preds))
            f1          = float(f1_score(loo_actuals, loo_preds, average="weighted", zero_division=0))

            result = {
                "horizon": forecast_horizon,
                "n_folds": n_folds,
                "accuracy": acc,
                "f1":       f1,
            }
            logger.info(
                f"LOOCV horizon {forecast_horizon}d — "
                f"Acc {acc * 100:.1f}% | F1 {f1:.3f} | n={n_folds} folds"
            )
            self.loocv_metrics[forecast_horizon] = result
            return result

        except Exception as e:
            logger.error(f"Error in LOOCV (horizon {forecast_horizon}): {e}")
            return {}
