"""
Walk-forward backtester for the ensemble classifier.

Each fold trains on an expanding window of prepared features and tests on the
next contiguous out-of-sample window.  A fresh StandardScaler is fit per fold
so no future statistics leak into the training distribution.

Returns per-fold and aggregate accuracy / F1 metrics that give a realistic
estimate of live out-of-sample performance — unlike the single 70/15/15 split,
this exercises the model across many distinct market conditions.
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import settings
from ml_engine.predictor import _align_proba, N_CLASSES

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Expanding-window walk-forward backtest.

    min_train   : minimum bars before the first test window begins
    step        : bars to advance the training cutoff each fold (~1 month = 21)
    test_window : bars in each out-of-sample evaluation window
    """

    def __init__(
        self,
        min_train: int = 200,
        step: int = 21,
        test_window: int = 21,
    ):
        self.min_train   = min_train
        self.step        = step
        self.test_window = test_window

    def _xgb_params(self) -> dict:
        p = dict(settings.XGBOOST_CLASSIFIER_PARAMS)
        p.pop("early_stopping_rounds", None)
        return p

    def _ternary(self, y_pct: np.ndarray, horizon: int) -> np.ndarray:
        db = settings.TARGET_DEADBAND.get(horizon, 0.5)
        return np.where(y_pct > db, 2, np.where(y_pct < -db, 0, 1))

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        horizon: int,
    ) -> Dict:
        """
        Run walk-forward backtest for one ticker / horizon pair.

        df must already have technical indicators and a 'Close' column.
        Returns a dict with per-fold results and aggregate statistics,
        or an empty dict if there is not enough data.
        """
        df_clean = df.dropna(subset=["Close"]).reset_index(drop=True)
        n = len(df_clean) - horizon
        if n < self.min_train + self.test_window:
            logger.warning(
                f"WF h{horizon}d: {n} usable bars — need at least "
                f"{self.min_train + self.test_window}. Skipping."
            )
            return {}

        X_raw = df_clean[feature_cols].fillna(0.0).values[:-horizon]
        y_pct = np.array([
            (df_clean.iloc[i + horizon]["Close"] - df_clean.iloc[i]["Close"])
            / df_clean.iloc[i]["Close"] * 100
            for i in range(n)
        ])
        y = self._ternary(y_pct, horizon)

        fold_results: List[Dict] = []
        cutoff = self.min_train

        while cutoff + self.test_window <= n:
            test_end = min(cutoff + self.test_window, n)

            X_tr, y_tr = X_raw[:cutoff],        y[:cutoff]
            X_te, y_te = X_raw[cutoff:test_end], y[cutoff:test_end]

            if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
                cutoff += self.step
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            try:
                xgb_clf = xgb.XGBClassifier(**self._xgb_params())
                xgb_clf.fit(X_tr_s, y_tr, verbose=False)

                lr = LogisticRegression(
                    max_iter=500, C=0.5, class_weight="balanced",
                    solver="lbfgs", random_state=42,
                )
                lr.fit(X_tr_s, y_tr)

                p_xgb = _align_proba(xgb_clf.predict_proba(X_te_s), xgb_clf.classes_)
                p_lr  = _align_proba(lr.predict_proba(X_te_s),       lr.classes_)
                preds = np.argmax(0.5 * p_xgb + 0.5 * p_lr, axis=1)

                fold_results.append({
                    "fold":       len(fold_results) + 1,
                    "train_bars": cutoff,
                    "test_bars":  len(y_te),
                    "accuracy":   float(accuracy_score(y_te, preds)),
                    "f1":         float(f1_score(y_te, preds, average="weighted", zero_division=0)),
                })
            except Exception as exc:
                logger.warning(f"WF fold {len(fold_results) + 1} (h{horizon}d) failed: {exc}")

            cutoff += self.step

        if not fold_results:
            return {}

        accs = [f["accuracy"] for f in fold_results]
        f1s  = [f["f1"]       for f in fold_results]
        result = {
            "horizon":       horizon,
            "n_folds":       len(fold_results),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "mean_f1":       float(np.mean(f1s)),
            "min_accuracy":  float(np.min(accs)),
            "max_accuracy":  float(np.max(accs)),
            "folds":         fold_results,
        }
        logger.info(
            f"Walk-forward h{horizon}d — "
            f"{len(fold_results)} folds | "
            f"mean acc {np.mean(accs):.1%} ± {np.std(accs):.1%} | "
            f"mean F1 {np.mean(f1s):.3f}"
        )
        return result
