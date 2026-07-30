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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
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

        # Per-test-sample accumulators (pooled across every fold). These drive
        # the selective / directional metrics below, which need sample-level
        # confidence and realized returns — not just per-fold accuracy.
        s_true: List[int]   = []
        s_pred: List[int]   = []
        s_conf: List[float] = []
        s_ret:  List[float] = []
        s_pup:  List[float] = []   # calibrated P(up)   per test sample
        s_pdn:  List[float] = []   # calibrated P(down) per test sample

        while cutoff + self.test_window <= n:
            test_end = min(cutoff + self.test_window, n)

            X_tr, y_tr = X_raw[:cutoff],        y[:cutoff]
            X_te, y_te = X_raw[cutoff:test_end], y[cutoff:test_end]
            r_te       = y_pct[cutoff:test_end]

            if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
                cutoff += self.step
                continue

            try:
                proba = self._fold_proba(X_tr, y_tr, X_te)
                preds = np.argmax(proba, axis=1)
                conf  = proba[np.arange(len(preds)), preds]

                s_true.extend(y_te.tolist())
                s_pred.extend(preds.tolist())
                s_conf.extend(conf.tolist())
                s_ret.extend(r_te.tolist())
                s_pup.extend(proba[:, 2].tolist())   # class 2 = up
                s_pdn.extend(proba[:, 0].tolist())   # class 0 = down

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
            "selective":     self._selective_metrics(
                np.array(s_true), np.array(s_pred),
                np.array(s_conf), np.array(s_ret),
                np.array(s_pup),  np.array(s_pdn), horizon,
            ),
        }
        sel = result["selective"]
        logger.info(
            f"Walk-forward h{horizon}d — "
            f"{len(fold_results)} folds | "
            f"mean acc {np.mean(accs):.1%} ± {np.std(accs):.1%} | "
            f"pooled {sel['pooled_accuracy']:.1%} | "
            f"gated@{sel['gate']:.2f} {sel['gated_accuracy']:.1%} (cov {sel['gated_coverage']:.1%}) | "
            f"dir@{sel['gate']:.2f} {sel['directional_gated_accuracy']:.1%} (cov {sel['directional_gated_coverage']:.1%})"
        )
        return result

    # ------------------------------------------------------------------
    # Per-fold ensemble → calibrated class probabilities
    # ------------------------------------------------------------------

    def _fold_proba(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
    ) -> np.ndarray:
        """Fit the XGB+LR ensemble on one expanding window and return calibrated
        class probabilities for the test window.

        When CALIBRATE_PROBABILITIES is on we hold out the last 15% of the
        training window as a calibration set (prefit sigmoid calibration, exactly
        like `MLPredictor._train_ensemble`) so the reported `confidence` is
        trustworthy — the selective metrics gate on it. A fresh StandardScaler is
        fit per fold on the fit slice only, so no future stats leak in.
        """
        cal_start = int(len(X_tr) * 0.85)
        Xin, yin = X_tr[:cal_start], y_tr[:cal_start]
        Xca, yca = X_tr[cal_start:], y_tr[cal_start:]
        calibrating = (
            settings.CALIBRATE_PROBABILITIES
            and len(Xca) >= 10
            and len(np.unique(yin)) >= 2
            and len(np.unique(yca)) >= 2
        )
        fit_X_raw, fit_y = (Xin, yin) if calibrating else (X_tr, y_tr)

        scaler   = StandardScaler()
        fit_X    = scaler.fit_transform(fit_X_raw)
        X_te_s   = scaler.transform(X_te)

        sw = (
            compute_sample_weight("balanced", fit_y)
            if settings.BALANCE_CLASS_WEIGHTS else None
        )
        xgb_clf = xgb.XGBClassifier(**self._xgb_params())
        xgb_clf.fit(fit_X, fit_y, sample_weight=sw, verbose=False)
        lr = LogisticRegression(
            max_iter=500, C=0.5, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )
        lr.fit(fit_X, fit_y)

        xgb_model, lr_model = xgb_clf, lr
        if calibrating:
            try:
                Xca_s = scaler.transform(Xca)
                xgb_model = CalibratedClassifierCV(
                    FrozenEstimator(xgb_clf), method="sigmoid").fit(Xca_s, yca)
                lr_model = CalibratedClassifierCV(
                    FrozenEstimator(lr), method="sigmoid").fit(Xca_s, yca)
            except Exception:
                xgb_model, lr_model = xgb_clf, lr

        p_xgb = _align_proba(xgb_model.predict_proba(X_te_s), xgb_model.classes_)
        p_lr  = _align_proba(lr_model.predict_proba(X_te_s),  lr_model.classes_)
        return 0.5 * p_xgb + 0.5 * p_lr

    # ------------------------------------------------------------------
    # Selective / directional metrics
    # ------------------------------------------------------------------

    def _selective_metrics(
        self,
        true: np.ndarray,
        pred: np.ndarray,
        conf: np.ndarray,
        ret:  np.ndarray,
        p_up:  np.ndarray = None,
        p_down: np.ndarray = None,
        horizon: int = None,
    ) -> Dict:
        """Compute four honest views of accuracy from pooled test samples:

        1. pooled full-coverage 3-class accuracy (every sample scored);
        2. confidence-gated 3-class accuracy — only samples whose calibrated
           confidence clears the (per-horizon) gate, plus coverage (fraction of
           days it fires) and a threshold sweep for the dashboard;
        3. directional accuracy — among samples the model calls a direction on
           (pred != flat), was the sign of the realized move right?
        4. FULL-COVERAGE binary up/down — collapse the calibrated probabilities
           to up-vs-down (argmax of P_up vs P_down) on EVERY day, judged by the
           sign of the realized return. This is the true 2-class metric: baseline
           50%, and unlike (3) it scores every day, not just non-flat calls.
        """
        # Per-horizon gate (falls back to the scalar default).
        gate = settings.WF_CONFIDENCE_GATE_BY_HORIZON.get(
            horizon, settings.WF_CONFIDENCE_GATE
        )
        total = len(true)
        if total == 0:
            return {"gate": gate}

        def _acc(mask):
            return float(accuracy_score(true[mask], pred[mask])) if mask.sum() else None

        gate_mask = conf >= gate
        # Directional (subset): fold to up(2)/down(0) by dropping "flat" calls.
        dir_mask  = pred != 1
        dir_gate  = dir_mask & gate_mask

        def _dir_acc(mask):
            if not mask.sum():
                return None
            truth = np.where(ret[mask] > 0, 2, 0)
            return float(accuracy_score(truth, pred[mask]))

        # ---- Full-coverage binary up/down (levers: binary target + gating) ----
        bin_block = {}
        if p_up is not None and p_down is not None and len(p_up) == total:
            bin_pred  = np.where(p_up >= p_down, 2, 0)
            bin_truth = np.where(ret > 0, 2, 0)
            denom     = p_up + p_down + 1e-9
            bin_conf  = np.maximum(p_up, p_down) / denom   # renormalised 2-class conf
            bin_gate  = bin_conf >= gate

            def _bin_acc(mask):
                return float(accuracy_score(bin_truth[mask], bin_pred[mask])) if mask.sum() else None

            bin_block = {
                "binary_accuracy":          _bin_acc(np.ones(total, dtype=bool)),
                "binary_gated_accuracy":    _bin_acc(bin_gate),
                "binary_gated_coverage":    float(bin_gate.mean()),
                "binary_gated_n":           int(bin_gate.sum()),
            }

        # Threshold sweep so the UI can plot the accuracy/coverage trade-off.
        sweep = []
        for th in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            m  = conf >= th
            dm = dir_mask & m
            row = {
                "threshold":            round(th, 2),
                "coverage":             float(m.mean()),
                "accuracy":             _acc(m),
                "n":                    int(m.sum()),
                "directional_coverage": float(dm.mean()),
                "directional_accuracy": _dir_acc(dm),
                "directional_n":        int(dm.sum()),
            }
            if bin_block:
                bm = bin_conf >= th
                row["binary_coverage"] = float(bm.mean())
                row["binary_accuracy"] = (
                    float(accuracy_score(bin_truth[bm], bin_pred[bm])) if bm.sum() else None
                )
                row["binary_n"] = int(bm.sum())
            sweep.append(row)

        return {
            "gate":                        gate,
            "n_samples":                   total,
            "pooled_accuracy":             _acc(np.ones(total, dtype=bool)),
            "gated_accuracy":              _acc(gate_mask),
            "gated_coverage":              float(gate_mask.mean()),
            "gated_n":                     int(gate_mask.sum()),
            "directional_accuracy":        _dir_acc(dir_mask),
            "directional_coverage":        float(dir_mask.mean()),
            "directional_gated_accuracy":  _dir_acc(dir_gate),
            "directional_gated_coverage":  float(dir_gate.mean()),
            "directional_gated_n":         int(dir_gate.sum()),
            **bin_block,
            "sweep":                       sweep,
        }
