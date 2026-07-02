"""
ExecutionEngine — ATR-based trade simulation with dynamic Stop-Loss / Take-Profit
and technical-indicator confidence blending.

Takes the ML model's directional prediction and converts it into a fully
specified simulated trade entry, including:

  • Entry price      — last Close of the provided OHLCV window
  • Stop-Loss (SL)   — entry ± (atr_multiplier_sl × ATR_14)
  • Take-Profit (TP) — entry ± (atr_multiplier_tp × ATR_14)
  • Position size    — shares = floor(dollar_risk / risk_per_share)
                       where dollar_risk = account_size × position_risk_pct
                       and   risk_per_share = atr_multiplier_sl × ATR_14

Risk-reward logic:
  With default multipliers (SL=2×ATR, TP=3×ATR) the risk-reward ratio is
  always 1.5:1, meaning we target $1.50 profit for every $1.00 risked.
  This is intentionally conservative relative to many quant strategies
  (which target 2:1 or higher) to account for slippage and model error.

ATR fallback hierarchy:
  1. ATR_14 column if present and non-NaN (set by TechnicalIndicatorEngine or
     the existing predictor.py `ta` integration).
  2. Rolling 14-bar mean of (High - Low) if High/Low columns exist.
  3. Rolling 14-bar mean of |pct_change(Close)| × Close as a last resort.
  If all three fail, or ATR resolves to zero, the trade is skipped.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Weight given to the technical confirmation score when blending with ML confidence.
# blended = ml_confidence × (1 − TECH_WEIGHT) + confirmation_score × TECH_WEIGHT
# At 0.35: a 52% ML signal + 3/4 confirming indicators → ~62% blended confidence.
TECH_WEIGHT: float = 0.35


class ExecutionEngine:
    """
    Converts ML directional predictions into simulated trade execution logs.

    Simulated — no real orders are sent.  The output dict is written to
    results/trade_logs.json and can be consumed by the Streamlit dashboard
    to show proposed entry / SL / TP levels on the price chart.

    Args:
        account_size:       Total account equity in dollars (default: $100,000).
        atr_multiplier_sl:  ATR multiple for Stop-Loss distance (default: 2×).
        atr_multiplier_tp:  ATR multiple for Take-Profit distance (default: 3×).
        position_risk_pct:  Maximum fraction of account to risk per trade (default: 1%).
        min_confidence:     Minimum model confidence to enter a trade (default: 55%).
                            Predictions below this threshold produce action="skip".
    """

    def __init__(
        self,
        account_size:       float = 100_000.0,
        atr_multiplier_sl:  float = 2.0,
        atr_multiplier_tp:  float = 3.0,
        position_risk_pct:  float = 0.01,
        min_confidence:     float = 0.50,   # lowered: blending adds ≥10% on top
        tech_weight:        float = TECH_WEIGHT,
    ) -> None:
        self.account_size      = account_size
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.position_risk_pct = position_risk_pct
        self.min_confidence    = min_confidence
        self.tech_weight       = tech_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_trade(
        self,
        prediction: dict,
        ohlcv_df:   pd.DataFrame,
    ) -> dict:
        """
        Simulate a trade based on the ML model's directional prediction.

        Args:
            prediction: Dict from MLPredictor.predict():
                {
                  "direction":  "up" | "down" | "flat",
                  "confidence": float (0.0 – 1.0),
                  "ticker":     str,
                  "horizon":    int (forecast days),
                }
            ohlcv_df:   OHLCV DataFrame for the ticker (daily or intraday).
                        Must have a "Close" column; "ATR_14" is used if present.

        Returns:
            Trade execution log dict with keys:
              ticker, direction, action, reason,
              entry_price, stop_loss, take_profit,
              sl_distance, tp_distance, risk_reward_ratio,
              position_size, dollar_risk, dollar_reward,
              atr_used, confidence, horizon,
              account_size, atr_multiplier_sl, atr_multiplier_tp,
              timestamp (ISO UTC string)

            action is one of: "long" | "short" | "skip"
        """
        ticker     = prediction.get("ticker",     "UNKNOWN")
        direction  = prediction.get("direction",  "flat")
        ml_conf    = float(prediction.get("confidence", 0.0))
        horizon    = int(prediction.get("horizon", 1))

        # ---- Pre-checks (before the expensive confirmation scoring) ----
        if ohlcv_df is None or ohlcv_df.empty:
            return {"ticker": ticker, "action": "skip", "reason": "empty_ohlcv",
                    "timestamp": datetime.now(timezone.utc).isoformat()}

        if direction == "flat":
            return {"ticker": ticker, "action": "skip", "reason": "direction_flat",
                    "direction": direction, "confidence": round(ml_conf, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat()}

        # ---- Technical-indicator confirmation (Ichimoku / ADX / BBW) ----
        # Blends ML model confidence with the fraction of indicator signals that
        # agree with the predicted direction.
        # Formula: blended = ml × (1 − w) + confirmation_score × w
        from feature_engine.tech_indicators import TechnicalIndicatorEngine
        tic          = TechnicalIndicatorEngine()
        confirmation = tic.compute_signal_confirmation(ohlcv_df, direction)
        conf_score   = confirmation["confirmation_score"]

        # Only blend when indicator columns are actually present.
        # If none were found (total_signals == 0) keep raw ML confidence —
        # blending 0.0 confirmation would penalise rather than boost.
        if confirmation["total_signals"] > 0:
            blended_conf = ml_conf * (1.0 - self.tech_weight) + conf_score * self.tech_weight
        else:
            blended_conf = ml_conf

        base_log = {
            "ticker":              ticker,
            "direction":           direction,
            "ml_confidence":       round(ml_conf,       4),
            "tech_conf_score":     round(conf_score,     4),
            "blended_confidence":  round(blended_conf,   4),
            "tech_signals":        confirmation["detail"],
            "confirming_signals":  f"{confirmation['confirming_count']}/{confirmation['total_signals']}",
            "horizon":             horizon,
            "account_size":        self.account_size,
            "atr_multiplier_sl":   self.atr_multiplier_sl,
            "atr_multiplier_tp":   self.atr_multiplier_tp,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }

        if blended_conf < self.min_confidence:
            return {
                **base_log,
                "action": "skip",
                "reason": f"low_blended_confidence ({blended_conf:.1%} < {self.min_confidence:.1%})",
            }

        # ---- Entry price (last close) ----
        if "Close" not in ohlcv_df.columns:
            return {**base_log, "action": "skip", "reason": "no_close_column"}

        entry_price = float(ohlcv_df["Close"].iloc[-1])
        if math.isnan(entry_price) or entry_price <= 0:
            return {**base_log, "action": "skip", "reason": "invalid_entry_price"}

        # ---- ATR ----
        atr = self._get_atr(ohlcv_df)
        if atr is None or math.isnan(atr) or atr <= 0:
            return {**base_log, "action": "skip", "reason": "zero_atr"}

        # ---- Stop-Loss and Take-Profit ----
        sl_distance = self.atr_multiplier_sl * atr
        tp_distance = self.atr_multiplier_tp * atr

        if direction == "up":
            action      = "long"
            stop_loss   = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # direction == "down"
            action      = "short"
            stop_loss   = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # ---- Position sizing (fractional Kelly-lite: risk a fixed % of account) ----
        dollar_risk    = self.account_size * self.position_risk_pct
        risk_per_share = sl_distance                  # dollars risked if stopped out
        position_size  = math.floor(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

        dollar_reward   = position_size * tp_distance
        risk_reward     = round(self.atr_multiplier_tp / self.atr_multiplier_sl, 4)

        return {
            **base_log,
            "action":            action,
            "reason":            "signal_accepted",
            "entry_price":       round(entry_price, 4),
            "stop_loss":         round(stop_loss,   4),
            "take_profit":       round(take_profit, 4),
            "sl_distance":       round(sl_distance, 4),
            "tp_distance":       round(tp_distance, 4),
            "risk_reward_ratio": risk_reward,
            "position_size":     position_size,         # shares
            "dollar_risk":       round(dollar_risk,   2),
            "dollar_reward":     round(dollar_reward, 2),
            "atr_used":          round(atr, 6),
        }

    # ------------------------------------------------------------------
    # Private: ATR resolution
    # ------------------------------------------------------------------

    def _get_atr(self, ohlcv_df: pd.DataFrame) -> Optional[float]:
        """
        Resolve ATR from the OHLCV DataFrame using a three-level fallback:

          1. ATR_14 column   — populated by TechnicalIndicatorEngine or
                               the existing `ta` library in predictor.py.
          2. Rolling H-L mean — (High - Low).rolling(14).mean(), a crude
                               but valid proxy when Wilder-smoothed ATR
                               is unavailable.
          3. Price-vol proxy  — rolling |pct_change| × Close, for DataFrames
                               that only have a Close column.

        Returns None if all three methods fail.
        """
        # --- Fallback 1: pre-computed ATR_14 column ---
        if "ATR_14" in ohlcv_df.columns:
            val = ohlcv_df["ATR_14"].dropna()
            if not val.empty:
                atr = float(val.iloc[-1])
                if not math.isnan(atr) and atr > 0:
                    return atr

        # --- Fallback 2: rolling High-Low mean ---
        if "High" in ohlcv_df.columns and "Low" in ohlcv_df.columns:
            hl_range = (ohlcv_df["High"] - ohlcv_df["Low"]).rolling(14).mean()
            val      = hl_range.dropna()
            if not val.empty:
                atr = float(val.iloc[-1])
                if not math.isnan(atr) and atr > 0:
                    logger.debug("ATR: using High-Low rolling mean fallback.")
                    return atr

        # --- Fallback 3: price-volatility proxy ---
        if "Close" in ohlcv_df.columns:
            close = ohlcv_df["Close"]
            proxy = close.pct_change().abs().rolling(14).mean() * close
            val   = proxy.dropna()
            if not val.empty:
                atr = float(val.iloc[-1])
                if not math.isnan(atr) and atr > 0:
                    logger.debug("ATR: using pct-change proxy fallback.")
                    return atr

        logger.warning("ATR could not be resolved from OHLCV data.")
        return None
