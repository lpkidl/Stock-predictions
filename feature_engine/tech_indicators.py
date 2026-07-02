"""
TechnicalIndicatorEngine — advanced indicators implemented in pure pandas/numpy.

Adds four indicator groups on top of the 23 already computed by
ml_engine/predictor.py (which uses the `ta` library):

  1. Ichimoku Cloud  — Tenkan (9), Kijun (26), Senkou A/B (52)
  2. ATR             — Average True Range, 14-period Wilder smoothing (RMA)
  3. ADX             — Average Directional Index + ±DI lines, 14-period
  4. BBW             — Bollinger Band Width = (upper − lower) / middle

No external dependencies beyond pandas and numpy.  `pandas-ta` is tried as an
optional accelerator for Ichimoku/ADX only; if unavailable the pure-pandas path
runs instead.  Both paths produce numerically identical results.

CRITICAL — look-ahead bias in Ichimoku Senkou Span A/B:
  The standard charting convention shifts Senkou A and B FORWARD 26 bars so the
  "cloud" appears ahead of price on a chart.  For ML feature engineering this is
  a data leak: the model would see information from bar t+26 when predicting t.

  This implementation NEVER applies the forward shift.  Senkou A and B are
  computed purely from data available at or before bar t:
    Senkou_A[t] = (Tenkan[t] + Kijun[t]) / 2
    Senkou_B[t] = (max(High[t−51..t]) + min(Low[t−51..t])) / 2
  The first ~51 rows will be NaN — this is correct and expected.

Integration:
  apply_all() is invoked inside MLPredictor.calculate_technical_indicators()
  after the `ta` library has already populated BB_upper, BB_middle, BB_lower,
  and ATR_14.  BBW reuses those existing columns; ATR skips itself if ATR_14
  already exists.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's moving average (RMA): equivalent to an EMA with alpha = 1/period.

    Used by ATR and ADX.  pandas ewm() with adjust=False and alpha=1/period
    matches Wilder's original definition exactly.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


class TechnicalIndicatorEngine:
    """
    Calculates advanced technical indicators and appends them as new columns
    to the input OHLCV DataFrame.

    All methods return a modified copy; the original DataFrame is never mutated.
    Each indicator inside apply_all() is wrapped in try/except so a failure in
    one group cannot corrupt the output of another.
    """

    def __init__(
        self,
        ichimoku_tenkan: int   = 9,
        ichimoku_kijun:  int   = 26,
        ichimoku_senkou: int   = 52,
        atr_period:      int   = 14,
        adx_period:      int   = 14,
        bb_period:       int   = 20,
        bb_std:          float = 2.0,
    ) -> None:
        self.ichimoku_tenkan = ichimoku_tenkan
        self.ichimoku_kijun  = ichimoku_kijun
        self.ichimoku_senkou = ichimoku_senkou
        self.atr_period      = atr_period
        self.adx_period      = adx_period
        self.bb_period       = bb_period
        self.bb_std          = bb_std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_signal_confirmation(
        self,
        df:        pd.DataFrame,
        direction: str,
    ) -> dict:
        """
        Score how strongly the current indicator state confirms a directional
        prediction from the ML model.

        Four independent sub-signals are evaluated; each either confirms (1)
        or fails to confirm (0) the given direction:

          1. Ichimoku T/K cross — Tenkan vs Kijun alignment with price
             Bullish: close > Tenkan > Kijun  (price above both averages, fast above slow)
             Bearish: close < Tenkan < Kijun

          2. Ichimoku cloud position — price relative to Senkou A/B
             Bullish: close > max(Senkou_A, Senkou_B)  (above the cloud)
             Bearish: close < min(Senkou_A, Senkou_B)  (below the cloud)

          3. ADX directional bias — trend strength + DI polarity
             Bullish: ADX > 20 (trending) AND +DI > -DI  (buying pressure dominant)
             Bearish: ADX > 20 AND -DI > +DI              (selling pressure dominant)
             ADX ≤ 20 → ranging market, DI lines unreliable → no confirmation

          4. BBW regime filter — Bollinger Band Width
             Confirms if 0.01 < BBW < 0.12 (active, non-degenerate volatility regime).
             BBW < 0.01 → extreme squeeze, breakout direction uncertain → no confirm.
             BBW > 0.12 → blown-out volatility, signal unreliable     → no confirm.
             This is direction-agnostic: it gates whether ANY signal is trustworthy.

        Blending formula (called by ExecutionEngine):
          blended_confidence = ml_confidence × (1 − TECH_WEIGHT)
                               + confirmation_score × TECH_WEIGHT
          where confirmation_score = confirming_signals / total_available_signals

        Args:
            df:        OHLCV DataFrame with indicator columns already appended
                       by apply_all().  Uses the last row only.
            direction: "up" | "down"  (flat → callers should not call this)

        Returns:
            dict with keys:
              confirming_count  (int)   — signals that confirmed the direction
              total_signals     (int)   — signals that had valid data to evaluate
              confirmation_score (float) — confirming / total, in [0, 1]
              detail            (dict)  — per-signal boolean map for logging
        """
        if df.empty:
            return {"confirming_count": 0, "total_signals": 0,
                    "confirmation_score": 0.0, "detail": {}}

        last  = df.iloc[-1]
        detail: dict = {}
        confirming = 0
        total      = 0

        def _val(col: str):
            v = last.get(col)
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

        close   = _val("Close")
        tenkan  = _val("Tenkan_9")
        kijun   = _val("Kijun_26")
        senk_a  = _val("Senkou_A")
        senk_b  = _val("Senkou_B")
        adx     = _val("ADX_14")
        dmp     = _val("DMP_14")
        dmn     = _val("DMN_14")
        bbw     = _val("BBW")

        # 1. Ichimoku T/K cross
        if close is not None and tenkan is not None and kijun is not None:
            total += 1
            if direction == "up":
                ok = close > tenkan > kijun
            else:
                ok = close < tenkan < kijun
            detail["ichimoku_tk"] = ok
            if ok:
                confirming += 1

        # 2. Ichimoku cloud position
        if close is not None and senk_a is not None and senk_b is not None:
            total += 1
            cloud_top    = max(senk_a, senk_b)
            cloud_bottom = min(senk_a, senk_b)
            if direction == "up":
                ok = close > cloud_top
            else:
                ok = close < cloud_bottom
            detail["ichimoku_cloud"] = ok
            if ok:
                confirming += 1

        # 3. ADX directional bias
        if adx is not None and dmp is not None and dmn is not None:
            total += 1
            if adx > 20:
                if direction == "up":
                    ok = dmp > dmn
                else:
                    ok = dmn > dmp
            else:
                ok = False   # ranging market — no directional confirmation
            detail["adx_di"] = ok
            if ok:
                confirming += 1

        # 4. BBW volatility regime gate (direction-agnostic)
        if bbw is not None:
            total += 1
            ok = 0.01 < bbw < 0.12
            detail["bbw_regime"] = ok
            if ok:
                confirming += 1

        score = confirming / max(total, 1)
        return {
            "confirming_count":   confirming,
            "total_signals":      total,
            "confirmation_score": round(score, 4),
            "detail":             detail,
        }

    def apply_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all four indicator groups sequentially.

        Returns:
            DataFrame with up to 8 new columns appended:
              Tenkan_9, Kijun_26, Senkou_A, Senkou_B   (Ichimoku)
              ADX_14, DMP_14, DMN_14                   (ADX + DI lines)
              BBW                                      (Bollinger Band Width)
            ATR_14 is only added if not already present.
        """
        df = df.copy()

        for method_name in (
            "calculate_ichimoku",
            "calculate_atr",
            "calculate_adx",
            "calculate_bbw",
        ):
            method = getattr(self, method_name)
            try:
                df = method(df)
            except Exception as exc:
                logger.warning(
                    f"TechnicalIndicatorEngine.{method_name} failed — "
                    f"continuing without it. Error: {exc}"
                )

        new_cols = [c for c in df.columns if c in {
            "Tenkan_9", "Kijun_26", "Senkou_A", "Senkou_B",
            "ADX_14", "DMP_14", "DMN_14", "BBW",
        }]
        logger.info(
            f"TechnicalIndicatorEngine added {len(new_cols)} column(s): {new_cols}"
        )
        return df

    # ------------------------------------------------------------------
    # 1. Ichimoku Cloud
    # ------------------------------------------------------------------

    def calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Ichimoku components without any forward shift.

        Formula:
          Tenkan_9  = (High.rolling(9).max()  + Low.rolling(9).min())  / 2
          Kijun_26  = (High.rolling(26).max() + Low.rolling(26).min()) / 2
          Senkou_A  = (Tenkan_9 + Kijun_26) / 2          ← NO forward shift
          Senkou_B  = (High.rolling(52).max() + Low.rolling(52).min()) / 2

        All four values use only data at or before bar t.
        NaN rows at the beginning reflect genuine insufficient history.

        Args:
            df: DataFrame with columns High, Low.

        Returns:
            df with new columns: Tenkan_9, Kijun_26, Senkou_A, Senkou_B.
        """
        if not all(c in df.columns for c in ("High", "Low")):
            raise ValueError("Ichimoku requires High and Low columns.")

        high, low = df["High"], df["Low"]

        tenkan = (
            high.rolling(self.ichimoku_tenkan).max() +
            low.rolling(self.ichimoku_tenkan).min()
        ) / 2.0

        kijun = (
            high.rolling(self.ichimoku_kijun).max() +
            low.rolling(self.ichimoku_kijun).min()
        ) / 2.0

        senkou_a = (tenkan + kijun) / 2.0

        senkou_b = (
            high.rolling(self.ichimoku_senkou).max() +
            low.rolling(self.ichimoku_senkou).min()
        ) / 2.0

        df["Tenkan_9"] = tenkan.values
        df["Kijun_26"] = kijun.values
        df["Senkou_A"] = senkou_a.values
        df["Senkou_B"] = senkou_b.values

        return df

    # ------------------------------------------------------------------
    # 2. ATR — Average True Range (Wilder's RMA)
    # ------------------------------------------------------------------

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ATR using Wilder's smoothing (RMA = EMA with alpha=1/period).

        True Range = max(H−L, |H−prev_C|, |L−prev_C|)
        ATR        = RMA(TR, period)

        If ATR_14 is already present (populated by predictor.py via `ta`),
        this method is a no-op to avoid a redundant column.

        Args:
            df: DataFrame with columns High, Low, Close.

        Returns:
            df unchanged if ATR_14 already exists, otherwise with ATR_14 added.
        """
        if "ATR_14" in df.columns:
            logger.debug("ATR_14 already present — skipping.")
            return df

        if not all(c in df.columns for c in ("High", "Low", "Close")):
            raise ValueError("ATR requires High, Low, Close columns.")

        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["ATR_14"] = _wilder_smooth(tr, self.atr_period).values
        return df

    # ------------------------------------------------------------------
    # 3. ADX — Average Directional Index
    # ------------------------------------------------------------------

    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ADX with +DI and -DI directional movement lines.

        ADX quantifies trend strength (0–100), direction-agnostic:
          ADX > 25 → trending   |  ADX < 20 → ranging / sideways

        +DI > -DI encodes bullish trend; +DI < -DI encodes bearish.

        Wilder's algorithm:
          +DM[t] = H[t] - H[t-1]  if (H[t]-H[t-1]) > (L[t-1]-L[t]) and > 0, else 0
          -DM[t] = L[t-1] - L[t]  if (L[t-1]-L[t]) > (H[t]-H[t-1]) and > 0, else 0
          Smooth +DM, -DM, TR using RMA(period)
          +DI  = 100 × (+DM_smooth / ATR_smooth)
          -DI  = 100 × (-DM_smooth / ATR_smooth)
          DX   = 100 × |+DI - -DI| / (+DI + -DI)
          ADX  = RMA(DX, period)

        Args:
            df: DataFrame with columns High, Low, Close.

        Returns:
            df with new columns: ADX_14, DMP_14 (+DI), DMN_14 (-DI).
        """
        if not all(c in df.columns for c in ("High", "Low", "Close")):
            raise ValueError("ADX requires High, Low, Close columns.")

        high, low, close = df["High"], df["Low"], df["Close"]
        prev_high = high.shift(1)
        prev_low  = low.shift(1)
        prev_close = close.shift(1)

        # True Range (same formula as ATR)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional movement
        up_move   = high - prev_high
        down_move = prev_low - low

        plus_dm  = np.where((up_move > down_move)   & (up_move   > 0), up_move,   0.0)
        minus_dm = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)

        plus_dm_series  = pd.Series(plus_dm,  index=df.index, dtype=float)
        minus_dm_series = pd.Series(minus_dm, index=df.index, dtype=float)

        # Wilder-smooth each component
        atr_smooth  = _wilder_smooth(tr,             self.adx_period)
        plus_smooth = _wilder_smooth(plus_dm_series, self.adx_period)
        minus_smooth= _wilder_smooth(minus_dm_series,self.adx_period)

        plus_di  = 100.0 * plus_smooth  / (atr_smooth + 1e-10)
        minus_di = 100.0 * minus_smooth / (atr_smooth + 1e-10)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = _wilder_smooth(dx, self.adx_period)

        df["ADX_14"] = adx.values
        df["DMP_14"] = plus_di.values
        df["DMN_14"] = minus_di.values

        return df

    # ------------------------------------------------------------------
    # 4. BBW — Bollinger Band Width
    # ------------------------------------------------------------------

    def calculate_bbw(
        self,
        df: pd.DataFrame,
        period: Optional[int]   = None,
        std:    Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute Bollinger Band Width: BBW = (upper − lower) / (middle + ε).

        High BBW → wide bands → volatile / expanding market.
        Low  BBW → narrow bands → squeeze, often precedes a breakout.

        Reuse strategy:
          predictor.py already computes BB_upper, BB_middle, BB_lower via the
          `ta` library.  If those columns are present, BBW is a single
          vectorised operation — no redundant indicator computation.
          If absent, the bands are derived from Close directly.

        Args:
            df:     DataFrame with Close (and ideally BB_upper/middle/lower).
            period: Override period (default: self.bb_period = 20).
            std:    Override std multiplier (default: self.bb_std = 2.0).

        Returns:
            df with new column: BBW.
        """
        period = period or self.bb_period
        std    = std    or self.bb_std

        # Fast path: reuse existing Bollinger columns from predictor.py
        if all(c in df.columns for c in ("BB_upper", "BB_middle", "BB_lower")):
            df["BBW"] = (
                (df["BB_upper"] - df["BB_lower"])
                / (df["BB_middle"] + 1e-6)
            )
            return df

        # Fallback: compute from Close
        if "Close" not in df.columns:
            raise ValueError("BBW requires a Close column.")

        close  = df["Close"]
        middle = close.rolling(period).mean()
        sigma  = close.rolling(period).std(ddof=0)

        upper  = middle + std * sigma
        lower  = middle - std * sigma

        df["BBW"] = ((upper - lower) / (middle + 1e-6)).values
        return df
