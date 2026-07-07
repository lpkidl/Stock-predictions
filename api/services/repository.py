"""Data readers: SQLite first (when populated and enabled), results/*.json
as fallback. Rollup math lives in pure functions so it can be unit-tested
and produces identical stats regardless of the source.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config import settings

from . import json_store

logger = logging.getLogger(__name__)

try:
    from db.models import DailyAccuracy, Prediction, PredictionOutcome, Trade
    from db.session import session_scope

    DB_IMPORTABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"DB layer unavailable, JSON only: {e}")
    DB_IMPORTABLE = False


def _db_ready() -> bool:
    return DB_IMPORTABLE and settings.DB_ENABLED


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


# ── Predictions ───────────────────────────────────────────────────────────────

def get_predictions() -> dict:
    """{ticker: {timestamp, horizons: {"1": {direction, confidence,
    probabilities{up,flat,down}, regime}, ...}}} — the predictions.json shape."""
    if _db_ready():
        try:
            with session_scope() as s:
                rows = s.query(Prediction).all()
                if rows:
                    latest: dict[tuple[str, int], Any] = {}
                    for r in rows:
                        k = (r.ticker, r.horizon_days)
                        if k not in latest or (
                            (r.predicted_at or "") and
                            (latest[k].predicted_at is None
                             or r.predicted_at > latest[k].predicted_at)
                        ):
                            latest[k] = r
                    out: dict[str, dict] = {}
                    for (ticker, horizon), r in latest.items():
                        entry = out.setdefault(ticker, {"horizons": {}, "timestamp": None})
                        entry["horizons"][str(horizon)] = {
                            "direction": r.direction,
                            "confidence": r.confidence,
                            "probabilities": {
                                "up": r.prob_up,
                                "flat": r.prob_flat,
                                "down": r.prob_down,
                            },
                            "regime": r.regime,
                        }
                        ts = _iso(r.predicted_at)
                        if ts and (entry["timestamp"] is None or ts > entry["timestamp"]):
                            entry["timestamp"] = ts
                    return out
        except Exception as e:
            logger.warning(f"DB predictions read failed, falling back to JSON: {e}")
    logger.info("Serving predictions from results/predictions.json")
    return json_store.load_json("predictions.json")


# ── Trades ────────────────────────────────────────────────────────────────────

def build_trades_response(logs: list[dict]) -> dict:
    """Split trade logs into active/skipped with summary totals.
    Mirrors ui/app.py tab4 (lines 946-965)."""
    active = [v for v in logs if v.get("action") not in ("skip", None)]
    skipped = [v for v in logs if v.get("action") == "skip"]
    last_run = next((v.get("timestamp") for v in logs if v.get("timestamp")), None)
    return {
        "last_run": last_run,
        "summary": {
            "evaluated": len(logs),
            "active": len(active),
            "skipped": len(skipped),
            "total_risk": round(sum(v.get("dollar_risk") or 0 for v in active), 2),
            "total_reward": round(sum(v.get("dollar_reward") or 0 for v in active), 2),
        },
        "active": active,
        "skipped": skipped,
    }


def _trade_row_to_dict(r) -> dict:
    tech_signals = {}
    if r.tech_signals:
        try:
            tech_signals = json.loads(r.tech_signals)
        except Exception:
            pass
    return {
        "ticker": r.ticker,
        "action": r.action,
        "reason": r.reason,
        "direction": r.direction,
        "horizon": r.horizon,
        "ml_confidence": r.ml_confidence,
        "tech_conf_score": r.tech_conf_score,
        "blended_confidence": r.blended_confidence,
        "confirming_signals": r.confirming_signals,
        "tech_signals": tech_signals,
        "entry_price": r.entry_price,
        "stop_loss": r.stop_loss,
        "take_profit": r.take_profit,
        "sl_distance": r.sl_distance,
        "tp_distance": r.tp_distance,
        "risk_reward_ratio": r.risk_reward_ratio,
        "position_size": r.position_size,
        "dollar_risk": r.dollar_risk,
        "dollar_reward": r.dollar_reward,
        "atr_used": r.atr_used,
        "account_size": r.account_size,
        "timestamp": _iso(r.executed_at),
    }


def get_trades() -> dict:
    if _db_ready():
        try:
            with session_scope() as s:
                from sqlalchemy import func

                max_run = s.query(func.max(Trade.run_id)).scalar()
                if max_run is not None:
                    rows = s.query(Trade).filter(Trade.run_id == max_run).all()
                    if rows:
                        return build_trades_response([_trade_row_to_dict(r) for r in rows])
        except Exception as e:
            logger.warning(f"DB trades read failed, falling back to JSON: {e}")
    logger.info("Serving trades from results/trade_logs.json")
    logs = json_store.load_json("trade_logs.json")
    return build_trades_response(list(logs.values()))


# ── Track record ──────────────────────────────────────────────────────────────

RANDOM_BASELINE = 1 / 3


def build_track_record(history: list[dict], daily_accuracy: list[dict] | None = None) -> dict:
    """All track-record rollups, computed once server-side.
    Mirrors ui/app.py tab5 (lines 1189-1305)."""
    resolved = [r for r in history if r.get("status") in ("correct", "incorrect")]
    pending = [r for r in history if r.get("status") == "pending"]
    n_correct = sum(1 for r in resolved if r["status"] == "correct")

    by_horizon = []
    for h in settings.FORECAST_HORIZONS:
        h_res = [r for r in resolved if r.get("horizon_days") == h]
        if not h_res:
            continue
        h_correct = sum(1 for r in h_res if r["status"] == "correct")
        by_horizon.append({
            "horizon_days": h,
            "evaluated": len(h_res),
            "correct": h_correct,
            "accuracy": round(h_correct / len(h_res), 4),
        })

    by_ticker = []
    for t in sorted({r["ticker"] for r in resolved}):
        t_res = [r for r in resolved if r["ticker"] == t]
        t_correct = sum(1 for r in t_res if r["status"] == "correct")
        by_ticker.append({
            "ticker": t,
            "evaluated": len(t_res),
            "correct": t_correct,
            "accuracy": round(t_correct / len(t_res), 4),
        })

    recent = sorted(resolved, key=lambda r: r.get("predicted_at") or "", reverse=True)[:30]
    pending_sorted = sorted(pending, key=lambda r: r.get("outcome_date") or "")

    return {
        "summary": {
            "overall_accuracy": round(n_correct / len(resolved), 4) if resolved else None,
            "correct": n_correct,
            "incorrect": len(resolved) - n_correct,
            "pending": len(pending),
            "random_baseline": round(RANDOM_BASELINE, 4),
        },
        "by_horizon": by_horizon,
        "by_ticker": by_ticker,
        "recent": recent,
        "pending": pending_sorted,
        "daily_accuracy": daily_accuracy or [],
    }


def _outcome_row_to_dict(r) -> dict:
    return {
        "id": r.id,
        "ticker": r.ticker,
        "horizon_days": r.horizon_days,
        "predicted_at": _iso(r.predicted_at),
        "predicted_direction": r.predicted_direction,
        "predicted_confidence": r.predicted_confidence,
        "entry_price": r.entry_price,
        "outcome_date": r.outcome_date.isoformat() if r.outcome_date else None,
        "actual_price": r.actual_price,
        "actual_direction": r.actual_direction,
        "actual_pct_change": r.actual_pct_change,
        "correct": r.correct,
        "status": r.status,
    }


def get_track_record() -> dict:
    if _db_ready():
        try:
            with session_scope() as s:
                rows = s.query(PredictionOutcome).all()
                if rows:
                    history = [_outcome_row_to_dict(r) for r in rows]
                    daily = [
                        {
                            "date": d.date.isoformat(),
                            "pct_correct": d.pct_correct,
                            "n_resolved": d.n_resolved,
                        }
                        for d in s.query(DailyAccuracy)
                        .filter(DailyAccuracy.ticker == "ALL", DailyAccuracy.horizon_days == 0)
                        .order_by(DailyAccuracy.date)
                        .all()
                    ]
                    return build_track_record(history, daily)
        except Exception as e:
            logger.warning(f"DB track-record read failed, falling back to JSON: {e}")
    logger.info("Serving track record from results/prediction_history.json")
    history = json_store.load_json("prediction_history.json", default=[])
    return build_track_record(history)


# ── JSON-only sources ─────────────────────────────────────────────────────────

def get_features(ticker: str, top_n: int) -> list[dict]:
    importance = json_store.load_json("feature_importance.json").get(ticker, {})
    return [{"name": k, "score": v} for k, v in list(importance.items())[:top_n]]


def get_metrics(ticker: str) -> dict:
    return json_store.load_json("model_metrics.json").get(ticker, {})
