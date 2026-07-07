"""
Best-effort write functions for the results database.

Every public function is wrapped with @_safe: if the DB is disabled,
unavailable, or a write fails for any reason, it logs a warning and returns
None — the JSON/CSV outputs remain the source of truth and the pipeline
must never fail because of the database.

Inputs are the same dict/list/DataFrame shapes the pipeline already produces
for its JSON files, so the backfill script reuses these functions verbatim.
"""

import functools
import json
import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser

from config import settings
from db.models import (
    DailyAccuracy,
    DailySentiment,
    PerformanceLedgerEntry,
    PipelineRun,
    Post,
    Prediction,
    PredictionOutcome,
    Trade,
)
from db.session import engine, session_scope

logger = logging.getLogger(__name__)

# Set to True when init_db() fails so later writes don't retry a broken engine
_disabled = False


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not settings.DB_ENABLED or _disabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                f"DB write failed in {fn.__name__} (JSON outputs unaffected): {exc}"
            )
            return None

    return wrapper


def _sqla_insert(model):
    """Dialect-aware insert supporting on_conflict_*. Single place to touch
    when moving from SQLite to Postgres (both dialects share the API)."""
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(model)


def _to_dt(value) -> Optional[datetime]:
    """Coerce ISO strings / datetimes / pandas Timestamps to naive datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = date_parser.parse(value)
        except (ValueError, OverflowError):
            return None
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    return None


def _to_date(value) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    dt = _to_dt(value)
    return dt.date() if dt else None


def _to_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pipeline run lifecycle
# ---------------------------------------------------------------------------

@_safe
def start_run(notes: Optional[str] = None) -> Optional[int]:
    with session_scope() as session:
        run = PipelineRun(
            started_at=datetime.utcnow(),
            status="running",
            tickers=settings.STOCK_TICKERS,
            notes=notes,
        )
        session.add(run)
        session.flush()
        return run.id


@_safe
def finish_run(run_id: Optional[int], status: str) -> None:
    if run_id is None:
        return
    with session_scope() as session:
        run = session.get(PipelineRun, run_id)
        if run:
            run.finished_at = datetime.utcnow()
            run.status = status


# ---------------------------------------------------------------------------
# Posts + sentiment
# ---------------------------------------------------------------------------

@_safe
def save_posts(run_id: Optional[int], sentiment_results: List[Dict]) -> None:
    """Insert per-post sentiment results (from process_ingestion_stream).
    Deduped on url: re-fetched posts are silently skipped."""
    if not sentiment_results:
        return
    rows = []
    skipped = 0
    for r in sentiment_results:
        url = r.get("url")
        if not url:
            skipped += 1
            continue
        probs = r.get("probabilities", {}) or {}
        rows.append({
            "run_id": run_id,
            "source": r.get("source", "unknown"),
            "ticker": r.get("ticker", "UNKNOWN"),
            "external_id": r.get("post_id"),
            "subreddit": r.get("subreddit"),
            "author": r.get("author"),
            "title": r.get("title"),
            "text": r.get("full_text") or r.get("text"),
            "url": url,
            "engagement_score": _to_int(r.get("engagement_score")),
            "posted_at": _to_dt(r.get("timestamp")),
            "sentiment_label": r.get("sentiment"),
            "sentiment_score": _to_float(r.get("score")),
            "sentiment_confidence": _to_float(r.get("confidence")),
            "prob_negative": _to_float(probs.get("negative")),
            "prob_neutral": _to_float(probs.get("neutral")),
            "prob_positive": _to_float(probs.get("positive")),
        })
    if skipped:
        logger.debug(f"save_posts: {skipped} result(s) had no URL — not stored")
    if not rows:
        return
    stmt = _sqla_insert(Post).on_conflict_do_nothing(index_elements=["url"])
    with session_scope() as session:
        session.execute(stmt, rows)
    logger.info(f"DB: stored {len(rows)} post(s) (duplicates skipped on url)")


@_safe
def save_daily_sentiment(run_id: Optional[int], sentiment_index) -> None:
    """Upsert daily sentiment index rows (DataFrame with columns
    date, ticker, sentiment_score, sentiment_std, post_count)."""
    if sentiment_index is None or getattr(sentiment_index, "empty", True):
        return
    with session_scope() as session:
        for rec in sentiment_index.to_dict("records"):
            d = _to_date(rec.get("date"))
            ticker = rec.get("ticker")
            if d is None or not ticker:
                continue
            stmt = _sqla_insert(DailySentiment).values(
                run_id=run_id,
                date=d,
                ticker=ticker,
                sentiment_score=_to_float(rec.get("sentiment_score")),
                sentiment_std=_to_float(rec.get("sentiment_std")),
                post_count=_to_int(rec.get("post_count")),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "ticker"],
                set_={
                    "run_id": stmt.excluded.run_id,
                    "sentiment_score": stmt.excluded.sentiment_score,
                    "sentiment_std": stmt.excluded.sentiment_std,
                    "post_count": stmt.excluded.post_count,
                },
            )
            session.execute(stmt)


# ---------------------------------------------------------------------------
# Predictions / outcomes / accuracy
# ---------------------------------------------------------------------------

@_safe
def save_predictions(run_id: Optional[int], predictions: Dict[str, Dict]) -> None:
    """Insert per-ticker-per-horizon predictions (predictions.json shape)."""
    if not predictions:
        return
    rows = []
    for ticker, data in predictions.items():
        predicted_at = _to_dt(data.get("timestamp"))
        for horizon_str, pred in (data.get("horizons") or {}).items():
            probs = pred.get("probabilities", {}) or {}
            rows.append(Prediction(
                run_id=run_id,
                ticker=ticker,
                horizon_days=int(horizon_str),
                direction=pred.get("direction"),
                confidence=_to_float(pred.get("confidence")),
                prob_up=_to_float(probs.get("up")),
                prob_down=_to_float(probs.get("down")),
                prob_flat=_to_float(probs.get("flat")),
                regime=pred.get("regime"),
                predicted_at=predicted_at,
            ))
    if not rows:
        return
    with session_scope() as session:
        session.add_all(rows)
    logger.info(f"DB: stored {len(rows)} prediction(s)")


@_safe
def upsert_prediction_outcomes(run_id: Optional[int], history: List[Dict]) -> None:
    """Upsert prediction_history.json records on their natural-key id so
    pending records get updated in place once outcomes resolve."""
    if not history:
        return
    with session_scope() as session:
        for r in history:
            rec_id = r.get("id")
            if not rec_id:
                continue
            values = {
                "id": rec_id,
                "ticker": r.get("ticker"),
                "horizon_days": _to_int(r.get("horizon_days")) or 0,
                "predicted_at": _to_dt(r.get("predicted_at")),
                "predicted_direction": r.get("predicted_direction"),
                "predicted_confidence": _to_float(r.get("predicted_confidence")),
                "entry_price": _to_float(r.get("entry_price")),
                "outcome_date": _to_date(r.get("outcome_date")),
                "actual_price": _to_float(r.get("actual_price")),
                "actual_direction": r.get("actual_direction"),
                "actual_pct_change": _to_float(r.get("actual_pct_change")),
                "correct": r.get("correct"),
                "status": r.get("status", "pending"),
            }
            stmt = _sqla_insert(PredictionOutcome).values(run_id=run_id, **values)
            update_cols = {k: getattr(stmt.excluded, k) for k in values if k != "id"}
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"], set_=update_cols
            )
            session.execute(stmt)
    logger.info(f"DB: upserted {len(history)} prediction outcome(s)")


@_safe
def upsert_daily_accuracy(history: List[Dict]) -> None:
    """Recompute percent-correct-per-day from the full history and upsert.

    Rows are written at three granularities per outcome date:
    (ticker, horizon), (ticker, ALL horizons -> 0), (ALL tickers -> "ALL", 0).
    Recomputing from scratch each run makes this self-healing.
    """
    resolved = [
        r for r in (history or [])
        if r.get("status") in ("correct", "incorrect") and r.get("outcome_date")
    ]
    if not resolved:
        return

    buckets: Dict[tuple, List[bool]] = defaultdict(list)
    for r in resolved:
        d = _to_date(r.get("outcome_date"))
        if d is None:
            continue
        is_correct = r.get("status") == "correct"
        horizon = _to_int(r.get("horizon_days")) or 0
        ticker = r.get("ticker", "UNKNOWN")
        buckets[(d, ticker, horizon)].append(is_correct)
        buckets[(d, ticker, 0)].append(is_correct)
        buckets[(d, "ALL", 0)].append(is_correct)

    with session_scope() as session:
        for (d, ticker, horizon), outcomes in buckets.items():
            n_resolved = len(outcomes)
            n_correct = sum(outcomes)
            stmt = _sqla_insert(DailyAccuracy).values(
                date=d,
                ticker=ticker,
                horizon_days=horizon,
                n_resolved=n_resolved,
                n_correct=n_correct,
                pct_correct=round(n_correct / n_resolved, 4),
                updated_at=datetime.utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "ticker", "horizon_days"],
                set_={
                    "n_resolved": stmt.excluded.n_resolved,
                    "n_correct": stmt.excluded.n_correct,
                    "pct_correct": stmt.excluded.pct_correct,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            session.execute(stmt)
    logger.info(f"DB: updated daily accuracy for {len(buckets)} (date, ticker, horizon) group(s)")


# ---------------------------------------------------------------------------
# Trades / performance ledger
# ---------------------------------------------------------------------------

@_safe
def save_trades(run_id: Optional[int], trade_logs: Dict[str, Dict]) -> None:
    """Insert trade logs (trade_logs.json shape: dict keyed by ticker).
    Skip-action logs lack price fields — all nullable."""
    if not trade_logs:
        return
    rows = []
    for ticker, t in trade_logs.items():
        tech_signals = t.get("tech_signals")
        rows.append(Trade(
            run_id=run_id,
            ticker=t.get("ticker", ticker),
            action=t.get("action"),
            reason=t.get("reason"),
            direction=t.get("direction"),
            horizon=_to_int(t.get("horizon")),
            ml_confidence=_to_float(t.get("ml_confidence")),
            tech_conf_score=_to_float(t.get("tech_conf_score")),
            blended_confidence=_to_float(t.get("blended_confidence")),
            confirming_signals=t.get("confirming_signals"),
            tech_signals=json.dumps(tech_signals) if tech_signals else None,
            entry_price=_to_float(t.get("entry_price")),
            stop_loss=_to_float(t.get("stop_loss")),
            take_profit=_to_float(t.get("take_profit")),
            sl_distance=_to_float(t.get("sl_distance")),
            tp_distance=_to_float(t.get("tp_distance")),
            risk_reward_ratio=_to_float(t.get("risk_reward_ratio")),
            position_size=_to_int(t.get("position_size")),
            dollar_risk=_to_float(t.get("dollar_risk")),
            dollar_reward=_to_float(t.get("dollar_reward")),
            atr_used=_to_float(t.get("atr_used")),
            account_size=_to_float(t.get("account_size")),
            executed_at=_to_dt(t.get("timestamp")),
        ))
    with session_scope() as session:
        session.add_all(rows)
    logger.info(f"DB: stored {len(rows)} trade log(s)")


@_safe
def save_performance_entries(run_id: Optional[int], entries: List[Dict]) -> None:
    """Insert performance-ledger entries (performance_ledger.json shape)."""
    if not entries:
        return
    rows = []
    for e in entries:
        wf = e.get("walk_forward", {}) or {}
        pred = e.get("prediction", {}) or {}
        rows.append(PerformanceLedgerEntry(
            run_id=run_id,
            ticker=e.get("ticker", "UNKNOWN"),
            horizon=_to_int(e.get("horizon")),
            timestamp=_to_dt(e.get("timestamp")),
            wf_mean_accuracy=_to_float(wf.get("mean_accuracy")),
            test_accuracy=_to_float(e.get("test_accuracy")),
            test_f1=_to_float(e.get("test_f1")),
            prediction_direction=pred.get("direction"),
            prediction_confidence=_to_float(pred.get("confidence")),
            raw=json.dumps(e, default=str),
        ))
    with session_scope() as session:
        session.add_all(rows)
