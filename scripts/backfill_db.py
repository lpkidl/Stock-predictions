"""
One-time (re-runnable) backfill: import existing results/*.json into the
database. Reuses the db.writer functions verbatim, so running this twice
leaves row counts unchanged (everything is an upsert or url-deduped insert).

Raw posts cannot be backfilled — they were never persisted before the DB
existed; only aggregates survive in the JSON outputs.

Usage:  python scripts/backfill_db.py
"""

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill")

from db import writer as db_writer  # noqa: E402
from db.session import init_db, session_scope  # noqa: E402
from db import models  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


def _load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        logger.info(f"skip: {name} not found")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning(f"skip: could not parse {name}: {exc}")
        return None


def _table_empty(model) -> bool:
    """Guard for plain-insert tables: re-running the backfill must not
    duplicate rows (outcomes/accuracy/sentiment are upserts and always safe)."""
    with session_scope() as session:
        count = session.query(model).count()
    if count:
        logger.info(f"skip: {model.__tablename__} already has {count} row(s)")
        return False
    return True


def main() -> None:
    init_db()
    run_id = db_writer.start_run(notes="backfill")
    logger.info(f"backfill run_id={run_id}")

    history = _load_json("prediction_history.json")
    if history:
        db_writer.upsert_prediction_outcomes(run_id, history)
        db_writer.upsert_daily_accuracy(history)

    trade_logs = _load_json("trade_logs.json")
    if trade_logs and _table_empty(models.Trade):
        db_writer.save_trades(run_id, trade_logs)

    predictions = _load_json("predictions.json")
    if predictions and _table_empty(models.Prediction):
        db_writer.save_predictions(run_id, predictions)

    ledger = _load_json("performance_ledger.json")
    if ledger and _table_empty(models.PerformanceLedgerEntry):
        db_writer.save_performance_entries(run_id, ledger)

    csv_path = RESULTS_DIR / "sentiment_index.csv"
    if csv_path.exists():
        import pandas as pd
        db_writer.save_daily_sentiment(run_id, pd.read_csv(csv_path))
    else:
        logger.info("skip: sentiment_index.csv not found")

    logger.info("raw posts cannot be backfilled (never persisted pre-DB)")

    db_writer.finish_run(run_id, "success")

    with session_scope() as session:
        for model in (
            models.PipelineRun, models.Post, models.DailySentiment,
            models.Prediction, models.PredictionOutcome, models.Trade,
            models.DailyAccuracy, models.PerformanceLedgerEntry,
        ):
            count = session.query(model).count()
            logger.info(f"{model.__tablename__:<22} {count} row(s)")


if __name__ == "__main__":
    main()
