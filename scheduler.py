"""
Scheduler / orchestration entry-point for the Stock Forecaster pipeline.

Usage
-----
# Run once immediately (e.g. ad-hoc backfill or CI test):
    python scheduler.py --once

# Run on the default cron from config (4 pm ET, Mon–Fri):
    python scheduler.py

# Override the cron expression:
    python scheduler.py --cron "0 9 * * 1-5"

The cron field order follows standard Unix cron: minute hour dom month dow.

Dependencies
------------
    pip install apscheduler>=3.10.0
"""
import argparse
import asyncio
import logging
import os
import sys

# XGBoost + OpenMP crash on macOS when spawned inside a ThreadPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline_once() -> int:
    """Synchronous wrapper so APScheduler can call this without async plumbing."""
    from main import StockForecasterPipeline

    pipeline = StockForecasterPipeline()
    try:
        success = asyncio.run(pipeline.run_full_pipeline())
        return 0 if success else 1
    except Exception as exc:
        logger.error(f"Pipeline run failed: {exc}", exc_info=True)
        return 1
    finally:
        pipeline.cleanup()


def _parse_cron(expr: str) -> dict:
    """
    Convert a 5-field cron string to kwargs for APScheduler's CronTrigger.
    Raises ValueError on bad format.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Cron expression must have exactly 5 fields "
            f"(minute hour dom month dow), got: {expr!r}"
        )
    keys = ["minute", "hour", "day", "month", "day_of_week"]
    return dict(zip(keys, parts))


def start_scheduler(cron_expr: str) -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "APScheduler is not installed. Run: pip install apscheduler>=3.10.0"
        )
        sys.exit(1)

    cron_kwargs = _parse_cron(cron_expr)
    scheduler   = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        run_pipeline_once,
        CronTrigger(**cron_kwargs, timezone="America/New_York"),
        name="stock_forecaster",
        max_instances=1,       # prevent overlapping runs
        misfire_grace_time=300,
    )

    logger.info(f"Scheduler started — cron: {cron_expr!r} (America/New_York)")
    logger.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stock Forecaster pipeline scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline once immediately and exit.",
    )
    parser.add_argument(
        "--cron",
        default=settings.SCHEDULER_CRON,
        metavar="EXPR",
        help=(
            "Cron expression (5 fields: minute hour dom month dow). "
            f"Default from config: {settings.SCHEDULER_CRON!r}"
        ),
    )
    args = parser.parse_args()

    if args.once:
        return run_pipeline_once()

    start_scheduler(args.cron)
    return 0


if __name__ == "__main__":
    sys.exit(main())
