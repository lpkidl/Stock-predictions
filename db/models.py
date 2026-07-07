"""
SQLAlchemy ORM models mirroring the pipeline's JSON/CSV outputs.

The database is written alongside the JSON files in results/ — JSON remains
the source of truth for v1, so every table here is derivable from those files
(except `posts`, which captures per-post data the JSON outputs discard).
"""

import datetime as dt
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    """One row per pipeline execution; other tables reference it via run_id."""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default="running")
    tickers: Mapped[Optional[str]] = mapped_column(String(256))
    notes: Mapped[Optional[str]] = mapped_column(Text)


class Post(Base):
    """A single Reddit/X post with its FinBERT sentiment inline (1:1)."""

    __tablename__ = "posts"
    __table_args__ = (
        UniqueConstraint("url", name="uq_posts_url"),
        Index("ix_posts_ticker_posted_at", "ticker", "posted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("pipeline_runs.id"), index=True
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)  # reddit / x
    ticker: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(64))  # reddit post_id
    subreddit: Mapped[Optional[str]] = mapped_column(String(64))
    author: Mapped[Optional[str]] = mapped_column(String(128))
    title: Mapped[Optional[str]] = mapped_column(Text)
    text: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    engagement_score: Mapped[Optional[int]] = mapped_column(Integer)  # upvotes / likes
    posted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(8))
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)  # [-1, 1]
    sentiment_confidence: Mapped[Optional[float]] = mapped_column(Float)
    prob_negative: Mapped[Optional[float]] = mapped_column(Float)
    prob_neutral: Mapped[Optional[float]] = mapped_column(Float)
    prob_positive: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class DailySentiment(Base):
    """Daily aggregated sentiment index per ticker (mirror of sentiment_index.csv)."""

    __tablename__ = "daily_sentiment"
    __table_args__ = (UniqueConstraint("date", "ticker", name="uq_daily_sentiment"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pipeline_runs.id"))
    date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(String(12), nullable=False)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    sentiment_std: Mapped[Optional[float]] = mapped_column(Float)
    post_count: Mapped[Optional[int]] = mapped_column(Integer)


class Prediction(Base):
    """One row per ticker x horizon per run (mirror of predictions.json)."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_lookup", "ticker", "horizon_days", "predicted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("pipeline_runs.id"), index=True
    )
    ticker: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    direction: Mapped[Optional[str]] = mapped_column(String(8))  # up / down / flat
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    prob_up: Mapped[Optional[float]] = mapped_column(Float)
    prob_down: Mapped[Optional[float]] = mapped_column(Float)
    prob_flat: Mapped[Optional[float]] = mapped_column(Float)
    regime: Mapped[Optional[str]] = mapped_column(String(32))
    predicted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)


class PredictionOutcome(Base):
    """Mirror of prediction_history.json; PK is the natural id
    "{ticker}_{date}_{horizon}d" so pending -> resolved upserts in place."""

    __tablename__ = "prediction_outcomes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pipeline_runs.id"))
    ticker: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    predicted_direction: Mapped[Optional[str]] = mapped_column(String(8))
    predicted_confidence: Mapped[Optional[float]] = mapped_column(Float)
    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    outcome_date: Mapped[Optional[dt.date]] = mapped_column(Date, index=True)
    actual_price: Mapped[Optional[float]] = mapped_column(Float)
    actual_direction: Mapped[Optional[str]] = mapped_column(String(8))
    actual_pct_change: Mapped[Optional[float]] = mapped_column(Float)
    correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(12), default="pending", index=True)


class Trade(Base):
    """One row per ticker per run from trade_logs.json (skips lack price fields)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("pipeline_runs.id"), index=True
    )
    ticker: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    action: Mapped[Optional[str]] = mapped_column(String(8))  # long / short / skip
    reason: Mapped[Optional[str]] = mapped_column(String(256))
    direction: Mapped[Optional[str]] = mapped_column(String(8))
    horizon: Mapped[Optional[int]] = mapped_column(Integer)
    ml_confidence: Mapped[Optional[float]] = mapped_column(Float)
    tech_conf_score: Mapped[Optional[float]] = mapped_column(Float)
    blended_confidence: Mapped[Optional[float]] = mapped_column(Float)
    confirming_signals: Mapped[Optional[str]] = mapped_column(String(8))  # e.g. "2/4"
    tech_signals: Mapped[Optional[str]] = mapped_column(Text)  # JSON dict of bools
    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    take_profit: Mapped[Optional[float]] = mapped_column(Float)
    sl_distance: Mapped[Optional[float]] = mapped_column(Float)
    tp_distance: Mapped[Optional[float]] = mapped_column(Float)
    risk_reward_ratio: Mapped[Optional[float]] = mapped_column(Float)
    position_size: Mapped[Optional[int]] = mapped_column(Integer)
    dollar_risk: Mapped[Optional[float]] = mapped_column(Float)
    dollar_reward: Mapped[Optional[float]] = mapped_column(Float)
    atr_used: Mapped[Optional[float]] = mapped_column(Float)
    account_size: Mapped[Optional[float]] = mapped_column(Float)
    executed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)


class DailyAccuracy(Base):
    """Percent of resolved predictions correct per outcome date.

    Rollups use sentinels ticker="ALL" and horizon_days=0 (not NULL — SQLite
    treats NULLs as distinct inside unique constraints)."""

    __tablename__ = "daily_accuracy"
    __table_args__ = (
        UniqueConstraint("date", "ticker", "horizon_days", name="uq_daily_accuracy"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(12), nullable=False)  # "ALL" = rollup
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 = rollup
    n_resolved: Mapped[int] = mapped_column(Integer, nullable=False)
    n_correct: Mapped[int] = mapped_column(Integer, nullable=False)
    pct_correct: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class PerformanceLedgerEntry(Base):
    """Mirror of performance_ledger.json entries (backtest metrics per run)."""

    __tablename__ = "performance_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("pipeline_runs.id"))
    ticker: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    horizon: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    wf_mean_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    test_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    test_f1: Mapped[Optional[float]] = mapped_column(Float)
    prediction_direction: Mapped[Optional[str]] = mapped_column(String(8))
    prediction_confidence: Mapped[Optional[float]] = mapped_column(Float)
    raw: Mapped[Optional[str]] = mapped_column(Text)  # full entry as JSON
