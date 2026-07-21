"""Recorded data sources: raw sentiment posts (with links) + recording status.

Posts live in the database only (never written to JSON), so this endpoint is
the window into what the pipeline is persisting run over run."""

from __future__ import annotations

from fastapi import APIRouter, Query

from api.services import repository

router = APIRouter()


@router.get("/data-sources")
def data_sources(
    limit: int = Query(default=60, ge=1, le=200),
    ticker: str | None = Query(default=None, max_length=12),
) -> dict:
    t = ticker.upper() if ticker else None
    return repository.get_data_sources(limit=limit, ticker=t)


@router.get("/sentiment-summary")
def sentiment_summary() -> dict:
    """Per-ticker latest sentiment score + short-term trend, for the tabs."""
    return repository.get_sentiment_summary()
