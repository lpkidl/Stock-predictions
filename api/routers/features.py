"""Per-ticker feature importance (JSON-only source)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import validate_ticker
from api.services import repository

router = APIRouter()


@router.get("/features/{ticker}")
def features(
    ticker: str = Depends(validate_ticker),
    top_n: int = Query(default=10, ge=3, le=32),
) -> dict:
    return {"ticker": ticker, "features": repository.get_features(ticker, top_n)}
