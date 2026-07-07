"""Per-ticker model metrics: val/test/LOOCV/walk-forward (JSON-only source)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import validate_ticker
from api.services import repository

router = APIRouter()


@router.get("/metrics/{ticker}")
def metrics(ticker: str = Depends(validate_ticker)) -> dict:
    return {"ticker": ticker, "horizons": repository.get_metrics(ticker)}
