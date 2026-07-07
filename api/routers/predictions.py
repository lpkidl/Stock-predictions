"""Latest ensemble predictions per ticker × horizon."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import validate_ticker
from api.services import repository

router = APIRouter()


@router.get("/predictions")
def all_predictions() -> dict:
    return repository.get_predictions()


@router.get("/predictions/{ticker}")
def ticker_predictions(ticker: str = Depends(validate_ticker)) -> dict:
    preds = repository.get_predictions()
    if ticker not in preds:
        raise HTTPException(status_code=404, detail=f"No predictions for {ticker}")
    return preds[ticker]
