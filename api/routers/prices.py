"""yfinance-backed prices and quotes — rate limited harder because they can
reach an external upstream on cache miss."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.deps import limiter, validate_period, validate_ticker
from api.services import prices as price_service

router = APIRouter()


@router.get("/quotes")
@limiter.limit("10/minute")
def quotes(request: Request) -> dict:
    return price_service.get_quotes()


@router.get("/prices/{ticker}")
@limiter.limit("10/minute")
def ticker_prices(
    request: Request,
    ticker: str = Depends(validate_ticker),
    period: str = Query(default="1y"),
) -> dict:
    period = validate_period(period)
    payload = price_service.get_prices(ticker, period)
    if payload is None:
        raise HTTPException(status_code=502, detail="Upstream price data unavailable")
    return payload
