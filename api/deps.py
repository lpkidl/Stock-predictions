"""Shared router dependencies: rate limiter and input validation."""

from __future__ import annotations

from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.services.prices import ALLOWED_PERIODS, allowed_tickers

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def validate_ticker(ticker: str) -> str:
    """Only tickers configured in STOCK_TICKERS are servable — blocks
    arbitrary-symbol yfinance fetches and cache poisoning."""
    ticker = ticker.upper()
    if ticker not in allowed_tickers():
        raise HTTPException(status_code=403, detail=f"Ticker not allowed: {ticker}")
    return ticker


def validate_period(period: str) -> str:
    if period not in ALLOWED_PERIODS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid period. Allowed: {', '.join(ALLOWED_PERIODS)}",
        )
    return period
