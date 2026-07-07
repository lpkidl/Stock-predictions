"""Latest run's trade-execution decisions."""

from __future__ import annotations

from fastapi import APIRouter

from api.services import repository

router = APIRouter()


@router.get("/trades")
def trades() -> dict:
    return repository.get_trades()
