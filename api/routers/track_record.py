"""Live prediction track record with server-side rollups."""

from __future__ import annotations

from fastapi import APIRouter

from api.services import repository

router = APIRouter()


@router.get("/track-record")
def track_record() -> dict:
    return repository.get_track_record()
