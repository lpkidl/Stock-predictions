"""Stock Forecaster API — app factory.

Security posture: loopback-only binding (see run command below), per-IP rate
limiting, whitelisted config exposure, ticker/period input validation, strict
CORS, and defensive response headers. Any future non-local deployment must sit
behind a TLS-terminating reverse proxy — never expose raw uvicorn publicly.

Run (dev):  venv/bin/uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.deps import limiter
from api.routers import (
    features,
    meta,
    metrics,
    predictions,
    prices,
    sources,
    track_record,
    trades,
)

app = FastAPI(title="Stock Forecaster API", docs_url="/api/docs", openapi_url="/api/openapi.json")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


for r in (meta, predictions, prices, features, metrics, trades, track_record, sources):
    app.include_router(r.router, prefix="/api")


# Production: serve the built React app from the same loopback process.
DIST_DIR = REPO_ROOT / "frontend" / "dist"
if DIST_DIR.is_dir():
    if (DIST_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = (DIST_DIR / full_path).resolve()
        # SPA fallback: unknown paths get index.html (client-side routing);
        # the resolve+prefix check blocks path traversal out of dist/.
        if (
            full_path
            and candidate.is_file()
            and str(candidate).startswith(str(DIST_DIR.resolve()))
        ):
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
