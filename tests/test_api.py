"""API accuracy + security tests.

Covers: track-record rollup math on a known fixture, trades summary totals,
RSI/SMA parity with the formulas the Streamlit UI used, input-validation
rejects, and no-secret-leak on /api/config.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import app
from api.services.prices import compute_rsi
from api.services.repository import build_track_record, build_trades_response

client = TestClient(app)


# ── Track-record rollups ──────────────────────────────────────────────────────

def _rec(ticker, horizon, status, predicted_at="2026-07-01T00:00:00"):
    return {
        "id": f"{ticker}_{predicted_at[:10]}_{horizon}d",
        "ticker": ticker,
        "horizon_days": horizon,
        "predicted_at": predicted_at,
        "status": status,
        "outcome_date": "2026-07-02",
    }


def test_track_record_rollups_exact():
    history = [
        _rec("AAPL", 1, "correct"),
        _rec("AAPL", 1, "incorrect", "2026-06-30T00:00:00"),
        _rec("AAPL", 3, "correct", "2026-06-29T00:00:00"),
        _rec("NVDA", 1, "correct", "2026-06-28T00:00:00"),
        _rec("NVDA", 5, "pending"),
        _rec("TSLA", 10, "pending"),
    ]
    out = build_track_record(history)

    assert out["summary"]["correct"] == 3
    assert out["summary"]["incorrect"] == 1
    assert out["summary"]["pending"] == 2
    assert out["summary"]["overall_accuracy"] == 0.75

    by_h = {r["horizon_days"]: r for r in out["by_horizon"]}
    assert by_h[1]["evaluated"] == 3 and by_h[1]["correct"] == 2
    assert by_h[1]["accuracy"] == round(2 / 3, 4)
    assert by_h[3]["accuracy"] == 1.0
    assert 5 not in by_h and 10 not in by_h  # pending-only horizons excluded

    by_t = {r["ticker"]: r for r in out["by_ticker"]}
    assert by_t["AAPL"]["accuracy"] == round(2 / 3, 4)
    assert by_t["NVDA"]["accuracy"] == 1.0
    assert "TSLA" not in by_t  # no resolved records

    # recent sorted newest-first, resolved only
    assert [r["predicted_at"][:10] for r in out["recent"]] == [
        "2026-07-01", "2026-06-30", "2026-06-29", "2026-06-28",
    ]


def test_track_record_empty_history():
    out = build_track_record([])
    assert out["summary"]["overall_accuracy"] is None
    assert out["summary"]["correct"] == 0
    assert out["by_horizon"] == [] and out["by_ticker"] == []


# ── Trades summary ────────────────────────────────────────────────────────────

def test_trades_summary_totals():
    logs = [
        {"ticker": "AAPL", "action": "long", "dollar_risk": 1000.0,
         "dollar_reward": 1500.0, "timestamp": "2026-07-03T00:00:00"},
        {"ticker": "NVDA", "action": "short", "dollar_risk": 999.5,
         "dollar_reward": 1499.25},
        {"ticker": "TSLA", "action": "skip", "reason": "direction_flat",
         "dollar_risk": 0, "dollar_reward": 0},
    ]
    out = build_trades_response(logs)
    s = out["summary"]
    assert s["evaluated"] == 3 and s["active"] == 2 and s["skipped"] == 1
    assert s["total_risk"] == 1999.5
    assert s["total_reward"] == 2999.25
    # totals must equal sums over the active list actually returned
    assert s["total_risk"] == round(sum(t["dollar_risk"] for t in out["active"]), 2)
    assert out["last_run"] == "2026-07-03T00:00:00"


# ── Indicator parity with ui/app.py formulas ─────────────────────────────────

def _fixture_close(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1.5, n)))


def test_rsi_matches_streamlit_formula():
    close = _fixture_close()
    # verbatim ui/app.py:558-562
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    expected = 100 - (100 / (1 + gain / (loss + 1e-9)))

    got = compute_rsi(close)
    pd.testing.assert_series_equal(got, expected)


def test_rsi_bounded():
    rsi = compute_rsi(_fixture_close()).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


# ── Input validation & security ───────────────────────────────────────────────

def test_unknown_ticker_403():
    for path in ("/api/predictions/EVIL", "/api/features/EVIL",
                 "/api/metrics/EVIL", "/api/prices/EVIL"):
        assert client.get(path).status_code == 403, path


def test_bad_period_422():
    assert client.get("/api/prices/AAPL?period=99y").status_code == 422


def test_top_n_bounds_422():
    assert client.get("/api/features/AAPL?top_n=2").status_code == 422
    assert client.get("/api/features/AAPL?top_n=33").status_code == 422


def test_config_leaks_no_secrets():
    body = client.get("/api/config").text.lower()
    for needle in ("password", "x_username", "x_email", "praw", "client_secret",
                   "alpha_vantage", "database_url"):
        assert needle not in body, f"secret-ish key leaked: {needle}"


def test_health_and_security_headers():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
