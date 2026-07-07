"""Health and whitelisted app configuration."""

from __future__ import annotations

from fastapi import APIRouter

from config import settings

router = APIRouter()

# The settings object holds credentials (X_*, PRAW_*) — this endpoint returns
# a hand-built whitelist only. Never serialize `settings` wholesale.

HORIZON_LABELS = {"1": "Tomorrow", "3": "3 Days", "5": "5 Days", "10": "10 Days"}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def config():
    p = settings.XGBOOST_CLASSIFIER_PARAMS
    model_settings = [
        {"setting": k, "value": str(v)}
        for k, v in {
            "Number of trees": p.get("n_estimators"),
            "Max tree depth": p.get("max_depth"),
            "Learning rate": p.get("learning_rate"),
            "L1 regularization": p.get("reg_alpha"),
            "L2 regularization": p.get("reg_lambda"),
            "Row sampling per tree": p.get("subsample"),
            "Feature sampling": p.get("colsample_bytree"),
            "Min leaf weight": p.get("min_child_weight"),
            "Early stopping": f"{p.get('early_stopping_rounds')} trees without improvement",
            "Total features": "32 (technical + macro + earnings + regime)",
            "Ensemble": "XGBClassifier + LogisticRegression (val-accuracy weighted)",
            "Target": "3-class: up / flat / down (per-horizon deadband)",
            "Forecast horizons": ", ".join(
                HORIZON_LABELS.get(str(h), f"{h}d") for h in settings.FORECAST_HORIZONS
            ),
        }.items()
    ]
    return {
        "tickers": [t.strip() for t in settings.STOCK_TICKERS.split(",") if t.strip()],
        "horizons": settings.FORECAST_HORIZONS,
        "deadbands": {str(k): v for k, v in settings.TARGET_DEADBAND.items()},
        "model_settings": model_settings,
    }
