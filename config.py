"""
Configuration management using Pydantic Settings.
Reads environment variables from .env file for secure credential storage.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """
    
    # Application settings
    DEBUG: bool = False
    APP_NAME: str = "Stock Forecaster"
    
    # Reddit settings
    REDDIT_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    REDDIT_REQUEST_TIMEOUT: int = 10
    
    # X/Twitter Twikit settings
    X_USERNAME: Optional[str] = None
    X_PASSWORD: Optional[str] = None
    X_EMAIL: Optional[str] = None
    X_SESSION_CACHE_PATH: str = ".twikit_session"
    
    # ML and data processing settings
    STOCK_TICKERS: str = "AAPL,NVDA,TSLA,GOOGL,MSFT"
    SENTIMENT_MODEL: str = "ProsusAI/finbert"
    SENTIMENT_BATCH_SIZE: int = 16

    # Data quantity — 2 years gives ~500 rows, enough for regime-specific models
    HISTORICAL_PERIOD: str = "2y"

    # Macro / market-context tickers fetched alongside stocks
    MACRO_TICKERS: list = ["^VIX", "^TNX", "^IRX", "QQQ", "XLK", "XLY"]

    # Maps each stock ticker to the sector ETF used for relative-strength features
    SECTOR_ETF_MAP: dict = {
        "AAPL": "QQQ",
        "NVDA": "QQQ",
        "MSFT": "XLK",
        "GOOGL": "QQQ",
        "TSLA": "XLY",
    }

    # Earnings proximity window (days)
    EARNINGS_WINDOW_DAYS: int = 5

    # Horizons to forecast (days ahead); one ensemble per horizon × regime
    FORECAST_HORIZONS: list = [1, 3, 5, 10]

    # Ternary-classification deadband per horizon (|return| < deadband → "flat")
    TARGET_DEADBAND: dict = {1: 0.3, 3: 0.7, 5: 1.0, 10: 1.5}

    # Minimum training rows required before training a regime-specific model
    REGIME_MIN_ROWS: int = 80

    # Rate limiting
    REDDIT_DELAY: float = 2.0
    X_DELAY: float = 3.0
    YFINANCE_DELAY: float = 1.0

    # XGBoost classifier hyperparameters (used for multi-class direction prediction)
    XGBOOST_CLASSIFIER_PARAMS: dict = {
        "n_estimators":          150,
        "max_depth":             4,
        "learning_rate":         0.05,
        "random_state":          42,
        "n_jobs":                1,
        "reg_alpha":             0.1,
        "reg_lambda":            1.0,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "min_child_weight":      5,
        "early_stopping_rounds": 20,
    }
    
    # Legacy — kept so old references don't break; not used by the new classifier
    TRAIN_TEST_SPLIT_RATIO: float = 0.8
    FORECAST_DAYS: int = 5

    # Streamlit settings
    STREAMLIT_PAGE_TITLE: str = "Stock Forecaster - AI-Powered Sentiment Analysis"
    STREAMLIT_PAGE_ICON: str = "📈"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Load settings
settings = Settings()
