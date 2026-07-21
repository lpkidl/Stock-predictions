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

    # Data quantity — 10 years (~2500 rows) so the model has enough history to
    # generalise instead of overfitting the ~350-row training slice of a 2y window.
    HISTORICAL_PERIOD: str = "10y"

    # Macro / market-context tickers fetched alongside stocks
    MACRO_TICKERS: list = ["^VIX", "^TNX", "^IRX", "QQQ", "XLK", "XLY"]

    # Company-name keywords for news relevance filtering. A fetched article is
    # kept for a ticker only if the symbol or one of these aliases appears in its
    # title/text — otherwise Yahoo's "related news" surfaces off-topic articles
    # (e.g. a Domino's piece tagged NVDA) whose sentiment pollutes the ticker.
    TICKER_ALIASES: dict = {
        "AAPL":  ["apple"],
        "NVDA":  ["nvidia"],
        "TSLA":  ["tesla"],
        "GOOGL": ["google", "alphabet"],
        "MSFT":  ["microsoft"],
    }

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

    # --- Model-quality overhaul knobs -------------------------------------
    # Per-regime sub-models split the (already small) training set into
    # bull/bear/sideways slices that mostly fell below REGIME_MIN_ROWS and fell
    # back anyway. Off by default: train ONE ensemble on all data and feed the
    # regime in as the numeric `regime_code` feature instead.
    USE_REGIME_MODELS: bool = False
    # Keep only the top-K features (by XGBoost gain on the training slice) to cut
    # overfitting from ~43 features on a few hundred rows. None = keep all.
    FEATURE_TOP_K: Optional[int] = 15
    # Wrap the base classifiers in probability calibration so `confidence` is
    # trustworthy — the execution engine gates trades on it.
    CALIBRATE_PROBABILITIES: bool = True
    # Balance class weights so the model doesn't collapse toward the majority
    # ("flat"/"up") class — the cause of the low F1 scores.
    BALANCE_CLASS_WEIGHTS: bool = True
    # Cap the number of expanding-window LOOCV folds. LOOCV retrains a model per
    # fold, so on the 10y window an unbounded loop is ~1.7k fits/horizon (tens of
    # minutes). Striding to this many evenly-spaced folds keeps the estimate
    # meaningful while bounding runtime; the 54-fold walk-forward is the primary
    # reliability signal regardless.
    LOOCV_MAX_FOLDS: int = 150

    # Alpha Vantage API (Layer 1 — dual data ingestion)
    ALPHA_VANTAGE_API_KEY: Optional[str] = None   # set in .env: ALPHA_VANTAGE_API_KEY=...

    # PRAW / Reddit OAuth credentials (Layer 1 — social stream via praw library)
    # Create a "script" app at https://www.reddit.com/prefs/apps, then set in .env:
    PRAW_CLIENT_ID:     Optional[str] = None
    PRAW_CLIENT_SECRET: Optional[str] = None
    # Must be descriptive per Reddit API rules: "AppName/Version by u/username"
    PRAW_USER_AGENT: str = "StockForecaster/1.0 by u/your_reddit_username"

    # ExecutionEngine risk parameters (Layer 4 — risk management)
    ATR_MULTIPLIER_SL: float = 2.0    # Stop-Loss = entry ± (2 × ATR)
    ATR_MULTIPLIER_TP: float = 3.0    # Take-Profit = entry ± (3 × ATR) → 1.5:1 R/R
    POSITION_RISK_PCT: float = 0.01   # Risk 1% of account per trade
    ACCOUNT_SIZE:      float = 100_000.0  # Simulated account equity ($)
    MIN_TRADE_CONFIDENCE: float = 0.50    # Applied to blended (ML + technical) confidence

    # Reddit .json scraping settings (no API key required)
    REDDIT_SUBREDDITS: list = ["wallstreetbets", "stocks"]
    REDDIT_FETCH_COMMENTS: bool = True    # append top comment bodies to post text
    REDDIT_COMMENT_LIMIT: int = 10        # max comments to fetch per post

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
    
    # Walk-forward backtesting
    WF_MIN_TRAIN: int = 250   # minimum bars before the first out-of-sample window
    WF_STEP: int = 42         # bars to advance the training cutoff each fold (~2 months);
                              # larger step keeps fold count sane on the 10y window
    WF_TEST_WINDOW: int = 21  # bars per out-of-sample evaluation window

    # Performance tracking
    PERFORMANCE_LEDGER_PATH: str = "results/performance_ledger.json"

    # Database (SQLAlchemy URL; point at Postgres later via .env, e.g.
    # DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname)
    DATABASE_URL: str = "sqlite:///results/stocks.db"
    DB_ENABLED: bool = True  # set False to disable all DB writes

    # Scheduler (cron expression used by scheduler.py)
    SCHEDULER_CRON: str = "0 16 * * 1-5"  # 4 pm ET, Mon–Fri (after US market close)

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
