# Architecture & Design

## System Overview

Stock Forecaster is a modular, production-grade system combining asynchronous data collection, NLP sentiment analysis, and ML-based forecasting for financial assets.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STOCK FORECASTER SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────┐
                  │                       │
          ┌───────▼────────┐      ┌──────▼──────────┐
          │  MAIN ORCHESTRATOR    │  STREAMLIT UI   │
          │  (main.py)            │  (ui/app.py)    │
          └───────┬────────┘      └─────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   ┌────────┐ ┌───────┐ ┌────────┐
   │ STAGE 1│ │STAGE 2│ │STAGE 3 │
   │        │ │       │ │        │
   │ Data   │ │  NLP  │ │   ML   │
   │ Ingest │ │Sent.  │ │Predict │
   └────────┘ └───────┘ └────────┘
```

## Module Architecture

### Module A: Data Ingestion (`ingestion/pipeline.py`)

**Purpose**: Asynchronously collect financial and social data from multiple sources

**Components**:
- `DataIngestionPipeline` class
- `fetch_yfinance_data()` - ThreadPoolExecutor-wrapped blocking calls
- `fetch_reddit_posts()` - Async HTTP requests to Reddit API
- `fetch_x_posts()` - Async X/Twitter integration with session caching
- `aggregate_all_data()` - Concurrent data gathering

**Data Flow**:
```
┌─────────────────────────────────────────────┐
│ Data Ingestion Pipeline                     │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ yfinance     │  │ Reddit API   │        │
│  │ (OHLCV data) │  │ (Posts)      │        │
│  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                │
│         ├─────────────────┤                │
│         │                 │                │
│         ▼                 ▼                │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ ThreadPool   │  │ AsyncClient  │        │
│  │ Executor     │  │ (httpx)      │        │
│  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                │
│         └────────┬────────┘                │
│                  │                         │
│                  ▼                         │
│         ┌──────────────────┐               │
│         │ Aggregated Data:  │               │
│         │ - stocks{}        │               │
│         │ - reddit[]        │               │
│         │ - x[]             │               │
│         └──────────────────┘               │
└─────────────────────────────────────────────┘
```

**Key Design Decisions**:
- ✅ ThreadPoolExecutor for yfinance (blocking library)
- ✅ httpx for async Reddit requests
- ✅ twikit for X/Twitter (handles auth + caching)
- ✅ Rate limiting with `asyncio.sleep()` delays
- ✅ Custom User-Agent headers to bypass Reddit 429 limits

**Rate Limiting Strategy**:
```python
# Between requests to same service
await asyncio.sleep(settings.REDDIT_DELAY)

# Between stages in main pipeline
await asyncio.sleep(2)

# Configurable via .env
REDDIT_DELAY = 2.0 seconds
X_DELAY = 3.0 seconds
YFINANCE_DELAY = 1.0 seconds
```

---

### Module B: NLP Sentiment (`nlp/sentiment.py`)

**Purpose**: Extract financial sentiment from social media text using FinBERT transformer model

**Components**:
- `SentimentAnalyzer` class
- `analyze_text()` - Single text sentiment analysis
- `analyze_batch()` - Optimized batch processing with `torch.no_grad()`
- `process_ingestion_stream()` - Process unified social media stream
- `calculate_daily_sentiment_index()` - Aggregate sentiment by date
- `_normalize_sentiment_score()` - Map outputs to (-1.0, 1.0) range

**Sentiment Score Mapping**:
```python
FinBERT Output → Normalized Score
─────────────────────────────────
Negative (0)   → -1.0 (or more nuanced)
Neutral (1)    → 0.0
Positive (2)   → 1.0

With confidence amplification:
Final = base_score × confidence
```

**Data Flow**:
```
┌──────────────────────────────────────────────┐
│ NLP Sentiment Engine                         │
├──────────────────────────────────────────────┤
│                                              │
│  ┌───────────────────────────────────────┐   │
│  │ Input: Social Media Posts              │   │
│  │ (Reddit & X posts)                     │   │
│  └─────────────┬─────────────────────────┘   │
│                │                              │
│  ┌─────────────▼─────────────┐                │
│  │ Text Extraction           │                │
│  │ - Combine title + text     │                │
│  │ - Filter empty strings     │                │
│  └─────────────┬─────────────┘                │
│                │                              │
│  ┌─────────────▼──────────────────┐           │
│  │ Batch Processing (16 at once)   │           │
│  │ with torch.no_grad() {          │           │
│  │   - Tokenization (512 max)      │           │
│  │   - Model inference              │           │
│  │   - Softmax probabilities         │           │
│  │ }                                │           │
│  └─────────────┬──────────────────┘           │
│                │                              │
│  ┌─────────────▼──────────────────┐           │
│  │ Score Normalization              │           │
│  │ Convert to (-1.0 to 1.0) range   │           │
│  └─────────────┬──────────────────┘           │
│                │                              │
│  ┌─────────────▼──────────────────┐           │
│  │ Daily Aggregation                │           │
│  │ Group by: ticker, date           │           │
│  │ Calculate: mean, std, count      │           │
│  └─────────────┬──────────────────┘           │
│                │                              │
│  ┌─────────────▼──────────────────┐           │
│  │ Output: DataFrame                │           │
│  │ Columns: date, ticker,           │           │
│  │          sentiment_score,        │           │
│  │          sentiment_std,          │           │
│  │          post_count               │           │
│  └────────────────────────────────┘           │
└──────────────────────────────────────────────┘
```

**Performance Optimizations**:
1. **Batch Processing**: Process 16 texts per batch
2. **torch.no_grad()**: Disable gradient computation for inference
3. **Model Caching**: Load model once, reuse for all texts
4. **GPU Support**: Auto-detect and use CUDA if available

**Memory Profile**:
- FinBERT model: ~500MB
- Per-batch processing: ~50-100MB
- Total typical usage: 1-2GB

---

### Module C: ML Prediction (`ml_engine/predictor.py`)

**Purpose**: Feature engineering and XGBoost-based stock price forecasting

**Components**:
- `MLPredictor` class
- `calculate_technical_indicators()` - RSI, MACD, Bollinger Bands
- `merge_feature_data()` - Align technical + sentiment features
- `prepare_training_data()` - Chronological split to prevent leakage
- `train_model()` - XGBoost training with validation
- `predict()` - Make predictions on new data
- `get_feature_importance()` - Identify key predictive features

**Technical Indicators**:
```python
RSI (14)              # Relative Strength Index (0-100)
MACD (12,26,9)        # Moving Average Convergence Divergence
Bollinger Bands (20)  # Upper, Middle, Lower bands + position
SMA (20, 50)          # Simple Moving Averages
ROC (14)              # Rate of Change
Volume indicators     # If available
```

**Chronological Data Split** (No Future Leakage):
```
Timeline: ─────────────────────────────────────────
          │     Past Data      │   Recent Data    │
          │ (80% - Training)   │ (20% - Testing)  │
          │                    │                  │
            ◄─ Temporal Flow ──►
            
Critical: No information from test period
         leaks into training
```

**Feature Merging Strategy**:
```python
Stock Features (Technical Indicators):
- OHLCV candles
- RSI, MACD, Bollinger Bands
- Moving averages
- Volume ratios

+ Sentiment Features:
- Daily sentiment score (-1 to 1)
- Sentiment standard deviation
- Number of posts

= Combined Feature Matrix
  for XGBoost training
```

**Data Flow**:
```
┌────────────────────────────────────────────┐
│ ML Prediction Pipeline                     │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │ Input: Stock OHLCV Data              │  │
│  └────────────┬─────────────────────────┘  │
│               │                            │
│  ┌────────────▼──────────────────────────┐ │
│  │ Technical Indicators Calculation      │ │
│  │ ├─ RSI(14)                           │ │
│  │ ├─ MACD(12,26,9)                     │ │
│  │ ├─ Bollinger Bands(20)               │ │
│  │ ├─ SMA(20,50)                        │ │
│  │ └─ Volume indicators                 │ │
│  └────────────┬──────────────────────────┘ │
│               │                            │
│               + Sentiment Index (from NLP) │
│               │                            │
│  ┌────────────▼──────────────────────────┐ │
│  │ Feature Merging (by date)            │ │
│  │ - Align stock & sentiment dates      │ │
│  │ - Forward-fill missing values        │ │
│  │ - Normalize features                 │ │
│  └────────────┬──────────────────────────┘ │
│               │                            │
│  ┌────────────▼──────────────────────────┐ │
│  │ Chronological Data Split             │ │
│  │ - 80% for training                   │ │
│  │ - 20% for testing                    │ │
│  │ - No future data leakage              │ │
│  └────────────┬──────────────────────────┘ │
│               │                            │
│  ┌────────────▼──────────────────────────┐ │
│  │ XGBoost Model Training               │ │
│  │ - 100 trees, depth 6                 │ │
│  │ - Learning rate 0.1                  │ │
│  │ - Output: next-day price change       │ │
│  └────────────┬──────────────────────────┘ │
│               │                            │
│  ┌────────────▼──────────────────────────┐ │
│  │ Prediction Pipeline                  │ │
│  │ - Predict on latest features         │ │
│  │ - Calculate confidence               │ │
│  │ - Determine direction (up/down)       │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

**Model Configuration** (from `config.py`):
```python
XGBOOST_PARAMS = {
    "n_estimators": 100,      # Number of trees
    "max_depth": 6,           # Tree depth
    "learning_rate": 0.1,     # Shrinkage factor
    "random_state": 42,       # Reproducibility
    "n_jobs": -1,             # Use all cores
}

TRAIN_TEST_SPLIT_RATIO = 0.8  # 80/20 split
```

**Metrics Generated**:
```python
Training Metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Test Metrics: (same calculations)

Feature Importance: Top 10 features
```

---

### Module D: Streamlit Dashboard (`ui/app.py`)

**Purpose**: Interactive web UI for visualization and exploration

**Components**:
- `render_sidebar()` - Configuration controls
- `render_key_metrics()` - Summary cards
- `render_price_sentiment_chart()` - Dual-axis visualization
- `render_sentiment_distribution()` - Pie chart
- `render_top_posts()` - Sortable post matrix
- `render_feature_importance()` - Feature contribution chart

**Dashboard Sections**:
```
┌─────────────────────────────────────────────┐
│ Stock Forecaster - Main UI                  │
├─────────────────────────────────────────────┤
│
│ Sidebar (Left):
│ ├─ Ticker Selector (dropdown)
│ ├─ Refresh Interval (slider)
│ ├─ Raw Posts Toggle
│ └─ Sentiment Threshold (slider)
│
│ Tabs:
│ ├─ Overview Tab
│ │  ├─ Key Metrics (4 cards)
│ │  │  ├─ Current Price
│ │  │  ├─ Forecast Direction
│ │  │  ├─ Expected Price Change
│ │  │  └─ Sentiment Score
│ │  └─ Technical Indicators
│ │
│ ├─ Sentiment Tab
│ │  ├─ Price vs Sentiment Chart (Plotly dual-axis)
│ │  └─ Sentiment Distribution (pie chart)
│ │
│ ├─ Social Posts Tab
│ │  ├─ Top Positive Posts (expandable)
│ │  └─ Top Negative Posts (expandable)
│ │
│ └─ Model Metrics Tab
│    ├─ Training Performance
│    ├─ Test Performance
│    └─ Feature Importance (bar chart)
│
└─────────────────────────────────────────────┘
```

**Visualization Libraries**:
- **Plotly**: Interactive 3D/dual-axis charts
- **Streamlit**: UI framework and component layout
- **Altair**: Statistical visualizations

---

## Main Orchestrator (`main.py`)

**Purpose**: Coordinate execution of all three stages in sequence

**Pipeline Execution Flow**:

```python
┌─────────────────────────────────────────────────────┐
│ StockForecasterPipeline.run_full_pipeline()         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Initialize()                                    │
│     ├─ DataIngestionPipeline()                      │
│     ├─ SentimentAnalyzer()                          │
│     └─ MLPredictor()                                │
│                                                     │
│  2. STAGE 1: Data Ingestion (async)                 │
│     ├─ Fetch yfinance data                          │
│     ├─ Fetch Reddit posts                           │
│     ├─ Fetch X/Twitter posts                        │
│     └─ Aggregate all data                           │
│                                                     │
│  wait 2 seconds (cool-down)                         │
│                                                     │
│  3. STAGE 2: NLP Sentiment Analysis                 │
│     ├─ Extract text from all posts                  │
│     ├─ Batch sentiment analysis                     │
│     ├─ Group by daily averages                      │
│     └─ Generate sentiment index                     │
│                                                     │
│  wait 2 seconds (cool-down)                         │
│                                                     │
│  4. STAGE 3: ML Feature Engineering & Prediction    │
│     For each ticker:                                │
│     ├─ Calculate technical indicators               │
│     ├─ Merge with sentiment data                    │
│     ├─ Prepare training data (chrono split)         │
│     ├─ Train XGBoost model                          │
│     ├─ Get feature importance                       │
│     └─ Generate predictions                         │
│                                                     │
│  5. Save Results                                    │
│     ├─ predictions.json                             │
│     ├─ feature_importance.json                      │
│     ├─ sentiment_index.csv                          │
│     └─ sentiment_summary.json                       │
│                                                     │
│  6. Cleanup                                         │
│     └─ Shutdown ThreadPoolExecutor                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Error Handling Strategy**:
```python
if not await run_ingestion_stage():
    logger.error("Ingestion failed")
    return False  # Stop pipeline

if not await run_nlp_stage():
    logger.warning("NLP failed")
    # Continue with stock data only
    
if not await run_ml_stage():
    logger.error("ML stage failed")
    return False  # Stop pipeline
```

---

## Asynchronous Architecture

### Event Loop Management

```python
async def main():
    pipeline = StockForecasterPipeline()
    
    # All async operations run on single event loop
    await pipeline.run_full_pipeline()
    # Event loop manages:
    # - Concurrent HTTP requests
    # - Non-blocking I/O
    # - Delay timers (asyncio.sleep)
```

### Concurrency Pattern

```
AsyncClient (httpx)          ThreadPoolExecutor
─────────────────────────────────────────────────
Reddit requests    ┐         yfinance calls
X queries          ├─ Concurrent    │
HTTP downloads     │  operations    └─ Blocking funcs
Async sleeps       ┘                  wrapped
                                      in thread
```

### Resource Management

```python
async with self.ingestion_pipeline as pipeline:
    # Context manager ensures:
    # - Session created on entry
    # - Session closed on exit
    # - Exception safe cleanup
    data = await pipeline.aggregate_all_data()
```

---

## Data Structures

### Stock Data (DataFrame)
```python
DataFrame columns:
├─ Date (index or column)
├─ Open, High, Low, Close, Volume (OHLCV)
├─ Technical Indicators
│  ├─ RSI
│  ├─ MACD, MACD_signal
│  ├─ BB_upper, BB_middle, BB_lower, BB_position
│  ├─ SMA_20, SMA_50
│  └─ ROC
└─ ticker (for reference)
```

### Sentiment Results (List of Dicts)
```python
[
  {
    "text": "Text preview...",
    "sentiment": "positive",
    "score": 0.85,
    "confidence": 0.92,
    "probabilities": {
      "negative": 0.05,
      "neutral": 0.03,
      "positive": 0.92
    },
    "source": "reddit",
    "ticker": "AAPL",
    "timestamp": "2024-01-15"
  },
  ...
]
```

### Daily Sentiment Index (DataFrame)
```python
DataFrame columns:
├─ date (datetime)
├─ ticker
├─ sentiment_score (mean)
├─ sentiment_std (standard deviation)
├─ post_count (number of posts)
└─ ... (grouped by date)
```

### ML Training Data (NumPy Arrays)
```python
X_train  : (n_samples, n_features) - Scaled features
X_test   : (n_samples, n_features) - Scaled features

y_train  : (n_samples,) - Target values (price change)
y_test   : (n_samples,) - Target values

Feature columns:
├─ RSI, MACD, MACD_signal, MACD_hist
├─ BB_upper, BB_middle, BB_lower, BB_position
├─ SMA_20, SMA_50, ROC
├─ Volume_SMA, Volume_ratio
├─ sentiment_score, sentiment_std, post_count
└─ ... (scaled by StandardScaler)
```

---

## Configuration Hierarchy

```
Environment Variable Defaults (in config.py)
            ↓
    .env file values (override)
            ↓
Runtime Configuration (settings object)
            ↓
Used by all modules
```

Example:
```env
# .env
STOCK_TICKERS="AAPL,MSFT"
REDDIT_DELAY=3.0

# config.py
REDDIT_DELAY: float = 2.0  # Default

# At runtime
settings.REDDIT_DELAY  # = 3.0 (from .env)
settings.STOCK_TICKERS # = "AAPL,MSFT" (from .env)
```

---

## Security & Best Practices

### Credentials Management
```python
# ✅ DO: Use environment variables
X_USERNAME = settings.X_USERNAME

# ❌ DON'T: Hardcode credentials
X_USERNAME = "my_secret_account"
```

### Rate Limiting
```python
# ✅ DO: Add delays between requests
async for ticker in tickers:
    await asyncio.sleep(settings.REDDIT_DELAY)
    await fetch_reddit_posts(ticker)

# ❌ DON'T: Rapid-fire requests
for ticker in tickers:
    fetch_reddit_posts(ticker)
```

### Error Handling
```python
# ✅ DO: Handle errors gracefully
try:
    data = await fetch_data(ticker)
except Exception as e:
    logger.error(f"Failed: {e}")
    continue  # Process other tickers

# ❌ DON'T: Let exceptions crash pipeline
for ticker in ticker_list:
    data = await fetch_data(ticker)  # Unhandled!
```

### Data Validation
```python
# ✅ DO: Validate before processing
if data is not None and not data.empty:
    # Process data
    
# ❌ DON'T: Assume data is valid
process(data)  # Could crash if empty
```

---

## Performance Characteristics

### Time Complexity
- **Data Ingestion**: O(T × P) where T = tickers, P = posts
- **NLP Analysis**: O(N) where N = total posts
- **ML Training**: O(N × F × D) where F = features, D = tree depth
- **Prediction**: O(1) per ticker

### Space Complexity
- **Models in Memory**: ~1-2GB (FinBERT + XGBoost)
- **Intermediate Data**: ~100-500MB
- **Results Storage**: ~10-50MB per run

### Typical Execution Profile
```
Ingestion:     30-90s  (network I/O limited)
NLP:          10-60s  (GPU limited if available, CPU otherwise)
ML:            5-20s  (CPU/RAM limited)
Total:        45-170s (~1-3 minutes)
```

---

## Extensibility Points

### Add New Data Source
```python
# In ingestion/pipeline.py
async def fetch_stocktwits_posts(self, ticker):
    # Implement similar to fetch_reddit_posts
    ...
```

### Add New Technical Indicator
```python
# In ml_engine/predictor.py
def calculate_technical_indicators(self, df):
    # Add new indicator
    df["NEW_INDICATOR"] = ta.new_indicator(df["Close"])
    ...
```

### Add Custom Sentiment Model
```python
# In config.py
SENTIMENT_MODEL = "alternative-model-name"

# Or override in nlp/sentiment.py
self.model = AutoModelForSequenceClassification.from_pretrained(custom_model)
```

### Add New ML Model
```python
# In ml_engine/predictor.py
from lightgbm import LGBMRegressor

# Train alternative model
self.model = LGBMRegressor(...)
self.model.fit(X_train, y_train)
```

---

## Monitoring & Observability

### Key Metrics to Track
1. Pipeline execution time
2. Data ingestion success rate
3. NLP inference latency
4. Model training time
5. Prediction accuracy (RMSE, MAE)
6. Feature importance changes

### Logging Levels
```
DEBUG   - Detailed execution flow
INFO    - Stage completions, success messages
WARNING - Recoverable errors, rate limits
ERROR   - Failures that halt pipeline
```

---

**Last Updated**: May 29, 2026

This architecture is designed for production use with focus on:
- ✅ Reliability (error handling, retries)
- ✅ Performance (async, batching, caching)
- ✅ Maintainability (modular design, clear separation of concerns)
- ✅ Scalability (easy to add new features/data sources)
