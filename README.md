# Stock Forecaster 📈

A production-ready, modular Python web application that aggregates free social media feeds and financial news, applies machine learning and natural language processing (NLP) to extract sentiment metrics, and forecasts short-term stock trends using AI.

## 🎯 Project Overview

Stock Forecaster is a comprehensive system that combines:

- **📊 Data Ingestion**: Asynchronous collection from yfinance, Reddit, and X (Twitter)
- **💭 Sentiment Analysis**: FinBERT-based financial sentiment extraction
- **🤖 ML Forecasting**: XGBoost-powered price prediction with technical indicators
- **📈 Interactive Dashboard**: Streamlit UI for visualization and analysis

## 🏗️ Architecture

```
/stock_forecaster
├── config.py                    # Pydantic Settings from .env
├── main.py                      # Async pipeline orchestrator
├── requirements.txt             # Production dependencies
│
├── /ingestion                   # Module A: Data Collection
│   ├── __init__.py
│   └── pipeline.py              # yfinance, Reddit, X scrapers
│
├── /nlp                         # Module B: Sentiment Engine
│   ├── __init__.py
│   └── sentiment.py             # FinBERT sentiment analysis
│
├── /ml_engine                   # Module C: ML Forecasting
│   ├── __init__.py
│   └── predictor.py             # Technical indicators + XGBoost
│
└── /ui                          # Module D: Dashboard
    └── app.py                   # Streamlit UI
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/rohinraina/Stock\ Forecaster/stock_forecaster
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings (optional - defaults work for basic usage)
# For X/Twitter data, you'll need to add credentials
```

### 3. Run the Full Pipeline

```bash
python main.py
```

The pipeline will:
1. Fetch stock data from yfinance
2. Scrape Reddit posts from r/wallstreetbets and r/stocks
3. Optionally fetch X/Twitter posts (if credentials provided)
4. Analyze sentiment using FinBERT
5. Calculate technical indicators
6. Train XGBoost models
7. Generate predictions
8. Save results to `./results/`

### 4. Launch the Dashboard

```bash
streamlit run ui/app.py
```

Access the dashboard at `http://localhost:8501`

## 📦 Module Specifications

### Module A: Data Ingestion (`/ingestion/pipeline.py`)

**Features:**
- ✅ Asynchronous data collection (no blocking I/O)
- ✅ yfinance integration with thread executor
- ✅ Reddit scraper with custom User-Agent to bypass rate limits
- ✅ X/Twitter integration via twikit with session caching
- ✅ Configurable rate limiting to protect endpoints
- ✅ Error handling and retry logic

**Usage:**
```python
async with DataIngestionPipeline() as pipeline:
    data = await pipeline.aggregate_all_data()
```

### Module B: NLP Sentiment Engine (`/nlp/sentiment.py`)

**Features:**
- ✅ FinBERT transformer model for financial sentiment
- ✅ Batch processing for fast inference
- ✅ `torch.no_grad()` optimization for speed
- ✅ Continuous sentiment scores (-1.0 to 1.0)
- ✅ Daily moving average sentiment index
- ✅ Multi-source post stream processing

**Usage:**
```python
analyzer = SentimentAnalyzer()
results, sentiment_index = analyzer.process_ingestion_stream(posts)
```

### Module C: ML Prediction Engine (`/ml_engine/predictor.py`)

**Features:**
- ✅ 32 features: technical indicators, macro series (VIX, yield spread, sector ETF momentum), earnings proximity flags
- ✅ Regime detection (bull / bear / sideways) via dual SMA crossover
- ✅ Ternary classification target (up / flat / down) with horizon-scaled deadband
- ✅ Ensemble: XGBClassifier + LogisticRegression weighted by validation accuracy
- ✅ Per-regime models with full-data fallback
- ✅ 3-way chronological split (70% train / 15% val / 15% test) — val used only for early stopping
- ✅ Temporal LOOCV for honest cross-validated performance estimates
- ✅ Feature importance analysis (XGBoost gain scores)

**Usage:**
```python
predictor = MLPredictor()
stock_data = predictor.calculate_technical_indicators(stock_data, ticker)
stock_data = predictor.merge_macro_features(stock_data, macro_data, ticker)
stock_data = predictor.add_earnings_features(stock_data, earnings_dates, ticker)
stock_data = predictor.detect_regime(stock_data)
loocv = predictor.loocv_validate(stock_data, forecast_horizon=5)
X_train, X_val, X_test, y_train, y_val, y_test, regimes, cols, dates = predictor.prepare_training_data(stock_data, forecast_horizon=5)
predictor.train_model(X_train, X_val, X_test, y_train, y_val, y_test, regimes, horizon=5)
pred = predictor.predict(X_test[-1:], regime=predictor.get_current_regime(stock_data), horizon=5)
# pred = {"direction": "up", "confidence": 0.64, "probabilities": {...}, "regime": "bull"}
```

### Module D: Streamlit UI (`/ui/app.py`)

**Features:**
- ✅ Interactive ticker selection
- ✅ Dual-axis price vs sentiment visualization
- ✅ Sentiment distribution analysis
- ✅ Top positive/negative posts display
- ✅ ML metrics and feature importance charts
- ✅ Real-time predictions with confidence scores

**Launch:**
```bash
streamlit run ui/app.py
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Application
DEBUG=False
STOCK_TICKERS="AAPL,NVDA,TSLA,GOOGL,MSFT"

# Data Sources
REDDIT_USER_AGENT="Mozilla/5.0..."
X_USERNAME=your_username
X_PASSWORD=your_password
X_EMAIL=your_email

# NLP
SENTIMENT_MODEL="ProsusAI/finbert"
SENTIMENT_BATCH_SIZE=16

# ML
TRAIN_TEST_SPLIT_RATIO=0.8

# Rate Limiting (seconds)
REDDIT_DELAY=2.0
X_DELAY=3.0
YFINANCE_DELAY=1.0
```

## 🔄 Operational Features

### Asynchronous Pipeline

All network operations use async/await to prevent blocking:

```python
async def run_ingestion_stage(self):
    async with self.ingestion_pipeline as pipeline:
        data = await pipeline.aggregate_all_data()
```

### Rate Limiting

Configurable delays protect third-party endpoints:

```python
await asyncio.sleep(settings.REDDIT_DELAY)  # Between requests
await asyncio.sleep(2)  # Between pipeline stages
```

### Chronological Data Splitting

Prevents future-data leakage in ML models:

```python
# Train on first 80% of dates, test on last 20%
split_idx = int(len(X) * split_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
```

### Session Caching for X/Twitter

Avoids repeated authentication:

```python
def _load_twikit_session(self) -> bool:
    # Load cached session to avoid re-authentication
    ...

def _save_twikit_session(self):
    # Save session for future runs
    ...
```

## 📊 Output Files

After running `python main.py`, results are saved to `./results/`:

- **predictions.json** - Model predictions with confidence scores
- **feature_importance.json** - Top important features per ticker
- **sentiment_index.csv** - Daily sentiment aggregation
- **sentiment_summary.json** - Sentiment statistics

## 🛠️ Technology Stack

### Core Libraries
- **asyncio** - Asynchronous network I/O
- **httpx** - Async HTTP client for Reddit/web scraping
- **yfinance** - Financial data collection
- **twikit** - X/Twitter API
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### ML & NLP
- **transformers** - Hugging Face FinBERT model
- **torch** - PyTorch deep learning
- **xgboost** - Gradient boosting for prediction
- **scikit-learn** - ML utilities and scaling
- **pandas-ta** - Technical analysis indicators

### UI & Visualization
- **streamlit** - Interactive dashboard
- **plotly** - Interactive charts
- **altair** - Statistical visualization

### Infrastructure
- **pydantic-settings** - Configuration management
- **python-dotenv** - Environment variable loading

## 📈 Data Flow

```
┌─────────────────────────────────────────────────────┐
│ 1. Data Ingestion (Async)                          │
│    - yfinance (stock data)                         │
│    - Reddit (wallstreetbets, stocks)               │
│    - X/Twitter (stock mentions)                    │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 2. NLP Sentiment Analysis                          │
│    - FinBERT model inference                       │
│    - Batch processing                              │
│    - Daily aggregation                             │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 3. ML Feature Engineering                          │
│    - Technical indicators (RSI, MACD, BB)          │
│    - Sentiment feature merging                     │
│    - Chronological data splitting                  │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 4. Model Training & Prediction                     │
│    - XGBoost regression                            │
│    - Performance evaluation                        │
│    - Next-day price forecasting                    │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 5. Streamlit Dashboard                             │
│    - Price vs Sentiment visualization              │
│    - Prediction cards                              │
│    - Top posts matrix                              │
└─────────────────────────────────────────────────────┘
```

## 🔐 Security Considerations

- **Environment Variables**: All sensitive credentials in `.env` (not committed)
- **User-Agent Headers**: Custom User-Agents bypass Reddit 429 rate limits
- **Rate Limiting**: Configurable delays prevent endpoint abuse
- **Session Caching**: X/Twitter credentials cached securely (local only)
- **No API Keys Hardcoded**: All secrets from environment

## ML Engine v2 — Improvements & Results

This section documents the upgraded ML engine (`ml_engine/predictor.py`) and the reasoning behind each change.

### What Changed

#### Data Quantity
Historical data window extended from 1 year to **2 years** (~500 daily rows, up from ~250). Early XGBoost stopping at tree 0–3 was the model signalling insufficient signal — more rows per regime is the primary fix.

#### New Features (32 total, up from 23)

| Feature | Source | Purpose |
|---|---|---|
| `VIX_close` | `^VIX` via yfinance | Market fear / volatility regime context |
| `yield_spread` | `^TNX − ^IRX` (10Y − 3M) | Yield curve shape; negative spread precedes recessions |
| `sector_mom_5d / 20d` | QQQ / XLK / XLY per ticker | Sector momentum (AAPL/NVDA/GOOGL → QQQ, MSFT → XLK, TSLA → XLY) |
| `rel_strength_5d` | Stock return − sector return | Stock's edge over its sector over 5 days |
| `days_to_earnings` | yfinance earnings calendar | Proximity to next earnings event (capped at 60 days) |
| `earnings_imminent` | Same | Binary flag: earnings within 5 days |
| `price_vs_SMA200_pct` | SMA_200 | Deviation from long-term trend |

#### Regime Detection
Each row is labelled **bull / bear / sideways** using a dual SMA crossover:

- **Bull**: `close > SMA_50 > SMA_200`
- **Bear**: `close < SMA_50 < SMA_200`
- **Sideways**: anything else (including the first ~200 rows where SMA_200 is warming up)

A separate ensemble is trained for each regime that has ≥ `REGIME_MIN_ROWS` (default 80) training samples. Regimes with fewer rows fall back to the full-data model. Reduce `REGIME_MIN_ROWS` in `config.py` to activate bull/bear models sooner.

#### Target Transformation — Ternary Classification
The model no longer predicts raw percentage return. Instead the target is a **3-class label** with a horizon-scaled deadband:

| Horizon | Deadband | Label |
|---|---|---|
| 1d | ±0.3% | up / flat / down |
| 3d | ±0.7% | up / flat / down |
| 5d | ±1.0% | up / flat / down |
| 10d | ±1.5% | up / flat / down |

Moves within the deadband are labelled **flat** and treated as unpredictable noise, reducing training signal contamination from micro-fluctuations.

#### Ensemble: XGBClassifier + Logistic Regression
Each regime model is an ensemble of:

1. **XGBClassifier** (depth-4, L1/L2 regularised, early stopping on val set)
2. **LogisticRegression** (C=0.5, balanced class weights, lbfgs)

The two models are weighted by their respective validation-set accuracy at fit time:
```
P(class) = w_xgb · P_xgb(class) + w_lr · P_lr(class)
```
The linear model is often competitive on 1-day because the signal is weak and a simpler hypothesis generalises better; XGBoost dominates on longer horizons where non-linear interactions matter more.

#### Data Split — 3-Way Chronological
```
|── 70% train ──|── 15% val ──|── 15% test ──|
```
- **Train**: XGBoost early stopping and LogisticRegression fit
- **Val**: early-stopping eval set and ensemble weight calculation — never seen by the test evaluator
- **Test**: final held-out accuracy/F1 reported after all training decisions are locked in

#### Cross-Validation — Temporal LOOCV
Walk-forward leave-one-out CV runs over the training portion only:

```
for i in range(min_train=60, train_end):
    train on [0 : i]   ← only past data
    predict row i      ← one step ahead
```

Each fold trains a fresh XGB + LR ensemble with equal weights (0.5/0.5) — no nested val loop — and predicts the single left-out observation. This gives ~290 out-of-sample predictions per ticker per horizon with no look-ahead bias.

### LOOCV Accuracy Results (2-year dataset, May 2026 run)

| Ticker | 1d acc | 3d acc | 5d acc | 10d acc |
|---|---|---|---|---|
| AAPL | 34% | 48% | 59% | 60% |
| NVDA | 40% | 50% | 61% | 66% |
| TSLA | 42% | 54% | 59% | 65% |
| GOOGL | 38% | 52% | 60% | 69% |
| MSFT | 30% | 44% | 55% | 63% |

**Interpretation**:
- 1-day accuracy is near random (~33% baseline for ternary) — consistent with academic literature where ~55% binary is considered good for liquid large-caps.
- 5- and 10-day accuracy reaches **59–69%**, well above the ternary baseline of 33%. This is where the signal is most exploitable. The model's confidence on these longer horizons is also more calibrated.
- MSFT's lower 1-day accuracy reflects its high "flat" class proportion — many 1-day moves fall within the ±0.3% deadband.

### Output Format Change

Predictions now return direction + confidence + full probability vector:

```json
{
  "direction": "up",
  "confidence": 0.64,
  "probabilities": {"down": 0.14, "flat": 0.22, "up": 0.64},
  "regime": "bull"
}
```

### Relevant Config Keys

```env
HISTORICAL_PERIOD=2y             # yfinance period string
MACRO_TICKERS=["^VIX","^TNX","^IRX","QQQ","XLK","XLY"]
SECTOR_ETF_MAP={"AAPL":"QQQ","NVDA":"QQQ","MSFT":"XLK","GOOGL":"QQQ","TSLA":"XLY"}
EARNINGS_WINDOW_DAYS=5           # days threshold for earnings_imminent flag
TARGET_DEADBAND={1:0.3,3:0.7,5:1.0,10:1.5}
REGIME_MIN_ROWS=80               # lower to enable bull/bear regime models sooner
```

---

## 📉 Limitations & Notes

1. **yfinance**: Uses free tier, may have rate limits
2. **Reddit**: Requires standard User-Agent header, 429 handling included
3. **X/Twitter**: Requires burner account credentials (twikit library limitation)
4. **ML Model**: Trained only on recent data, add more historical data for better accuracy
5. **Sentiment**: FinBERT tuned for financial text, general sentiment may vary

## 🚀 Production Deployment

### Pre-Deployment Checklist

- [ ] Configure `.env` with production values
- [ ] Test with your target stock tickers
- [ ] Verify sentiment analysis quality
- [ ] Validate ML predictions on historical data
- [ ] Set up logging to persistent storage
- [ ] Configure rate limiting appropriately
- [ ] Test error handling (network failures, etc.)

### Deployment Options

1. **Docker**: Create Dockerfile to containerize
2. **Cloud**: Deploy on AWS Lambda, Azure Functions, or GCP Cloud Run
3. **VPS**: Run as background service on Linux VPS
4. **Scheduled**: Use cron/systemd for periodic execution

### Monitoring

Add monitoring for:
- Pipeline execution time
- Data ingestion success rate
- NLP inference latency
- ML prediction accuracy
- Dashboard uptime

## 📝 Example Usage

### Run Complete Pipeline

```bash
python main.py
```

### Run Dashboard Only

```bash
streamlit run ui/app.py
```

### Custom Script Integration

```python
import asyncio
from main import StockForecasterPipeline

async def custom_analysis():
    pipeline = StockForecasterPipeline()
    await pipeline.initialize()
    
    # Run individual stages
    await pipeline.run_ingestion_stage()
    await pipeline.run_nlp_stage()
    await pipeline.run_ml_stage()
    
    # Access results
    predictions = pipeline.predictions
    sentiment = pipeline.sentiment_index
    
    pipeline.cleanup()

asyncio.run(custom_analysis())
```

## 🐛 Troubleshooting

### ImportError: transformers not found
```bash
pip install transformers torch
```

### Reddit 429 Rate Limit Error
- Verify User-Agent in `.env`
- Increase `REDDIT_DELAY` in configuration
- Check network connectivity

### Model Training Fails
- Ensure sufficient historical data (100+ rows)
- Verify `date` column is present in data
- Check for NaN values in features

### Streamlit Port Already in Use
```bash
streamlit run ui/app.py --server.port 8502
```

## 📚 References

- [FinBERT Model](https://huggingface.co/ProsusAI/finbert)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Async Python Guide](https://docs.python.org/3/library/asyncio.html)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources (news APIs, earnings calendars)
- Enhanced sentiment models (custom fine-tuning)
- More ML models (LSTM, attention mechanisms)
- Portfolio optimization features
- Risk assessment modules

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs in application output
3. Verify `.env` configuration
4. Test individual modules independently

---

**Last Updated**: May 30, 2026  
**Status**: Production-Ready ✅
