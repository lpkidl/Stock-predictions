# Quick Start Guide

Get Stock Forecaster running in 5 minutes!

## ⚡ TL;DR

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python3 main.py

# 4. View dashboard (in another terminal)
streamlit run ui/app.py

# 5. Check results
cat results/predictions.json
```

That's it! 🚀

---

## Step-by-Step Setup

### Prerequisites
- Python 3.9+
- 4GB RAM
- Internet connection

### 1. Navigate to Project

```bash
cd /Users/rohinraina/Stock\ Forecaster/stock_forecaster
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate it
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 2-5 minutes depending on internet speed.

### 4. Configure (Optional)

```bash
# Copy example config
cp .env.example .env

# Edit if needed (defaults work fine for basic use)
# nano .env  # or use your editor
```

### 5. Run Pipeline

```bash
python3 main.py
```

Expected output:
```
INFO - Initializing Stock Forecaster Pipeline...
INFO - STAGE 1: DATA INGESTION
INFO - Successfully fetched 250 rows for AAPL
...
INFO - STAGE 2: NLP SENTIMENT ANALYSIS
INFO - Analyzed 150 posts
...
INFO - STAGE 3: ML FEATURE ENGINEERING & PREDICTION
INFO - Training metrics: {'mse': 0.0245, ...}
INFO - 🔮 PREDICTION for AAPL: Direction: UP, Expected price change: $2.35
```

### 6. View Dashboard

In another terminal:

```bash
# Activate same venv
source venv/bin/activate

# Navigate to project
cd /Users/rohinraina/Stock\ Forecaster/stock_forecaster

# Launch dashboard
streamlit run ui/app.py
```

Opens at: `http://localhost:8501`

### 7. Check Results

```bash
# See all results
ls -la results/

# View predictions
cat results/predictions.json

# View sentiment data
head results/sentiment_index.csv

# View model metrics
cat results/feature_importance.json
```

---

## Common Tasks

### Use Different Tickers

Edit `.env`:
```env
STOCK_TICKERS="GOOGL,AMZN,META"
```

Then run:
```bash
python3 main.py
```

### Use Less Data (Faster)

Edit `.env`:
```env
HISTORICAL_DAYS=30  # Instead of 365
STOCK_TICKERS="AAPL"  # Just one ticker
```

### Run Multiple Times

Results save to `results/` folder. Each run overwrites previous results. To keep history:

```bash
# Before running, backup results
mkdir results_backup_$(date +%s)
cp results/* results_backup_*/
```

### Enable X/Twitter Data

1. Create burner X/Twitter account
2. Edit `.env`:
```env
X_USERNAME=your_username
X_PASSWORD=your_password
X_EMAIL=your_email
```
3. Run pipeline

### Use Custom Stock List

Edit `.env`:
```env
STOCK_TICKERS="AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,NFLX,PYPL,AMD"
```

### Debug Issues

```bash
# See detailed logs
python3 main.py 2>&1 | tee debug.log

# Check specific errors
grep ERROR debug.log
grep WARNING debug.log
```

---

## Understanding the Output

After running `python3 main.py`, the `results/` folder contains:

### `predictions.json`
```json
{
  "AAPL": {
    "price_change": 2.35,
    "direction": "up",
    "confidence": 0.72,
    "timestamp": "2024-01-15T10:30:45"
  },
  ...
}
```

### `sentiment_index.csv`
```
date,ticker,sentiment_score,sentiment_std,post_count
2024-01-15,AAPL,0.65,0.12,45
2024-01-14,AAPL,0.52,0.18,38
```

### `feature_importance.json`
```json
{
  "AAPL": {
    "sentiment_score": 0.25,
    "RSI": 0.18,
    "MACD": 0.15,
    ...
  },
  ...
}
```

### `sentiment_summary.json`
```json
{
  "total_posts": 234,
  "positive": 156,
  "neutral": 45,
  "negative": 33,
  "average_score": 0.62
}
```

---

## Dashboard Features

Once you launch Streamlit (`streamlit run ui/app.py`):

1. **Sidebar** - Select ticker, adjust settings
2. **Overview Tab** - Key metrics and current forecast
3. **Sentiment Tab** - Charts showing sentiment trends
4. **Social Posts Tab** - Top positive/negative posts
5. **Model Metrics Tab** - ML performance and feature importance

---

## Troubleshooting

### "No data returned for ticker"
- Ticker might be wrong (try AAPL)
- Run during market hours (9:30 AM - 4:00 PM ET)

### "Reddit 429 error"
- Increase delay in `.env`:
```env
REDDIT_DELAY=5.0
```

### "Out of memory"
- Reduce tickers or use less historical data
- Run on machine with more RAM

### "Model loading takes forever"
- Normal on first run (downloads~500MB model)
- Subsequent runs are faster

### Port 8501 already in use
```bash
streamlit run ui/app.py --server.port=8502
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more issues.

---

## Next Steps

1. **Understand the System**
   - Read [README.md](README.md) for full overview
   - Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

2. **Customize Configuration**
   - Edit `.env` for your tickers
   - Adjust rate limits if needed
   - Add X/Twitter credentials (optional)

3. **Run on Schedule**
   - Set up cron job for daily execution
   - Use Docker for containerization
   - Deploy to cloud (AWS, Azure, GCP)

4. **Integrate with Other Systems**
   - Import results into your trading platform
   - Add to notifications/alerts
   - Build custom dashboards

5. **Improve Predictions**
   - Add more data sources
   - Tune ML model parameters
   - Collect historical predictions to validate

---

## File Structure Reference

```
stock_forecaster/
├── main.py                 # Run this: python3 main.py
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
├── .env.example          # Copy to .env
│
├── ingestion/            # Data collection (async)
│   └── pipeline.py
├── nlp/                  # Sentiment analysis
│   └── sentiment.py
├── ml_engine/            # ML forecasting
│   └── predictor.py
├── ui/                   # Dashboard
│   └── app.py           # Run this: streamlit run ui/app.py
│
└── results/              # Output (auto-created)
    ├── predictions.json
    ├── sentiment_index.csv
    └── ...
```

---

## Performance Tips

- **First run**: Takes 2-3 minutes (downloads models)
- **Subsequent runs**: ~1-2 minutes
- **Make it faster**:
  - Use fewer tickers
  - Reduce historical days
  - Reduce batch size (if RAM limited)

---

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Check logs: `python3 main.py 2>&1 | grep ERROR`
4. Read inline code comments in module files

---

## Common Makefiles Commands

```bash
# Install everything
make install-dev

# Run pipeline
make run

# Launch dashboard in another terminal
make run-ui

# View results
make view-results

# Clean cache
make clean
```

---

## Success Checklist

- ✅ Python 3.9+ installed
- ✅ Virtual environment created
- ✅ Dependencies installed
- ✅ `.env` configured (or using defaults)
- ✅ Pipeline runs without errors
- ✅ Dashboard launches
- ✅ Results folder populated
- ✅ Predictions visible

---

## What to Do Next

### Option A: Run Daily
```bash
# Set up cron job (macOS/Linux)
crontab -e

# Add this line:
0 9 * * * cd /Users/rohinraina/Stock\ Forecaster/stock_forecaster && \
  source venv/bin/activate && python3 main.py
```

### Option B: Use Docker
```bash
# Build image
docker build -t stock-forecaster .

# Run
docker run stock-forecaster
```

### Option C: Customize
- Edit `.env` for different tickers
- Modify ML parameters in `config.py`
- Add data sources in `ingestion/pipeline.py`
- Create custom visualizations in `ui/app.py`

---

## Quick Reference

**Run Everything:**
```bash
python3 main.py  # Ingestion + NLP + ML
```

**Just View Results:**
```bash
streamlit run ui/app.py
```

**Both (separate terminals):**
```bash
# Terminal 1
python3 main.py

# Terminal 2
streamlit run ui/app.py
```

**Just Ingestion:**
```python
from ingestion.pipeline import DataIngestionPipeline
import asyncio

async def main():
    async with DataIngestionPipeline() as pipeline:
        data = await pipeline.aggregate_all_data()

asyncio.run(main())
```

---

## Estimated Time

- Installation: 5-10 minutes
- First run: 2-3 minutes
- Dashboard: instant
- **Total**: 10-15 minutes to see results

---

**Ready to start?** Go to step 1 above! 🚀

For detailed information, see [README.md](README.md)
