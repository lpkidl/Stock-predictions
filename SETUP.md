# Stock Forecaster - Setup and Install Guide

This file contains instructions for setting up and running Stock Forecaster from scratch.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git (optional, for version control)
- At least 4GB RAM (for ML model loading)
- macOS, Linux, or Windows

## Installation Steps

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/rohinraina/Stock\ Forecaster/stock_forecaster

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 2: Upgrade pip and Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install production dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

### Step 3: Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env

# Edit .env file with your settings
# - For basic usage, defaults work fine
# - For X/Twitter data, add your burner account credentials
# - To add more tickers, modify STOCK_TICKERS

nano .env  # or use your preferred editor
```

### Step 4: Verify Installation

```bash
# Test imports
python3 -c "import asyncio, httpx, yfinance, torch, xgboost, streamlit; print('✓ All imports successful')"

# Run a quick test
python3 main.py --help  # or just: python3 main.py
```

## Usage

### Run Complete Pipeline

```bash
python3 main.py
```

This will:
1. Fetch stock data from yfinance
2. Scrape posts from Reddit
3. Scrape posts from X/Twitter (if configured)
4. Analyze sentiment with FinBERT
5. Calculate technical indicators
6. Train ML models
7. Generate predictions
8. Save results to `./results/`

### Launch Interactive Dashboard

```bash
streamlit run ui/app.py
```

The dashboard will open at `http://localhost:8501`

### View Results

After running the pipeline, check the `results/` folder:

```bash
ls -la results/
# predictions.json
# feature_importance.json
# sentiment_index.csv
# sentiment_summary.json
```

## Configuration Guide

### Stock Tickers

Edit `.env`:
```env
STOCK_TICKERS="AAPL,NVDA,TSLA,GOOGL,MSFT"
```

Add or remove tickers as needed (comma-separated).

### Data Sources

**Reddit** (Always Available):
- Default configuration works out of the box
- No credentials needed

**X/Twitter** (Optional):
- Requires burner account credentials
- Add to `.env`:
```env
X_USERNAME=your_username
X_PASSWORD=your_password
X_EMAIL=your_email
```

### Rate Limiting

If you get "Too Many Requests" errors, increase delays in `.env`:
```env
REDDIT_DELAY=5.0      # Increase from 2.0
X_DELAY=5.0           # Increase from 3.0
YFINANCE_DELAY=2.0    # Increase from 1.0
```

### Data Retention

Adjust how much historical data to fetch:
```env
HISTORICAL_DAYS=365   # 1 year
HISTORICAL_DAYS=90    # 3 months for faster execution
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xxx'"

**Solution:** Ensure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Issue: "Reddit 429 Too Many Requests"

**Solution:** Increase REDDIT_DELAY in .env:
```env
REDDIT_DELAY=5.0
```

Also verify User-Agent header is set correctly.

### Issue: "CUDA/GPU errors"

**Solution:** PyTorch will fall back to CPU automatically. If issues persist:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Memory error" during model loading

**Solution:** This is normal for large models like FinBERT. Reduce SENTIMENT_BATCH_SIZE in .env:
```env
SENTIMENT_BATCH_SIZE=8    # Reduce from 16
```

### Issue: "Streamlit port already in use"

**Solution:** Use a different port:
```bash
streamlit run ui/app.py --server.port 8502
```

## Project Structure

```
stock_forecaster/
├── config.py                 # Configuration management
├── main.py                   # Pipeline orchestrator
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── .env.example             # Example configuration
├── .env                     # Your configuration (not in git)
├── .gitignore               # Git ignore patterns
├── README.md                # Full documentation
├── SETUP.md                 # This file
│
├── ingestion/               # Data collection
│   ├── __init__.py
│   └── pipeline.py          # yfinance, Reddit, X scrapers
│
├── nlp/                     # Sentiment analysis
│   ├── __init__.py
│   └── sentiment.py         # FinBERT processor
│
├── ml_engine/               # ML forecasting
│   ├── __init__.py
│   └── predictor.py         # Technical indicators + XGBoost
│
├── ui/                      # Dashboard
│   └── app.py               # Streamlit UI
│
└── results/                 # Output (auto-created)
    ├── predictions.json
    ├── feature_importance.json
    ├── sentiment_index.csv
    └── sentiment_summary.json
```

## System Requirements

### Minimum (Testing)
- 4GB RAM
- 2GB disk space
- Python 3.9+
- Single core

### Recommended (Production)
- 8GB+ RAM
- 10GB+ disk space
- Python 3.10+
- Multi-core processor
- GPU (optional, for faster inference)

## Performance Notes

### Typical Execution Times

With 5 tickers and default configuration:

- **Ingestion**: ~30-60 seconds (depends on network)
- **NLP Processing**: ~10-30 seconds (depends on post count)
- **ML Training**: ~5-15 seconds
- **Total Pipeline**: ~1-2 minutes

### Optimization Tips

1. **Reduce Number of Tickers**
   ```env
   STOCK_TICKERS="AAPL,MSFT"
   ```

2. **Reduce Historical Days**
   ```env
   HISTORICAL_DAYS=90
   ```

3. **Reduce Sentiment Batch Size** (if OOM errors)
   ```env
   SENTIMENT_BATCH_SIZE=8
   ```

4. **Use GPU** (if available)
   - PyTorch will auto-detect and use GPU
   - Verify: `python3 -c "import torch; print(torch.cuda.is_available())"`

## Advanced Usage

### Running Individual Components

```python
# Just ingestion
from ingestion.pipeline import DataIngestionPipeline
async with DataIngestionPipeline() as pipeline:
    data = await pipeline.aggregate_all_data()

# Just sentiment analysis
from nlp.sentiment import SentimentAnalyzer
analyzer = SentimentAnalyzer()
results = analyzer.analyze_batch(texts)

# Just ML prediction
from ml_engine.predictor import MLPredictor
predictor = MLPredictor()
predictor.train_model(X_train, y_train)
predictions = predictor.predict(X_test)
```

### Custom Tickers and Timeframes

Edit `.env` to add custom tickers:
```env
STOCK_TICKERS="AAPL,MSFT,GOOGL,META,AMZN,NVDA,TSLA"
HISTORICAL_DAYS=180
```

### Continuous Monitoring

Set up a cron job to run pipeline periodically:

```bash
# Edit crontab
crontab -e

# Add daily execution at 9:00 AM
0 9 * * * cd /Users/rohinraina/Stock\ Forecaster/stock_forecaster && source venv/bin/activate && python3 main.py >> logs/pipeline.log 2>&1
```

## Support and Resources

- **Documentation**: See `README.md`
- **Configuration**: Check `config.py` for all available settings
- **Dependencies**: Review `requirements.txt` for version info
- **Issues**: Check logs in terminal output
- **Examples**: See code comments in each module

## Next Steps

1. ✅ Complete the installation
2. ✅ Run `python3 main.py` to test the pipeline
3. ✅ Launch `streamlit run ui/app.py` to view results
4. ✅ Explore the `results/` folder for predictions
5. ✅ Customize `.env` for your use case

## Quick Reference

```bash
# Activate virtual environment
source venv/bin/activate

# Run pipeline
python3 main.py

# Launch dashboard
streamlit run ui/app.py

# View results
cat results/predictions.json

# Deactivate when done
deactivate
```

---

**Installation Complete!** 🎉

You're now ready to use Stock Forecaster. Happy analyzing!

For more details, see `README.md`.
