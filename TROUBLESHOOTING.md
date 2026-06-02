# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: "ModuleNotFoundError: No module named 'xxx'"

**Symptom**: Command fails with missing module error

**Solutions**:
1. Verify virtual environment is activated
   ```bash
   which python3  # Should show path in venv/
   ```

2. Reinstall dependencies
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. Check Python version (3.9+)
   ```bash
   python3 --version
   ```

4. Install specific package
   ```bash
   pip install transformers torch xgboost
   ```

---

#### Issue: "CUDA out of memory" or GPU not found

**Symptom**: Model loading fails with GPU errors

**Solutions**:
1. Force CPU-only mode (PyTorch auto-falls back):
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   python3 main.py
   ```

2. Reinstall PyTorch for CPU:
   ```bash
   pip uninstall torch -y
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. Reduce batch size in `.env`:
   ```env
   SENTIMENT_BATCH_SIZE=4
   ```

---

### Data Collection Issues

#### Issue: "Reddit 429 Too Many Requests"

**Symptom**: Reddit requests return 429 status code

**Solutions**:
1. Increase delay between requests in `.env`:
   ```env
   REDDIT_DELAY=5.0  # Increase from 2.0
   REDDIT_DELAY=10.0 # Even more conservative
   ```

2. Verify User-Agent header:
   ```env
   REDDIT_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
   ```

3. Add retry logic:
   ```python
   # Implemented in ingestion/pipeline.py
   # Automatically retries after 60 seconds if rate limited
   ```

4. Run during off-peak hours (late night, early morning)

---

#### Issue: "No data returned for ticker XXX"

**Symptom**: yfinance returns empty data

**Solutions**:
1. Verify ticker is valid
   ```bash
   python3 -c "import yfinance as yf; print(yf.Ticker('AAPL').info)"
   ```

2. Check market hours:
   - yfinance may not return data on weekends/holidays
   - Try running during market hours

3. Increase timeout in `.env`:
   ```env
   REDDIT_REQUEST_TIMEOUT=20  # Increase from 10
   ```

4. Try alternative tickers
   ```env
   STOCK_TICKERS="AAPL,MSFT"
   ```

---

#### Issue: "X/Twitter authentication failed"

**Symptom**: "Invalid credentials" or "Login failed"

**Solutions**:
1. Verify credentials in `.env`:
   ```env
   X_USERNAME=correct_username
   X_PASSWORD=correct_password
   X_EMAIL=registered_email
   ```

2. Use burner account (X/Twitter deprecated API access)
   - Create separate account for scraping
   - Use app password if available

3. Clear session cache and retry:
   ```bash
   rm -f .twikit_session
   python3 main.py
   ```

4. Check if account is rate limited
   - Wait 1-2 hours before retrying

---

### NLP / Sentiment Analysis Issues

#### Issue: "Model loading takes forever" or "Process hanging"

**Symptom**: Application seems stuck at "Loading sentiment analyzer"

**Solutions**:
1. Check network connection (model is being downloaded)
   ```bash
   ping huggingface.co
   ```

2. Pre-download model:
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
   model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
   ```

3. Use smaller model alternative:
   ```env
   SENTIMENT_MODEL="distilbert-base-uncased-finetuned-sst-2-english"
   ```

4. Check disk space (models downloaded to `~/.cache/huggingface/`)
   ```bash
   du -sh ~/.cache/huggingface/
   ```

---

#### Issue: "CUDA out of memory" during NLP

**Symptom**: RuntimeError about GPU memory

**Solutions**:
1. Reduce batch size in `.env`:
   ```env
   SENTIMENT_BATCH_SIZE=4    # Reduce from 16
   SENTIMENT_BATCH_SIZE=1    # Minimum (slowest)
   ```

2. Clear GPU cache:
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

3. Force CPU mode:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   python3 main.py
   ```

4. Restart Python kernel (for Jupyter):
   ```
   Kernel -> Restart & Clear Output
   ```

---

### ML / Training Issues

#### Issue: "Insufficient data" error

**Symptom**: Training fails with "Insufficient data: only XX rows"

**Solutions**:
1. Increase historical days:
   ```env
   HISTORICAL_DAYS=365  # Increase from current
   ```

2. Add more tickers:
   ```env
   STOCK_TICKERS="AAPL,MSFT,GOOGL,AMZN"
   ```

3. Combine data from multiple sources
   - Add financial news APIs
   - Include longer historical periods

4. Reduce minimum threshold (edit `ml_engine/predictor.py`):
   ```python
   if len(df_clean) < 50:  # Was 100
       logger.error(...)
   ```

---

#### Issue: "Training RMSE very high" (bad predictions)

**Symptom**: Test metrics show RMSE > 1.0 or predictions are inaccurate

**Solutions**:
1. **Data quality issues**
   - Check sentiment data quality
   - Verify technical indicators are calculated correctly
   - Look for outliers or gaps in data

2. **Feature engineering**
   - Add more indicators (`BB`, `ATR`, `Volume`)
   - Include lagged features (previous day's values)
   - Normalize features properly

3. **Model tuning** (edit `config.py`):
   ```python
   XGBOOST_PARAMS: dict = {
       "n_estimators": 200,        # Increase from 100
       "max_depth": 8,             # Increase from 6
       "learning_rate": 0.05,      # Decrease from 0.1
   }
   ```

4. **Training data issues**
   - Ensure chronological split (no future leakage)
   - Remove outliers
   - Use more training data

---

#### Issue: "Model predictions are always the same"

**Symptom**: All predictions show "UP" or "DOWN", no variation

**Solutions**:
1. Check if training data is too imbalanced
   ```python
   # In ml_engine/predictor.py
   print("Positive samples:", np.sum(y > 0))
   print("Negative samples:", np.sum(y < 0))
   ```

2. Use class weighting (for classification):
   ```python
   self.model = xgb.XGBClassifier(
       scale_pos_weight=class_weight,
       **settings.XGBOOST_PARAMS
   )
   ```

3. Reduce model complexity:
   ```python
   "max_depth": 3,  # Reduce from 6
   "learning_rate": 0.05,  # Reduce from 0.1
   ```

---

### Streamlit Dashboard Issues

#### Issue: "StreamlitAPIException: Data type not understood"

**Symptom**: Dashboard fails to load with TypeError

**Solutions**:
1. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```

2. Check data types in chart rendering:
   ```python
   # Ensure datetime columns
   df["date"] = pd.to_datetime(df["date"])
   ```

3. Restart Streamlit:
   ```bash
   streamlit run ui/app.py --logger.level=debug
   ```

---

#### Issue: "Streamlit port already in use"

**Symptom**: "Port 8501 already in use"

**Solutions**:
1. Use different port:
   ```bash
   streamlit run ui/app.py --server.port=8502
   ```

2. Kill existing process:
   ```bash
   lsof -i :8501  # Find process ID
   kill -9 <PID>
   ```

3. Wait and retry (or restart computer):
   ```bash
   sleep 30 && streamlit run ui/app.py
   ```

---

#### Issue: "Dashboard shows no data"

**Symptom**: UI loads but charts are empty

**Solutions**:
1. Verify pipeline has been run:
   ```bash
   ls -la results/
   ```

2. Check if results files exist and are valid:
   ```bash
   cat results/predictions.json
   ```

3. Manually load and inspect data:
   ```python
   import json
   import pandas as pd
   
   with open("results/predictions.json") as f:
       print(json.load(f))
   ```

---

### Performance Issues

#### Issue: "Pipeline is very slow"

**Symptom**: Execution takes > 5 minutes

**Solutions**:
1. **Reduce scope**:
   ```env
   STOCK_TICKERS="AAPL,MSFT"    # Fewer tickers
   HISTORICAL_DAYS=90           # Less history
   ```

2. **Increase parallelization**:
   - Ensure async operations work correctly
   - Verify no blocking calls in pipeline

3. **Monitor CPU/Memory**:
   ```bash
   top -o %CPU    # Monitor CPU
   top -o %MEM    # Monitor memory
   ```

4. **Profile code**:
   ```python
   import cProfile
   cProfile.run('pipeline.run_full_pipeline()')
   ```

---

#### Issue: "Out of memory" errors

**Symptom**: "MemoryError" or system swap kicks in

**Solutions**:
1. Reduce data scope:
   ```env
   HISTORICAL_DAYS=90
   SENTIMENT_BATCH_SIZE=4
   ```

2. Process data in chunks
   - Implement streaming for NLP
   - Handle tickers sequentially

3. Upgrade system RAM
   - Simple solution for compute-heavy operations

---

### Network Issues

#### Issue: "Connection timeout" errors

**Symptom**: "HTTPError: connection timeout" or "Failed to connect"

**Solutions**:
1. Check internet connection:
   ```bash
   ping 8.8.8.8
   ping reddit.com
   ping huggingface.co
   ```

2. Increase timeout in `.env`:
   ```env
   REDDIT_REQUEST_TIMEOUT=30  # Increase from 10
   ```

3. Check firewall/proxy:
   ```bash
   curl -I https://reddit.com
   ```

4. Try VPN if IP is blocked:
   - Some services block certain IP ranges

---

#### Issue: "SSL/TLS certificate error"

**Symptom**: "certificate verify failed" or similar

**Solutions**:
1. Update certificates:
   ```bash
   pip install --upgrade certifi
   ```

2. Bypass SSL (NOT RECOMMENDED FOR PRODUCTION):
   ```python
   import urllib3
   urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
   ```

3. Check system time (SSL depends on correct time):
   ```bash
   date
   ```

---

### Configuration Issues

#### Issue: ".env file not loaded"

**Symptom**: Settings show defaults, not configured values

**Solutions**:
1. Verify `.env` exists in correct location:
   ```bash
   ls -la .env
   ```

2. Ensure file is in project root:
   ```bash
   pwd  # Should show stock_forecaster directory
   ```

3. Check file permissions:
   ```bash
   chmod 644 .env
   ```

4. Reload configuration:
   ```bash
   # Edit and save .env, then restart application
   ```

---

#### Issue: "Invalid configuration value"

**Symptom**: "Validation error" when starting

**Solutions**:
1. Check `.env` syntax:
   ```env
   # Correct format
   KEY=value
   DELAY=2.0
   TICKERS="A,B,C"
   
   # Avoid quotes for numbers
   SENTIMENT_BATCH_SIZE=16  # ✓ Correct
   SENTIMENT_BATCH_SIZE="16"  # ✗ Wrong
   ```

2. Validate with:
   ```python
   from config import settings
   print(settings.dict())
   ```

---

### Logging and Debugging

#### Enable Debug Mode

```env
DEBUG=True
LOG_LEVEL=DEBUG
```

Then run:
```bash
python3 main.py 2>&1 | tee debug.log
```

#### View Multiple Log Levels

```bash
# Just errors
grep ERROR debug.log

# Just warnings
grep WARNING debug.log

# Specific component
grep "ingestion" debug.log

# Timeline
tail -f debug.log
```

---

## Getting Help

1. **Check logs first**
   ```bash
   tail -100 logs/pipeline.log
   python3 main.py 2>&1 | grep -i error
   ```

2. **Verify configuration**
   ```bash
   python3 -c "from config import settings; print(settings.dict())"
   ```

3. **Test individual components**
   ```python
   # Test ingestion
   from ingestion.pipeline import DataIngestionPipeline
   # ... test code
   
   # Test sentiment
   from nlp.sentiment import SentimentAnalyzer
   # ... test code
   ```

4. **Check system resources**
   ```bash
   df -h          # Disk space
   free -h        # Memory
   top -b -n1    # CPU usage
   ```

---

## Performance Baseline

For reference, typical execution times:

- **Ingestion**: 30-90 seconds
- **NLP Processing**: 10-60 seconds
- **ML Training**: 5-20 seconds
- **Total**: 1-3 minutes

If significantly slower, check:
- Network performance
- System resource availability
- Model download times (first run)

---

**Last Updated**: May 29, 2026

For issues not covered here, check application logs and `README.md` for more information.
