# Deployment Guide

This document provides instructions for deploying Stock Forecaster to production environments.

## Pre-Deployment Checklist

- [ ] All tests pass (`make test`)
- [ ] Code is linted and formatted (`make lint`, `make format`)
- [ ] `.env` is configured with production values
- [ ] Minimum Python version 3.9 verified
- [ ] SSL/TLS certificates ready (if applicable)
- [ ] Database backups configured (for persistent results)
- [ ] Monitoring and alerting set up
- [ ] Rollback plan documented

## Deployment Options

### Option 1: Local VPS / Dedicated Server

**Best for**: Personal use, small teams, on-premises deployment

#### Setup

1. **Copy project to server**
   ```bash
   scp -r stock_forecaster/ user@server:/opt/
   ```

2. **Create service file** (`/etc/systemd/system/stock-forecaster.service`)
   ```ini
   [Unit]
   Description=Stock Forecaster Pipeline
   After=network.target
   
   [Service]
   Type=simple
   User=stock_user
   WorkingDirectory=/opt/stock_forecaster
   ExecStart=/opt/stock_forecaster/venv/bin/python3 main.py
   Restart=on-failure
   RestartSec=10
   StandardOutput=journal
   StandardError=journal
   
   [Install]
   WantedBy=multi-user.target
   ```

3. **Enable service**
   ```bash
   sudo systemctl enable stock-forecaster
   sudo systemctl start stock-forecaster
   ```

4. **Monitor**
   ```bash
   sudo journalctl -u stock-forecaster -f
   ```

#### Dashboard Service File

Create `/etc/systemd/system/stock-forecaster-ui.service`:
```ini
[Unit]
Description=Stock Forecaster Streamlit UI
After=network.target

[Service]
Type=simple
User=stock_user
WorkingDirectory=/opt/stock_forecaster
ExecStart=/opt/stock_forecaster/venv/bin/streamlit run ui/app.py --server.port=8501
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Option 2: Docker Container

**Best for**: Cloud deployment, orchestration, consistency across environments

#### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY stock_forecaster/ /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create results directory
RUN mkdir -p results logs

# Expose Streamlit port (optional, comment out if only running pipeline)
EXPOSE 8501

# Run pipeline
CMD ["python3", "main.py"]
```

#### Build and Run

```bash
# Build image
docker build -t stock-forecaster:latest .

# Run container (pipeline only)
docker run -v /path/to/results:/app/results stock-forecaster:latest

# Run with Streamlit UI
docker run -p 8501:8501 -v /path/to/results:/app/results stock-forecaster:latest \
  streamlit run ui/app.py --server.address=0.0.0.0
```

#### Docker Compose for Full Stack

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  pipeline:
    build: .
    image: stock-forecaster:latest
    container_name: stock-forecaster-pipeline
    volumes:
      - ./results:/app/results
      - ./logs:/app/logs
    environment:
      - DEBUG=False
      - STOCK_TICKERS=AAPL,NVDA,TSLA
      - REDDIT_DELAY=2.0
    restart: on-failure
    restart_policy:
      max_retries: 3

  ui:
    build: .
    image: stock-forecaster:latest
    container_name: stock-forecaster-ui
    ports:
      - "8501:8501"
    volumes:
      - ./results:/app/results
    command: streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501
    depends_on:
      - pipeline
    restart: always
```

Run with:
```bash
docker-compose up -d
```

### Option 3: Cloud Functions / Lambda

**Best for**: Scheduled execution, low-cost, serverless

#### AWS Lambda Example

1. **Create deployment package**
   ```bash
   mkdir lambda_deployment
   pip install -r requirements.txt -t lambda_deployment/
   cp -r stock_forecaster/* lambda_deployment/
   cd lambda_deployment && zip -r ../lambda.zip . && cd ..
   ```

2. **Create Lambda handler** (`lambda_handler.py`)
   ```python
   import asyncio
   import json
   from main import StockForecasterPipeline
   
   async def run():
       pipeline = StockForecasterPipeline()
       await pipeline.run_full_pipeline()
       return {"statusCode": 200, "body": json.dumps({"status": "success"})}
   
   def lambda_handler(event, context):
       try:
           result = asyncio.run(run())
           return result
       except Exception as e:
           return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
   ```

3. **Deploy to Lambda**
   ```bash
   aws lambda create-function \
     --function-name stock-forecaster \
     --runtime python3.11 \
     --role arn:aws:iam::ACCOUNT_ID:role/lambda-role \
     --handler lambda_handler.lambda_handler \
     --zip-file fileb://lambda.zip \
     --timeout 300 \
     --memory-size 2048
   ```

4. **Set up CloudWatch scheduled trigger**
   ```bash
   aws events put-rule \
     --name stock-forecaster-daily \
     --schedule-expression "cron(0 9 * * ? *)"
   
   aws events put-targets \
     --rule stock-forecaster-daily \
     --targets "Id"="1","Arn"="arn:aws:lambda:REGION:ACCOUNT_ID:function:stock-forecaster"
   ```

### Option 4: Kubernetes

**Best for**: Enterprise, high availability, multi-tenant

#### Create Kubernetes Manifests

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-forecaster-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: stock-forecaster
  template:
    metadata:
      labels:
        app: stock-forecaster
    spec:
      containers:
      - name: pipeline
        image: stock-forecaster:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: DEBUG
          value: "False"
        - name: STOCK_TICKERS
          value: "AAPL,NVDA,TSLA"
        volumeMounts:
        - name: results
          mountPath: /app/results
      volumes:
      - name: results
        emptyDir: {}
```

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: stock-forecaster-ui
spec:
  type: LoadBalancer
  selector:
    app: stock-forecaster
  ports:
  - protocol: TCP
    port: 8501
    targetPort: 8501
```

Deploy with:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get service stock-forecaster-ui
```

## Production Configuration

### Environment File Template

```env
# Production settings
DEBUG=False
APP_NAME="Stock Forecaster"

# Data sources
STOCK_TICKERS="AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA"
REDDIT_USER_AGENT="Mozilla/5.0..."
REDDIT_REQUEST_TIMEOUT=15

# X/Twitter (if enabled)
X_USERNAME=<your_burner_account>
X_PASSWORD=<secure_password>
X_EMAIL=<burner_email>

# ML settings
SENTIMENT_BATCH_SIZE=16
TRAIN_TEST_SPLIT_RATIO=0.8

# Rate limiting
REDDIT_DELAY=3.0
X_DELAY=4.0
YFINANCE_DELAY=1.5

# Historical data
HISTORICAL_DAYS=365

# Logging
LOG_LEVEL=INFO
```

### Monitoring Setup

#### Prometheus + Grafana

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'stock-forecaster'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

#### Key Metrics to Monitor

1. **Pipeline Execution Time**
   - `pipeline_execution_duration_seconds`
   - Alert if > 5 minutes

2. **Data Quality**
   - `ingestion_success_rate`
   - `posts_collected_total`
   - Alert if success rate < 80%

3. **ML Model Performance**
   - `model_training_duration_seconds`
   - `test_rmse`
   - Alert if RMSE increases > 10%

4. **Error Rate**
   - `pipeline_errors_total`
   - Alert if error rate > 5%

### Security Considerations

1. **Environment Variables**
   - Store secrets in secure secret management (AWS Secrets Manager, HashiCorp Vault)
   - Rotate credentials regularly
   - Never commit `.env` to version control

2. **Network**
   - Use HTTPS/TLS for UI dashboard
   - Restrict API access to known IPs
   - Use VPN for remote access

3. **Data Protection**
   - Encrypt data at rest and in transit
   - Regular backups of results
   - GDPR compliance if storing user data

4. **Access Control**
   - Use IAM roles and policies
   - Implement RBAC for Streamlit dashboards
   - Audit all access logs

## Maintenance

### Regular Tasks

**Daily**:
- Monitor logs for errors
- Check data quality
- Verify predictions accuracy

**Weekly**:
- Review model performance metrics
- Check system resource usage
- Backup results

**Monthly**:
- Retrain models with new data
- Update dependencies
- Review and update rate limits

### Backup Strategy

```bash
# Backup results
tar -czf backup_results_$(date +%Y%m%d).tar.gz results/

# Backup to S3
aws s3 cp backup_results_*.tar.gz s3://my-backup-bucket/

# Retention: Keep last 90 days
find . -name "backup_results_*.tar.gz" -mtime +90 -delete
```

### Log Rotation

Create `/etc/logrotate.d/stock-forecaster`:
```
/opt/stock_forecaster/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 stock_user stock_user
}
```

## Scaling Considerations

### Horizontal Scaling

For running multiple ticker pools:

```yaml
# Run multiple instances with different tickers
Instance 1: AAPL,MSFT,GOOGL
Instance 2: AMZN,NVDA,TSLA
Instance 3: META,NFLX,PYPL
```

### Vertical Scaling

If single instance needs more power:

1. Increase memory allocation
2. Use GPU for faster NLP inference
3. Implement caching (Redis) for API responses
4. Use connection pooling for database

## Troubleshooting Deployment

### Service Won't Start

```bash
# Check service status
sudo systemctl status stock-forecaster

# View logs
sudo journalctl -u stock-forecaster -n 50

# Restart service
sudo systemctl restart stock-forecaster
```

### High Memory Usage

```bash
# Reduce batch size in .env
SENTIMENT_BATCH_SIZE=8

# Monitor memory
watch free -h
```

### Network Errors

1. Check Internet connectivity
2. Verify firewall rules
3. Check DNS resolution
4. Review rate limits

### API Rate Limits

```bash
# Add delays to rate limiting
REDDIT_DELAY=5.0
X_DELAY=6.0

# Implement caching to reduce calls
# Consider using API pools if available
```

## Rollback Procedure

1. **Stop current deployment**
   ```bash
   sudo systemctl stop stock-forecaster
   ```

2. **Restore previous version**
   ```bash
   git checkout previous-version
   ```

3. **Test locally**
   ```bash
   python3 main.py --dry-run
   ```

4. **Restart service**
   ```bash
   sudo systemctl start stock-forecaster
   ```

## Support

For deployment issues:
- Check application logs
- Verify configuration file
- Test with smaller dataset
- Contact support team

---

**Last Updated**: May 29, 2026
