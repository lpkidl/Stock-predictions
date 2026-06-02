.PHONY: help install install-dev run run-ui lint format test clean logs docs

help:
	@echo "Stock Forecaster - Makefile Commands"
	@echo "===================================="
	@echo "make install      - Install production dependencies"
	@echo "make install-dev  - Install all dependencies (prod + dev)"
	@echo "make run          - Run the full pipeline"
	@echo "make run-ui       - Launch Streamlit dashboard"
	@echo "make lint         - Run code linting (flake8, pylint)"
	@echo "make format       - Format code (black, isort)"
	@echo "make test         - Run test suite"
	@echo "make clean        - Remove cache and build files"
	@echo "make logs         - View application logs"
	@echo "make docs         - Generate documentation"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Production dependencies installed"

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✓ All dependencies installed (prod + dev)"

run:
	python3 main.py

run-ui:
	streamlit run ui/app.py

lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@echo "Running pylint..."
	pylint *.py ingestion/*.py nlp/*.py ml_engine/*.py ui/*.py --disable=all --enable=E,F || true

format:
	@echo "Formatting with black..."
	black . --line-length=88
	@echo "Sorting imports with isort..."
	isort . --profile black

test:
	pytest tests/ -v --cov=. --cov-report=html

clean:
	find . -type d -name __pycache__ -exec rm -r {} + || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ dist/ build/ *.egg-info/
	@echo "✓ Cache and build files cleaned"

logs:
	tail -f logs/pipeline.log

docs:
	sphinx-build -b html docs/ docs/_build/
	@echo "✓ Documentation built in docs/_build/"

setup-venv:
	python3 -m venv venv
	source venv/bin/activate && make install
	@echo "✓ Virtual environment created and activated"

display-config:
	@echo "Current Configuration:"
	@python3 -c "from config import settings; print(f'Debug: {settings.DEBUG}'); print(f'Tickers: {settings.STOCK_TICKERS}'); print(f'REDdit delay: {settings.REDDIT_DELAY}s'); print(f'X delay: {settings.X_DELAY}s')"

view-results:
	@echo "=== Predictions ==="
	@cat results/predictions.json 2>/dev/null || echo "No predictions found"
	@echo ""
	@echo "=== Feature Importance ==="
	@cat results/feature_importance.json 2>/dev/null | head -20 || echo "No feature importance found"
	@echo ""
	@echo "=== Sentiment Summary ==="
	@cat results/sentiment_summary.json 2>/dev/null || echo "No sentiment summary found"

watch:
	@echo "Watching for results changes..."
	@while true; do clear; date; ls -lh results/ 2>/dev/null || echo "No results yet"; sleep 5; done

# Utility targets
.SILENT: help
