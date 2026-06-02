"""
Main Application Entry Point.
Orchestrates the async pipeline: data ingestion -> NLP sentiment -> ML prediction.
Coordinates data flow through all modules and makes results accessible to UI.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict
import json
from pathlib import Path

# XGBoost + OpenMP crash on macOS when spawned inside a ThreadPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from config import settings
from ingestion.pipeline import DataIngestionPipeline
from nlp.sentiment import SentimentAnalyzer
from ml_engine.predictor import MLPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StockForecasterPipeline:
    """
    Main orchestrator for the stock forecasting pipeline.
    Manages data flow through ingestion, NLP, and ML modules.
    """
    
    def __init__(self):
        """Initialize the pipeline with all sub-components."""
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.ingestion_pipeline: DataIngestionPipeline = None
        self.sentiment_analyzer: SentimentAnalyzer = None
        self.ml_predictor: MLPredictor = None
        
        self.aggregated_data: Dict = {}
        self.sentiment_results: list = []
        self.sentiment_index: pd.DataFrame = None
        self.predictions: Dict = {}
        self.feature_importance: Dict = {}
        self.model_metrics: Dict = {}
        
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing Stock Forecaster Pipeline...")
        
        # Initialize data ingestion
        self.ingestion_pipeline = DataIngestionPipeline(
            executor=self.executor
        )
        
        # Initialize sentiment analyzer
        logger.info("Loading sentiment analyzer (this may take a moment)...")
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize ML predictor
        self.ml_predictor = MLPredictor()
        
        logger.info("Pipeline initialization complete")
    
    async def run_ingestion_stage(self) -> bool:
        """
        Stage 1: Data Ingestion
        Fetch historical stock data and social media posts.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STAGE 1: DATA INGESTION")
            logger.info("=" * 60)
            
            async with self.ingestion_pipeline as pipeline:
                self.aggregated_data = await pipeline.aggregate_all_data()
            
            logger.info(f"Ingestion complete. Aggregated data keys: {self.aggregated_data.keys()}")
            return True
            
        except Exception as e:
            logger.error(f"Error in ingestion stage: {str(e)}")
            return False
    
    async def run_nlp_stage(self) -> bool:
        """
        Stage 2: NLP Sentiment Analysis
        Process social media posts and generate sentiment index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STAGE 2: NLP SENTIMENT ANALYSIS")
            logger.info("=" * 60)
            
            if not self.aggregated_data:
                logger.warning("No aggregated data available for NLP stage")
                return False
            
            # Combine all social posts
            all_posts = (
                self.aggregated_data.get("reddit", []) +
                self.aggregated_data.get("x", [])
            )
            
            if not all_posts:
                logger.warning("No social media posts to analyze - skipping NLP, ML will use technical indicators only")
                self.sentiment_results = []
                self.sentiment_index = None
                return True
            
            # Process sentiment stream
            self.sentiment_results, self.sentiment_index = (
                self.sentiment_analyzer.process_ingestion_stream(all_posts)
            )
            
            logger.info(
                f"NLP stage complete. "
                f"Analyzed {len(self.sentiment_results)} posts. "
                f"Generated sentiment index for {len(self.sentiment_index)} dates"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error in NLP stage: {str(e)}")
            return False
    
    async def run_ml_stage(self) -> bool:
        """
        Stage 3: ML Feature Engineering & Prediction
        Calculate technical indicators, merge sentiment data, train model, predict.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info("STAGE 3: ML FEATURE ENGINEERING & PREDICTION")
            logger.info("=" * 60)
            
            if not self.aggregated_data.get("stocks"):
                logger.warning("No stock data available for ML stage")
                return False
            
            macro_data    = self.aggregated_data.get("macro", {})
            earnings_data = self.aggregated_data.get("earnings", {})
            tickers       = list(self.aggregated_data["stocks"].keys())

            for ticker in tickers:
                logger.info(f"\nProcessing ticker: {ticker}")

                stock_data = self.aggregated_data["stocks"][ticker]

                # Technical indicators (includes SMA_200 for regime detection)
                stock_data = self.ml_predictor.calculate_technical_indicators(stock_data, ticker)

                # Macro features: VIX, yield spread, sector ETF momentum
                stock_data = self.ml_predictor.merge_macro_features(stock_data, macro_data, ticker)

                # Earnings proximity flags
                stock_data = self.ml_predictor.add_earnings_features(
                    stock_data, earnings_data.get(ticker, []), ticker
                )

                # Regime labels for per-regime model training
                stock_data = self.ml_predictor.detect_regime(stock_data)

                # Merge sentiment if available
                if self.sentiment_index is not None:
                    merged_data = self.ml_predictor.merge_feature_data(
                        stock_data, self.sentiment_index, ticker
                    )
                else:
                    merged_data = stock_data

                if merged_data.empty:
                    logger.warning(f"Merged data is empty for {ticker}")
                    continue

                regime_dist = merged_data["regime"].value_counts().to_dict() if "regime" in merged_data.columns else {}
                logger.info(
                    f"Features ready for {ticker}: shape {merged_data.shape} | "
                    f"regimes: {regime_dist}"
                )

                horizons       = settings.FORECAST_HORIZONS
                ticker_preds   = {}
                ticker_metrics = {}

                for horizon in horizons:
                    prep_result = self.ml_predictor.prepare_training_data(
                        merged_data, forecast_horizon=horizon,
                    )
                    if prep_result is None:
                        logger.warning(f"Could not prepare training data for {ticker} h{horizon}d")
                        continue

                    (X_train, X_val, X_test,
                     y_train, y_val,  y_test,
                     reg_train, feature_cols, date_ranges) = prep_result

                    logger.info(
                        f"Horizon {horizon}d — Train {X_train.shape[0]}, "
                        f"Val {X_val.shape[0]}, Test {X_test.shape[0]} | "
                        f"{date_ranges['train_start'][:10]} → {date_ranges['test_end'][:10]}"
                    )

                    # LOOCV before fitting the final model
                    loocv = self.ml_predictor.loocv_validate(
                        merged_data, forecast_horizon=horizon
                    )

                    self.ml_predictor.train_model(
                        X_train, X_val, X_test,
                        y_train, y_val, y_test,
                        reg_train, horizon=horizon,
                    )

                    metrics = self.ml_predictor.get_metrics_report(horizon=horizon)
                    ticker_metrics[str(horizon)] = {
                        "val":   metrics["val_metrics"],
                        "test":  metrics["test_metrics"],
                        "loocv": loocv,
                    }

                    if len(X_test) > 0:
                        current_regime = self.ml_predictor.get_current_regime(merged_data)
                        pred = self.ml_predictor.predict(
                            X_test[-1:], regime=current_regime, horizon=horizon
                        )
                        if "direction" in pred:
                            ticker_preds[str(horizon)] = {
                                "direction":     pred["direction"],
                                "confidence":    pred["confidence"],
                                "probabilities": pred["probabilities"],
                                "regime":        current_regime,
                            }

                self.predictions[ticker] = {
                    "horizons":  ticker_preds,
                    "timestamp": datetime.now().isoformat(),
                }
                self.model_metrics[ticker] = ticker_metrics
                self.feature_importance[ticker] = (
                    self.ml_predictor.get_feature_importance(horizon=horizons[0], top_n=10)
                )

                logger.info(f"\n🔮 PREDICTIONS for {ticker}:")
                for h, p in ticker_preds.items():
                    probs = p.get("probabilities", {})
                    logger.info(
                        f"   {h:>2} day(s): {p['direction'].upper():<5}  "
                        f"conf {p['confidence']:.1%}  "
                        f"[down {probs.get('down', 0):.2f} "
                        f"flat {probs.get('flat', 0):.2f} "
                        f"up {probs.get('up', 0):.2f}]  "
                        f"regime={p.get('regime', '?')}"
                    )
            
            logger.info("\nML stage complete")
            return True
            
        except Exception as e:
            logger.error(f"Error in ML stage: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self, output_dir: str = "./results"):
        """
        Save pipeline results to JSON files for analysis and UI consumption.
        
        Args:
            output_dir: Directory to save results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            if self.predictions:
                with open(output_path / "predictions.json", "w") as f:
                    json.dump(self.predictions, f, indent=2)
                logger.info(f"Saved predictions to {output_path / 'predictions.json'}")
            
            # Save feature importance
            if self.feature_importance:
                with open(output_path / "feature_importance.json", "w") as f:
                    json.dump(self.feature_importance, f, indent=2)
                logger.info(f"Saved feature importance to {output_path / 'feature_importance.json'}")

            # Save model metrics
            if self.model_metrics:
                # Convert numpy types to native Python for JSON serialisation
                def _to_python(obj):
                    if hasattr(obj, "item"):
                        return obj.item()
                    return obj
                serialisable = {
                    ticker: {
                        split: {k: _to_python(v) for k, v in m.items()}
                        for split, m in ticker_metrics.items()
                    }
                    for ticker, ticker_metrics in self.model_metrics.items()
                }
                with open(output_path / "model_metrics.json", "w") as f:
                    json.dump(serialisable, f, indent=2)
                logger.info(f"Saved model metrics to {output_path / 'model_metrics.json'}")
            
            # Save sentiment index
            if self.sentiment_index is not None and not self.sentiment_index.empty:
                self.sentiment_index.to_csv(
                    output_path / "sentiment_index.csv",
                    index=False
                )
                logger.info(f"Saved sentiment index to {output_path / 'sentiment_index.csv'}")
            
            # Save sentiment results summary
            if self.sentiment_results:
                sentiment_summary = {
                    "total_posts": len(self.sentiment_results),
                    "positive": sum(1 for r in self.sentiment_results if r.get("sentiment") == "positive"),
                    "neutral": sum(1 for r in self.sentiment_results if r.get("sentiment") == "neutral"),
                    "negative": sum(1 for r in self.sentiment_results if r.get("sentiment") == "negative"),
                    "average_score": float(
                        sum(r.get("score", 0) for r in self.sentiment_results) / len(self.sentiment_results)
                    )
                }
                
                with open(output_path / "sentiment_summary.json", "w") as f:
                    json.dump(sentiment_summary, f, indent=2)
                logger.info(f"Saved sentiment summary to {output_path / 'sentiment_summary.json'}")
            
            logger.info(f"All results saved to {output_dir}/")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    async def run_full_pipeline(self):
        """
        Execute the complete pipeline: ingestion -> NLP -> ML.
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING STOCK FORECASTER PIPELINE")
        logger.info(f"Time: {datetime.now()}")
        logger.info("=" * 60 + "\n")
        
        # Initialize
        await self.initialize()
        
        # Stage 1: Ingestion
        if not await self.run_ingestion_stage():
            logger.error("Ingestion stage failed")
            return False
        
        # Cool-down between stages
        logger.info("Cool-down: waiting 2 seconds...")
        await asyncio.sleep(2)
        
        # Stage 2: NLP (optional - ML stage continues even without sentiment data)
        if not await self.run_nlp_stage():
            logger.warning("NLP stage failed - ML will run with technical indicators only")

        # Cool-down between stages
        logger.info("Cool-down: waiting 2 seconds...")
        await asyncio.sleep(2)

        # Stage 3: ML
        if not await self.run_ml_stage():
            logger.error("ML stage failed")
            return False
        
        # Save results
        self.save_results()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE EXECUTION COMPLETE")
        logger.info(f"Time: {datetime.now()}")
        logger.info("=" * 60)
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.executor.shutdown(wait=True)
        logger.info("Cleanup complete")


async def main():
    """Main async entry point."""
    pipeline = StockForecasterPipeline()
    
    try:
        success = await pipeline.run_full_pipeline()
        if not success:
            logger.error("Pipeline execution failed")
            return 1
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        pipeline.cleanup()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
