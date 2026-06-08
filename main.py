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
from ml_engine.walk_forward import WalkForwardBacktester
from performance.tracker import PerformanceTracker

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
        self.trade_logs: Dict = {}           # populated by run_execution_stage()
        self.walk_forward_results: Dict = {} # populated by run_ml_stage()
        self.perf_tracker = PerformanceTracker(
            ledger_path=settings.PERFORMANCE_LEDGER_PATH
        )
        
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

                # Write enriched DataFrame back so Stage 4 ExecutionEngine
                # can access ATR_14, Ichimoku, ADX, and BBW columns.
                self.aggregated_data["stocks"][ticker] = merged_data

                regime_dist = merged_data["regime"].value_counts().to_dict() if "regime" in merged_data.columns else {}
                logger.info(
                    f"Features ready for {ticker}: shape {merged_data.shape} | "
                    f"regimes: {regime_dist}"
                )

                horizons       = settings.FORECAST_HORIZONS
                ticker_preds   = {}
                ticker_metrics = {}
                ticker_wf      = {}

                wf_backtester = WalkForwardBacktester(
                    min_train   = settings.WF_MIN_TRAIN,
                    step        = settings.WF_STEP,
                    test_window = settings.WF_TEST_WINDOW,
                )

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

                    # Walk-forward backtest (primary reliability signal)
                    logger.info(f"Running walk-forward backtest for {ticker} h{horizon}d ...")
                    wf_result = wf_backtester.run(merged_data, feature_cols, horizon)
                    ticker_wf[horizon] = wf_result

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
                        "val":          metrics["val_metrics"],
                        "test":         metrics["test_metrics"],
                        "loocv":        loocv,
                        "walk_forward": wf_result,
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
                self.model_metrics[ticker]        = ticker_metrics
                self.walk_forward_results[ticker] = ticker_wf
                self.feature_importance[ticker]   = (
                    self.ml_predictor.get_feature_importance(horizon=horizons[0], top_n=10)
                )

                # Persist metrics to the performance ledger
                self.perf_tracker.record_batch(
                    ticker=ticker,
                    walk_forward_by_horizon=ticker_wf,
                    test_metrics_by_horizon={
                        str(h): ticker_metrics.get(str(h), {}).get("test", {})
                        for h in horizons
                    },
                    predictions_by_horizon=ticker_preds,
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
    
    async def run_execution_stage(self) -> bool:
        """
        Stage 4: Trade Simulation (Layer 4 — Risk Management & Execution).

        For each ticker takes the 1-day horizon prediction from the ML stage,
        resolves ATR from the stock OHLCV data, and calls ExecutionEngine to
        produce a fully-specified trade entry log (entry price, stop-loss,
        take-profit, position size).

        Results are stored in self.trade_logs and later written to
        results/trade_logs.json by save_results().

        Returns:
            True if the stage ran without a fatal error.
            False only if an unexpected exception prevents the stage from
            producing any output.
        """
        try:
            logger.info("=" * 60)
            logger.info("STAGE 4: TRADE SIMULATION (ATR-BASED EXECUTION)")
            logger.info("=" * 60)

            if not self.predictions:
                logger.warning("No predictions available — skipping execution stage.")
                return True

            from feature_engine.execution import ExecutionEngine
            engine = ExecutionEngine(
                account_size      = settings.ACCOUNT_SIZE,
                atr_multiplier_sl = settings.ATR_MULTIPLIER_SL,
                atr_multiplier_tp = settings.ATR_MULTIPLIER_TP,
                position_risk_pct = settings.POSITION_RISK_PCT,
                min_confidence    = settings.MIN_TRADE_CONFIDENCE,
            )

            stocks = self.aggregated_data.get("stocks", {})

            for ticker, ticker_data in self.predictions.items():
                horizon_preds = ticker_data.get("horizons", {})
                # Use the 1-day horizon prediction as the primary trade signal
                pred_1d = horizon_preds.get("1") or horizon_preds.get(1)
                if pred_1d is None:
                    logger.debug(f"No 1-day prediction for {ticker} — skipping.")
                    continue

                stock_df = stocks.get(ticker)
                if stock_df is None or stock_df.empty:
                    logger.warning(f"No stock data for {ticker} — skipping execution.")
                    continue

                prediction = {
                    "direction":  pred_1d["direction"],
                    "confidence": pred_1d["confidence"],
                    "ticker":     ticker,
                    "horizon":    1,
                }

                trade_log = engine.simulate_trade(prediction, stock_df)
                self.trade_logs[ticker] = trade_log

                action = trade_log["action"]
                if action != "skip":
                    logger.info(
                        f"TRADE {ticker}: {action.upper():<5} "
                        f"entry={trade_log['entry_price']:.2f}  "
                        f"SL={trade_log['stop_loss']:.2f}  "
                        f"TP={trade_log['take_profit']:.2f}  "
                        f"size={trade_log['position_size']} shares  "
                        f"R/R={trade_log['risk_reward_ratio']}  "
                        f"ML={trade_log.get('ml_confidence',0):.1%}  "
                        f"tech={trade_log.get('tech_conf_score',0):.1%}  "
                        f"blended={trade_log.get('blended_confidence',0):.1%}  "
                        f"signals={trade_log.get('confirming_signals','?')}"
                    )
                else:
                    logger.info(
                        f"TRADE {ticker}: SKIP — {trade_log.get('reason', 'n/a')}  "
                        f"(ML={trade_log.get('ml_confidence', trade_log.get('confidence', 0)):.1%}  "
                        f"tech={trade_log.get('tech_conf_score', 0):.1%}  "
                        f"blended={trade_log.get('blended_confidence', 0):.1%}  "
                        f"signals={trade_log.get('confirming_signals', '?')})"
                    )

            logger.info(
                f"Execution stage complete. "
                f"{sum(1 for t in self.trade_logs.values() if t['action'] != 'skip')} "
                f"active trade(s) / "
                f"{len(self.trade_logs)} ticker(s) processed."
            )
            return True

        except Exception as exc:
            logger.error(f"Error in execution stage: {exc}")
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
            
            # Save trade execution logs
            if self.trade_logs:
                with open(output_path / "trade_logs.json", "w") as f:
                    json.dump(self.trade_logs, f, indent=2, default=str)
                logger.info(f"Saved trade logs to {output_path / 'trade_logs.json'}")

            # Save walk-forward backtest results
            if self.walk_forward_results:
                with open(output_path / "walk_forward.json", "w") as f:
                    json.dump(self.walk_forward_results, f, indent=2, default=str)
                logger.info(f"Saved walk-forward results to {output_path / 'walk_forward.json'}")

            # Generate and save aggregate performance report from the full ledger
            try:
                report = self.perf_tracker.generate_report()
                with open(output_path / "performance_report.json", "w") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Saved performance report to {output_path / 'performance_report.json'}")
                self.perf_tracker.print_report()
            except Exception as exc:
                logger.warning(f"Could not generate performance report: {exc}")

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

        # Stage 4: Execution simulation (non-fatal — pipeline continues on failure)
        if not await self.run_execution_stage():
            logger.warning("Execution stage failed — results saved without trade logs.")

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
