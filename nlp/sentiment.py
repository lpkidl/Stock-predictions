"""
Financial Sentiment Analysis Engine.
Uses FinBERT transformer model for financial text sentiment analysis
with performance optimization using torch.no_grad() and batch processing.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment in financial texts using FinBERT.
    Provides batch processing and optimization for fast inference.
    """
    
    def __init__(self, model_name: str = settings.SENTIMENT_MODEL):
        """
        Initialize the sentiment analyzer with FinBERT model.
        Model is loaded lazily on first use to avoid crashing when
        no posts need to be analyzed.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = None
        self.model = None

        # FinBERT label mapping (negative, neutral, positive)
        self.label_mapping = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    def _ensure_model_loaded(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is not None:
            return
        logger.info(f"Loading {self.model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()
        logger.info("Sentiment analyzer initialized successfully")
    
    def _normalize_sentiment_score(
        self,
        label_idx: int,
        confidence: float
    ) -> float:
        """
        Convert FinBERT output to continuous sentiment score (-1.0 to 1.0).
        
        Args:
            label_idx: Label index (0=negative, 1=neutral, 2=positive)
            confidence: Confidence score from model
            
        Returns:
            Normalized sentiment score (-1.0 to 1.0)
        """
        # Map label to base score
        label_scores = {
            0: -1.0,   # negative
            1: 0.0,    # neutral
            2: 1.0     # positive
        }
        
        base_score = label_scores.get(label_idx, 0.0)
        
        # Amplify the score based on confidence (but keep in [-1, 1] range)
        if base_score != 0.0:
            return base_score * confidence
        else:
            # For neutral, range from -0.5 to 0.5 based on confidence
            return (confidence - 0.5)
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text string.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "text": text,
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "error": "Invalid or empty text"
            }

        self._ensure_model_loaded()
        try:
            with torch.no_grad():  # Optimization: disable gradient computation
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Model inference
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                # Convert to sentiment score
                sentiment_score = self._normalize_sentiment_score(
                    int(predicted_class),
                    confidence
                )
                
                return {
                    "text": text[:100],  # Store first 100 chars
                    "sentiment": self.label_mapping[int(predicted_class)],
                    "score": round(sentiment_score, 4),
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "negative": round(float(probabilities[0]), 4),
                        "neutral": round(float(probabilities[1]), 4),
                        "positive": round(float(probabilities[2]), 4)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {
                "text": text[:100] if text else "",
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = settings.SENTIMENT_BATCH_SIZE
    ) -> List[Dict]:
        """
        Analyze sentiment of multiple texts efficiently using batching.
        Optimized with torch.no_grad() for faster inference.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of sentiment analysis results
        """
        self._ensure_model_loaded()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                with torch.no_grad():
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        max_length=512,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Model inference on batch
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Process batch results
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                    predicted_classes = np.argmax(probabilities, axis=1)
                    confidences = np.max(probabilities, axis=1)
                    
                    for text, pred_class, confidence, probs in zip(
                        batch_texts,
                        predicted_classes,
                        confidences,
                        probabilities
                    ):
                        sentiment_score = self._normalize_sentiment_score(
                            int(pred_class),
                            float(confidence)
                        )
                        
                        results.append({
                            "text": text[:100],
                            "sentiment": self.label_mapping[int(pred_class)],
                            "score": round(sentiment_score, 4),
                            "confidence": round(float(confidence), 4),
                            "probabilities": {
                                "negative": round(float(probs[0]), 4),
                                "neutral": round(float(probs[1]), 4),
                                "positive": round(float(probs[2]), 4)
                            }
                        })
                
            except Exception as e:
                logger.error(f"Error in batch analysis: {str(e)}")
                for text in batch_texts:
                    results.append({
                        "text": text[:100] if text else "",
                        "sentiment": "neutral",
                        "score": 0.0,
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        logger.info(f"Analyzed {len(results)} texts")
        return results
    
    def calculate_daily_sentiment_index(
        self,
        sentiment_results: List[Dict],
        ticker: str
    ) -> pd.DataFrame:
        """
        Calculate daily moving average sentiment index from individual predictions.
        Maps sentiment scores to stock dates for ML feature engineering.
        
        Args:
            sentiment_results: List of sentiment analysis dictionaries
            ticker: Stock ticker for this sentiment data
            
        Returns:
            DataFrame with date and daily average sentiment score
        """
        if not sentiment_results:
            logger.warning(f"No sentiment data for {ticker}")
            return pd.DataFrame(columns=["date", "ticker", "sentiment_score"])
        
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_results)
        
        # Add timestamp if not present (assume current date)
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now()
        
        # Convert to datetime if string
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Extract date (for daily aggregation)
        df["date"] = df["timestamp"].dt.date
        
        # Group by date and calculate average sentiment
        daily_sentiment = df.groupby("date").agg({
            "score": ["mean", "std", "count"],
        }).reset_index()
        
        daily_sentiment.columns = ["date", "sentiment_score", "sentiment_std", "post_count"]
        daily_sentiment["ticker"] = ticker
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
        
        logger.info(
            f"Generated daily sentiment index for {ticker}: "
            f"{len(daily_sentiment)} days, "
            f"avg sentiment: {daily_sentiment['sentiment_score'].mean():.4f}"
        )
        
        return daily_sentiment
    
    def process_ingestion_stream(
        self,
        social_posts: List[Dict]
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Process a unified stream of social media posts from all sources.
        Extract text fields, analyze sentiment, and generate sentiment index.
        
        Args:
            social_posts: List of post dictionaries from ingestion pipeline
            
        Returns:
            Tuple of (analyzed sentiments list, grouped daily sentiment DataFrame)
        """
        if not social_posts:
            logger.warning("No social posts to process")
            return [], pd.DataFrame(
                columns=["date", "ticker", "sentiment_score", "sentiment_std", "post_count"]
            )
        
        # Extract texts from posts
        texts = []
        post_metadata = []
        
        for post in social_posts:
            # Combine relevant text fields
            text_parts = []
            if "title" in post and post["title"]:
                text_parts.append(post["title"])
            if "text" in post and post["text"]:
                text_parts.append(post["text"])
            
            combined_text = " ".join(text_parts)
            
            if combined_text.strip():
                texts.append(combined_text)
                post_metadata.append({
                    "source": post.get("source", "unknown"),
                    "ticker": post.get("ticker", "UNKNOWN"),
                    "timestamp": post.get("timestamp", datetime.now())
                })
        
        if not texts:
            logger.warning("No valid text extracted from posts")
            return [], pd.DataFrame(
                columns=["date", "ticker", "sentiment_score", "sentiment_std", "post_count"]
            )
        
        # Analyze sentiments in batch
        sentiment_results = self.analyze_batch(texts)
        
        # Attach metadata to results
        for result, metadata in zip(sentiment_results, post_metadata):
            result.update(metadata)
        
        # Group by ticker and calculate daily sentiment index
        daily_sentiment_dfs = []
        tickers = set(post.get("ticker") for posts in [sentiment_results] for post in posts)
        
        for ticker in tickers:
            ticker_posts = [
                result for result in sentiment_results
                if result.get("ticker") == ticker
            ]
            if ticker_posts:
                daily_df = self.calculate_daily_sentiment_index(
                    ticker_posts,
                    ticker
                )
                daily_sentiment_dfs.append(daily_df)
        
        # Combine all daily sentiment data
        if daily_sentiment_dfs:
            combined_daily_sentiment = pd.concat(
                daily_sentiment_dfs,
                ignore_index=True
            ).sort_values(["ticker", "date"])
        else:
            combined_daily_sentiment = pd.DataFrame(
                columns=["date", "ticker", "sentiment_score", "sentiment_std", "post_count"]
            )
        
        logger.info(
            f"Processed {len(sentiment_results)} posts, "
            f"generated sentiment index for {len(tickers)} tickers"
        )
        
        return sentiment_results, combined_daily_sentiment
