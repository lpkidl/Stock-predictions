"""
NewsAnalyzer — fetches financial news via NewsAPI and scores it with Groq (Llama 3.3).

Flow per ticker per run:
  1. Fetch the last 7 days of English-language news articles mentioning the ticker
     from NewsAPI.org (free tier: 100 req/day).
  2. Bundle titles + descriptions into a single prompt and call Groq's
     Llama 3.3 70B model (free tier: 14,400 req/day).
  3. Parse the structured JSON response into four numerical features:
       news_sentiment      — overall tone of coverage  (-1.0 to +1.0)
       news_confidence     — how certain the LLM is     ( 0.0 to  1.0)
       news_risk_score     — prominence of risk themes  ( 0.0 to  1.0)
       news_catalyst_score — prominence of +ve catalysts( 0.0 to  1.0)
  4. Return a date-indexed DataFrame that can be merged into the stock feature
     matrix by MLPredictor.merge_news_features().

Because NewsAPI free tier returns articles from the last 30 days, the features
will be non-zero for recent rows and 0-filled for older historical rows.
XGBoost handles sparse features like this naturally — the model learns that
a news_sentiment of 0 simply means "no news data for this date."

Requirements:
  pip install groq          # Groq Python SDK
  NEWSAPI_KEY   in .env     # free at https://newsapi.org/register
  GROQ_API_KEY  in .env     # free at https://console.groq.com
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
import pandas as pd

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

logger = logging.getLogger(__name__)

# Groq model to use — llama-3.3-70b-versatile is free and capable
GROQ_MODEL = "llama-3.3-70b-versatile"

# NewsAPI endpoint
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Prompt template sent to the LLM
_SYSTEM_PROMPT = (
    "You are a quantitative financial analyst. "
    "You will be given a list of recent news headlines and summaries about a stock. "
    "Respond ONLY with a single valid JSON object — no markdown, no explanation."
)

_USER_TEMPLATE = """\
Stock ticker: {ticker}

Recent news articles (title — description):
{articles_text}

Analyse the sentiment and content of these articles and return a JSON object with exactly these keys:
{{
  "sentiment": <float between -1.0 (very bearish) and 1.0 (very bullish)>,
  "confidence": <float between 0.0 (uncertain) and 1.0 (very certain)>,
  "risk_score": <float between 0.0 (no risk signals) and 1.0 (high risk/negative events)>,
  "catalyst_score": <float between 0.0 (no positive drivers) and 1.0 (strong positive catalysts)>,
  "themes": <list of up to 5 short strings describing the key news themes>
}}
"""


class NewsAnalyzer:
    """
    Fetches financial news from NewsAPI and scores it with Groq / Llama 3.3.

    Args:
        newsapi_key:  NewsAPI.org API key (free tier, 100 req/day).
        groq_api_key: Groq API key (free tier, 14,400 req/day).
        lookback_days: How many days of articles to pull per ticker (default 7).
        max_articles:  Maximum articles to include in the LLM prompt (default 10).
    """

    def __init__(
        self,
        newsapi_key: str,
        groq_api_key: str,
        lookback_days: int = 7,
        max_articles: int = 10,
    ):
        self.newsapi_key   = newsapi_key
        self.groq_api_key  = groq_api_key
        self.lookback_days = lookback_days
        self.max_articles  = max_articles

    # ------------------------------------------------------------------
    # Step 1 — fetch articles
    # ------------------------------------------------------------------

    def fetch_articles(self, ticker: str) -> List[Dict]:
        """
        Fetch recent English-language news articles mentioning `ticker`.
        Returns a list of dicts with keys: title, description, publishedAt, url.
        """
        from_date = (
            datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        ).strftime("%Y-%m-%d")

        params = {
            "q":        ticker,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": self.max_articles,
            "from":     from_date,
            "apiKey":   self.newsapi_key,
        }

        try:
            resp = httpx.get(NEWSAPI_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            logger.info(f"NewsAPI: {len(articles)} articles for {ticker}")
            return articles
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 426:
                logger.warning(
                    f"NewsAPI: free tier does not support this query for {ticker} "
                    f"(HTTP 426). Check your plan."
                )
            elif exc.response.status_code == 401:
                logger.warning("NewsAPI: invalid API key — check NEWSAPI_KEY in .env")
            else:
                logger.warning(f"NewsAPI error for {ticker}: {exc}")
            return []
        except Exception as exc:
            logger.warning(f"NewsAPI fetch failed for {ticker}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Step 2 — LLM scoring
    # ------------------------------------------------------------------

    def score_with_llm(self, ticker: str, articles: List[Dict]) -> Optional[Dict]:
        """
        Send article headlines + descriptions to Groq / Llama 3.3 and parse
        the structured JSON response.

        Returns a dict with keys: sentiment, confidence, risk_score,
        catalyst_score, themes.  Returns None on failure.
        """
        if not articles:
            return None

        lines = []
        for i, art in enumerate(articles[: self.max_articles], 1):
            title = (art.get("title") or "").strip()
            desc  = (art.get("description") or "").strip()
            if title:
                lines.append(f"{i}. {title}" + (f" — {desc}" if desc else ""))

        if not lines:
            return None

        articles_text = "\n".join(lines)
        user_msg      = _USER_TEMPLATE.format(
            ticker=ticker,
            articles_text=articles_text,
        )

        try:
            if Groq is None:
                raise ImportError("groq package not installed")
            client   = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)

            # Clamp all numeric fields to valid ranges
            def _clamp(v, lo, hi):
                try:
                    return max(lo, min(hi, float(v)))
                except (TypeError, ValueError):
                    return 0.0

            return {
                "sentiment":      _clamp(result.get("sentiment",      0.0), -1.0, 1.0),
                "confidence":     _clamp(result.get("confidence",     0.5),  0.0, 1.0),
                "risk_score":     _clamp(result.get("risk_score",     0.0),  0.0, 1.0),
                "catalyst_score": _clamp(result.get("catalyst_score", 0.0),  0.0, 1.0),
                "themes":         result.get("themes", []),
            }

        except Exception as exc:
            logger.warning(f"Groq LLM scoring failed for {ticker}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Step 3 — build feature DataFrame
    # ------------------------------------------------------------------

    def analyze_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch news and score with LLM for one ticker.

        Returns a single-row DataFrame indexed by today's date with columns:
          news_sentiment, news_confidence, news_risk_score, news_catalyst_score
        (or None if news fetch or LLM call failed).
        """
        articles = self.fetch_articles(ticker)
        if not articles:
            logger.info(f"No news articles found for {ticker} — skipping LLM")
            return None

        scores = self.score_with_llm(ticker, articles)
        if scores is None:
            return None

        today = pd.Timestamp(datetime.now(timezone.utc).date())
        df = pd.DataFrame([{
            "date":                 today,
            "ticker":               ticker,
            "news_sentiment":       scores["sentiment"],
            "news_confidence":      scores["confidence"],
            "news_risk_score":      scores["risk_score"],
            "news_catalyst_score":  scores["catalyst_score"],
            "news_themes":          ", ".join(scores.get("themes", [])),
            "news_article_count":   len(articles),
        }])

        logger.info(
            f"News scores for {ticker}: "
            f"sentiment={scores['sentiment']:+.2f}  "
            f"confidence={scores['confidence']:.2f}  "
            f"risk={scores['risk_score']:.2f}  "
            f"catalyst={scores['catalyst_score']:.2f}  "
            f"themes={scores.get('themes', [])}"
        )
        return df

    def analyze_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Run analyze_ticker for every ticker and concatenate results.

        Returns a DataFrame with one row per ticker that had news today.
        Empty DataFrame if nothing was fetched.
        """
        frames = []
        for ticker in tickers:
            df = self.analyze_ticker(ticker)
            if df is not None:
                frames.append(df)

        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=[
            "date", "ticker", "news_sentiment", "news_confidence",
            "news_risk_score", "news_catalyst_score",
            "news_themes", "news_article_count",
        ])
