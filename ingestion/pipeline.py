"""
Asynchronous Data Ingestion Pipeline.
Fetches data from yfinance (stocks), Reddit, and X (Twitter).
All network operations are asynchronous to prevent blocking.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json

import httpx
import yfinance as yf
import pandas as pd
from config import settings

# For X/Twitter integration
try:
    from twikit import Client
except ImportError:
    Client = None

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Asynchronous data ingestion from multiple sources:
    - yfinance: Historical stock OHLCV data
    - Reddit: Posts from r/wallstreetbets and r/stocks
    - X/Twitter: Posts using twikit library with session caching
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            executor: Optional ThreadPoolExecutor for blocking operations
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=5)
        self.session: Optional[httpx.AsyncClient] = None
        self.twikit_client: Optional[Client] = None
        self.tickers: List[str] = settings.STOCK_TICKERS.split(",")
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=settings.REDDIT_REQUEST_TIMEOUT,
            headers={"User-Agent": settings.REDDIT_USER_AGENT}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
        if self.twikit_client:
            # Clean up X client if needed
            pass
    
    async def fetch_yfinance_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data from yfinance using thread executor.
        Prevents blocking the async loop on I/O-bound operations.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1y", "3mo")
            interval: Data interval (e.g., "1d", "1h")
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor,
                lambda: yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False
                )
            )
            
            if data is not None and not data.empty:
                # yfinance 1.x returns MultiIndex columns — flatten to single level
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data["ticker"] = ticker
                logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
                await asyncio.sleep(settings.YFINANCE_DELAY)
                return data
            else:
                logger.warning(f"No data returned for ticker {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {ticker}: {str(e)}")
            return None
    
    async def fetch_reddit_posts(
        self,
        ticker: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch posts from Reddit r/wallstreetbets and r/stocks.

        Uses PRAW (official Reddit API) when PRAW_CLIENT_ID and
        PRAW_CLIENT_SECRET are configured in .env.  Falls back to a
        lightweight httpx scrape otherwise; if Reddit returns 403 (which
        happens when the raw JSON API is blocked) the method returns an
        empty list so the pipeline continues without social data.

        Args:
            ticker: Stock ticker to search for
            limit: Number of posts to retrieve

        Returns:
            List of post dictionaries containing text and metadata
        """
        # ── PRAW path (preferred when credentials are set) ───────────────
        if settings.PRAW_CLIENT_ID and settings.PRAW_CLIENT_SECRET:
            return await self._fetch_reddit_via_praw(ticker, limit)

        # ── Legacy raw-HTTP path ─────────────────────────────────────────
        if not self.session:
            logger.error("HTTP session not initialized")
            return []

        posts = []
        subreddits = ["wallstreetbets", "stocks"]

        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {"q": ticker, "sort": "new", "limit": limit}

                response = await self.session.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                if "data" in data and "children" in data["data"]:
                    for post in data["data"]["children"]:
                        if post["kind"] == "t3":
                            post_data = post["data"]
                            posts.append({
                                "source": "reddit",
                                "subreddit": subreddit,
                                "ticker": ticker,
                                "title": post_data.get("title", ""),
                                "text": post_data.get("selftext", ""),
                                "score": post_data.get("score", 0),
                                "timestamp": datetime.fromtimestamp(
                                    post_data.get("created_utc", 0)
                                ),
                                "url": f"https://reddit.com{post_data.get('permalink', '')}"
                            })

                logger.info(
                    f"Fetched {len(posts)} posts for {ticker} from r/{subreddit}"
                )
                await asyncio.sleep(settings.REDDIT_DELAY)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limited by Reddit — waiting 60s before retry")
                    await asyncio.sleep(60)
                elif e.response.status_code == 403:
                    logger.warning(
                        "Reddit raw JSON API returned 403. "
                        "Set PRAW_CLIENT_ID + PRAW_CLIENT_SECRET in .env to use the "
                        "official Reddit API and restore social data collection."
                    )
                    break
                else:
                    logger.error(f"HTTP error fetching Reddit data: {str(e)}")
            except Exception as e:
                logger.error(f"Error fetching Reddit posts for {ticker}: {str(e)}")

        return posts

    async def _fetch_reddit_via_praw(
        self,
        ticker: str,
        limit: int,
    ) -> List[Dict]:
        """Fetch Reddit posts using the official PRAW library."""
        try:
            import praw as _praw
            loop = asyncio.get_event_loop()

            def _sync_fetch():
                reddit = _praw.Reddit(
                    client_id=settings.PRAW_CLIENT_ID,
                    client_secret=settings.PRAW_CLIENT_SECRET,
                    user_agent=settings.PRAW_USER_AGENT,
                )
                posts_out = []
                for sub_name in ["wallstreetbets", "stocks"]:
                    sub = reddit.subreddit(sub_name)
                    for submission in sub.search(ticker, sort="new", limit=limit):
                        posts_out.append({
                            "source": "reddit",
                            "subreddit": sub_name,
                            "ticker": ticker,
                            "title": submission.title or "",
                            "text": submission.selftext or "",
                            "score": submission.score,
                            "timestamp": datetime.fromtimestamp(submission.created_utc),
                            "url": f"https://reddit.com{submission.permalink}",
                        })
                return posts_out

            posts = await loop.run_in_executor(self.executor, _sync_fetch)
            logger.info(f"PRAW fetched {len(posts)} posts for {ticker}")
            await asyncio.sleep(settings.REDDIT_DELAY)
            return posts

        except Exception as e:
            logger.warning(f"PRAW fetch failed for {ticker}: {e}")
            return []
    
    async def initialize_twikit_client(self) -> bool:
        """
        Initialize X/Twitter client with session caching.
        Loads a saved cookie file if available; otherwise authenticates
        using X_USERNAME / X_PASSWORD / X_EMAIL from .env and saves cookies.

        Returns:
            True if initialization successful, False otherwise
        """
        if not Client:
            logger.warning("twikit library not installed")
            return False

        if not all([settings.X_USERNAME, settings.X_PASSWORD, settings.X_EMAIL]):
            logger.info("X/Twitter credentials not configured in .env — skipping X data")
            return False

        try:
            self.twikit_client = Client()

            # Try to restore a saved cookie session first
            cookie_path = settings.X_SESSION_CACHE_PATH + ".json"
            try:
                self.twikit_client.load_cookies(cookie_path)
                logger.info("Loaded cached X/Twitter cookies")
                return True
            except Exception:
                pass  # No cache or stale cache — fall through to fresh login

            # Full login
            await self.twikit_client.login(
                auth_info_1=settings.X_USERNAME,
                auth_info_2=settings.X_EMAIL,
                password=settings.X_PASSWORD,
            )
            self.twikit_client.save_cookies(cookie_path)
            logger.info("Successfully authenticated with X/Twitter and saved cookies")
            return True

        except Exception as e:
            logger.warning(f"X/Twitter init failed: {e} — X data will be skipped")
            self.twikit_client = None
            return False

    async def fetch_x_posts(
        self,
        ticker: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Fetch posts from X (Twitter) using twikit (fully async).
        Uses saved cookie cache to avoid re-authentication on every run.

        Args:
            ticker: Stock ticker/keyword to search for
            limit: Number of posts to retrieve

        Returns:
            List of post dictionaries containing text and metadata
        """
        posts = []

        if not self.twikit_client:
            if not await self.initialize_twikit_client():
                return posts

        try:
            search_results = await self.twikit_client.search_tweet(
                query=f"${ticker} lang:en",
                product="Latest",
                count=limit,
            )

            for tweet in search_results:
                user = getattr(tweet, "user", None)
                screen_name = getattr(user, "screen_name", "") if user else ""
                tweet_id    = getattr(tweet, "id", "")
                posts.append({
                    "source":    "x",
                    "ticker":    ticker,
                    "text":      getattr(tweet, "full_text", getattr(tweet, "text", "")),
                    "author":    getattr(user, "name", "") if user else "",
                    "likes":     getattr(tweet, "favorite_count", 0),
                    "timestamp": datetime.now(),
                    "url":       f"https://x.com/{screen_name}/status/{tweet_id}",
                })

            logger.info(f"Fetched {len(posts)} posts for {ticker} from X")
            await asyncio.sleep(settings.X_DELAY)

        except Exception as e:
            logger.warning(f"Error fetching X posts for {ticker}: {e}")

        return posts
    
    async def fetch_macro_data(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch macro / market-context series: VIX, treasury yields, sector ETFs.
        Reuses fetch_yfinance_data with the same period as stock data.
        """
        tasks = [
            self.fetch_yfinance_data(t, period=settings.HISTORICAL_PERIOD)
            for t in settings.MACRO_TICKERS
        ]
        results = await asyncio.gather(*tasks)
        macro = {}
        for ticker, df in zip(settings.MACRO_TICKERS, results):
            if df is not None:
                macro[ticker] = df
                logger.info(f"Fetched macro data for {ticker}: {len(df)} rows")
            else:
                logger.warning(f"Could not fetch macro data for {ticker}")
        return macro

    async def fetch_earnings_dates(self, ticker: str) -> List:
        """
        Fetch historical + upcoming earnings dates via yfinance.
        Returns a list of pd.Timestamps (empty list on failure).
        """
        try:
            loop = asyncio.get_event_loop()
            ticker_obj = yf.Ticker(ticker)
            earnings_df = await loop.run_in_executor(
                self.executor,
                lambda: ticker_obj.earnings_dates
            )
            if earnings_df is not None and not earnings_df.empty:
                dates = earnings_df.index.tolist()
                logger.info(f"Fetched {len(dates)} earnings dates for {ticker}")
                return dates
            return []
        except Exception as e:
            logger.warning(f"Could not fetch earnings dates for {ticker}: {e}")
            return []

    async def aggregate_all_data(self) -> Dict:
        """
        Aggregate data from all sources for all configured tickers.
        Runs data fetching tasks concurrently.
        """
        aggregated_data = {
            "stocks":   {},
            "reddit":   [],
            "x":        [],
            "macro":    {},
            "earnings": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Fetch stock data (2y), macro data, and earnings concurrently
        stock_tasks    = [self.fetch_yfinance_data(t, period=settings.HISTORICAL_PERIOD) for t in self.tickers]
        macro_task     = self.fetch_macro_data()
        earnings_tasks = [self.fetch_earnings_dates(t) for t in self.tickers]

        stock_results, macro_data, earnings_results = await asyncio.gather(
            asyncio.gather(*stock_tasks),
            macro_task,
            asyncio.gather(*earnings_tasks),
        )

        for ticker, data in zip(self.tickers, stock_results):
            if data is not None:
                aggregated_data["stocks"][ticker] = data

        aggregated_data["macro"] = macro_data

        for ticker, dates in zip(self.tickers, earnings_results):
            aggregated_data["earnings"][ticker] = dates

        # Fetch social media data concurrently
        reddit_results, x_results = await asyncio.gather(
            asyncio.gather(*[self.fetch_reddit_posts(t) for t in self.tickers]),
            asyncio.gather(*[self.fetch_x_posts(t) for t in self.tickers]),
        )

        aggregated_data["reddit"] = [p for posts in reddit_results for p in posts]
        aggregated_data["x"]      = [p for posts in x_results      for p in posts]

        # ----------------------------------------------------------------
        # Optional Layer 1-A: Alpha Vantage intraday OHLCV
        # Only runs when ALPHA_VANTAGE_API_KEY is set in .env.
        # Stores 60-min OHLCV DataFrames (UTC-indexed) per ticker.
        # ----------------------------------------------------------------
        aggregated_data["intraday_ohlcv"] = {}
        if settings.ALPHA_VANTAGE_API_KEY:
            try:
                from feature_engine.alpha_vantage import AlphaVantageClient
                async with AlphaVantageClient(settings.ALPHA_VANTAGE_API_KEY) as av_client:
                    av_tasks = [
                        av_client.fetch_intraday_ohlcv(t, interval="60min")
                        for t in self.tickers
                    ]
                    av_results = await asyncio.gather(*av_tasks, return_exceptions=True)
                    for ticker, result in zip(self.tickers, av_results):
                        if isinstance(result, Exception):
                            logger.warning(f"AV intraday fetch failed for {ticker}: {result}")
                        elif result is not None and not result.empty:
                            aggregated_data["intraday_ohlcv"][ticker] = result
                            logger.info(f"AV intraday: stored {len(result)} bars for {ticker}")
            except ImportError:
                logger.warning("feature_engine not importable — skipping Alpha Vantage step.")
            except Exception as exc:
                logger.error(f"Alpha Vantage intraday block failed: {exc}")
        else:
            logger.debug("ALPHA_VANTAGE_API_KEY not set — skipping Alpha Vantage intraday.")

        # ----------------------------------------------------------------
        # Optional Layer 1-B: PRAW Reddit social stream
        # Only runs when PRAW_CLIENT_ID + PRAW_CLIENT_SECRET are set in .env.
        # Dispatched via run_in_executor because praw is a synchronous library.
        # Stores hourly-resampled social-volume DataFrames per ticker.
        # ----------------------------------------------------------------
        aggregated_data["reddit_social"] = {}
        if settings.PRAW_CLIENT_ID and settings.PRAW_CLIENT_SECRET:
            try:
                from feature_engine.reddit_stream import RedditSocialStream
                praw_stream = RedditSocialStream(
                    client_id=settings.PRAW_CLIENT_ID,
                    client_secret=settings.PRAW_CLIENT_SECRET,
                    user_agent=settings.PRAW_USER_AGENT,
                )
                loop = asyncio.get_event_loop()
                praw_tasks = [
                    loop.run_in_executor(
                        self.executor,
                        lambda t=t: praw_stream.fetch_submissions(t)
                    )
                    for t in self.tickers
                ]
                praw_results = await asyncio.gather(*praw_tasks, return_exceptions=True)
                for ticker, result in zip(self.tickers, praw_results):
                    if isinstance(result, Exception):
                        logger.warning(f"PRAW fetch failed for {ticker}: {result}")
                    elif result is not None and not result.empty:
                        aggregated_data["reddit_social"][ticker] = result
                        logger.info(f"PRAW social: stored {len(result)} hourly buckets for {ticker}")
            except ImportError:
                logger.warning("feature_engine not importable — skipping PRAW step.")
            except Exception as exc:
                logger.error(f"PRAW social block failed: {exc}")
        else:
            logger.debug("PRAW credentials not set — skipping PRAW social stream.")

        logger.info(
            f"Aggregated data: {len(aggregated_data['stocks'])} stocks, "
            f"{len(aggregated_data['macro'])} macro series, "
            f"{len(aggregated_data['reddit'])} Reddit posts, "
            f"{len(aggregated_data['x'])} X posts"
        )
        return aggregated_data
