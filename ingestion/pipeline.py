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

import threading

import httpx
import yfinance as yf
import pandas as pd
from config import settings

# yfinance (1.x, curl_cffi-based) is NOT thread-safe: concurrent downloads can
# return another ticker's data or a merged multi-ticker frame. Serialize all
# yf.download calls through this lock.
_YF_LOCK = threading.Lock()

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
            headers={
                "User-Agent": settings.REDDIT_USER_AGENT,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
            follow_redirects=True,
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

            def _download():
                with _YF_LOCK:
                    return yf.download(
                        ticker,
                        period=period,
                        interval=interval,
                        progress=False
                    )

            data = await loop.run_in_executor(self.executor, _download)

            if data is not None and not data.empty:
                # yfinance 1.x returns MultiIndex columns (field, ticker).
                # Select this ticker's columns explicitly — never accept a
                # frame that belongs to (or mixes in) a different ticker.
                if isinstance(data.columns, pd.MultiIndex):
                    level1 = set(data.columns.get_level_values(1))
                    if ticker in level1:
                        data = data.xs(ticker, axis=1, level=1)
                    elif level1 == {""}:
                        data.columns = data.columns.get_level_values(0)
                    else:
                        logger.error(
                            f"yfinance returned wrong data for {ticker} "
                            f"(contains {sorted(level1)}) — discarding"
                        )
                        return None
                if data.columns.duplicated().any():
                    logger.error(f"Duplicate columns in {ticker} data — discarding")
                    return None
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

        # ── Free .json path (no API key required) ───────────────────────
        if not self.session:
            logger.error("HTTP session not initialized")
            return []

        posts = []

        for subreddit in settings.REDDIT_SUBREDDITS:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": ticker,
                    "sort": "new",
                    "limit": limit,
                    "restrict_sr": "on",  # keep results within this subreddit
                }

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
                                "post_id": post_data.get("id", ""),
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
                        f"Reddit .json returned 403 for r/{subreddit}. "
                        "The subreddit may be private or require authentication."
                    )
                else:
                    logger.error(f"HTTP error fetching Reddit data: {str(e)}")
            except Exception as e:
                logger.error(f"Error fetching Reddit posts for {ticker}: {str(e)}")

        # Optionally enrich top posts with comment text for deeper sentiment signal
        if settings.REDDIT_FETCH_COMMENTS and posts:
            top_posts = sorted(posts, key=lambda p: p.get("score", 0), reverse=True)[:5]
            for post in top_posts:
                if post.get("post_id") and post.get("subreddit"):
                    comments_text = await self._fetch_post_comments(
                        post["post_id"], post["subreddit"], settings.REDDIT_COMMENT_LIMIT
                    )
                    if comments_text:
                        post["text"] = (post["text"] + " " + comments_text).strip()
                    await asyncio.sleep(settings.REDDIT_DELAY)

        return posts

    async def _fetch_post_comments(
        self,
        post_id: str,
        subreddit: str,
        limit: int,
    ) -> str:
        """Fetch top comments for a post and return them as a single text string."""
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "depth": 1, "sort": "top"}
        try:
            response = await self.session.get(url, params=params)
            if response.status_code != 200:
                return ""
            data = response.json()
            # data[1] holds the comment listing
            if not (isinstance(data, list) and len(data) > 1):
                return ""
            bodies = []
            for item in data[1].get("data", {}).get("children", []):
                if item.get("kind") == "t1":
                    body = item.get("data", {}).get("body", "")
                    if body and body not in ("[deleted]", "[removed]"):
                        bodies.append(body)
            return " ".join(bodies)
        except Exception as e:
            logger.debug(f"Comment fetch failed for post {post_id}: {e}")
            return ""

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
                for sub_name in settings.REDDIT_SUBREDDITS:
                    sub = reddit.subreddit(sub_name)
                    for submission in sub.search(ticker, sort="new", limit=limit):
                        posts_out.append({
                            "source": "reddit",
                            "subreddit": sub_name,
                            "ticker": ticker,
                            "post_id": submission.id,
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

    async def fetch_news_articles(self, ticker: str, limit: int = 25) -> List[Dict]:
        """
        Fetch financial news headlines for a ticker — no API key required.

        Two free sources, merged and deduped by URL:
          • yfinance Ticker.news (Yahoo Finance)
          • Google News RSS search

        Returns post dicts shaped like the social sources so they flow through
        the same FinBERT sentiment pipeline (which was trained on financial
        news, so these score better than social posts):
        { source:"news", ticker, title, text, author (publisher),
          timestamp (datetime), url }
        """
        articles: List[Dict] = []
        seen_urls = set()

        # ── Source 1: Yahoo Finance via yfinance ──────────────────────
        try:
            loop = asyncio.get_event_loop()

            def _fetch_yf_news():
                with _YF_LOCK:
                    return yf.Ticker(ticker).news or []

            raw_items = await loop.run_in_executor(self.executor, _fetch_yf_news)
            for item in raw_items[:limit]:
                # yfinance >=1.x nests fields under "content"; older versions
                # are flat — support both.
                content = item.get("content", item)
                title = content.get("title", "")
                url = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or item.get("link", "")
                )
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or item.get("publisher", "")
                )
                pub_date = content.get("pubDate") or item.get("providerPublishTime")
                if isinstance(pub_date, (int, float)):
                    timestamp = datetime.fromtimestamp(pub_date)
                else:
                    try:
                        timestamp = pd.to_datetime(pub_date).to_pydatetime()
                        if timestamp.tzinfo:
                            timestamp = timestamp.replace(tzinfo=None)
                    except Exception:
                        timestamp = datetime.now()
                if not title or not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append({
                    "source": "news",
                    "ticker": ticker,
                    "title": title,
                    "text": content.get("summary", "") or "",
                    "author": publisher,
                    "timestamp": timestamp,
                    "url": url,
                })
        except Exception as e:
            logger.warning(f"Yahoo Finance news fetch failed for {ticker}: {e}")

        # ── Source 2: Google News RSS ─────────────────────────────────
        try:
            import xml.etree.ElementTree as ET
            from email.utils import parsedate_to_datetime

            rss_url = (
                "https://news.google.com/rss/search"
                f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            )
            resp = await self.session.get(rss_url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                for item in root.iter("item"):
                    if len(articles) >= 2 * limit:
                        break
                    title = (item.findtext("title") or "").strip()
                    url = (item.findtext("link") or "").strip()
                    source_el = item.find("source")
                    publisher = source_el.text if source_el is not None else "Google News"
                    try:
                        timestamp = parsedate_to_datetime(item.findtext("pubDate", ""))
                        if timestamp.tzinfo:
                            timestamp = timestamp.replace(tzinfo=None)
                    except Exception:
                        timestamp = datetime.now()
                    if not title or not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    articles.append({
                        "source": "news",
                        "ticker": ticker,
                        "title": title,
                        "text": "",
                        "author": publisher,
                        "timestamp": timestamp,
                        "url": url,
                    })
            else:
                logger.warning(
                    f"Google News RSS returned {resp.status_code} for {ticker}"
                )
        except Exception as e:
            logger.warning(f"Google News RSS fetch failed for {ticker}: {e}")

        logger.info(f"Fetched {len(articles)} news article(s) for {ticker}")
        return articles

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
            "news":     [],
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

        # Fetch social media + news data concurrently
        reddit_results, x_results, news_results = await asyncio.gather(
            asyncio.gather(*[self.fetch_reddit_posts(t) for t in self.tickers]),
            asyncio.gather(*[self.fetch_x_posts(t) for t in self.tickers]),
            asyncio.gather(*[self.fetch_news_articles(t) for t in self.tickers]),
        )

        aggregated_data["reddit"] = [p for posts in reddit_results for p in posts]
        aggregated_data["x"]      = [p for posts in x_results      for p in posts]
        aggregated_data["news"]   = [p for posts in news_results   for p in posts]

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
        # Layer 1-B: Reddit social stream (hourly volume + polarity buckets)
        # Uses PRAW when credentials are set; falls back to the free .json
        # API automatically — no credentials required for the HTTP path.
        # ----------------------------------------------------------------
        aggregated_data["reddit_social"] = {}
        try:
            from feature_engine.reddit_stream import RedditSocialStream
            reddit_stream = RedditSocialStream(
                client_id=settings.PRAW_CLIENT_ID,
                client_secret=settings.PRAW_CLIENT_SECRET,
                user_agent=settings.PRAW_USER_AGENT,
            )
            loop = asyncio.get_event_loop()
            reddit_tasks = [
                loop.run_in_executor(
                    self.executor,
                    lambda t=t: reddit_stream.fetch_submissions(t)
                )
                for t in self.tickers
            ]
            reddit_results = await asyncio.gather(*reddit_tasks, return_exceptions=True)
            for ticker, result in zip(self.tickers, reddit_results):
                if isinstance(result, Exception):
                    logger.warning(f"Reddit social fetch failed for {ticker}: {result}")
                elif result is not None and not result.empty:
                    aggregated_data["reddit_social"][ticker] = result
                    logger.info(f"Reddit social: stored {len(result)} hourly buckets for {ticker}")
        except ImportError:
            logger.warning("feature_engine not importable — skipping Reddit social stream.")
        except Exception as exc:
            logger.error(f"Reddit social block failed: {exc}")

        logger.info(
            f"Aggregated data: {len(aggregated_data['stocks'])} stocks, "
            f"{len(aggregated_data['macro'])} macro series, "
            f"{len(aggregated_data['reddit'])} Reddit posts, "
            f"{len(aggregated_data['x'])} X posts, "
            f"{len(aggregated_data['news'])} news articles"
        )
        return aggregated_data
