"""Alpha Vantage financial-news MCP tool.

The tool is deliberately independent of the HTTP API layer so it can also be
registered with an MCP server/agent host.  Its public tool contract is
``financial_news(ticker, limit)`` and it returns normalized article data only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import re
from server.services.audit_logger import audit_context

from server.config import settings

logger = logging.getLogger(__name__)
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
RELEVANCE_THRESHOLD = 0.6


@dataclass
class _CachedNews:
    expires_at: float
    payload: dict[str, Any]


class FinancialNewsMCPTool:
    """Reusable MCP tool that fetches and normalizes Alpha Vantage news."""

    name = "financial_news"
    _cache: dict[str, _CachedNews] = {}
    _cache_lock = threading.Lock()

    def __init__(self, *, api_key: str | None = None, cache_ttl: int | None = None):
        self._api_key = api_key if api_key is not None else settings.ALPHA_VANTAGE_API_KEY
        self._cache_ttl = (
            cache_ttl if cache_ttl is not None else settings.ALPHA_VANTAGE_NEWS_CACHE_TTL
        )

    async def financial_news(
        self,
        ticker: str | None = None,
        limit: int = 10,
        query: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return normalized news for ``ticker`` or ``query``.

        The complete Alpha Vantage feed is normalized, filtered for the requested
        ticker, ranked, and aggregated before the requested number of articles is
        returned. Network I/O runs in a worker thread so an async caller can fetch
        financial news alongside RAG.
        """
        resolved_ticker = (ticker or kwargs.get("ticker") or "").strip().upper()
        if not resolved_ticker and query:
            q_clean = query.strip()
            if re.fullmatch(r"[A-Za-z]{1,5}", q_clean):
                resolved_ticker = q_clean.upper()
            else:
                words = [w.strip(".,;:!?()[]") for w in q_clean.split()]
                candidate_tickers = [w for w in words if w.isupper() and 1 <= len(w) <= 5 and w.isalpha()]
                if candidate_tickers:
                    resolved_ticker = candidate_tickers[0]

        limit = max(1, min(int(limit), 50))
        if not resolved_ticker:
            return self._with_limit(self._empty(ticker=query or "GENERAL"), limit, cached=False)

        ticker = resolved_ticker

        with audit_context(
            action="FINANCIAL_NEWS_MCP_CALL",
            metadata={"ticker": ticker, "limit": limit, "cached": False}
        ) as meta:
            cached = self._get_cached(ticker)
            if cached is not None:
                meta["cached"] = True
                return self._with_limit(cached, limit, cached=True)
            if not self._api_key:
                logger.warning("financial_news: Alpha Vantage key is not configured")
                meta["error"] = "API key not configured"
                meta["status"] = "failed"
                return self._with_limit(self._empty(ticker), limit, cached=False)

            try:
                payload = await asyncio.to_thread(self._fetch, ticker)
                return self._with_limit(payload, limit, cached=False)
            except Exception as exc:
                logger.exception("financial_news: request failed for ticker=%s", ticker)
                meta["error"] = str(exc)
                meta["status"] = "failed"
                return self._with_limit(self._empty(ticker), limit, cached=False)

    def _get_cached(self, ticker: str) -> dict[str, Any] | None:
        with self._cache_lock:
            cached = self._cache.get(ticker)
            if cached and cached.expires_at > time.monotonic():
                return cached.payload
            self._cache.pop(ticker, None)
        return None

    def _fetch(self, ticker: str) -> dict[str, Any]:
        query = urlencode(
            {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": "200", "apikey": self._api_key}
        )
        with urlopen(f"{ALPHA_VANTAGE_URL}?{query}", timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Alpha Vantage returned a non-object response")
        if data.get("Error Message") or data.get("Note") or data.get("Information"):
            logger.warning("financial_news: Alpha Vantage did not return feed data")
            return self._empty(ticker)
        return self._normalize(ticker, data)

    @classmethod
    def filter_by_ticker(cls, feed: list[dict[str, Any]], ticker: str, relevance_threshold: float = 0.6) -> list[dict[str, Any]]:
        """Filter feed items for the requested ticker, extracting the ticker-specific metadata and applying the relevance threshold."""
        articles = []
        ticker = ticker.strip().upper()
        for item in feed:
            if not isinstance(item, dict):
                continue
            ticker_sentiment = next(
                (
                    row
                    for row in item.get("ticker_sentiment") or []
                    if str(row.get("ticker", "")).upper() == ticker
                ),
                None,
            )
            if not ticker_sentiment:
                continue
            relevance = cls._as_float(ticker_sentiment.get("relevance_score"))
            if relevance is None or relevance < relevance_threshold:
                continue

            # Read ticker sentiment and overall sentiment scores/labels
            ticker_score = cls._as_float(ticker_sentiment.get("ticker_sentiment_score"))
            ticker_label = ticker_sentiment.get("ticker_sentiment_label")
            overall_score = cls._as_float(item.get("overall_sentiment_score"))
            overall_label = item.get("overall_sentiment_label")

            # Use ticker sentiment score/label with overall as fallback
            score = ticker_score if ticker_score is not None else overall_score
            sentiment = ticker_label if ticker_label is not None else overall_label

            articles.append(
                {
                    "title": item.get("title") or "",
                    "summary": item.get("summary") or "",
                    "url": item.get("url") or "",
                    "published_at": item.get("time_published") or "",
                    "sentiment": sentiment or "Unknown",
                    "sentiment_score": score,
                    "ticker_sentiment_score": ticker_score,
                    "ticker_sentiment_label": ticker_label,
                    "overall_sentiment": overall_label,
                    "overall_sentiment_score": overall_score,
                    "relevance_score": relevance,
                    "publisher": item.get("source") or "Alpha Vantage",
                }
            )
        return articles

    @classmethod
    def rank_articles(cls, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort articles using:
        1. relevance_score (highest first)
        2. publication time (newest first)
        """
        ranked = list(articles)
        ranked.sort(
            key=lambda article: (
                article.get("relevance_score") or 0.0,
                cls._publication_sort_value(article.get("published_at") or ""),
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def format_pub_date(value: str) -> str:
        """Format the publication date to YYYY-MM-DD format."""
        if not value:
            return "Unknown"
        for pattern in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value[: len(datetime.now().strftime(pattern))], pattern)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Fallback: if it's 8 digits like 20260712
        if len(value) >= 8 and value[:8].isdigit():
            return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
        return value

    @classmethod
    def calculate_sentiment_statistics(cls, articles: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate sentiment counts and average score based on article metadata."""
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        scores: list[float] = []

        for article in articles:
            label = str(article.get("sentiment") or "").lower()
            if "bullish" in label or label in {"positive", "pos"}:
                counts["positive"] += 1
            elif "bearish" in label or label in {"negative", "neg"}:
                counts["negative"] += 1
            else:
                counts["neutral"] += 1
            if article.get("sentiment_score") is not None:
                scores.append(article["sentiment_score"])

        average = sum(scores) / len(scores) if scores else None

        # Calculate date range
        dates = []
        for article in articles:
            pub_date = cls.format_pub_date(article.get("published_at"))
            if pub_date and pub_date != "Unknown":
                dates.append(pub_date)
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range = f"{min_date} to {max_date}" if min_date != max_date else min_date
        else:
            date_range = "Unknown"

        return {
            **counts,
            "average_sentiment_score": average,
            "overall_sentiment_label": cls._aggregate_label(average, counts),
            "articles_analyzed": len(articles),
            "date_range": date_range,
        }

    @classmethod
    def _normalize(cls, ticker: str, data: dict[str, Any]) -> dict[str, Any]:
        feed = data.get("feed") or []
        filtered = cls.filter_by_ticker(feed, ticker, relevance_threshold=RELEVANCE_THRESHOLD)
        ranked = cls.rank_articles(filtered)
        return {
            "source": "Alpha Vantage",
            "ticker": ticker,
            "articles": ranked,
            "sentiment_summary": cls.calculate_sentiment_statistics(ranked),
        }

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _publication_sort_value(value: str) -> float:
        """Return a sortable timestamp while tolerating Alpha Vantage variants."""
        if not value:
            return 0.0
        for pattern in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value[: len(datetime.now().strftime(pattern))], pattern).timestamp()
            except ValueError:
                continue
        # Alpha Vantage's canonical timestamp is lexically chronological.
        return float("".join(character for character in value if character.isdigit()) or 0)

    @staticmethod
    def _aggregate_label(average: float | None, counts: dict[str, int]) -> str:
        """Derive a deterministic aggregate label from API ticker metadata."""
        if average is None:
            if counts["positive"] > counts["negative"]:
                return "Bullish"
            if counts["negative"] > counts["positive"]:
                return "Bearish"
            return "Neutral"
        if average >= 0.35:
            return "Bullish"
        if average >= 0.15:
            return "Somewhat Bullish"
        if average <= -0.35:
            return "Bearish"
        if average <= -0.15:
            return "Somewhat Bearish"
        return "Neutral"

    @classmethod
    def _empty(cls, ticker: str) -> dict[str, Any]:
        return {
            "source": "Alpha Vantage",
            "ticker": ticker,
            "articles": [],
            "sentiment_summary": cls.calculate_sentiment_statistics([]),
        }

    @classmethod
    def _with_limit(cls, payload: dict[str, Any], limit: int, *, cached: bool) -> dict[str, Any]:
        articles = payload.get("articles", [])[:limit]
        return {
            **payload,
            "articles": articles,
            "sentiment_summary": cls.calculate_sentiment_statistics(articles),
            "cached": cached,
        }


_financial_news_tool = FinancialNewsMCPTool()


async def financial_news(
    ticker: str | None = None,
    limit: int = 10,
    query: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """MCP tool entry point: ``financial_news({ticker, limit, query})``."""
    return await _financial_news_tool.financial_news(ticker=ticker, limit=limit, query=query, **kwargs)


def get_financial_news_tool() -> FinancialNewsMCPTool:
    return _financial_news_tool
