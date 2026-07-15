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
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from server.config import settings

logger = logging.getLogger(__name__)
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


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

    async def financial_news(self, ticker: str, limit: int = 10) -> dict[str, Any]:
        """Return normalized news for ``ticker``.

        Results are cached by ticker for the configured TTL.  Network I/O runs in
        a worker thread so an async caller can fetch financial news alongside RAG.
        """
        ticker = ticker.strip().upper()
        if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
            raise ValueError("ticker must be a non-empty market symbol")
        limit = max(1, min(int(limit), 50))

        cached = self._get_cached(ticker)
        if cached is not None:
            return self._with_limit(cached, limit, cached=True)
        if not self._api_key:
            logger.warning("financial_news: Alpha Vantage key is not configured")
            return self._with_limit(self._empty(ticker), limit, cached=False)

        try:
            payload = await asyncio.to_thread(self._fetch, ticker)
        except Exception:
            logger.exception("financial_news: request failed for ticker=%s", ticker)
            return self._with_limit(self._empty(ticker), limit, cached=False)

        with self._cache_lock:
            self._cache[ticker] = _CachedNews(
                expires_at=time.monotonic() + max(0, self._cache_ttl), payload=payload
            )
        return self._with_limit(payload, limit, cached=False)

    def _get_cached(self, ticker: str) -> dict[str, Any] | None:
        with self._cache_lock:
            cached = self._cache.get(ticker)
            if cached and cached.expires_at > time.monotonic():
                return cached.payload
            self._cache.pop(ticker, None)
        return None

    def _fetch(self, ticker: str) -> dict[str, Any]:
        query = urlencode(
            {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": self._api_key}
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
    def _normalize(cls, ticker: str, data: dict[str, Any]) -> dict[str, Any]:
        articles: list[dict[str, Any]] = []
        for item in data.get("feed") or []:
            if not isinstance(item, dict):
                continue
            ticker_sentiment = next(
                (
                    row
                    for row in item.get("ticker_sentiment") or []
                    if str(row.get("ticker", "")).upper() == ticker
                ),
                {},
            )
            score = ticker_sentiment.get(
                "ticker_sentiment_score", item.get("overall_sentiment_score")
            )
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None
            articles.append(
                {
                    "title": item.get("title") or "",
                    "summary": item.get("summary") or "",
                    "url": item.get("url") or "",
                    "published_at": item.get("time_published") or "",
                    "sentiment": ticker_sentiment.get(
                        "ticker_sentiment_label", item.get("overall_sentiment_label")
                    ),
                    "sentiment_score": score,
                    "publisher": item.get("source") or "Alpha Vantage",
                }
            )
        return {"source": "Alpha Vantage", "ticker": ticker, "articles": articles}

    @staticmethod
    def _empty(ticker: str) -> dict[str, Any]:
        return {"source": "Alpha Vantage", "ticker": ticker, "articles": []}

    @staticmethod
    def _with_limit(payload: dict[str, Any], limit: int, *, cached: bool) -> dict[str, Any]:
        return {**payload, "articles": payload.get("articles", [])[:limit], "cached": cached}


_financial_news_tool = FinancialNewsMCPTool()


async def financial_news(ticker: str, limit: int = 10) -> dict[str, Any]:
    """MCP tool entry point: ``financial_news({ticker, limit})``."""
    return await _financial_news_tool.financial_news(ticker=ticker, limit=limit)


def get_financial_news_tool() -> FinancialNewsMCPTool:
    return _financial_news_tool
