"""Roic.ai ticker-search MCP tool; endpoint additions belong in this module."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from server.config import settings

logger = logging.getLogger(__name__)
ROIC_SEARCH_URL = "https://api.roic.ai/v2/tickers/search"
ROIC_LATEST_TRANSCRIPT_URL = "https://api.roic.ai/v2/company/earnings-calls/latest"


@dataclass
class _CacheEntry:
    expires_at: float
    payload: dict[str, Any]


class RoicAIMCPTool:
    """Reusable Roic.ai client with normalized ticker-search output."""

    name = "roic_ai"
    _cache: dict[str, _CacheEntry] = {}
    _lock = threading.Lock()

    def __init__(self, *, api_key: str | None = None, cache_ttl: int | None = None):
        self._api_key = settings.ROIC_API_KEY if api_key is None else api_key
        self._cache_ttl = settings.ROIC_CACHE_TTL if cache_ttl is None else cache_ttl

    async def roic_ai(self, query: str, limit: int = 5, transcript: bool = False) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return self._error("invalid_request", "Provide a company name or ticker.")
        limit = max(1, min(int(limit), 20))
        key = f"{query.upper()}:{transcript}"
        cached = self._read_cache(key)
        if cached:
            return {**cached, "cached": True}
        if not self._api_key:
            return self._error("not_configured", "ROIC_API_KEY is not configured.")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    ROIC_SEARCH_URL,
                    params={"apikey": self._api_key, "query": query},
                )
                if response.status_code == 429:
                    return self._error("rate_limited", "Roic.ai rate limit reached. Please try again shortly.")
                response.raise_for_status()
                result = self._normalize(query, response.json(), limit)
                if transcript and result["results"]:
                    ticker = result["results"][0].get("ticker")
                    if ticker:
                        transcript_response = await client.get(
                            f"{ROIC_LATEST_TRANSCRIPT_URL}/{ticker}",
                            params={"apikey": self._api_key},
                        )
                        if transcript_response.status_code == 429:
                            return self._error("rate_limited", "Roic.ai rate limit reached. Please try again shortly.")
                        transcript_response.raise_for_status()
                        result["latest_transcript"] = self._normalize_transcript(
                            ticker, transcript_response.json()
                        )
            self._write_cache(key, result)
            logger.info("roic.search: query=%r results=%d cache=false", query, len(result["results"]))
            return {**result, "cached": False}
        except httpx.HTTPStatusError as exc:
            logger.warning("roic.search: API error status=%s", exc.response.status_code)
            return self._error("api_error", "Roic.ai could not process this request.")
        except Exception:
            logger.exception("roic.search: request failed query=%r", query)
            return self._error("unavailable", "Roic.ai is temporarily unavailable.")

    @staticmethod
    def _normalize(query: str, payload: Any, limit: int) -> dict[str, Any]:
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("results") or payload.get("tickers") or []
            if isinstance(rows, dict):
                rows = [rows]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        results = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            ticker = row.get("ticker") or row.get("symbol") or row.get("code")
            name = row.get("company_name") or row.get("name") or row.get("company")
            if ticker or name:
                results.append({
                    "ticker": ticker,
                    "company": name,
                    "exchange": row.get("exchange") or row.get("market"),
                    "country": row.get("country"),
                    "currency": row.get("currency"),
                })
        return {"source": "Roic.ai", "query": query, "results": results}

    @staticmethod
    def _normalize_transcript(ticker: str, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict) or not payload.get("content"):
            return None
        return {
            "ticker": payload.get("symbol") or ticker,
            "year": payload.get("year"),
            "quarter": payload.get("quarter"),
            "date": payload.get("date"),
            "content": str(payload["content"]),
        }

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry.expires_at > time.monotonic():
                return entry.payload
            self._cache.pop(key, None)
        return None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = _CacheEntry(time.monotonic() + max(0, self._cache_ttl), payload)

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {"source": "Roic.ai", "error": {"code": code, "message": message}, "results": []}


_roic_tool = RoicAIMCPTool()


async def roic_ai(query: str, limit: int = 5, transcript: bool = False) -> dict[str, Any]:
    """MCP entry point: search a company or ticker on Roic.ai."""
    return await _roic_tool.roic_ai(query=query, limit=limit, transcript=transcript)
