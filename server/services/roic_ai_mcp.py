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
        from server.services.audit_logger import audit_context

        query = query.strip()
        if not query:
            return self._error("invalid_request", "Provide a company name or ticker.")
        limit = max(1, min(int(limit), 20))

        with audit_context(
            action="ROIC_AI_MCP_CALL",
            metadata={
                "query": query,
                "limit": limit,
                "transcript": transcript,
            }
        ) as meta:
            key = f"{query.upper()}:{transcript}"
            cached = self._read_cache(key)
            if cached:
                return {**cached, "cached": True}
            if not self._api_key:
                meta["error"] = "ROIC_API_KEY is not configured."
                meta["status"] = "failed"
                return self._error("not_configured", "ROIC_API_KEY is not configured.")
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(
                        ROIC_SEARCH_URL,
                        params={"apikey": self._api_key, "query": query},
                    )
                    if response.status_code == 429:
                        meta["error"] = "Roic.ai rate limit reached."
                        meta["status"] = "failed"
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
                                meta["error"] = "Roic.ai rate limit reached."
                                meta["status"] = "failed"
                                return self._error("rate_limited", "Roic.ai rate limit reached. Please try again shortly.")
                            transcript_response.raise_for_status()
                            result["latest_transcript"] = self._normalize_transcript(
                                ticker, transcript_response.json()
                            )
                    # Fetch financial metrics/ratios for the resolved ticker
                    if result["results"]:
                        ticker = result["results"][0].get("ticker")
                        if ticker:
                            result["metrics"] = await self._fetch_financial_metrics(client, ticker)
                self._write_cache(key, result)
                logger.info("roic.search: query=%r results=%d cache=false", query, len(result["results"]))
                return {**result, "cached": False}
            except httpx.HTTPStatusError as exc:
                logger.warning("roic.search: API error status=%s", exc.response.status_code)
                meta["error"] = f"API error status={exc.response.status_code}"
                meta["status"] = "failed"
                return self._error("api_error", "Roic.ai could not process this request.")
            except Exception as exc:
                logger.exception("roic.search: request failed query=%r", query)
                meta["error"] = str(exc)
                meta["status"] = "failed"
                return self._error("unavailable", "Roic.ai is temporarily unavailable.")

    async def _fetch_financial_metrics(self, client: httpx.AsyncClient, ticker: str) -> dict[str, Any] | None:
        """Fetch all fundamental ratios and statements for ticker concurrently and merge them by year."""
        import asyncio
        endpoints = {
            "profitability": f"https://api.roic.ai/v2/fundamental/ratios/profitability/{ticker}",
            "credit": f"https://api.roic.ai/v2/fundamental/ratios/credit/{ticker}",
            "liquidity": f"https://api.roic.ai/v2/fundamental/ratios/liquidity/{ticker}",
            "working_capital": f"https://api.roic.ai/v2/fundamental/ratios/working-capital/{ticker}",
            "multiples": f"https://api.roic.ai/v2/fundamental/multiples/{ticker}",
            "income_statement": f"https://api.roic.ai/v2/fundamental/income-statement/{ticker}",
            "balance_sheet": f"https://api.roic.ai/v2/fundamental/balance-sheet/{ticker}",
            "cash_flow": f"https://api.roic.ai/v2/fundamental/cash-flow/{ticker}",
        }

        async def fetch_one(name, url):
            try:
                resp = await client.get(url, params={"apikey": self._api_key})
                if resp.status_code == 429:
                    await asyncio.sleep(1)
                    resp = await client.get(url, params={"apikey": self._api_key})
                if resp.status_code == 200:
                    return name, resp.json()
                return name, None
            except Exception:
                return name, None

        tasks = [fetch_one(name, url) for name, url in endpoints.items()]
        fetched = await asyncio.gather(*tasks)
        fetched_dict = dict(fetched)

        # Merge by fiscal_year
        merged_by_year = {}
        for category, data in fetched_dict.items():
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                fy = item.get("fiscal_year")
                if not fy:
                    continue
                fy_str = str(fy)
                if fy_str not in merged_by_year:
                    merged_by_year[fy_str] = {}
                merged_by_year[fy_str].update(item)

        if not merged_by_year:
            return None

        # Compute derived metrics
        for fy_str, record in merged_by_year.items():
            if "pr_to_free_cash_flow" in record and record["pr_to_free_cash_flow"]:
                record["fcf_yield"] = (1.0 / record["pr_to_free_cash_flow"]) * 100
            
            if "bs_sh_out" in record and "pr_last" in record:
                record["market_cap"] = record["bs_sh_out"] * record["pr_last"]
            
            if "is_sales_revenue_turnover" in record and "bs_tot_asset" in record and record["bs_tot_asset"]:
                record["asset_turnover"] = record["is_sales_revenue_turnover"] / record["bs_tot_asset"]

        sorted_years = sorted(merged_by_year.keys(), key=int)
        for i in range(1, len(sorted_years)):
            curr_yr = sorted_years[i]
            prev_yr = sorted_years[i-1]
            curr_rec = merged_by_year[curr_yr]
            prev_rec = merged_by_year[prev_yr]

            # Revenue growth
            if "is_sales_revenue_turnover" in curr_rec and "is_sales_revenue_turnover" in prev_rec:
                prev_val = prev_rec["is_sales_revenue_turnover"]
                if prev_val:
                    curr_rec["revenue_growth"] = ((curr_rec["is_sales_revenue_turnover"] - prev_val) / prev_val) * 100

            # EBITDA growth
            curr_ebitda = curr_rec.get("ebitda")
            prev_ebitda = prev_rec.get("ebitda")
            if curr_ebitda is not None and prev_ebitda:
                curr_rec["ebitda_growth"] = ((curr_ebitda - prev_ebitda) / prev_ebitda) * 100

            # EPS growth
            if "diluted_eps" in curr_rec and "diluted_eps" in prev_rec:
                prev_val = prev_rec["diluted_eps"]
                if prev_val:
                    curr_rec["eps_growth"] = ((curr_rec["diluted_eps"] - prev_val) / prev_val) * 100

        return merged_by_year

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
