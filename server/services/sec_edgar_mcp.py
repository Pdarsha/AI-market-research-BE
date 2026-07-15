"""Reusable SEC EDGAR MCP tools and endpoint client.

``SECEDGARClient`` centralizes SEC transport, headers, caching and company
resolution. New SEC endpoint services (submissions, concepts, frames) can reuse
its :meth:`_get_json` method without changing the chat or MCP controllers.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from server.config import settings

logger = logging.getLogger(__name__)
SEC_DATA_URL = "https://data.sec.gov"
SEC_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"

_METRICS = {
    "revenue": (("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"), "USD"),
    "net_income": (("NetIncomeLoss",), "USD"),
    "assets": (("Assets",), "USD"),
    "liabilities": (("Liabilities",), "USD"),
    "operating_cash_flow": (("NetCashProvidedByUsedInOperatingActivities",), "USD"),
    "investing_cash_flow": (("NetCashProvidedByUsedInInvestingActivities",), "USD"),
    "financing_cash_flow": (("NetCashProvidedByUsedInFinancingActivities",), "USD"),
    "cash_and_equivalents": (("CashAndCashEquivalentsAtCarryingValue",), "USD"),
    "basic_eps": (("EarningsPerShareBasic",), "USD/shares"),
    "diluted_eps": (("EarningsPerShareDiluted",), "USD/shares"),
    "shareholders_equity": (("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "USD"),
    "shares_outstanding": (("EntityCommonStockSharesOutstanding",), "shares"),
}
_FORM_PRIORITY = {"10-K": 0, "10-Q": 1, "8-K": 2}


@dataclass
class _CacheEntry:
    expires_at: float
    payload: Any


class SECEDGARClient:
    """Async-friendly SEC client with a shared mapping and facts cache."""

    _cache: dict[str, _CacheEntry] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        mapping_cache_ttl: int | None = None,
        facts_cache_ttl: int | None = None,
    ) -> None:
        self._user_agent = user_agent or settings.SEC_EDGAR_USER_AGENT
        self._mapping_cache_ttl = mapping_cache_ttl or settings.SEC_EDGAR_MAPPING_CACHE_TTL
        self._facts_cache_ttl = facts_cache_ttl or settings.SEC_EDGAR_FACTS_CACHE_TTL

    async def company_facts(
        self,
        *,
        company: str | None = None,
        ticker: str | None = None,
        period: str = "latest",
        metrics: list[str] | None = None,
        years: int | None = None,
    ) -> dict[str, Any]:
        """Resolve a company and return its normalized SEC company facts."""
        query = (ticker or company or "").strip()
        if not query:
            return self._error("invalid_request", "Provide either company or ticker.")

        started = time.monotonic()
        try:
            resolved = await self.resolve_company(company=company, ticker=ticker)
            if "error" in resolved:
                return resolved
            cik = resolved["cik"]
            cache_key = f"facts:{cik}"
            facts_data = self._read_cache(cache_key)
            cached = facts_data is not None
            if facts_data is None:
                endpoint = f"{SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik}.json"
                facts_data = await self._get_json(endpoint)
                self._write_cache(cache_key, facts_data, self._facts_cache_ttl)
            result = self._normalize_facts(
                resolved, facts_data, period=period, metrics=metrics, years=years
            )
            result["cached"] = cached
            logger.info(
                "sec.company_facts: query=%r ticker=%s company=%r cik=%s cache=%s elapsed_ms=%d",
                query, resolved["ticker"], resolved["company"], cik, cached,
                (time.monotonic() - started) * 1000,
            )
            return result
        except ValueError as exc:
            return self._error("invalid_request", str(exc))
        except Exception as exc:
            logger.exception("sec.company_facts: failed query=%r", query)
            return self._error("sec_unavailable", "SEC EDGAR is unavailable. Please try again later.", str(exc))

    async def resolve_company(
        self, *, company: str | None = None, ticker: str | None = None
    ) -> dict[str, Any]:
        """Resolve ticker or company title to a 10-digit CIK using SEC mapping."""
        query = (ticker or company or "").strip()
        if not query:
            return self._error("invalid_request", "Provide either company or ticker.")
        mapping = self._read_cache("company_mapping")
        if mapping is None:
            mapping = await self._get_json(SEC_MAPPING_URL)
            self._write_cache("company_mapping", mapping, self._mapping_cache_ttl)

        rows = mapping.values() if isinstance(mapping, dict) else []
        normalized = " ".join(query.upper().replace(",", " ").split())
        ticker_query = (ticker or query).strip().upper()
        matches = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_ticker = str(row.get("ticker") or "").upper()
            title = str(row.get("title") or "")
            title_normalized = " ".join(title.upper().replace(",", " ").split())
            if row_ticker == ticker_query or title_normalized == normalized:
                matches = [row]
                break
            if not ticker and normalized and normalized in title_normalized:
                matches.append(row)

        if len(matches) != 1:
            message = "Company not found." if not matches else "Company name is ambiguous; provide its ticker."
            return self._error("company_not_found" if not matches else "ambiguous_company", message)
        row = matches[0]
        try:
            cik = str(int(row["cik_str"])).zfill(10)
        except (KeyError, TypeError, ValueError):
            return self._error("invalid_cik", "SEC returned an invalid company identifier.")
        return {"company": row.get("title") or query, "ticker": row.get("ticker") or ticker_query, "cik": cik}

    async def _get_json(self, endpoint: str) -> dict[str, Any]:
        """GET an SEC endpoint with the required descriptive User-Agent header."""
        logger.info("sec.request: endpoint=%s", endpoint)
        async with httpx.AsyncClient(
            headers={"User-Agent": self._user_agent}, timeout=20
        ) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("SEC returned a non-object response")
        return payload

    @classmethod
    def _normalize_facts(
        cls,
        resolved: dict[str, str],
        data: dict[str, Any],
        *,
        period: str,
        metrics: list[str] | None,
        years: int | None,
    ) -> dict[str, Any]:
        """Produce one coherent reporting period (or an annual trend) from XBRL.

        The SEC can include the same fact in original/amended filings and across
        taxonomies. Entries are deduplicated by period, then ranked by period end,
        filing form, and filed date—never by their incidental JSON order.
        """
        period = period.lower()
        if period not in {"latest", "annual", "quarterly"}:
            raise ValueError("period must be latest, annual, or quarterly")
        selected = cls._selected_metrics(metrics)
        taxonomies = data.get("facts") or {}
        entries_by_metric = {
            name: cls._metric_entries(taxonomies, name, period) for name in selected
        }
        anchor = cls._latest_entry(entries_by_metric.values())
        if anchor is None:
            return {
                **resolved,
                "source": "SEC EDGAR",
                "error": {
                    "code": "financial_data_unavailable",
                    "message": "No validated SEC facts are available for the requested metric and reporting period.",
                },
            }

        metadata = {
            "report_type": anchor["form"],
            "period_end": anchor["end"],
            "filed_date": anchor.get("filed"),
        }
        financials: dict[str, Any] = {}
        if years:
            years = max(1, min(int(years), 10))
            for name, entries in entries_by_metric.items():
                history = cls._annual_history(entries, years)
                if history:
                    financials[name] = {"history": [cls._output_value(item) for item in history]}
        else:
            for name, entries in entries_by_metric.items():
                entry = cls._for_anchor(entries, anchor)
                if entry:
                    financials[name] = cls._output_value(entry)
        if not financials:
            return {
                **resolved,
                "source": "SEC EDGAR",
                "error": {
                    "code": "financial_data_unavailable",
                    "message": "No validated SEC facts are available for the requested metric and reporting period.",
                },
            }
        return {**resolved, "source": "SEC EDGAR", **metadata, "financials": financials}

    @staticmethod
    def _selected_metrics(metrics: list[str] | None) -> list[str]:
        if not metrics:
            return list(_METRICS)
        unknown = set(metrics) - set(_METRICS)
        if unknown:
            raise ValueError(f"Unsupported metrics: {', '.join(sorted(unknown))}")
        return list(dict.fromkeys(metrics))

    @classmethod
    def _metric_entries(cls, taxonomies: dict, metric: str, period: str) -> list[dict[str, Any]]:
        concepts, unit = _METRICS[metric]
        allowed_forms = {"annual": {"10-K"}, "quarterly": {"10-Q"}}.get(period, {"10-K", "10-Q", "8-K"})
        taxonomy = taxonomies.get("dei" if metric == "shares_outstanding" else "us-gaap") or {}
        for concept_name in concepts:
            concept = taxonomy.get(concept_name) or {}
            values = (concept.get("units") or {}).get(unit) or []
            entries = [
                {**item, "unit": unit, "concept": concept_name}
                for item in values
                if cls._is_valid_entry(item, unit=unit, allowed_forms=allowed_forms)
            ]
            deduped = cls._deduplicate_entries(entries)
            if deduped:
                return deduped
        return []

    @staticmethod
    def _is_valid_entry(item: dict[str, Any], *, unit: str, allowed_forms: set[str]) -> bool:
        """Validate filing, unit, numeric value, period, and filed date before use."""
        return (
            item.get("form") in allowed_forms
            and item.get("end")
            and item.get("filed")
            and isinstance(item.get("val"), (int, float))
            and bool(unit)
        )

    @staticmethod
    def _deduplicate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the best filing for each fiscal period, including amendments."""
        best: dict[tuple[Any, ...], dict[str, Any]] = {}
        for entry in entries:
            key = (entry.get("end"), entry.get("fy"), entry.get("fp"), entry.get("form"))
            current = best.get(key)
            if current is None or (entry.get("filed") or "") > (current.get("filed") or ""):
                best[key] = entry
        return sorted(best.values(), key=lambda item: (item["end"], -_FORM_PRIORITY.get(item["form"], 99), item.get("filed") or ""), reverse=True)

    @staticmethod
    def _latest_entry(entry_groups) -> dict[str, Any] | None:
        entries = [entry for group in entry_groups for entry in group]
        return max(entries, key=lambda item: (item["end"], -_FORM_PRIORITY.get(item["form"], 99), item.get("filed") or ""), default=None)

    @staticmethod
    def _for_anchor(entries: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any] | None:
        exact = [entry for entry in entries if entry["end"] == anchor["end"] and entry["form"] == anchor["form"]]
        if exact:
            return max(exact, key=lambda item: item.get("filed") or "")
        same_period = [entry for entry in entries if entry["end"] == anchor["end"]]
        if same_period:
            return max(same_period, key=lambda item: (-_FORM_PRIORITY.get(item["form"], 99), item.get("filed") or ""))
        # Balance-sheet and income-statement filing dates may differ slightly.
        return min(entries, key=lambda item: abs(int(item["end"].replace("-", "")) - int(anchor["end"].replace("-", ""))), default=None)

    @staticmethod
    def _annual_history(entries: list[dict[str, Any]], years: int) -> list[dict[str, Any]]:
        annual = [entry for entry in entries if entry.get("form") == "10-K"]
        return sorted(annual, key=lambda item: item["end"])[-years:]

    @staticmethod
    def _output_value(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "value": entry["val"],
            "currency": "USD/share" if entry["unit"] == "USD/shares" else entry["unit"],
            "period_end": entry["end"],
            "report_type": entry["form"],
            "filed_date": entry.get("filed"),
        }

    @classmethod
    def _read_cache(cls, key: str) -> Any | None:
        with cls._cache_lock:
            entry = cls._cache.get(key)
            if entry and entry.expires_at > time.monotonic():
                return entry.payload
            cls._cache.pop(key, None)
        return None

    @classmethod
    def _write_cache(cls, key: str, payload: Any, ttl: int) -> None:
        with cls._cache_lock:
            cls._cache[key] = _CacheEntry(time.monotonic() + max(0, ttl), payload)

    @staticmethod
    def _error(code: str, message: str, detail: str | None = None) -> dict[str, Any]:
        response: dict[str, Any] = {"error": {"code": code, "message": message}}
        if detail:
            response["error"]["detail"] = detail
        return response


_sec_edgar_client = SECEDGARClient()


async def sec_company_facts(
    company: str | None = None,
    ticker: str | None = None,
    period: str = "latest",
    metrics: list[str] | None = None,
    years: int | None = None,
) -> dict[str, Any]:
    """MCP tool entry point for normalized SEC company facts."""
    return await _sec_edgar_client.company_facts(
        company=company, ticker=ticker, period=period, metrics=metrics, years=years
    )


def get_sec_edgar_client() -> SECEDGARClient:
    return _sec_edgar_client
