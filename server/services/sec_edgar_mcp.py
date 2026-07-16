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
from datetime import datetime
from typing import Any

import httpx

from server.config import settings

logger = logging.getLogger(__name__)
SEC_DATA_URL = "https://data.sec.gov"
SEC_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"

_METRICS = {
    "revenue": (("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"), "USD"),
    "net_income": (("NetIncomeLoss",), "USD"),
    "operating_income": (("OperatingIncomeLoss",), "USD"),
    "assets": (("Assets",), "USD"),
    "liabilities": (("Liabilities",), "USD"),
    "cash_and_equivalents": (("CashAndCashEquivalentsAtCarryingValue",), "USD"),
    "operating_cash_flow": (("NetCashProvidedByUsedInOperatingActivities",), "USD"),
    "investing_cash_flow": (("NetCashProvidedByUsedInInvestingActivities",), "USD"),
    "financing_cash_flow": (("NetCashProvidedByUsedInFinancingActivities",), "USD"),
    "basic_eps": (("EarningsPerShareBasic",), "USD/shares"),
    "diluted_eps": (("EarningsPerShareDiluted",), "USD/shares"),
    "shareholders_equity": (("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "USD"),
    "shares_outstanding": (("EntityCommonStockSharesOutstanding",), "shares"),
}

_METRIC_ALIASES = {
    "revenue": "revenue",
    "sales": "revenue",
    "net income": "net_income",
    "net_income": "net_income",
    "profit": "net_income",
    "operating income": "operating_income",
    "operating_income": "operating_income",
    "assets": "assets",
    "liabilities": "liabilities",
    "cash": "cash_and_equivalents",
    "cash_and_equivalents": "cash_and_equivalents",
    "operating cash flow": "operating_cash_flow",
    "operating_cash_flow": "operating_cash_flow",
    "free cash flow": "free_cash_flow",
    "free_cash_flow": "free_cash_flow",
    "eps": ("basic_eps", "diluted_eps"),
    "basic_eps": "basic_eps",
    "diluted_eps": "diluted_eps",
    "shareholders_equity": "shareholders_equity",
    "equity": "shareholders_equity",
    "shares_outstanding": "shares_outstanding",
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
        request_type: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a company and return its normalized SEC company facts or submissions."""
        query = (ticker or company or "").strip()
        if not query:
            return self._error("invalid_request", "Provide either company or ticker.")

        started = time.monotonic()
        try:
            resolved = await self.resolve_company(company=company, ticker=ticker)
            if "error" in resolved:
                return resolved
            cik = resolved["cik"]

            # Route based on request_type
            if request_type == "submissions":
                cache_key = f"submissions:{cik}"
                submissions_data = self._read_cache(cache_key)
                cached = submissions_data is not None
                if submissions_data is None:
                    endpoint = f"{SEC_DATA_URL}/submissions/CIK{cik}.json"
                    submissions_data = await self._get_json(endpoint)
                    self._write_cache(cache_key, submissions_data, self._facts_cache_ttl)
                result = self._normalize_submissions(resolved, submissions_data)
                result["cached"] = cached
                logger.info(
                    "sec.submissions: query=%r ticker=%s company=%r cik=%s cache=%s elapsed_ms=%d",
                    query, resolved["ticker"], resolved["company"], cik, cached,
                    (time.monotonic() - started) * 1000,
                )
                return result

            # Otherwise, use companyfacts
            cache_key = f"facts:{cik}"
            facts_data = self._read_cache(cache_key)
            cached = facts_data is not None
            if facts_data is None:
                endpoint = f"{SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik}.json"
                facts_data = await self._get_json(endpoint)
                self._write_cache(cache_key, facts_data, self._facts_cache_ttl)

            sub_cache_key = f"submissions:{cik}"
            submissions_data = self._read_cache(sub_cache_key)
            if submissions_data is None:
                try:
                    sub_endpoint = f"{SEC_DATA_URL}/submissions/CIK{cik}.json"
                    submissions_data = await self._get_json(sub_endpoint)
                    self._write_cache(sub_cache_key, submissions_data, self._facts_cache_ttl)
                except Exception:
                    logger.warning("sec.company_facts: failed to fetch submissions for CIK %s", cik)
                    submissions_data = None

            result = self._normalize_facts(
                resolved, facts_data, period=period, metrics=metrics, years=years, submissions_data=submissions_data
            )
            result["cached"] = cached
            result["request_type"] = "facts"
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
    def _normalize_submissions(cls, resolved: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
        """Normalize filings from the SEC submissions endpoint."""
        filings_dict = data.get("filings", {})
        recent = filings_dict.get("recent", {})
        if not recent or not recent.get("accessionNumber"):
            return {
                **resolved,
                "source": "SEC EDGAR",
                "error": {
                    "code": "filings_unavailable",
                    "message": "No recent filings are available in the SEC Submissions data.",
                },
            }

        acc_nums = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        forms = recent.get("form", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_doc_descs = recent.get("primaryDocDescription", [])

        normalized_filings = []
        cik = resolved["cik"]

        for i in range(len(acc_nums)):
            form = forms[i]
            acc_num = acc_nums[i]
            acc_num_no_dashes = acc_num.replace("-", "")
            prim_doc = primary_docs[i]

            link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num_no_dashes}/{prim_doc}" if prim_doc else None
            filing_url = f"https://www.sec.gov/edgar/data/{cik}/{acc_num_no_dashes}" if cik else None
            filing_index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num_no_dashes}/{acc_num}-index.htm" if cik else None

            normalized_filings.append({
                "form": form,
                "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                "period_end": report_dates[i] if i < len(report_dates) else None,
                "description": primary_doc_descs[i] if i < len(primary_doc_descs) else form,
                "link": link,
                "accession_number": acc_num,
                "filing_url": filing_url,
                "filing_html_url": link,
                "filing_index_url": filing_index_url,
                "filing_document_url": link,
            })

        latest_filing = normalized_filings[0] if normalized_filings else None

        latest_financial_filing = None
        if latest_filing and latest_filing.get("form") not in ("10-K", "10-Q"):
            for f in normalized_filings:
                if f.get("form") in ("10-K", "10-Q"):
                    latest_financial_filing = f
                    break

        recent_list = normalized_filings[:5]

        return {
            **resolved,
            "source": "SEC EDGAR",
            "request_type": "submissions",
            "latest_filing": latest_filing,
            "latest_financial_filing": latest_financial_filing,
            "recent_filings": recent_list
        }

    @classmethod
    def _normalize_facts(
        cls,
        resolved: dict[str, str],
        data: dict[str, Any],
        *,
        period: str,
        metrics: list[str] | None,
        years: int | None,
        submissions_data: dict[str, Any] | None = None,
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

        cik = resolved.get("cik") or ""
        accn_to_prim_doc = {}
        if submissions_data:
            filings_dict = submissions_data.get("filings", {})
            recent = filings_dict.get("recent", {})
            if recent:
                acc_nums = recent.get("accessionNumber", [])
                primary_docs = recent.get("primaryDocument", [])
                for i in range(min(len(acc_nums), len(primary_docs))):
                    accn_to_prim_doc[acc_nums[i]] = primary_docs[i]

        metadata = {
            "report_type": anchor["form"],
            "period_end": anchor["end"],
            "filed_date": anchor.get("filed"),
        }
        financials: dict[str, Any] = {}
        if years:
            years = max(1, min(int(years), 10))
            for name, entries in entries_by_metric.items():
                valid_entries = [e for e in entries if cls._is_valid_final_record(e)]
                history = cls._annual_history(valid_entries, years)
                history = [h for h in history if cls._is_valid_final_record(h)]
                if history:
                    financials[name] = {"history": [cls._output_value(item, cik, accn_to_prim_doc) for item in history]}
                else:
                    financials[name] = {
                        "error": "The requested metric is not available in the SEC Company Facts data for the latest filing."
                    }
        else:
            for name, entries in entries_by_metric.items():
                valid_entries = [e for e in entries if cls._is_valid_final_record(e)]
                if not valid_entries:
                    financials[name] = {
                        "error": "The requested metric is not available in the SEC Company Facts data for the latest filing."
                    }
                    continue
                entry = cls._for_anchor(valid_entries, anchor)
                if entry and cls._is_valid_final_record(entry):
                    financials[name] = cls._output_value(entry, cik, accn_to_prim_doc)
                else:
                    latest_available = valid_entries[0]
                    val_out = cls._output_value(latest_available, cik, accn_to_prim_doc)
                    val_out["warning"] = f"Value is from an earlier reporting period: {latest_available.get('end')} ({latest_available.get('form')})"
                    financials[name] = val_out

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

    @classmethod
    def _selected_metrics(cls, metrics: list[str] | None) -> list[str]:
        if not metrics:
            return list(_METRICS)
        resolved_metrics = []
        for m in metrics:
            m_lower = m.lower().replace("_", " ")
            key = _METRIC_ALIASES.get(m.lower()) or _METRIC_ALIASES.get(m_lower) or _METRIC_ALIASES.get(m)
            if key:
                if isinstance(key, (list, tuple)):
                    resolved_metrics.extend(key)
                else:
                    resolved_metrics.append(key)
            elif m in _METRICS:
                resolved_metrics.append(m)
            else:
                found = False
                for std_key, (concepts, _) in _METRICS.items():
                    if m in concepts or m.lower() in [c.lower() for c in concepts]:
                        resolved_metrics.append(std_key)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Unsupported metric: {m}")
        return list(dict.fromkeys(resolved_metrics))

    @classmethod
    def _metric_entries(cls, taxonomies: dict, metric: str, period: str) -> list[dict[str, Any]]:
        if metric == "free_cash_flow":
            ocf_entries = cls._get_concept_entries(taxonomies, ("NetCashProvidedByUsedInOperatingActivities",), "USD", period)
            capex_entries = cls._get_concept_entries(taxonomies, ("PaymentsToAcquirePropertyPlantAndEquipment",), "USD", period)

            capex_map = {
                (c.get("end"), c.get("fy"), c.get("fp"), c.get("form")): c
                for c in capex_entries
            }

            fcf_entries = []
            for ocf in ocf_entries:
                key = (ocf.get("end"), ocf.get("fy"), ocf.get("fp"), ocf.get("form"))
                if key in capex_map:
                    capex = capex_map[key]
                    fcf_entries.append({
                        **ocf,
                        "val": ocf["val"] - capex["val"],
                        "concept": "FreeCashFlow",
                    })
            return fcf_entries

        concepts, unit = _METRICS[metric]
        return cls._get_concept_entries(taxonomies, concepts, unit, period)

    @classmethod
    def _get_concept_entries(cls, taxonomies: dict, concept_names: str | list[str] | tuple[str, ...], unit: str, period: str) -> list[dict[str, Any]]:
        if isinstance(concept_names, str):
            concept_names = (concept_names,)
        # Never use Form 4 or Form 8-K to answer financial metric questions.
        allowed_forms = {"annual": {"10-K"}, "quarterly": {"10-Q"}}.get(period, {"10-K", "10-Q"})

        taxonomy = {}
        for tax_name in ("us-gaap", "dei"):
            if tax_name in taxonomies:
                taxonomy.update(taxonomies[tax_name])

        for concept_name in concept_names:
            concept = taxonomy.get(concept_name) or {}
            values = (concept.get("units") or {}).get(unit) or []
            entries = [
                {**item, "unit": item["unit"] if "unit" in item else unit, "concept": concept_name}
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
        # Verify that all key fields belong to the same SEC fact record (dictionary)
        return (
            isinstance(item, dict)
            and item.get("form") in allowed_forms
            and item.get("end") is not None
            and item.get("filed") is not None
            and item.get("fy") is not None
            and item.get("fp") is not None
            and isinstance(item.get("val"), (int, float))
            and bool(unit)
            and ("unit" not in item or item["unit"] is not None)
        )

    @classmethod
    def _deduplicate_entries(cls, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the best filing for each fiscal period, including amendments."""
        best: dict[tuple[Any, ...], dict[str, Any]] = {}
        for entry in entries:
            key = (entry.get("fy"), entry.get("fp"))
            current = best.get(key)
            if current is None or (entry.get("filed") or "") > (current.get("filed") or ""):
                best[key] = entry
        return sorted(best.values(), key=cls._entry_sort_key, reverse=True)

    @staticmethod
    def _entry_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        filed = entry.get("filed") or ""
        end = entry.get("end") or ""
        fy = entry.get("fy") or 0
        fp = entry.get("fp") or ""
        form = entry.get("form") or ""

        # Prioritize 10-K > 10-Q as a tie-breaker
        form_priority = 2 if form == "10-K" else (1 if form == "10-Q" else 0)

        fp_order = {"FY": 4, "Q3": 3, "Q2": 2, "Q1": 1}
        fp_val = fp_order.get(str(fp).upper(), 0)

        return (filed, end, form_priority, fy, fp_val)

    @classmethod
    def _latest_entry(cls, entry_groups) -> dict[str, Any] | None:
        entries = [entry for group in entry_groups for entry in group]
        if not entries:
            return None
        return max(entries, key=cls._entry_sort_key)

    @classmethod
    def _for_anchor(cls, entries: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any] | None:
        exact = [entry for entry in entries if entry["end"] == anchor["end"] and entry["form"] == anchor["form"]]
        if exact:
            return max(exact, key=cls._entry_sort_key)
        same_period = [entry for entry in entries if entry["end"] == anchor["end"]]
        if same_period:
            return max(same_period, key=cls._entry_sort_key)

        closest = min(entries, key=lambda item: abs(int((item.get("end") or "0").replace("-", "")) - int((anchor.get("end") or "0").replace("-", ""))), default=None)
        if closest:
            try:
                d1 = datetime.strptime(closest["end"], "%Y-%m-%d")
                d2 = datetime.strptime(anchor["end"], "%Y-%m-%d")
                if abs((d1 - d2).days) <= 45:
                    return closest
            except Exception:
                pass
        return None

    @staticmethod
    def _annual_history(entries: list[dict[str, Any]], years: int) -> list[dict[str, Any]]:
        annual = [entry for entry in entries if entry.get("form") == "10-K"]
        return sorted(annual, key=lambda item: item["end"])[-years:]

    @classmethod
    def _is_valid_final_record(cls, entry: dict[str, Any]) -> bool:
        """Verify that all fields are part of the same fact record and present."""
        if not isinstance(entry, dict):
            return False
        required_keys = ["filed", "end", "fy", "fp", "val", "form", "unit"]
        return all(entry.get(k) is not None for k in required_keys)

    @classmethod
    def _format_human_friendly(cls, val: int | float, unit: str | None) -> str:
        """Format large financial values in a human-friendly format (e.g. $219.66 billion),
        while preserving the unit context.
        """
        if not isinstance(val, (int, float)):
            return str(val)

        is_usd = False
        if unit:
            unit_str = str(unit).upper()
            if "USD" in unit_str or "CURRENCY" in unit_str:
                is_usd = True

        abs_val = abs(val)
        prefix = "$" if is_usd else ""

        if abs_val >= 1e12:
            num_str = f"{abs_val / 1e12:.2f}"
            suffix = " trillion"
        elif abs_val >= 1e9:
            num_str = f"{abs_val / 1e9:.2f}"
            suffix = " billion"
        elif abs_val >= 1e6:
            num_str = f"{abs_val / 1e6:.2f}"
            suffix = " million"
        else:
            if isinstance(val, int):
                num_str = f"{abs_val:,}"
            else:
                num_str = f"{abs_val:,.2f}"
            suffix = ""

        sign = "-" if val < 0 else ""
        return f"{sign}{prefix}{num_str}{suffix}"

    @classmethod
    def _output_value(cls, entry: dict[str, Any], cik: str = "", accn_to_prim_doc: dict[str, str] | None = None) -> dict[str, Any]:
        raw_val = entry["val"]
        formatted_val = cls._format_human_friendly(raw_val, entry.get("unit"))
        accn = entry.get("accn")
        
        filing_url = None
        filing_html_url = None
        filing_index_url = None
        filing_document_url = None
        
        if accn and cik:
            accn_no_dashes = accn.replace("-", "")
            filing_url = f"https://www.sec.gov/edgar/data/{cik}/{accn_no_dashes}"
            filing_index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_no_dashes}/{accn}-index.htm"
            prim_doc = (accn_to_prim_doc or {}).get(accn)
            if prim_doc:
                filing_html_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_no_dashes}/{prim_doc}"
                filing_document_url = filing_html_url
                
        return {
            "value": raw_val,  # Keep for backward compatibility
            "raw_value": raw_val,
            "formatted_value": formatted_val,
            "unit": "USD/share" if entry["unit"] == "USD/shares" else entry["unit"],
            "fiscal_year": entry.get("fy"),
            "fiscal_period": entry.get("fp"),  # Keep for backward compatibility
            "fiscal_quarter": entry.get("fp"),
            "period_end": entry["end"],
            "filed_date": entry.get("filed"),
            "sec_source": entry.get("concept"),
            "report_type": entry["form"],  # Keep for backward compatibility
            "filing_type": entry["form"],
            "filing_url": filing_url,
            "filing_html_url": filing_html_url,
            "filing_index_url": filing_index_url,
            "filing_document_url": filing_document_url,
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
    request_type: str | None = None,
) -> dict[str, Any]:
    """MCP tool entry point for normalized SEC company facts."""
    return await _sec_edgar_client.company_facts(
        company=company, ticker=ticker, period=period, metrics=metrics, years=years, request_type=request_type
    )


def get_sec_edgar_client() -> SECEDGARClient:
    return _sec_edgar_client
