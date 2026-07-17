"""Companies House MCP tool.

Allows querying company status, registered office address, and officers/directors
from the UK registry (Companies House).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from server.config import settings

logger = logging.getLogger(__name__)


# Static lookup for common UK SIC 2007 codes
SIC_DESCRIPTIONS = {
    "01110": "Growing of cereals, leguminous crops and oil seeds",
    "10890": "Manufacture of other food products not elsewhere classified",
    "21200": "Manufacture of pharmaceutical preparations",
    "26200": "Manufacture of computers and peripheral equipment",
    "26300": "Manufacture of communication equipment",
    "41100": "Development of building projects",
    "41200": "Construction of residential and non-residential buildings",
    "43210": "Electrical installation",
    "43220": "Plumbing, heat and air-conditioning installation",
    "43390": "Other building completion and finishing",
    "45200": "Maintenance and repair of motor vehicles",
    "46900": "Non-specialised wholesale trade",
    "47110": "Retail sale in non-specialised stores with food, beverages or tobacco predominating",
    "47910": "Retail sale via mail order houses or via Internet",
    "55100": "Hotels and similar accommodation",
    "56101": "Licensed restaurants",
    "56102": "Unlicensed restaurants and cafes",
    "56103": "Take-away food shops",
    "56302": "Public houses and bars",
    "62010": "Computer programming activities",
    "62020": "Information technology consultancy activities",
    "62090": "Other information technology and computer service activities",
    "63110": "Data processing, hosting and related activities",
    "63120": "Web portals",
    "64191": "Banks",
    "64209": "Activities of other holding companies not elsewhere classified",
    "64301": "Activities of investment trusts",
    "64304": "Activities of open-ended investment companies",
    "64921": "Credit granting by non-deposit taking finance houses and other specialist consumer credit providers",
    "64922": "Activities of mortgage finance companies",
    "64999": "Financial intermediation not elsewhere classified",
    "68100": "Buying and selling of own real estate",
    "68209": "Other letting and operating of own or leased real estate",
    "69100": "Legal activities",
    "69201": "Accounting, and auditing activities",
    "69202": "Book-keeping activities",
    "69203": "Tax consultancy",
    "70100": "Activities of head offices",
    "70221": "Financial management",
    "70229": "Management consultancy activities other than financial management",
    "71111": "Architectural activities",
    "71129": "Other engineering activities",
    "72190": "Other research and experimental development on natural sciences and engineering",
    "73110": "Advertising agencies",
    "73120": "Media representation services",
    "74909": "Other professional, scientific and technical activities not elsewhere classified",
    "78109": "Activities of employment agencies",
    "78200": "Temporary employment agency activities",
    "81210": "General cleaning of buildings",
    "81229": "Other industrial and building cleaning",
    "81300": "Landscape service activities",
    "82110": "Office administrative service activities",
    "82990": "Other business support service activities not elsewhere classified",
    "85310": "General secondary education",
    "85421": "First degree level higher education",
    "85422": "Post-graduate level higher education",
    "85590": "Other education not elsewhere classified",
    "86210": "General medical practice activities",
    "86220": "Specialist medical practice activities",
    "86230": "Dental practice activities",
    "86900": "Other human health activities",
    "88990": "Other social work activities without accommodation not elsewhere classified",
    "90010": "Performing arts",
    "90020": "Support activities to performing arts",
    "90030": "Artistic creation",
    "93110": "Operation of sports facilities",
    "93120": "Activities of sport clubs",
    "93130": "Fitness facilities",
    "93199": "Other sports activities",
    "93290": "Other amusement and recreation activities not elsewhere classified",
}


@dataclass
class _CacheEntry:
    expires_at: float
    payload: dict[str, Any]


class CompaniesHouseClient:
    """Async-friendly Companies House client with caching."""

    _cache: dict[str, _CacheEntry] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_ttl: int | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.COMPANIES_HOUSE_API_KEY
        self._base_url = base_url if base_url is not None else settings.COMPANIES_HOUSE_BASE_URL
        self._cache_ttl = cache_ttl if cache_ttl is not None else settings.COMPANIES_HOUSE_CACHE_TTL

    async def search_companies(self, query: str, items_per_page: int = 10) -> dict[str, Any]:
        """Search UK companies by query. Maps to GET /search/companies."""
        query = query.strip()
        if not query:
            return self._error("invalid_request", "Provide a search query.")

        items_per_page = max(1, min(int(items_per_page), 50))
        cache_key = f"search:{query}:{items_per_page}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        payload = await self._get_json(
            "search/companies",
            params={"q": query, "items_per_page": items_per_page}
        )
        if "error" not in payload:
            self._write_cache(cache_key, payload, self._cache_ttl)
        return payload

    async def get_company_profile(self, company_number: str) -> dict[str, Any]:
        """Retrieve profile details for a company. Maps to GET /company/{number}."""
        company_number = company_number.strip()
        if not company_number:
            return self._error("invalid_request", "Provide a company number.")

        cache_key = f"profile:{company_number}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        payload = await self._get_json(f"company/{company_number}")
        if "error" not in payload:
            self._write_cache(cache_key, payload, self._cache_ttl)
        return payload

    async def get_officers(self, company_number: str) -> dict[str, Any]:
        """Retrieve officers/directors list for a company. Maps to GET /company/{number}/officers."""
        company_number = company_number.strip()
        if not company_number:
            return self._error("invalid_request", "Provide a company number.")

        cache_key = f"officers:{company_number}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        payload = await self._get_json(f"company/{company_number}/officers")
        if "error" not in payload:
            self._write_cache(cache_key, payload, self._cache_ttl)
        return payload

    async def resolve_and_profile(self, query: str) -> dict[str, Any]:
        """Resolve a query by search, determine the company number, fetch its profile, and combine them."""
        query = query.strip()
        logger.info("companies_house.query: incoming_query=%r", query)

        if not query:
            return self._error("invalid_request", "Provide a search query.")

        # 1. Always call search_companies(query)
        search_res = await self.search_companies(query, items_per_page=10)
        
        search_key = f"search:{query}:10"
        search_cached = self._read_cache(search_key) is not None
        
        search_items = []
        if "error" not in search_res:
            search_items = search_res.get("items", [])
        
        logger.info("companies_house.search: query=%r results_count=%d cached=%s", query, len(search_items), search_cached)

        # 2. Check if query itself looks like a valid company number
        is_num = self._is_valid_company_number(query)
        
        # 3. Determine the company_number to use for the profile call
        company_number = None
        resolution_method = "none"
        exact_match = None
        if is_num:
            company_number = query.upper()
            resolution_method = "supplied_directly"
        else:
            if search_items:
                q_norm = " ".join(query.upper().split())
                for item in search_items:
                    title_norm = " ".join((item.get("title") or "").upper().split())
                    if title_norm == q_norm:
                        exact_match = item
                        break
                top_match = exact_match or search_items[0]
                company_number = top_match.get("company_number")
                resolution_method = "exact_name_match" if exact_match else "first_search_result"

        logger.info("companies_house.resolution: query=%r resolved_company_number=%s method=%s", query, company_number, resolution_method)

        # 4. If search returns zero results AND no valid company number was found, return a friendly message
        if not search_items and not company_number:
            logger.info("companies_house.response: query=%r resolved=False (no results)", query)
            return {
                "query": query,
                "search_results": [],
                "profile": None,
                "resolved_company_number": None,
                "message": f"No Companies House records were found matching '{query}'.",
                "source": "Companies House"
            }

        # 5. Always call get_company_profile(company_number) if determined
        profile_res = None
        profile_cached = False
        summary_object = None
        raw_officers = []
        normalized_officers = []
        
        if company_number:
            profile_res = await self.get_company_profile(company_number)
            profile_key = f"profile:{company_number}"
            profile_cached = self._read_cache(profile_key) is not None
            
            logger.info("companies_house.profile: company_number=%s cached=%s success=%s", company_number, profile_cached, "error" not in profile_res)

            if "error" in profile_res:
                profile_res = None
            else:
                officers_res = await self.get_officers(company_number)
                if "error" not in officers_res:
                    raw_officers = officers_res.get("items", [])
                    profile_res["officers"] = raw_officers
                else:
                    profile_res["officers"] = []

                # Normalize officers: active only, sorted by appointment date (newest first), limit to 15
                active_officers = [
                    off for off in raw_officers
                    if not off.get("resigned_on")
                ]
                active_officers.sort(key=lambda x: x.get("appointed_on") or "0000-00-00", reverse=True)
                
                for off in active_officers[:15]:
                    normalized_officers.append({
                        "name": off.get("name"),
                        "role": off.get("officer_role"),
                        "appointed_on": off.get("appointed_on")
                    })
                
                logger.info("companies_house.officers: company_number=%s active_count=%d total_count=%d", company_number, len(normalized_officers), len(raw_officers))

                # Normalize address
                addr_dict = profile_res.get("registered_office_address") or {}
                street_parts = []
                if addr_dict.get("address_line_1"):
                    street_parts.append(addr_dict["address_line_1"])
                if addr_dict.get("address_line_2"):
                    street_parts.append(addr_dict["address_line_2"])
                street = ", ".join(street_parts) if street_parts else "N/A"
                city = addr_dict.get("locality") or "N/A"
                postal_code = addr_dict.get("postal_code") or "N/A"
                country = addr_dict.get("country") or "N/A"

                # Resolve SIC codes
                resolved_sic_codes = []
                for code in profile_res.get("sic_codes") or []:
                    desc = SIC_DESCRIPTIONS.get(code, "Specialised business or registry classification")
                    resolved_sic_codes.append({
                        "code": code,
                        "description": desc
                    })

                # Name history
                name_history = []
                for p in profile_res.get("previous_company_names") or []:
                    name_history.append({
                        "name": p.get("name"),
                        "ceased_on": p.get("ceased_on"),
                        "effective_from": p.get("effective_from")
                    })

                # Accounts & confirmation statement
                accs = profile_res.get("accounts") or {}
                accounts_info = {
                    "next_due": accs.get("next_due"),
                    "overdue": accs.get("overdue", False)
                }
                conf = profile_res.get("confirmation_statement") or {}
                confirmation_info = {
                    "next_due": conf.get("next_due"),
                    "overdue": conf.get("overdue", False)
                }

                # Construct rich summary object
                summary_object = {
                    "company_name": profile_res.get("company_name"),
                    "company_number": company_number,
                    "status": profile_res.get("company_status"),
                    "company_type": profile_res.get("type"),
                    "incorporated_on": profile_res.get("date_of_creation"),
                    "registered_office": {
                        "street": street,
                        "city": city,
                        "postal_code": postal_code,
                        "country": country
                    },
                    "sic_codes": resolved_sic_codes,
                    "officers": normalized_officers,
                    "name_history": name_history,
                    "accounts": accounts_info,
                    "confirmation_statement": confirmation_info,
                    "resolution_method": resolution_method,
                    "source_url": f"https://find-and-update.company-information.service.gov.uk/company/{company_number}"
                }

        # Build list of other matches
        other_matches = []
        for item in search_items:
            # Avoid listing the resolved company in other matches
            if item.get("company_number") != company_number:
                other_matches.append({
                    "title": item.get("title"),
                    "company_number": item.get("company_number"),
                    "company_status": item.get("company_status"),
                    "address_snippet": item.get("address_snippet") or (item.get("address") or {}).get("locality"),
                })
        
        if summary_object:
            summary_object["other_matches"] = other_matches

        logger.info("companies_house.response: query=%r resolved=%s", query, bool(profile_res))

        return {
            "query": query,
            "search_results": search_items,
            "profile": profile_res,
            "resolved_company_number": company_number,
            "summary": summary_object,
            "cached": {"search": search_cached, "profile": profile_cached},
            "source": "Companies House"
        }

    def _is_valid_company_number(self, q: str) -> bool:
        q = q.strip().upper()
        return bool(re.match(r"^(?:[A-Z]{2}\d{6}|\d{8})$", q))

    async def _get_json(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a Companies House endpoint using HTTP Basic Auth (API key as username, empty password)."""
        if not self._api_key:
            logger.warning("companies_house: API key is not configured")
            return self._error("not_configured", "Companies House API key is not configured.")

        # Ensure base URL has correct format
        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        headers = {"Accept": "application/json"}
        auth = (self._api_key, "")

        logger.info("companies_house.request: url=%s params=%s", url, params)

        try:
            retries = 1
            while retries >= 0:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(url, params=params, headers=headers, auth=auth)

                    if response.status_code == 429:
                        if retries > 0:
                            logger.info("companies_house.retry: url=%s retry_count=1 (rate limited)", url)
                            await asyncio.sleep(1.0)
                            retries -= 1
                            continue
                        else:
                            return self._error("rate_limited", "Companies House API rate limit reached. Please try again later.")

                    if response.status_code == 401:
                        return self._error(
                            "unauthorized",
                            "Authentication failed. Verify if your Companies House API key is correct and matches the environment (sandbox vs live)."
                        )

                    if response.status_code == 404:
                        return self._error(
                            "not_found",
                            "The requested resource was not found on Companies House."
                        )

                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Companies House returned a non-object response")
                    return payload
        except httpx.HTTPStatusError as exc:
            logger.exception("companies_house.request: HTTP status error")
            return self._error(
                "http_error",
                f"Companies House returned HTTP error status: {exc.response.status_code}",
                str(exc)
            )
        except Exception as exc:
            logger.exception("companies_house.request: unexpected error")
            return self._error(
                "api_unavailable",
                "Companies House registry is currently unavailable.",
                str(exc)
            )

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


_companies_house_client = CompaniesHouseClient()


async def companies_house_search(query: str, items_per_page: int = 10) -> dict[str, Any]:
    """MCP tool entry point: search Companies House by company name."""
    return await _companies_house_client.search_companies(query=query, items_per_page=items_per_page)


async def companies_house_profile(query: str) -> dict[str, Any]:
    """MCP tool entry point: retrieve company details and officers by company query."""
    return await _companies_house_client.resolve_and_profile(query=query)


def get_companies_house_client() -> CompaniesHouseClient:
    return _companies_house_client
