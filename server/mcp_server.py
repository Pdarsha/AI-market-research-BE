"""Standalone MCP server exposing financial-news and SEC EDGAR tools.

Run with ``python -m server.mcp_server`` after installing requirements.  The RAG
API uses the same underlying tool directly, so MCP clients and chat requests get
identical normalization, cache semantics, and API-key handling.
"""

from fastmcp import FastMCP

from server.services.financial_news_mcp import financial_news as _financial_news
from server.services.sec_edgar_mcp import sec_company_facts as _sec_company_facts
from server.services.roic_ai_mcp import roic_ai as _roic_ai
from server.services.companies_house_mcp import (
    companies_house_search as _companies_house_search,
    companies_house_profile as _companies_house_profile,
)

mcp = FastMCP("capitaledge-financial-data")


@mcp.tool(name="financial_news")
async def financial_news(ticker: str, limit: int = 10) -> dict:
    """Retrieve normalized live financial news and sentiment for a stock ticker."""
    return await _financial_news(ticker=ticker, limit=limit)


@mcp.tool(name="roic_ai")
async def roic_ai(query: str, limit: int = 5, transcript: bool = False) -> dict:
    """Search Roic.ai by public-company ticker or company name."""
    return await _roic_ai(query=query, limit=limit, transcript=transcript)


@mcp.tool(name="sec_company_facts")
async def sec_company_facts(
    company: str | None = None,
    ticker: str | None = None,
    period: str = "latest",
    metrics: list[str] | None = None,
    years: int | None = None,
) -> dict:
    """Retrieve normalized annual, quarterly, latest, or historical SEC facts."""
    return await _sec_company_facts(
        company=company, ticker=ticker, period=period, metrics=metrics, years=years
    )


@mcp.tool(name="companies_house_search")
async def companies_house_search(query: str, items_per_page: int = 10) -> dict:
    """Search Companies House (UK company registry) by company name."""
    return await _companies_house_search(query=query, items_per_page=items_per_page)


@mcp.tool(name="companies_house_profile")
async def companies_house_profile(query: str) -> dict:
    """Retrieve UK company search results and resolved company details."""
    return await _companies_house_profile(query=query)
if __name__ == "__main__":
    mcp.run(transport="stdio")
