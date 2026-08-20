"""Agent Tool Registry for Agent Workflow Execution.

Provides a unified interface to discover, bind, and execute tools (RAG, SEC EDGAR,
Roic.ai, Financial News, Companies House, Web Search, and Custom Endpoints)
during multi-step Agent execution.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import httpx

from server.config import settings
from server.services.rag_service import get_rag_service
from server.services.sec_edgar_mcp import sec_company_facts
from server.services.roic_ai_mcp import roic_ai
from server.services.financial_news_mcp import financial_news
from server.services.companies_house_mcp import companies_house_search, companies_house_profile

logger = logging.getLogger(__name__)


class AgentTool:
    """Descriptor and runner for an executable Agent tool."""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        category: str,
        runner: Callable,
        parameters_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.category = category
        self.runner = runner
        self.parameters_schema = parameters_schema or {}

    async def execute(self, query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the tool runner safely with error boundary."""
        try:
            return await self.runner(query=query, context=context, **kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{self.name}': {str(e)}", exc_info=True)
            return {
                "success": False,
                "tool": self.name,
                "error": str(e),
                "data": None,
            }


class AgentToolRegistry:
    """Central registry of all built-in and dynamic tools accessible to Agent Workflows."""

    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}
        self._register_default_tools()

    def register_tool(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[AgentTool]:
        if not name:
            return None
        norm_name = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if norm_name in self._tools:
            return self._tools[norm_name]

        alias_map = {
            "web": "web_search",
            "search": "web_search",
            "google": "web_search",
            "bing": "web_search",
            "websearch": "web_search",
            "internet": "web_search",
            "internet_search": "web_search",
            "browse": "web_search",
            "web_crawler": "web_search",
            "online_search": "web_search",
            "rag": "rag",
            "workspace_documents": "rag",
            "workspace": "rag",
            "document": "rag",
            "documents": "rag",
            "document_search": "rag",
            "workspace_search": "rag",
            "retrieval": "rag",
            "docs": "rag",
            "pdf_search": "rag",
            "sec": "sec_edgar",
            "edgar": "sec_edgar",
            "sec_edgar": "sec_edgar",
            "sec_filings": "sec_edgar",
            "10k": "sec_edgar",
            "10q": "sec_edgar",
            "financial_statements": "sec_edgar",
            "roic": "roic_ai",
            "roic_ai": "roic_ai",
            "transcript": "roic_ai",
            "transcripts": "roic_ai",
            "earnings_call": "roic_ai",
            "earnings": "roic_ai",
            "valuation": "roic_ai",
            "news": "financial_news",
            "financial_news": "financial_news",
            "market_news": "financial_news",
            "alpha_vantage": "financial_news",
            "companies_house": "companies_house",
            "companies_house_uk": "companies_house",
            "uk_companies": "companies_house",
            "uk_registry": "companies_house",
            "company_registry": "companies_house",
        }

        target = alias_map.get(norm_name)
        if target and target in self._tools:
            return self._tools[target]

        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "display_name": t.display_name,
                "description": t.description,
                "category": t.category,
                "parameters_schema": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def _register_default_tools(self) -> None:
        # 1. RAG / Workspace Document Retrieval
        async def run_rag(query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            rag_service = get_rag_service()
            db = context.get("db")
            workspace_id = context.get("workspace_id")
            document_ids = context.get("document_ids") or []
            limit_past_7_days = context.get("limit_docs_past_7_days", False)
            user_id = context.get("user_id")

            db_created = False
            if db is None:
                from server.db.session import SessionLocal
                db = SessionLocal()
                db_created = True

            try:
                # Ensure document_ids is a clean list of strings
                if isinstance(document_ids, (str, int)):
                    document_ids = [str(document_ids)]
                elif isinstance(document_ids, list):
                    document_ids = [str(d) for d in document_ids if d]
                else:
                    document_ids = []

                # Handle limit_past_7_days filter if enabled
                if limit_past_7_days and db is not None and user_id:
                    from datetime import datetime, timedelta
                    from server.models.workspace import Document
                    seven_days_ago = datetime.utcnow() - timedelta(days=7)
                    doc_query = db.query(Document.id).filter(
                        Document.user_id == str(user_id),
                        Document.created_at >= seven_days_ago,
                    )
                    if workspace_id:
                        doc_query = doc_query.filter(Document.workspace_id == workspace_id)
                    if document_ids:
                        doc_query = doc_query.filter(Document.id.in_(document_ids))
                    valid_doc_ids = [str(r[0]) for r in doc_query.all()]
                    document_ids = valid_doc_ids
                    if not document_ids:
                        return {
                            "success": True,
                            "tool": "rag",
                            "query": query,
                            "chunks": [],
                            "source_documents": [],
                            "summary": "No documents found within the past 7 days.",
                        }

                top_k = kwargs.get("top_k") or settings.RAG_TOP_K

                # Perform RAG retrieval via rag_service.retrieve in a worker thread
                chunks = await asyncio.to_thread(
                    rag_service.retrieve,
                    db=db,
                    query=query,
                    user_id=str(user_id) if user_id else "",
                    document_ids=document_ids,
                    workspace_id=workspace_id,
                    top_k=top_k,
                )


                # Format retrieved chunks with complete citation metadata
                chunks_data = []
                for c in (chunks or []):
                    doc_name = getattr(c, "file_name", "Document")
                    content_text = getattr(c, "content", "") or ""
                    chunks_data.append({
                        "id": getattr(c, "id", None),
                        "chunk_id": getattr(c, "id", None),
                        "document_id": getattr(c, "document_id", None),
                        "file_name": doc_name,
                        "title": doc_name,
                        "document_name": doc_name,
                        "content": content_text,
                        "excerpt": content_text,
                        "page_number": getattr(c, "page_number", None),
                        "page": getattr(c, "page_number", None),
                        "section_title": getattr(c, "section_title", None),
                        "section": getattr(c, "section_title", None),
                        "document_url": getattr(c, "document_url", None) or getattr(c, "document_path", None),
                        "document_path": getattr(c, "document_path", None),
                        "start_offset": getattr(c, "start_offset", None),
                        "end_offset": getattr(c, "end_offset", None),
                        "type": "document",
                        "source_type": "document",
                    })

                source_documents = list(dict.fromkeys(c.file_name for c in (chunks or []) if getattr(c, "file_name", None)))
                context_summary = rag_service.build_context(chunks) if chunks else ""

                return {
                    "success": True,
                    "tool": "rag",
                    "query": query,
                    "chunks": chunks_data,
                    "sources": chunks_data,
                    "source_documents": source_documents,
                    "summary": context_summary,
                }
            except Exception as e:
                logger.error(f"RAG execution failed: {str(e)}", exc_info=True)
                return {
                    "success": False,
                    "tool": "rag",
                    "query": query,
                    "error": str(e),
                    "chunks": [],
                    "source_documents": [],
                    "summary": "",
                }
            finally:
                if db_created and db is not None:
                    db.close()

        self.register_tool(
            AgentTool(
                name="rag",
                display_name="Workspace Document Search",
                description="Search and retrieve relevant text chunks from attached workspace documents and PDFs.",
                category="retrieval",
                runner=run_rag,
            )
        )
        self.register_tool(
            AgentTool(
                name="workspace_documents",
                display_name="Workspace Document Search",
                description="Search workspace documents.",
                category="retrieval",
                runner=run_rag,
            )
        )

        # 2. SEC EDGAR MCP Tool
        async def run_sec(query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            ticker = kwargs.get("ticker")
            company = kwargs.get("company")
            if not ticker and not company:
                q_clean = query.strip()
                if re.fullmatch(r"[A-Za-z]{1,5}", q_clean):
                    ticker = q_clean.upper()
                else:
                    words = [w.strip(".,;:!?()[]") for w in q_clean.split()]
                    candidate_tickers = [w for w in words if w.isupper() and 1 <= len(w) <= 5 and w.isalpha()]
                    if candidate_tickers:
                        ticker = candidate_tickers[0]
                    else:
                        company = q_clean
            period = kwargs.get("period", "latest")
            metrics = kwargs.get("metrics")
            facts = await sec_company_facts(company=company, ticker=ticker, period=period, metrics=metrics)
            rag_service = get_rag_service()
            sec_results = [facts] if isinstance(facts, dict) else (facts or [])
            sec_context, sec_sources = (
                rag_service.build_sec_contexts(sec_results, start_number=1)
                if facts and not (isinstance(facts, dict) and facts.get("error"))
                else ("", [])
            )
            return {
                "success": True if not (isinstance(facts, dict) and facts.get("error")) else False,
                "tool": "sec_edgar",
                "ticker": ticker or company,
                "data": facts,
                "sources": sec_sources,
                "summary": sec_context,
            }

        self.register_tool(
            AgentTool(
                name="sec_edgar",
                display_name="SEC EDGAR Financials",
                description="Retrieve standardized 10-K/10-Q financial statements and company facts from US SEC EDGAR.",
                category="financial_data",
                runner=run_sec,
            )
        )

        # 3. Roic.ai MCP Tool
        async def run_roic(query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            transcript = kwargs.get("transcript", False)
            res = await roic_ai(query=query, transcript=transcript)
            rag_service = get_rag_service()
            roic_context, roic_sources = (
                rag_service.build_roic_context(res, start_number=1, query=query)
                if res and not (isinstance(res, dict) and res.get("error"))
                else ("", [])
            )
            return {
                "success": True if not (isinstance(res, dict) and res.get("error")) else False,
                "tool": "roic_ai",
                "query": query,
                "data": res,
                "sources": roic_sources,
                "summary": roic_context,
            }

        self.register_tool(
            AgentTool(
                name="roic_ai",
                display_name="Roic.ai Market & Financial Data",
                description="Access deep global equity analytics, valuation metrics, and earnings call transcripts.",
                category="financial_data",
                runner=run_roic,
            )
        )

        # 4. Financial News MCP Tool
        async def run_financial_news(query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            ticker = kwargs.get("ticker")
            limit = int(kwargs.get("limit", settings.ALPHA_VANTAGE_NEWS_MAX_RESULTS))
            if not ticker:
                q_clean = query.strip()
                if re.fullmatch(r"[A-Za-z]{1,5}", q_clean):
                    ticker = q_clean.upper()
                else:
                    words = [w.strip(".,;:!?()[]") for w in q_clean.split()]
                    candidate_tickers = [w for w in words if w.isupper() and 1 <= len(w) <= 5 and w.isalpha()]
                    if candidate_tickers:
                        ticker = candidate_tickers[0]

            res = None
            try:
                if ticker:
                    res = await financial_news(ticker=ticker, limit=limit)
                else:
                    res = await financial_news(query=query, limit=limit)
            except Exception as fn_err:
                logger.warning("financial_news execution error: %s", fn_err)
                res = None

            # If Alpha Vantage returns no articles for a general query, augment with web search
            if not res or not res.get("articles"):
                web_tool = self.get_tool("web_search")
                if web_tool:
                    return await web_tool.execute(query=query, context=context)

            rag_service = get_rag_service()
            news_context, news_sources = (
                rag_service.build_financial_news_context(res, start_number=1)
                if res and not (isinstance(res, dict) and res.get("error"))
                else ("", [])
            )
            return {
                "success": True if not (isinstance(res, dict) and res.get("error")) else False,
                "tool": "financial_news",
                "ticker": ticker,
                "query": query,
                "data": res,
                "sources": news_sources,
                "summary": news_context,
            }

        self.register_tool(
            AgentTool(
                name="financial_news",
                display_name="Financial News & Sentiment",
                description="Fetch real-time news headlines, earnings coverage, and market sentiment.",
                category="news",
                runner=run_financial_news,
            )
        )

        # 5. Companies House MCP Tool
        async def run_companies_house(query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            company_number = kwargs.get("company_number")
            if company_number:
                res = await companies_house_profile(query=company_number)
            else:
                res = await companies_house_profile(query=query)
            rag_service = get_rag_service()
            ch_context, ch_sources = (
                rag_service.build_companies_house_context(res, start_number=1)
                if res and not (isinstance(res, dict) and res.get("error"))
                else ("", [])
            )
            return {
                "success": True if not (isinstance(res, dict) and res.get("error")) else False,
                "tool": "companies_house",
                "query": query,
                "data": res,
                "sources": ch_sources,
                "summary": ch_context,
            }

        self.register_tool(
            AgentTool(
                name="companies_house",
                display_name="UK Companies House",
                description="Search UK registered company profiles, officer appointments, and filing history.",
                category="corporate_registry",
                runner=run_companies_house,
            )
        )

        # 6. Web Search Tool
        async def run_web_search(query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            rag_service = get_rag_service()
            try:
                context_str, web_sources = await asyncio.to_thread(
                    rag_service.web_search, query
                )
                normalized_sources = []
                for s in (web_sources or []):
                    link = s.get("link") or s.get("url") or ""
                    title = s.get("title") or "Web Source"
                    snippet = s.get("snippet") or s.get("excerpt") or ""
                    date_val = s.get("date") or s.get("published_date")
                    normalized_sources.append({
                        "type": "web",
                        "source_type": "web",
                        "title": title,
                        "source_title": title,
                        "url": link,
                        "link": link,
                        "snippet": snippet,
                        "excerpt": snippet,
                        "description": snippet,
                        "date": date_val,
                        "published_date": date_val,
                    })
                return {
                    "success": True,
                    "tool": "web_search",
                    "query": query,
                    "data": normalized_sources,
                    "sources": normalized_sources,
                    "summary": context_str,
                }
            except Exception as e:
                logger.error(f"Web search tool error: {e}", exc_info=True)
                return {
                    "success": False,
                    "tool": "web_search",
                    "query": query,
                    "data": [],
                    "sources": [],
                    "summary": "",
                    "error": str(e),
                }

        self.register_tool(
            AgentTool(
                name="web_search",
                display_name="Web Search",
                description="Search public internet for recent information, press releases, and articles.",
                category="search",
                runner=run_web_search,
            )
        )

    async def execute_custom_source(
        self, custom_source: Dict[str, Any], query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Safely query an external custom HTTP JSON source configured by the user."""
        url = custom_source.get("url")
        if not url:
            return {"success": False, "error": "Custom source missing URL"}

        method = custom_source.get("method", "GET").upper()
        headers = custom_source.get("headers", {})
        timeout_sec = float(custom_source.get("timeout_seconds", 10.0))

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                if method == "POST":
                    resp = await client.post(
                        url,
                        json={"query": query, "context": context},
                        headers=headers,
                    )
                else:
                    resp = await client.get(
                        url,
                        params={"query": query},
                        headers=headers,
                    )

                if resp.status_code == 200:
                    return {
                        "success": True,
                        "tool": custom_source.get("name", "custom_source"),
                        "data": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
                    }
                return {
                    "success": False,
                    "tool": custom_source.get("name", "custom_source"),
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                }
        except Exception as e:
            return {
                "success": False,
                "tool": custom_source.get("name", "custom_source"),
                "error": str(e),
            }


# Singleton instance
agent_tool_registry = AgentToolRegistry()


def get_agent_tool_registry() -> AgentToolRegistry:
    return agent_tool_registry
