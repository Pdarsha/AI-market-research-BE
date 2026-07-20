"""RAG retrieval service.

Orchestrates the query side of the pipeline with two-stage retrieval:

    query -> embed query (OpenAI) -> pgvector cosine search (candidate pool)
          -> cross-encoder re-rank -> top-k chunks -> build context string

Retrieval is scoped to the requesting user and, when supplied, a single
workspace. Stage 1 ordering is done in-database by pgvector's cosine distance
operator; stage 2 re-scores the candidates with a cross-encoder
(:mod:`server.services.rerank_service`) for sharper relevance before the
top-k are passed to the LLM.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

import serpapi
from sqlalchemy import text
from sqlalchemy.orm import Session

from server.config import settings
from server.models.workspace import DocumentChunk
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.services.rerank_service import get_rerank_service
from server.services.financial_news_mcp import FinancialNewsMCPTool
from server.utils.prompt import (
    FOLLOWUP_MARKER,
    ai_assistent,
    augmented_system_prompt,
    classification_prompt,
    financial_news_classification_prompt,
    followup_instruction_block,
    sec_edgar_classification_prompt,
    roic_ai_classification_prompt,
    web_search_classification_prompt,
    companies_house_classification_prompt,
)

logger = logging.getLogger(__name__)

serp_client = serpapi.Client(api_key=settings.SERPAPI_KEY)


class RAGService:
    """Embed a query and retrieve the most relevant document chunks."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedder = embedding_service

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        user_id: str,
        top_k: int | None = None,
        document_ids: list[str],
        workspace_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Return the top-k most relevant :class:`DocumentChunk` rows for ``query``.

        Two-stage retrieval:
          1. **Vector search** — fetch a broad candidate pool ordered by pgvector
             cosine distance (fast, approximate).
          2. **Re-ranking** — a cross-encoder re-scores the candidates against the
             query and the top ``top_k`` are kept (slower, more accurate).

        When re-ranking is disabled (``settings.RAG_RERANK_ENABLED``) the vector
        top-k is returned directly.
        """
        import time
        from server.services.audit_logger import AuditLogger

        start_time = time.time()
        try:
            top_k = top_k or settings.RAG_TOP_K

            if document_ids and len(document_ids) > top_k:
                top_k = min(len(document_ids), settings.RAG_MAX_CONTEXT_CHUNKS)

            query_embedding = self._embedder.embed_text(query)

            self._widen_hnsw_search(db)
            distance = DocumentChunk.embedding.cosine_distance(query_embedding)

            def _scoped():
                q = db.query(DocumentChunk).filter(DocumentChunk.user_id == str(user_id))
                if workspace_id:
                    q = q.filter(DocumentChunk.workspace_id == str(workspace_id))
                return q

            candidate_k = (
                max(top_k, settings.RAG_RERANK_CANDIDATES)
                if settings.RAG_RERANK_ENABLED
                else top_k
            )

            if document_ids and len(document_ids) > 1:
                import math

                per_doc = max(1, math.ceil(candidate_k / len(document_ids)))
                rows: list[DocumentChunk] = []
                for doc_id in document_ids:
                    rows.extend(
                        _scoped()
                        .filter(DocumentChunk.document_id == doc_id)
                        .order_by(distance)
                        .limit(per_doc)
                        .all()
                    )
            else:
                stmt = _scoped()
                if document_ids:
                    stmt = stmt.filter(DocumentChunk.document_id.in_(document_ids))
                rows = stmt.order_by(distance).limit(candidate_k).all()

            if not rows:
                final = []
            else:
                if settings.RAG_RERANK_ENABLED:
                    ordered = get_rerank_service().rerank(
                        query,
                        rows,
                        top_k=len(rows),
                        text_getter=lambda chunk: chunk.content or "",
                    )
                else:
                    ordered = rows

                final = self._ensure_document_representation(
                    ordered, document_ids=document_ids, top_k=top_k
                )

            elapsed = time.time() - start_time
            AuditLogger.log(
                action="RAG_RETRIEVAL",
                user_id=str(user_id),
                workspace_id=str(workspace_id) if workspace_id else None,
                status="success",
                processing_time=elapsed,
                metadata={
                    "query": query,
                    "document_ids": document_ids,
                    "chunks_count": len(final),
                },
            )
            return final
        except Exception as e:
            elapsed = time.time() - start_time
            AuditLogger.log(
                action="RAG_RETRIEVAL",
                user_id=str(user_id),
                workspace_id=str(workspace_id) if workspace_id else None,
                status="error",
                processing_time=elapsed,
                metadata={
                    "query": query,
                    "document_ids": document_ids,
                    "error": str(e),
                },
            )
            raise


    @staticmethod
    def _ensure_document_representation(
        ordered_rows: list[DocumentChunk],
        *,
        document_ids: list[str],
        top_k: int,
    ) -> list[DocumentChunk]:
        """Pick the final ``top_k`` chunks, guaranteeing that every attached
        document contributes at least its best chunk (when several are attached).

        Without this, a global top-k lets a many-chunk document fill every slot and
        silently drop a few-chunk document (e.g. an uploaded image) from the answer
        context. Slots left after the per-document floor are filled by global rank.
        """
        if not ordered_rows:
            return []
        if not document_ids or len(document_ids) <= 1:
            return ordered_rows[:top_k]

        rank = {id(row): index for index, row in enumerate(ordered_rows)}
        best_by_doc: dict[str, DocumentChunk] = {}
        for row in ordered_rows:
            best_by_doc.setdefault(row.document_id, row)

        selected: list[DocumentChunk] = []
        selected_ids: set[int] = set()

        for doc_id in document_ids:
            row = best_by_doc.get(doc_id)
            if row is not None and id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))

        for row in ordered_rows:
            if len(selected) >= top_k:
                break
            if id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))

        selected.sort(key=lambda row: rank[id(row)])
        return selected[:top_k]

    @staticmethod
    def _widen_hnsw_search(db: Session) -> None:
        """Raise ``hnsw.ef_search`` for the current transaction (PostgreSQL only).

        HNSW applies the ``WHERE`` filter (user/workspace/document scope) only
        *after* pulling its ``ef_search`` nearest neighbours from the whole table,
        so a small default window (40) can miss a specific document's chunks when
        other documents sit nearer in vector space — the cause of "first upload
        returns nothing, re-upload works". A wider window restores recall for
        filtered searches. ``iterative_scan`` (pgvector >= 0.8) makes it exact;
        it's attempted in a savepoint so older versions degrade gracefully.
        """
        try:
            if db.bind is None or db.bind.dialect.name != "postgresql":
                return
        except Exception:
            return

        try:
            db.execute(
                text("SET LOCAL hnsw.ef_search = :ef").bindparams(
                    ef=int(settings.RAG_HNSW_EF_SEARCH)
                )
            )
        except Exception:
            logger.debug("rag: could not set hnsw.ef_search", exc_info=True)
            return

        try:
            with db.begin_nested():
                db.execute(
                    text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                )
        except Exception:
            logger.debug("rag: hnsw.iterative_scan unavailable", exc_info=True)

    @staticmethod
    def build_context(chunks: list[DocumentChunk]) -> str:
        """Concatenate chunk contents into a single context string for the LLM."""
        return "\n\n---\n\n".join(chunk.content for chunk in chunks if chunk.content)

    @staticmethod
    def build_context_with_citations(chunks: list[DocumentChunk]) -> str:
        """Build context string with numbered markers for chunks.

        The chunks are numbered [1], [2], etc., so the LLM can cite them
        explicitly in its response using these markers. This enables automatic
        citation mapping through the CitationService.

        Returns:
            A context string with chunks numbered [1], [2], etc.
        """
        if not chunks:
            return ""

        numbered_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            if chunk.content:
                # Include a numbered marker so the LLM knows which chunk is which
                numbered_chunks.append(f"[{i}] {chunk.content}")

        return "\n\n---\n\n".join(numbered_chunks)

    @staticmethod
    def build_context_with_chunks(
        chunks: list[DocumentChunk],
    ) -> tuple[str, list[DocumentChunk]]:
        """Return both context string and the original chunks for citation mapping.

        The chunks are returned separately so the citation service can map LLM-
        generated citations back to the original retrieved chunks with full metadata.

        Returns:
            (context_string, chunks_list)
        """
        context = "\n\n---\n\n".join(chunk.content for chunk in chunks if chunk.content)
        return context, chunks

    @staticmethod
    def build_messages(
        *,
        history: list[dict],
        question: str,
        retrieved_context: str | None = None,
        web_context: str | None = None,
        financial_news_context: str | None = None,
        sec_context: str | None = None,
        roic_context: str | None = None,
        companies_house_context: str | None = None,
    ) -> list[dict]:
        """Prompt generation.

        Grounds the answer in whatever context is available:
          * ``retrieved_context`` — RAG document chunks;
          * ``web_context``       — web search results.
        When either (or both) is present the augmented system prompt is used;
        with neither, the plain-assistant prompt is used. Previous chat history is
        always included for continuity, and the current question is appended last.

        The follow-up-questions instruction is appended to the system prompt so
        every answer is followed by a parseable list of suggested follow-ups
        (see :meth:`split_answer_and_followups`).
        """
        if (
            retrieved_context is not None
            or web_context is not None
            or financial_news_context is not None
            or sec_context is not None
            or roic_context is not None
            or companies_house_context is not None
        ):
            system_prompt = augmented_system_prompt(
                document_context=retrieved_context,
                web_context=web_context,
                financial_news_context=financial_news_context,
                sec_context=sec_context,
                roic_context=roic_context,
                companies_house_context=companies_house_context,
            )
        else:
            system_prompt = ai_assistent()

        system_prompt += followup_instruction_block()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def split_answer_and_followups(text: str) -> tuple[str, list[str]]:
        """Split a raw generation into ``(answer, follow_up_questions)``.

        The model is instructed to append ``FOLLOWUP_MARKER`` followed by a JSON
        array of questions. This strips that block from the answer and parses the
        questions. If the marker is missing or the block is unparseable, the full
        text is returned as the answer with an empty follow-up list — never
        placeholder text.
        """
        if not text or FOLLOWUP_MARKER not in text:
            return (text or "").strip(), []

        answer, _, tail = text.partition(FOLLOWUP_MARKER)

        questions: list[str] = []
        match = re.search(r"\[.*\]", tail, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    questions = [
                        str(q).strip() for q in parsed if str(q).strip()
                    ]
            except (ValueError, TypeError):
                questions = []

        return answer.strip(), questions[:4]

    @staticmethod
    def source_file_names(chunks: list[DocumentChunk]) -> list[str]:
        """Return the distinct source file names behind ``chunks`` (order-stable)."""
        return list(dict.fromkeys(c.file_name for c in chunks if c.file_name))

    def retrieve_context(
        self,
        db: Session,
        *,
        query: str,
        user_id: str,
        top_k: int | None = None,
        document_ids: list[str],
        workspace_id: str | None = None,
    ) -> tuple[str, list[str]]:
        """Convenience wrapper returning ``(context_string, source_file_names)``."""
        chunks = self.retrieve(
            db,
            query=query,
            user_id=user_id,
            top_k=top_k,
            document_ids=document_ids,
            workspace_id=workspace_id,
        )
        return self.build_context(chunks), self.source_file_names(chunks)

    def retrieve_context_with_chunks(
        self,
        db: Session,
        *,
        query: str,
        user_id: str,
        top_k: int | None = None,
        document_ids: list[str],
        workspace_id: str | None = None,
    ) -> tuple[str, list[str], list[DocumentChunk]]:
        """Convenience wrapper returning context, source names, and full chunks.

        Unlike retrieve_context, this also returns the DocumentChunk objects with
        full metadata, enabling the citation service to map answer text back to
        the original chunks for citation generation.

        Returns:
            (context_string, source_file_names, chunks_list)
        """
        chunks = self.retrieve(
            db,
            query=query,
            user_id=user_id,
            top_k=top_k,
            document_ids=document_ids,
            workspace_id=workspace_id,
        )
        return self.build_context(chunks), self.source_file_names(chunks), chunks

    @staticmethod
    def generate_response(client, *, messages: list[dict], model: str) -> str:
        """LLM response generation using streaming, returning the full text.

        Streaming is preserved (``stream=True``) exactly as the existing chat flow;
        the chunks are aggregated into the final response string.
        """
        ai_response = ""
        completion = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1024, stream=True, temperature=0
        )
        for chunk in completion:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                ai_response += chunk.choices[0].delta.content
        return ai_response

    @staticmethod
    def classify_requires_document(client, question: str, model: str) -> bool:
        """Ask the LLM whether ``question`` needs the uploaded document(s).

        Sends a lightweight one-word (``YES``/``NO``) routing prompt. On any error the
        call fails safe by returning ``True`` (prefer grounding the answer in the
        documents rather than silently ignoring them).
        """
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": classification_prompt()},
                    {"role": "user", "content": f'Question:\n"{question}"\nAnswer:'},
                ],
                max_tokens=1024,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            requires = answer.startswith("Y")
            logger.info(
                "rag.classify: verdict=%s (raw=%r) question_len=%d",
                requires,
                answer,
                len(question),
            )
            return requires
        except Exception:
            logger.exception(
                "rag.classify: classification failed; defaulting to RAG (use documents)"
            )
            return True

    @staticmethod
    def classify_requires_web_search(client, question: str, model: str) -> bool:
        """Ask the LLM whether ``question`` needs a live web search.

        Sends a lightweight one-word (``YES``/``NO``) routing prompt. On any error
        the call fails safe by returning ``False`` (skip the search rather than add
        latency / spurious context).
        """
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": web_search_classification_prompt()},
                    {"role": "user", "content": f'Question:\n"{question}"\nAnswer:'},
                ],
                max_tokens=1024,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            requires = answer.startswith("Y")
            logger.info(
                "rag.classify_web: verdict=%s (raw=%r) question_len=%d",
                requires,
                answer,
                len(question),
            )
            return requires
        except Exception:
            logger.exception(
                "rag.classify_web: classification failed; defaulting to no web search"
            )
            return False

    @staticmethod
    def classify_financial_news_ticker(
        client, question: str, model: str
    ) -> str | None:
        """Return the ticker for a live-finance question, or ``None``.

        This is deliberately separate from generic web-search routing so a finance
        question uses the Alpha Vantage tool instead of an unrelated web search.
        """
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": financial_news_classification_prompt()},
                    {"role": "user", "content": question},
                ],
                max_tokens=16,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            match = re.fullmatch(r"YES:([A-Z][A-Z0-9.\-]{0,9})", answer)
            ticker = match.group(1) if match else None
            logger.info("rag.classify_finance: ticker=%s question_len=%d", ticker, len(question))
            return ticker
        except Exception:
            logger.exception("rag.classify_finance: classification failed; skipping finance news")
            return None

    @staticmethod
    def classify_roic_ai_query(client, question: str, model: str) -> str | None:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": roic_ai_classification_prompt()}, {"role": "user", "content": question}],
                max_tokens=16,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            match = re.fullmatch(r"YES:([A-Z][A-Z0-9.\-]{0,9})", answer)
            return match.group(1) if match else None
        except Exception:
            logger.exception("rag.classify_roic: classification failed")
            return None

    @staticmethod
    def classify_companies_house_query(client, question: str, model: str) -> str | None:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": companies_house_classification_prompt()},
                    {"role": "user", "content": question},
                ],
                max_tokens=16,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip()
            if answer.upper().startswith("YES:"):
                return answer[4:].strip()
            return None
        except Exception:
            logger.exception("rag.classify_companies_house: classification failed")
            return None

    @staticmethod
    def is_earnings_transcript_query(question: str) -> bool:
        """Recognize transcript intent even when the lightweight planner misses it."""
        normalized = question.lower()
        phrases = (
            "earnings call", "earnings transcript", "conference call",
            "management commentary", "management say", "management said",
            "ceo say", "ceo said", "cfo say", "cfo said", "guidance",
            "forward guidance", "management discussion",
        )
        return any(phrase in normalized for phrase in phrases)

    @staticmethod
    def transcript_company_query(question: str) -> str:
        """Return a company-like Roic lookup term; keep the query as a fallback."""
        ticker = re.search(r"\$([A-Za-z]{1,10})\b", question)
        if ticker:
            return ticker.group(1).upper()
        companies = (
            "Apple", "Microsoft", "Tesla", "Nvidia", "Amazon", "Meta",
            "Alphabet", "Google", "Netflix", "AMD", "Intel",
        )
        for company in companies:
            if re.search(rf"\b{re.escape(company)}\b", question, re.IGNORECASE):
                return company
        return question

    @staticmethod
    def build_roic_context(data: dict, *, start_number: int, query: str = "") -> tuple[str, list[dict]]:
        lines, sources = [], []
        
        results = data.get("results") or []
        resolved_ticker = results[0].get("ticker") if results else None
        
        roic_metrics_keywords = [
            "gross margin", "gross profit margin", "gross%", "gross_margin", "grossmargin", "gross margins", "gross profit margins",
            "operating margin", "operating profit margin", "ebit margin", "ebit_margin", "oper_margin", "operating_margin", "operatingmargin", "operating_profit_margin", "ebitmargin", "operating margins", "operating profit margins", "ebit margins",
            "ebitda margin", "ebitda_margin", "ebitdamargin", "ebitda margins",
            "ebitda", "ebitdavalue",
            "net margin", "net profit margin", "net_margin", "netmargin", "net profit margins", "net margins", "profit margin", "profit margins",
            "roic", "return_on_inv_capital", "returnoninvestedcapital", "return on invested capital",
            "roe", "return_com_eqy", "returnonequity", "return on equity",
            "roa", "return_on_asset", "returnonassets", "return on assets",
            "return on capital", "return_on_cap", "return on capitals",
            "free cash flow", "free_cash_flow", "cf_free_cash_flow", "fcf", "free cash flows",
            "fcf yield", "fcf_yield", "fcf yields",
            "pe ratio", "p/e ratio", "p/e", "pe", "pe_ratio", "pe ratios", "p/e ratios",
            "ev/ebitda", "ev / ebitda", "ev_to_ttm_ebitda",
            "price/sales", "price / sales", "p/s", "p/s ratio", "pr_to_sales_ratio", "price/sales ratio", "price/sales ratios",
            "enterprise value", "enterprise_value", "enterprise values",
            "market cap", "market_cap", "market capitalization", "market caps",
            "current ratio", "current_ratio", "cur_ratio", "current ratios",
            "quick ratio", "quick_ratio", "quick ratios",
            "debt/equity", "debt to equity", "debt-to-equity", "tot_debt_to_tot_eqy", "debt/equity ratio", "debt/equity ratios",
            "asset turnover", "asset_turnover", "asset turnovers",
            "inventory turnover", "inventory_turnover", "invent_turn", "inventory turnovers",
            "revenue growth", "revenue_growth", "revenue growths",
            "ebitda growth", "ebitda_growth", "ebitda growths",
            "eps growth", "eps_growth", "eps growths"
        ]
        query_lower = query.lower()
        has_roic_metric = any(keyword in query_lower for keyword in roic_metrics_keywords)
        
        metrics = data.get("metrics")
        
        api_aliases = {
            "gross_margin": ["gross_margin", "grossMargin", "gross_profit_margin", "gross%"],
            "oper_margin": ["oper_margin", "operating_margin", "operatingMargin", "operating_profit_margin", "ebit_margin", "EBITMargin", "ebitmargin"],
            "ebitda_margin": ["ebitda_margin", "ebitdaMargin", "ebitda_profit_margin"],
            "profit_margin": ["profit_margin", "profitMargin", "net_margin", "netMargin", "net_profit_margin"],
            "return_on_inv_capital": ["return_on_inv_capital", "returnOnInvCapital", "roic", "ROIC"],
            "return_com_eqy": ["return_com_eqy", "returnComEqy", "roe", "ROE"],
            "return_on_asset": ["return_on_asset", "returnOnAsset", "roa", "ROA"],
            "return_on_cap": ["return_on_cap", "returnOnCap", "return_on_capital"],
            "cf_free_cash_flow": ["cf_free_cash_flow", "cfFreeCashFlow", "free_cash_flow", "freeCashFlow"],
            "fcf_yield": ["fcf_yield", "fcfYield", "free_cash_flow_yield"],
            "pe_ratio": ["pe_ratio", "peRatio", "p_e_ratio", "pe", "PE"],
            "ev_to_ttm_ebitda": ["ev_to_ttm_ebitda", "evToTtmEbitda", "ev_ebitda", "evEbitda", "ev/ebitda"],
            "pr_to_sales_ratio": ["pr_to_sales_ratio", "prToSalesRatio", "price_sales", "priceSales", "price/sales"],
            "enterprise_value": ["enterprise_value", "enterpriseValue", "ev", "EV"],
            "market_cap": ["market_cap", "marketCap", "market_capitalization", "marketCapitalization"],
            "cur_ratio": ["cur_ratio", "curRatio", "current_ratio", "currentRatio"],
            "quick_ratio": ["quick_ratio", "quickRatio"],
            "tot_debt_to_tot_eqy": ["tot_debt_to_tot_eqy", "totDebtToTotEqy", "debt_equity", "debtEquity", "debt/equity"],
            "asset_turnover": ["asset_turnover", "assetTurnover"],
            "invent_turn": ["invent_turn", "inventTurn", "inventory_turnover", "inventoryTurnover"],
            "revenue_growth": ["revenue_growth", "revenueGrowth", "sales_growth", "salesGrowth"],
            "ebitda_growth": ["ebitda_growth", "ebitdaGrowth"],
            "eps_growth": ["eps_growth", "epsGrowth"],
            "ebitda": ["ebitda", "EBITDA", "EBITDAValue", "ebitdaValue"]
        }

        def get_metric_val(record: dict, key: str) -> tuple[Any, str | None]:
            if record.get(key) is not None:
                return record[key], key
            for alias in api_aliases.get(key, []):
                if record.get(alias) is not None:
                    return record[alias], alias
                for k, v in record.items():
                    if k.lower() == alias.lower() and v is not None:
                        return v, k
            return None, None

        metric_keywords_map = {
            ("gross margin", "gross profit margin", "gross%", "gross_margin", "grossmargin", "gross margins", "gross profit margins"): "gross_margin",
            ("operating margin", "operating profit margin", "ebit margin", "ebit_margin", "oper_margin", "operating_margin", "operatingmargin", "operating_profit_margin", "ebitmargin", "operating margins", "operating profit margins", "ebit margins"): "oper_margin",
            ("ebitda margin", "ebitda_margin", "ebitdamargin", "ebitda margins"): "ebitda_margin",
            ("net margin", "net profit margin", "net_margin", "netmargin", "net profit margins", "net margins", "profit margin", "profit margins"): "profit_margin",
            ("roic", "return_on_inv_capital", "returnoninvestedcapital", "return on invested capital"): "return_on_inv_capital",
            ("roe", "return_com_eqy", "returnonequity", "return on equity"): "return_com_eqy",
            ("roa", "return_on_asset", "returnonassets", "return on assets"): "return_on_asset",
            ("return on capital", "return_on_cap", "return on capitals"): "return_on_cap",
            ("free cash flow", "free_cash_flow", "cf_free_cash_flow", "fcf", "free cash flows"): "cf_free_cash_flow",
            ("fcf yield", "fcf_yield", "fcf yields"): "fcf_yield",
            ("pe ratio", "p/e ratio", "p/e", "pe", "pe_ratio", "pe ratios", "p/e ratios"): "pe_ratio",
            ("ev/ebitda", "ev / ebitda", "ev_to_ttm_ebitda"): "ev_to_ttm_ebitda",
            ("price/sales", "price / sales", "p/s", "p/s ratio", "pr_to_sales_ratio", "price/sales ratio", "price/sales ratios"): "pr_to_sales_ratio",
            ("enterprise value", "enterprise_value", "enterprise values"): "enterprise_value",
            ("market cap", "market_cap", "market capitalization", "market caps"): "market_cap",
            ("current ratio", "current_ratio", "cur_ratio", "current ratios"): "cur_ratio",
            ("quick ratio", "quick_ratio", "quick ratios"): "quick_ratio",
            ("debt/equity", "debt to equity", "debt-to-equity", "tot_debt_to_tot_eqy", "debt/equity ratio", "debt/equity ratios"): "tot_debt_to_tot_eqy",
            ("asset turnover", "asset_turnover", "asset turnovers"): "asset_turnover",
            ("inventory turnover", "inventory_turnover", "invent_turn", "inventory turnovers"): "invent_turn",
            ("revenue growth", "revenue_growth", "revenue growths"): "revenue_growth",
            ("ebitda growth", "ebitda_growth", "ebitda growths"): "ebitda_growth",
            ("eps growth", "eps_growth", "eps growths"): "eps_growth",
            ("ebitda", "ebitdavalue"): "ebitda"
        }

        requested_and_missing = False
        resolved_details = []
        if has_roic_metric:
            for terms, key in metric_keywords_map.items():
                if any(term in query_lower for term in terms):
                    has_metric = False
                    field_found = None
                    if metrics:
                        sorted_years = sorted(metrics.keys(), key=int, reverse=True)
                        for fy in sorted_years:
                            record = metrics[fy]
                            val, field_name = get_metric_val(record, key)
                            if val is not None:
                                has_metric = True
                                field_found = field_name
                                break
                    if not has_metric:
                        requested_and_missing = True
                    resolved_details.append({
                        "user_metric": [t for t in terms if t in query_lower][0],
                        "normalized": key,
                        "roic_field": field_found,
                        "has_val": has_metric
                    })

            logger.info("roic.provider_response: ticker=%s years=%s", resolved_ticker, list(metrics.keys()) if metrics else None)
            for detail in resolved_details:
                logger.info(
                    "roic.metric_resolution: user_metric=%r normalized=%s roic_field=%s has_val=%s fallback=%s",
                    detail["user_metric"],
                    detail["normalized"],
                    detail["roic_field"],
                    detail["has_val"],
                    "none" if detail["has_val"] else "not_available"
                )

        if has_roic_metric and (requested_and_missing or not resolved_ticker or data.get("error")):
            number = start_number + len(sources)
            lines.append(f"[{number}] ROIC.ai Error:\nThe requested metric is not available from the ROIC AI data provider.")
            sources.append({
                "title": "ROIC.ai Data Provider Error",
                "link": f"https://roic.ai/company/{resolved_ticker}" if resolved_ticker else "https://roic.ai/",
                "snippet": "The requested metric is not available from the ROIC AI data provider.",
                "publisher": "ROIC.ai",
                "source_type": "roic_ai",
                "ticker": resolved_ticker
            })
            return "\n\n".join(lines), sources

        if data.get("error") or not results:
            return "", []

        for result in results:
            number = start_number + len(sources)
            title = result.get("company") or result.get("ticker")
            details = [f"[{number}] {title}", f"Ticker: {result.get('ticker')}", f"Exchange: {result.get('exchange') or 'Unknown'}"]
            lines.append("\n".join(details))
            sources.append({"title": title, "link": f"https://roic.ai/company/{result.get('ticker')}" if result.get('ticker') else "https://roic.ai/", "snippet": "Roic.ai company search result.", "publisher": "Roic.ai", "source_type": "roic_ai", "ticker": result.get("ticker")})

        if metrics:
            sorted_years = sorted(metrics.keys(), key=int, reverse=True)[:3]
            number = start_number + len(sources)
            
            metric_display_names = {
                "gross_margin": "Gross Margin",
                "oper_margin": "Operating Margin",
                "ebitda_margin": "EBITDA Margin",
                "profit_margin": "Net Margin",
                "return_on_inv_capital": "ROIC",
                "return_com_eqy": "ROE",
                "return_on_asset": "ROA",
                "return_on_cap": "Return on Capital",
                "cf_free_cash_flow": "Free Cash Flow",
                "fcf_yield": "FCF Yield",
                "pe_ratio": "PE Ratio",
                "ev_to_ttm_ebitda": "EV/EBITDA",
                "pr_to_sales_ratio": "Price/Sales",
                "enterprise_value": "Enterprise Value",
                "market_cap": "Market Cap",
                "cur_ratio": "Current Ratio",
                "quick_ratio": "Quick Ratio",
                "tot_debt_to_tot_eqy": "Debt/Equity",
                "asset_turnover": "Asset Turnover",
                "invent_turn": "Inventory Turnover",
                "revenue_growth": "Revenue Growth",
                "ebitda_growth": "EBITDA Growth",
                "eps_growth": "EPS Growth",
                "ebitda": "EBITDA"
            }
            
            def format_value(name: str, val: Any) -> str:
                if val is None:
                    return "N/A"
                if not isinstance(val, (int, float)):
                    return str(val)
                percentage_metrics = {
                    "gross_margin", "ebitda_margin", "oper_margin", "profit_margin",
                    "return_com_eqy", "return_on_asset", "return_on_cap", "return_on_inv_capital",
                    "fcf_yield", "revenue_growth", "ebitda_growth", "eps_growth"
                }
                currency_metrics = {
                    "ebitda", "cf_free_cash_flow", "enterprise_value", "market_cap"
                }
                if name in percentage_metrics:
                    return f"{val:.2f}%"
                if name in currency_metrics:
                    abs_val = abs(val)
                    prefix = "$"
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
                        num_str = f"{abs_val:,.2f}"
                        suffix = ""
                    sign = "-" if val < 0 else ""
                    return f"{sign}{prefix}{num_str}{suffix}"
                return f"{val:.2f}"

            metric_lines = [f"[{number}] ROIC.ai Financial Metrics for {data.get('query')} ({resolved_ticker}):"]
            for fy in sorted_years:
                record = metrics[fy]
                metric_lines.append(f"Fiscal Year: {fy} (Currency: {record.get('currency') or 'USD'}):")
                for key, display_name in metric_display_names.items():
                    val, _ = get_metric_val(record, key)
                    if val is not None:
                        formatted_val = format_value(key, val)
                        metric_lines.append(f"  - {display_name}: {formatted_val}")
            
            lines.append("\n".join(metric_lines))
            sources.append({
                "title": f"ROIC.ai Financial Metrics — {resolved_ticker}",
                "link": f"https://roic.ai/company/{resolved_ticker}" if resolved_ticker else "https://roic.ai/",
                "snippet": f"Consolidated financial metrics from ROIC.ai for {resolved_ticker}.",
                "publisher": "ROIC.ai",
                "source_type": "roic_ai",
                "ticker": resolved_ticker
            })

        transcript = data.get("latest_transcript")
        if transcript:
            terms = {word for word in re.findall(r"[a-z]{4,}", query.lower())} - {"what", "about", "latest", "management", "company"}
            paragraphs = [p.strip() for p in transcript["content"].split("\n\n") if p.strip()]
            relevant = [p for p in paragraphs if any(term in p.lower() for term in terms)]
            excerpt = "\n\n".join((relevant or paragraphs)[:8])[:12000]
            number = start_number + len(sources)
            lines.append(f"[{number}] Latest earnings-call transcript — {transcript['ticker']} Q{transcript.get('quarter')} {transcript.get('year')}\nDate: {transcript.get('date')}\n{excerpt}")
            sources.append({"title": f"Roic.ai earnings-call transcript — {transcript['ticker']}", "link": f"https://roic.ai/transcripts/{transcript['ticker']}" if transcript.get('ticker') else "https://roic.ai/", "snippet": excerpt[:500], "publisher": "Roic.ai", "source_type": "roic_ai", "ticker": transcript["ticker"], "navigation_type": "api", "navigation_target": f"/v1/citations/roic/transcript/{transcript['ticker']}"})
        return "\n\n".join(lines), sources

    @staticmethod
    def classify_sec_edgar_ticker(client, question: str, model: str) -> str | None:
        """Backward-compatible single-ticker SEC router."""
        tickers = RAGService.classify_sec_edgar_tickers(client, question, model)
        return tickers[0] if tickers else None

    @staticmethod
    def classify_sec_edgar_tickers(client, question: str, model: str) -> list[str]:
        """Return one or two tickers for SEC questions, including comparisons."""
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sec_edgar_classification_prompt()},
                    {"role": "user", "content": question},
                ],
                max_tokens=16,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            match = re.fullmatch(
                r"YES:([A-Z][A-Z0-9.\-]{0,9}(?:,[A-Z][A-Z0-9.\-]{0,9})?)", answer
            )
            tickers = match.group(1).split(",") if match else []
            logger.info("rag.classify_sec: tickers=%s question_len=%d", tickers, len(question))
            return tickers[:2]
        except Exception:
            logger.exception("rag.classify_sec: classification failed; skipping SEC retrieval")
            return []

    @staticmethod
    def sec_request_options(question: str) -> dict:
        """Derive deterministic SEC tool options from an already-routed query."""
        lowered = question.lower()

        # Check if the query is asking about the latest filing itself or financial metrics
        metric_keywords = {
            "revenue", "sales", "net income", "profit", "assets", "liabilities",
            "equity", "cash", "eps", "earnings per share", "operating income", "free cash flow"
        }
        has_metrics = any(term in lowered for term in metric_keywords)

        filing_keywords = {
            "filing", "filings", "submission", "submissions", "filed", "history",
            "date", "link", "links", "url", "urls", "recent", "latest", "last", "form"
        }
        has_filings = any(term in lowered for term in filing_keywords)

        intent_phrases = ["latest sec filing", "latest submission", "most recent filing"]
        is_filing_intent = any(phrase in lowered for phrase in intent_phrases)

        if is_filing_intent or (has_filings and not has_metrics):
            request_type = "submissions"
        else:
            request_type = "facts"

        years_match = re.search(r"last\s+(\d+)\s+years?", lowered)
        word_years = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        if not years_match:
            years_match = re.search(r"last\s+(one|two|three|four|five)\s+years?", lowered)
        years = None
        if years_match:
            value = years_match.group(1)
            years = int(value) if value.isdigit() else word_years[value]

        if years or any(term in lowered for term in ("annual", "10-k", "yearly")):
            period = "annual"
        elif any(term in lowered for term in ("quarter", "quarterly", "10-q")):
            period = "quarterly"
        else:
            period = "latest"

        metrics: list[str] = []
        if "revenue" in lowered or "sales" in lowered:
            metrics.append("revenue")
        if "net income" in lowered or "profit" in lowered:
            metrics.append("net_income")
        if "operating income" in lowered:
            metrics.append("operating_income")
        if "balance sheet" in lowered:
            metrics.extend(["assets", "liabilities", "shareholders_equity"])
        else:
            if "assets" in lowered:
                metrics.append("assets")
            if "liabilities" in lowered:
                metrics.append("liabilities")
            if "equity" in lowered:
                metrics.append("shareholders_equity")
        if "free cash flow" in lowered:
            metrics.append("free_cash_flow")
        elif "cash flow" in lowered:
            metrics.extend(["operating_cash_flow", "investing_cash_flow", "financing_cash_flow"])
        elif "cash" in lowered:
            metrics.append("cash_and_equivalents")
        if "eps" in lowered or "earnings per share" in lowered:
            metrics.extend(["basic_eps", "diluted_eps"])

        return {
            "period": period,
            "metrics": list(dict.fromkeys(metrics)) or None,
            "years": years,
            "request_type": request_type
        }

    @staticmethod
    def build_financial_news_context(
        news: dict, *, start_number: int
    ) -> tuple[str, list[dict]]:
        """Make normalized Alpha Vantage news compact and citation-addressable."""
        summary_stats = news.get("sentiment_summary") or {}
        ticker = news.get("ticker") or "Unknown"

        # Build structured headers exactly as requested
        header_lines = [
            "Company:",
            ticker,
            "",
            "Articles analyzed:",
            str(summary_stats.get("articles_analyzed", 0)),
            "",
            "Average sentiment:",
            f"{summary_stats.get('average_sentiment_score'):.2f}" if summary_stats.get("average_sentiment_score") is not None else "N/A",
            "",
            "Overall sentiment:",
            str(summary_stats.get("overall_sentiment_label") or "Neutral"),
            "",
            "Positive:",
            str(summary_stats.get("positive", 0)),
            "",
            "Neutral:",
            str(summary_stats.get("neutral", 0)),
            "",
            "Negative:",
            str(summary_stats.get("negative", 0)),
            "",
            "Articles:",
            ""
        ]

        lines: list[str] = ["\n".join(header_lines)]
        sources: list[dict] = []
        for article in news.get("articles") or []:
            url = (article.get("url") or "").strip()
            if not url:
                continue
            number = start_number + len(sources)
            title = article.get("title") or "Untitled article"
            summary = article.get("summary") or ""

            pub_date = article.get("published_at") or "Unknown"
            formatted_date = FinancialNewsMCPTool.format_pub_date(pub_date)

            article_block = [
                f"[{number}]",
                "Title:",
                title,
                "",
                "Published:",
                formatted_date,
                "",
                "Source:",
                article.get("publisher") or "Alpha Vantage",
                "",
                "Summary:",
                summary,
                "",
                "Sentiment:",
                article.get("sentiment") or "Unknown",
                "",
                "Relevance:",
                str(article.get("relevance_score") or "0.0"),
                "",
                "URL:",
                url
            ]
            lines.append("\n".join(article_block))
            sources.append(
                {
                    "title": title,
                    "link": url,
                    "snippet": summary,
                    "date": pub_date,
                    "publisher": article.get("publisher") or "Alpha Vantage",
                    "source_type": "financial_news",
                    "ticker": ticker,
                }
            )
        return "\n\n".join(lines), sources

    @staticmethod
    def build_companies_house_context(
        data: dict, *, start_number: int
    ) -> tuple[str, list[dict]]:
        """Format Companies House search results and resolved profile into a combined cited context block."""
        if not data:
            return "", []

        number = start_number
        query = data.get("query") or "Unknown query"

        # If data represents no results found
        if "message" in data:
            link = "https://find-and-update.company-information.service.gov.uk"
            sources = [{
                "title": "Companies House: No Records Found",
                "link": link,
                "snippet": f"No Companies House records were found matching '{query}'.",
                "publisher": "Companies House",
                "source_type": "companies_house",
            }]
            return f"[{number}] Companies House Registry:\nNo Companies House records were found matching '{query}'.", sources

        summary = data.get("summary")
        resolved_number = data.get("resolved_company_number")

        lines = [
            f"[{number}] Companies House Registry Results for query: '{query}'"
        ]

        if summary:
            company_name = summary.get("company_name") or "Unknown Company"
            comp_num = summary.get("company_number") or "Unknown"
            status = summary.get("status") or "Unknown"
            company_type = summary.get("company_type") or "Unknown"
            inc_date = summary.get("incorporated_on") or "Unknown"
            
            office = summary.get("registered_office") or {}
            street = office.get("street") or "N/A"
            city = office.get("city") or "N/A"
            postal_code = office.get("postal_code") or "N/A"
            country = office.get("country") or "N/A"

            lines.extend([
                "\nResolved Company:",
                f"  - Company Name: {company_name}",
                f"  - Company Number: {comp_num}",
                f"  - Company Type: {company_type}",
                f"  - Resolution Method: {summary.get('resolution_method')}",
                "\nCompany Status:",
                f"  - Status: {status}",
            ])
            
            accs = summary.get("accounts") or {}
            conf = summary.get("confirmation_statement") or {}
            accs_due = accs.get("next_due") or "N/A"
            accs_overdue = "Overdue" if accs.get("overdue") else "Up to date"
            conf_due = conf.get("next_due") or "N/A"
            conf_overdue = "Overdue" if conf.get("overdue") else "Up to date"

            lines.extend([
                f"  - Accounts Status: {accs_overdue} (Next due: {accs_due})",
                f"  - Confirmation Statement Status: {conf_overdue} (Next due: {conf_due})",
                f"\nIncorporated: {inc_date}",
            ])

            history = summary.get("name_history") or []
            if history:
                lines.append("\nCompany Name History:")
                for name_entry in history:
                    lines.append(f"  - Previous Name: {name_entry.get('name')} (Effective: {name_entry.get('effective_from')} to {name_entry.get('ceased_on')})")

            lines.extend([
                "\nRegistered Office:",
                f"  - Street: {street}",
                f"  - City: {city}",
                f"  - Postal Code: {postal_code}",
                f"  - Country: {country}",
            ])

            sic_codes = summary.get("sic_codes") or []
            if sic_codes:
                lines.append("\nSIC Codes:")
                for sic in sic_codes:
                    lines.append(f"  - {sic.get('code')}: {sic.get('description')}")
            else:
                lines.append("\nSIC Codes: None listed")

            officers = summary.get("officers") or []
            if officers:
                lines.append("\nCurrent Officers:")
                for off in officers:
                    lines.append(f"  - Name: {off.get('name')}, Role: {off.get('role')}, Appointed On: {off.get('appointed_on')}")
            else:
                lines.append("\nCurrent Officers: None listed")

            lines.append(f"\nSource: {summary.get('source_url')}")

            other_matches = summary.get("other_matches") or []
            if other_matches:
                lines.append("\nOther Search Matches:")
                for item in other_matches:
                    title = item.get("title") or "Unknown"
                    num = item.get("company_number") or "Unknown"
                    status_str = item.get("company_status") or "Unknown"
                    addr = item.get("address_snippet") or "Unknown Address"
                    lines.append(f"  - {title} (Company Number: {num}, Status: {status_str}, Address: {addr})")
        else:
            lines.append("No resolved company profile could be retrieved.")

        if resolved_number:
            link = f"https://find-and-update.company-information.service.gov.uk/company/{resolved_number}"
            company_name_title = "Unknown Company"
            if summary:
                company_name_title = summary.get("company_name") or "Unknown Company"
            title_text = f"Companies House: {company_name_title}"
        else:
            link = "https://find-and-update.company-information.service.gov.uk"
            title_text = f"Companies House Search: {query}"

        sources = [{
            "title": title_text,
            "link": link,
            "snippet": f"Companies House Registry search and profile results for query '{query}'.",
            "publisher": "Companies House",
            "source_type": "companies_house",
            "company_number": resolved_number
        }]

        return "\n".join(lines), sources

    @staticmethod
    def build_sec_context(sec_data: dict, *, start_number: int) -> tuple[str, list[dict]]:
        """Format normalized SEC facts as one cited, compact context block."""
        if sec_data.get("error"):
            return "", []

        cik = sec_data.get("cik")
        company = sec_data.get("company")
        ticker = sec_data.get("ticker")
        cik_numeric = str(int(cik)) if cik else ""

        if sec_data.get("request_type") == "submissions":
            latest = sec_data.get("latest_filing") or {}
            form = latest.get("form") or "Unknown"
            filing_date = latest.get("filing_date") or "Unknown"
            period_end = latest.get("period_end") or "Unknown"
            description = latest.get("description") or "Unknown"
            link = latest.get("link") or "Unknown"

            lines = [
                f"[{start_number}] Company: {company}",
                f"Ticker: {ticker}",
                f"CIK: {cik}",
                "Request Type: Submissions",
                f"Filing Type: {form}",
                f"Filing Date: {filing_date}",
                f"Period End: {period_end}",
                f"Filing Description: {description}",
                f"SEC Filing Link: {link}",
                f"Brief Summary: This is the latest official SEC submission ({form}) filed by {company} on {filing_date} for the period ended {period_end}.",
            ]

            if sec_data.get("latest_financial_filing"):
                lff = sec_data["latest_financial_filing"]
                lines.append("\nLatest Financial Filing (10-Q or 10-K):")
                lines.append(f"Filing Type: {lff.get('form') or 'Unknown'}")
                lines.append(f"Filing Date: {lff.get('filing_date') or 'Unknown'}")
                lines.append(f"Period End: {lff.get('period_end') or 'Unknown'}")
                lines.append(f"Filing Description: {lff.get('description') or 'Unknown'}")
                lines.append(f"SEC Filing Link: {lff.get('link') or 'Unknown'}")

            if sec_data.get("recent_filings"):
                lines.append("\nRecent Filings History:")
                for rf in sec_data["recent_filings"][:5]:
                    lines.append(f"- Form {rf.get('form')} filed on {rf.get('filing_date')} (Period End: {rf.get('period_end')})")

            f_url = latest.get("filing_url") or link
            if not f_url or f_url == "Unknown":
                f_url = f"https://www.sec.gov/edgar/data/{cik_numeric}" if cik_numeric else "https://www.sec.gov/edgar/searchedgar/companysearch"

            return "\n".join(lines), [{
                "title": f"SEC EDGAR: {company} ({form})",
                "link": f_url,
                "snippet": f"Company: {company}\nFiling Type: {form}\nFiled: {filing_date}\nPeriod End: {period_end}",
                "publisher": "SEC EDGAR",
                "source_type": "sec_edgar",
                "ticker": ticker,
            }]

        financials = sec_data.get("financials") or {}
        lines = [
            f"[{start_number}] Company: {company}",
            f"Ticker: {ticker}",
            f"CIK: {cik}",
            f"Report: {sec_data.get('report_type')} ended {sec_data.get('period_end')}",
            f"Filed: {sec_data.get('filed_date')}",
        ]
        has_financials = False
        for name, data in financials.items():
            if not data:
                continue
            has_financials = True
            label = name.replace("_", " ").title()

            if isinstance(data, dict) and "error" in data:
                lines.append(f"{label}: {data['error']}")
            elif "history" in data:
                values = []
                for item in data["history"]:
                    val_display = item.get("formatted_value") or f"{item['value']} {item.get('unit') or item.get('currency')}"
                    metadata_str = f"Filing Type: {item.get('filing_type') or item.get('report_type')}, Fiscal Year: {item.get('fiscal_year')}, Fiscal Quarter: {item.get('fiscal_quarter') or 'N/A'}, Period End: {item.get('period_end')}, Filed Date: {item.get('filed_date')}"
                    val_str = f"{item['period_end']}: {val_display} ({metadata_str}, Source: {item.get('sec_source')})"
                    values.append(val_str)
                lines.append(f"{label} History: " + "; ".join(values))
            else:
                val_display = data.get("formatted_value") or f"{data['value']} {data.get('unit') or data.get('currency')}"
                metadata_str = f"Filing Type: {data.get('filing_type') or data.get('report_type')}, Fiscal Year: {data.get('fiscal_year')}, Fiscal Quarter: {data.get('fiscal_quarter') or 'N/A'}, Period End: {data.get('period_end')}, Filed Date: {data.get('filed_date')}"
                val_str = f"{val_display} ({metadata_str}, Source: {data.get('sec_source')})"
                if "warning" in data:
                    val_str += f" (Warning: {data['warning']})"
                lines.append(f"{label}: {val_str}")

        if not has_financials:
            return "", []

        # Extract unique filing URLs from the fact records
        unique_filings = {}
        for name, data in financials.items():
            if not isinstance(data, dict):
                continue
            if "history" in data:
                for item in data["history"]:
                    f_key = (item.get("filing_type"), item.get("filed_date"))
                    if f_key not in unique_filings and item.get("filing_url"):
                        unique_filings[f_key] = item
            elif data.get("filing_url"):
                f_key = (data.get("filing_type"), data.get("filed_date"))
                if f_key not in unique_filings:
                    unique_filings[f_key] = data

        sec_sources = []
        if unique_filings:
            for (f_type, f_date), item in unique_filings.items():
                p_end = item.get("period_end")
                f_url = item.get("filing_url") or item.get("filing_index_url") or item.get("filing_html_url")
                sec_sources.append({
                    "title": f"SEC EDGAR: {company} ({f_type})",
                    "link": f_url,
                    "snippet": f"Company: {company}\nFiling Type: {f_type}\nFiled: {f_date}\nPeriod End: {p_end}",
                    "publisher": "SEC EDGAR",
                    "source_type": "sec_edgar",
                    "ticker": ticker,
                })
        else:
            fallback_url = f"https://www.sec.gov/edgar/data/{cik_numeric}" if cik_numeric else "https://www.sec.gov/edgar/searchedgar/companysearch"
            sec_sources.append({
                "title": f"SEC EDGAR: {company}",
                "link": fallback_url,
                "snippet": f"Company: {company}\nOfficial SEC EDGAR filings page.",
                "publisher": "SEC EDGAR",
                "source_type": "sec_edgar",
                "ticker": ticker,
            })

        return "\n".join(lines), sec_sources

    @staticmethod
    def build_sec_contexts(sec_results: list[dict], *, start_number: int) -> tuple[str, list[dict]]:
        """Build separate, sequentially cited SEC blocks for each company."""
        contexts: list[str] = []
        sources: list[dict] = []
        for result in sec_results:
            context, result_sources = RAGService.build_sec_context(
                result, start_number=start_number + len(sources)
            )
            if context:
                contexts.append(context)
                sources.extend(result_sources)
        return "\n\n---\n\n".join(contexts), sources

    @staticmethod
    def _format_scalar_fields(data: dict, skip: tuple[str, ...]) -> str:
        """Flatten the scalar (str/number) fields of a SerpAPI widget to text.

        Nested objects, images and link metadata (in ``skip``) are ignored so the
        digest stays compact and LLM-friendly.
        """
        parts: list[str] = []
        for key, value in data.items():
            if key in skip:
                continue
            if isinstance(value, (str, int, float)) and str(value).strip():
                label = key.replace("_", " ").capitalize()
                parts.append(f"{label}: {value}")
        return "\n".join(parts)

    @classmethod
    def _extract_direct_answer(cls, results: dict) -> str:
        """Pull the direct-answer widgets (answer box / knowledge graph) out of a
        SerpAPI response. This is where live data (weather, prices, scores,
        definitions) lives — organic results usually only have links."""
        sections: list[str] = []
        _skip = (
            "type",
            "thumbnail",
            "link",
            "displayed_link",
            "images",
            "list",
            "table",
            "source",
            "serpapi_link",
            "people_also_search_for",
        )

        answer_box = results.get("answer_box")
        if isinstance(answer_box, dict):
            text = cls._format_scalar_fields(answer_box, skip=_skip)
            if text:
                sections.append("Direct answer:\n" + text)

        knowledge_graph = results.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            text = cls._format_scalar_fields(
                knowledge_graph,
                skip=_skip + ("kgmid", "knowledge_graph_search_link"),
            )
            if text:
                sections.append("Knowledge graph:\n" + text)

        return "\n\n".join(sections)

    @classmethod
    def web_search(
        cls, query: str, max_results: int | None = None, start_number: int = 1
    ) -> tuple[str, list[dict]]:
        """Run a web search and return ``(context_string, sources)``.

        ``start_number`` is the first citation marker to assign. Document chunks
        are numbered first ([1]..[n]), so web results continue from ``n + 1`` and
        the LLM sees ONE unbroken [n] sequence across the whole CONTEXT.

        Uses the configured engine (Google by default) and extracts, in priority
        order: the direct-answer widgets (answer box / knowledge graph — where
        live data such as weather and prices lives) followed by the top organic
        results (title + snippet + link). Best-effort: any failure (or no
        results) returns ``("", [])`` so the chat flow can continue.
        """
        max_results = max_results or settings.WEB_SEARCH_RESULTS
        try:
            params = {
                "engine": settings.WEB_SEARCH_ENGINE,
                "q": query,
                "gl": settings.WEB_SEARCH_COUNTRY,
                "hl": settings.WEB_SEARCH_LANGUAGE,
                "num": max_results,
            }
            results = serp_client.search(params)
        except Exception:
            logger.exception(
                "rag.web_search: search failed for query_len=%d", len(query)
            )
            return "", []

        sections: list[str] = []

        direct = cls._extract_direct_answer(results)
        if direct:
            sections.append(direct)

        organic = (results.get("organic_results") or [])[:max_results]
        lines: list[str] = []
        sources: list[dict] = []
        # Number link-bearing results with [n] markers, continuing the document
        # chunk sequence, so the LLM can cite web sources the same way it cites
        # uploaded documents. CitationService maps the markers back to `sources`.
        for result in organic:
            link = (result.get("link") or "").strip()
            if not link:
                continue
            title = (result.get("title") or "").strip()
            snippet = (result.get("snippet") or "").strip()
            number = start_number + len(sources)
            lines.append(f"[{number}] {title}\n{snippet}\nURL: {link}")
            sources.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "date": result.get("date"),
                    "source_type": "web",
                }
            )
        if lines:
            sections.append("Top results:\n" + "\n\n".join(lines))

        context = "\n\n".join(sections)
        logger.info(
            "rag.web_search: engine=%s query_len=%d direct_answer=%s organic=%d",
            settings.WEB_SEARCH_ENGINE,
            len(query),
            bool(direct),
            len(sources),
        )
        return context, sources


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return the shared :class:`RAGService` singleton."""
    return RAGService(embedding_service=get_embedding_service())
