from typing import List
import logging
from concurrent.futures import ThreadPoolExecutor
from server.config import settings
from server.utils.chat_helpers import get_llm_kwargs, is_reasoning_model
from server.utils.llm_client import execute_with_llm_retry

logger = logging.getLogger(__name__)

class DeepResearchOrchestrator:
    def __init__(self, client, rag_service, db, current_user, session, generation=None):
        self.client = client
        self.rag = rag_service
        self.db = db
        self.current_user = current_user
        self.session = session
        # The generation handle for this turn, when the caller has one. Deep
        # research is the longest-running stage in the pipeline -- several
        # rounds of LLM planning and retrieval -- so it has to be interruptible
        # or Stop does nothing for as long as it runs.
        self.generation = generation

    def _check_cancelled(self, stage: str) -> None:
        if self.generation is not None:
            self.generation.raise_if_cancelled(f"deep_research.{stage}")

    def generate_sub_queries(self, main_query: str) -> List[str]:
        prompt = f"""Analyze the following research request and determine the user's objective (e.g., Information request, Comparison, Recommendation, Investigation, Market/Technical/Financial/Legal research). 
Based on this objective, break the request into 3-5 logical research topics or sub-questions that must be independently investigated to fully address the prompt.

Request: {main_query}

Format your response strictly as a simple list of topics/questions, one per line, with no bullet points or numbering."""
        try:
            kwargs = get_llm_kwargs(
                settings.DEEP_RESEARCH_MODEL,
                temperature=0.3,
                reasoning_effort=settings.DEEP_RESEARCH_REASONING_EFFORT,
            )
            def _do_sub_queries():
                try:
                    return self.client.chat.completions.create(
                        model=settings.DEEP_RESEARCH_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                except Exception as api_err:
                    if "temperature" in str(api_err).lower() and "temperature" in kwargs:
                        retry_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
                        return self.client.chat.completions.create(
                            model=settings.DEEP_RESEARCH_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            **retry_kwargs
                        )
                    raise api_err

            completion = execute_with_llm_retry(
                _do_sub_queries,
                max_attempts=3,
                initial_delay=1.0,
                operation_name="Deep Research Sub-queries",
            )
            response = completion.choices[0].message.content or ""
            queries = [q.strip("- *").strip() for q in response.split("\n") if q.strip()]
            return queries if queries else [main_query]
        except Exception:
            return [main_query]

    def execute_retrieval_for_query(self, query: str, chat_id: str, document_ids: list, history: list) -> str:
        self._check_cancelled("sub_query")
        optimized_prompt, conversation_company, _ = self.rag.rewrite_query(
            self.client, query, history, model=settings.DEEP_RESEARCH_MODEL
        )
        resolved_company, resolved_ticker, is_sec_registrant = self.rag.resolve_company(
            self.client, conversation_company or optimized_prompt, model=settings.DEEP_RESEARCH_MODEL
        )
        
        is_document_question = False
        if document_ids:
            is_document_question = self.rag.classify_requires_document(
                self.client, question=optimized_prompt, model=settings.DEEP_RESEARCH_MODEL
            )
            
        intent = self.rag.classify_mcp_intent(self.client, question=optimized_prompt, model=settings.DEEP_RESEARCH_MODEL)
        
        sec_tickers = []
        sec_options = {}
        roic_query = None
        financial_ticker = None
        companies_house_query = None
        requires_web = False
        
        active_id = resolved_ticker or resolved_company
        lowered = optimized_prompt.lower()
        
        if active_id:
            # SEC
            sec_keywords = ["10-k", "10-q", "8-k", "filing", "annual report", "quarterly report", "sec submission", "company filing"]
            if intent == "SEC_FILINGS" or any(kw in lowered for kw in sec_keywords):
                if is_sec_registrant:
                    sec_tickers = [active_id]
                    sec_options = self.rag.sec_request_options(optimized_prompt)
            
            # Alpha Vantage
            alpha_keywords = ["stock price", "market price", "revenue", "eps", "financial metrics", "market data", "dividend", "financial performance"]
            if intent == "FINANCIAL_MARKET_DATA" or any(kw in lowered for kw in alpha_keywords):
                if resolved_ticker:
                    financial_ticker = resolved_ticker
                    
            # ROIC
            roic_keywords = ["company fundamental", "balance sheet", "income statement", "cash flow", "valuation", "ratio", "business fundamental"]
            if intent in ["ROIC", "COMPANY_INFORMATION"] or any(kw in lowered for kw in roic_keywords):
                roic_query = active_id
                
            # Companies House
            ch_keywords = ["uk company", "companies house", "british company", "officers", "filing history", "uk registration"]
            if intent == "COMPANIES_HOUSE" or any(kw in lowered for kw in ch_keywords):
                companies_house_query = active_id
                
        requires_web = self.rag.classify_requires_web_search(
            self.client, question=optimized_prompt, model=settings.DEEP_RESEARCH_MODEL
        )
        
        futures = {}
        user_id_str = str(self.current_user.id)
        ws_id = self.session.workspace_id
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            if is_document_question:
                futures["rag"] = executor.submit(
                    self.rag.retrieve_documents_context,
                    self.db, False, "", user_id_str, document_ids, ws_id, optimized_prompt, chat_id
                )
            if sec_tickers:
                futures["sec"] = executor.submit(
                    self.rag.retrieve_sec_company_facts_context,
                    chat_id, user_id_str, ws_id, sec_tickers, sec_options
                )
            if roic_query:
                futures["roic"] = executor.submit(
                    self.rag.retrieve_roic_ai_context,
                    chat_id, user_id_str, ws_id, roic_query
                )
            if financial_ticker:
                futures["news"] = executor.submit(
                    self.rag.retrieve_financial_news_context,
                    chat_id, user_id_str, ws_id, financial_ticker
                )
            if companies_house_query:
                futures["companies_house"] = executor.submit(
                    self.rag.retrieve_companies_house_context,
                    chat_id, user_id_str, ws_id, companies_house_query
                )
                
            retrieved_chunks = None
            if "rag" in futures:
                try:
                    _, retrieved_chunks = futures["rag"].result()
                except Exception:
                    pass
            sec_res = None
            if "sec" in futures:
                try:
                    sec_res = futures["sec"].result()
                except Exception:
                    pass
            roic_res = None
            if "roic" in futures:
                try:
                    roic_res = futures["roic"].result()
                except Exception:
                    pass
            news_res = None
            if "news" in futures:
                try:
                    news_res = futures["news"].result()
                except Exception:
                    pass
            ch_res = None
            if "companies_house" in futures:
                try:
                    ch_res = futures["companies_house"].result()
                except Exception:
                    pass
                    
        news_src, sec_src, roic_src, ch_src, web_sources = [], [], [], [], []
        contexts = []
        if retrieved_chunks:
            contexts.append(self.rag.build_context_with_citations(retrieved_chunks) or "")
        if news_res:
            ctx, news_src = self.rag.build_financial_news_context(news_res, start_number=1)
            if ctx: contexts.append(ctx)
        if sec_res:
            ctx, sec_src = self.rag.build_sec_contexts(sec_res if isinstance(sec_res, list) else [sec_res], start_number=1)
            if ctx: contexts.append(ctx)
        if roic_res:
            ctx, roic_src = self.rag.build_roic_context(roic_res, start_number=1, query=optimized_prompt)
            if ctx: contexts.append(ctx)
        if ch_res:
            ctx, ch_src = self.rag.build_companies_house_context(ch_res, start_number=1)
            if ctx: contexts.append(ctx)
            
        mcp_executed = bool(news_res or sec_res or roic_res or ch_res)
        has_data = any([
            news_res and self.rag.has_useful_data(news_res),
            sec_res and self.rag.has_useful_data(sec_res),
            roic_res and self.rag.has_useful_data(roic_res),
            ch_res and self.rag.has_useful_data(ch_res)
        ])
        
        if mcp_executed and not has_data and sec_res is None:
            requires_web = True
            
        if requires_web:
            web_ctx, websrc = self.rag.web_search(optimized_prompt, start_number=1)
            if web_ctx:
                contexts.append(web_ctx)
                if websrc: web_sources.extend(websrc)
                
        return "\n\n".join(contexts), retrieved_chunks, news_src, sec_src, roic_src, ch_src, web_sources

    def validate_and_consolidate(self, all_contexts: List[str]) -> str:
        if not all_contexts:
            return ""
        
        combined = "\n\n=== SOURCE BATCH ===\n\n".join([c for c in all_contexts if c])
        if not combined.strip():
            return ""
            
        prompt = f"""You are an expert research analyst. You have collected evidence from multiple independent retrieval pipelines.

Your task is to validate and consolidate this evidence into a single, unified, highly structured research context.

Instructions:
1. Validate Evidence: Identify duplicate information, supporting evidence, conflicting evidence, weak evidence, and missing evidence.
2. Prioritize Evidence: Prefer multiple independent confirmations over primary sources, over official documentation, over research papers, over reliable web sources, over single-source claims.
3. Remove Duplicates: Merge identical findings.
4. Handle Contradictions: Do not ignore conflicting information. Explicitly explain it.
5. Identify Evidence Gaps: Highlight weak evidence or missing information.
6. PRESERVE ALL CITATIONS (e.g. [1], [2], [company_facts], [financial_news]). Do not lose citation markers.
7. Provide a clean, consolidated text that combines findings logically (relationships, trends, patterns, contradictions, trade-offs, risks) to be directly used for generating a final research report.

Raw Collected Evidence:
{combined}
"""
        try:
            kwargs = get_llm_kwargs(
                settings.DEEP_RESEARCH_MODEL,
                temperature=0.3,
                reasoning_effort=settings.DEEP_RESEARCH_REASONING_EFFORT,
            )
            def _do_synth():
                try:
                    return self.client.chat.completions.create(
                        model=settings.DEEP_RESEARCH_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                except Exception as api_err:
                    if "temperature" in str(api_err).lower() and "temperature" in kwargs:
                        retry_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
                        return self.client.chat.completions.create(
                            model=settings.DEEP_RESEARCH_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            **retry_kwargs
                        )
                    raise api_err

            completion = execute_with_llm_retry(
                _do_synth,
                max_attempts=3,
                initial_delay=1.0,
                operation_name="Deep Research Synthesis",
            )
            return completion.choices[0].message.content or combined
        except Exception:
            return combined

    def execute(self, initial_query: str, chat_id: str, document_ids: list, history: list, initial_context: str) -> tuple:
        self._check_cancelled("plan")
        sub_queries = self.generate_sub_queries(initial_query)
        self._check_cancelled("dispatch")
        contexts = [initial_context] if initial_context else []
        
        all_chunks, all_news, all_sec, all_roic, all_ch, all_web = [], [], [], [], [], []
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(sub_queries))) as executor:
            future_to_sq = {executor.submit(self.execute_retrieval_for_query, sq, chat_id, document_ids, history): sq for sq in sub_queries}
            for future in concurrent.futures.as_completed(future_to_sq):
                if self.generation is not None and self.generation.cancelled:
                    # Stop pressed mid-research: drop the remaining sub-queries
                    # instead of waiting out the whole fan-out.
                    for pending in future_to_sq:
                        pending.cancel()
                    self._check_cancelled("retrieval")
                try:
                    ctx, chunks, news_src, sec_src, roic_src, ch_src, web_src = future.result()
                    if ctx: contexts.append(ctx)
                    if chunks: all_chunks.extend(chunks)
                    if news_src: all_news.extend(news_src)
                    if sec_src: all_sec.extend(sec_src)
                    if roic_src: all_roic.extend(roic_src)
                    if ch_src: all_ch.extend(ch_src)
                    if web_src: all_web.extend(web_src)
                except Exception as exc:
                    logger.error(f"Error executing retrieval for subquery: {exc}")
                
        self._check_cancelled("consolidate")
        final_context = self.validate_and_consolidate(contexts)
        return final_context, all_chunks, all_news, all_sec, all_roic, all_ch, all_web
