import json
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timezone
from typing import Optional

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI

from server import crud, schemas
from server.config import settings
from server.services.audit_logger import AuditLogger
from server.services.citation_service import get_citation_service
from server.services.deep_research import DeepResearchOrchestrator
from server.services.document_takeaways_service import document_takeaways_service
from server.services.rag_service import ThinkLongerUnavailable, get_rag_service
from server.utils.chat_helpers import (
    build_routing_log,
    check_spurious_roic_query,
    check_user_typed_context,
    is_explicit_document_request,
)
from server.utils.helpers import (
    document_sources_only,
    resolve_active_document_ids,
    strip_citation_markers,
)
from server.utils.prompt import (
    AGENT_PROMPTS,
    followup_instruction_block,
    think_longer_reasoning_prompt,
)
from server.utils.source_attribution import (
    cited_document_names,
    source_attribution_from_citations,
)
from server.services.agent_trigger_service import agent_trigger_service
from server.services.skill_selection_service import get_skill_selection_service
from server.services.skill_instruction_service import get_skill_instruction_service
from server.utils.llm_client import get_openai_client

logger = logging.getLogger(__name__)

client = get_openai_client()
rag = get_rag_service()
citation_svc = get_citation_service()

# Shared thread pool executor for parallel task execution
_planner_executor = ThreadPoolExecutor(max_workers=4)
_retrieval_executor = ThreadPoolExecutor(max_workers=6)


class ChatOrchestrationService:
    """Modularized & optimized service layer for AI Chat message orchestration."""

    @staticmethod
    def _build_valid_history(prior_messages: list) -> list[dict]:
        """Extract clean user/assistant conversation history turns."""
        valid_history = []
        for i, msg in enumerate(prior_messages):
            if msg.role == "user":
                if i + 1 < len(prior_messages) and prior_messages[i + 1].role == "assistant":
                    valid_history.append({"role": msg.role, "content": msg.content})
            elif msg.role in ("assistant", "system"):
                valid_history.append({"role": msg.role, "content": msg.content})
        return valid_history

    @staticmethod
    def _handle_multi_document_grid(
        db,
        current_user,
        chat_id: str,
        session,
        request: schemas.ChatMessageRequest,
        unique_doc_ids: list[str],
        context_reason: str,
        background_tasks: Optional[BackgroundTasks],
    ) -> StreamingResponse:
        """Create and stream a multi-document analysis grid response."""
        ws_id = session.workspace_id or request.workspace_id
        if not ws_id and unique_doc_ids:
            doc_rec = crud.document.get(db, unique_doc_ids[0])
            if doc_rec:
                ws_id = doc_rec.workspace_id

        if not ws_id:
            raise HTTPException(
                status_code=400,
                detail="Workspace ID required for multi-document analysis",
            )

        valid_docs = crud.document.get_multi_by_workspace(db, workspace_id=ws_id)
        valid_doc_ids = {str(d.id) for d in valid_docs}
        for d_id in unique_doc_ids:
            if d_id not in valid_doc_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {d_id} not found in workspace",
                )

        if background_tasks is None:
            background_tasks = BackgroundTasks()

        try:
            grid = document_takeaways_service.create_grid(
                db=db,
                background_tasks=background_tasks,
                workspace_id=ws_id,
                name="Document Analysis Grid",
                document_ids=unique_doc_ids,
                user_id=str(current_user.id),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        grid_data = document_takeaways_service.serialize_grid(grid, db)
        user_message = crud.chat_message.create(
            db=db,
            obj_in=schemas.ChatMessageCreate(
                chat_id=chat_id,
                content=request.content,
                role="user",
                user_id=str(current_user.id),
                workspace_id=ws_id,
                document_ids=unique_doc_ids,
            ),
        )

        grid_citations = []
        try:
            grid_chunks = []
            for d_id in unique_doc_ids:
                doc_chunks = crud.document_chunk.search_nearest_chunks(
                    db,
                    query_embedding=[0.0] * 1536,
                    user_id=str(current_user.id),
                    workspace_id=ws_id,
                    document_ids=[d_id],
                    candidate_k=1,
                )
                if doc_chunks:
                    grid_chunks.extend(doc_chunks)
            if grid_chunks:
                grid_citations = citation_svc.generate_citations_from_chunks(grid_chunks)
        except Exception as e:
            logger.warning(f"Failed to generate citations for multi-document grid: {e}")

        grid_source_attribution = source_attribution_from_citations(grid_citations)

        ai_response = f"Analysis Grid created for {len(unique_doc_ids)} documents."
        assistant_message = crud.chat_message.create(
            db=db,
            obj_in=schemas.ChatMessageCreate(
                chat_id=chat_id,
                content=ai_response,
                role="assistant",
                user_id=str(current_user.id),
                workspace_id=ws_id,
                document_ids=unique_doc_ids,
                source_documents=[
                    d.file_name
                    for d in valid_docs
                    if str(d.id) in unique_doc_ids and d.file_name
                ],
                citations=grid_citations,
                mode_used=getattr(request, "mode", "auto"),
            ),
        )

        grid_followups = [
            f"What are the key comparative takeaways across these {len(unique_doc_ids)} documents?",
            "Which document reports the strongest growth metrics or positive findings?",
            "Can you summarize the main differences and risk factors across the files?",
        ]

        message_obj = {
            "id": assistant_message.id,
            "chat_id": assistant_message.chat_id,
            "role": assistant_message.role,
            "content": assistant_message.content,
            "workspace_id": assistant_message.workspace_id,
            "document_ids": assistant_message.document_ids,
            "uploaded_document_ids": (
                unique_doc_ids if context_reason == "new_upload" else []
            ),
            "attachments": [],
            "source_documents": [
                d.file_name
                for d in valid_docs
                if str(d.id) in unique_doc_ids and d.file_name
            ],
            "citations": grid_citations,
            "sources": grid_citations,
            "source_attribution": grid_source_attribution,
            "follow_up_questions": grid_followups,
            "created_at": (
                assistant_message.created_at.isoformat()
                if assistant_message.created_at
                else None
            ),
            "updated_at": (
                assistant_message.updated_at.isoformat()
                if assistant_message.updated_at
                else None
            ),
        }

        final_data = {
            "success": True,
            "error": None,
            "data": {
                "chat_session_id": chat_id,
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "response": ai_response,
                "document_ids": unique_doc_ids,
                "active_document_ids": unique_doc_ids,
                "mode_used": getattr(request, "mode", "auto"),
                "is_grid": True,
                "grid": grid_data,
                "used_documents": True,
                "used_web_search": False,
                "source_documents": [
                    d.file_name
                    for d in valid_docs
                    if str(d.id) in unique_doc_ids and d.file_name
                ],
                "citations": grid_citations,
                "sources": grid_citations,
                "source_attribution": grid_source_attribution,
                "web_sources": None,
                "financial_news": None,
                "sec_company_facts": None,
                "roic_ai": None,
                "companies_house": None,
                "follow_up_questions": grid_followups,
                "message": message_obj,
            },
            "message": "Analysis Grid created successfully for selected documents.",
        }

        def grid_event_stream():
            yield f"data: {json.dumps({'type': 'complete', 'data': final_data})}\n\n"

        return StreamingResponse(grid_event_stream(), media_type="text/event-stream")

    @classmethod
    def process_chat_message(
        cls,
        db,
        current_user,
        chat_id: str,
        session,
        request: schemas.ChatMessageRequest,
        background_tasks: Optional[BackgroundTasks] = None,
    ):
        """Execute the AI chat orchestration pipeline."""
        prior_messages = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)
        history = cls._build_valid_history(prior_messages)

        is_workspace_analysis = (
            request.content.strip().startswith(
                "Perform a comprehensive, structured workspace analysis for"
            )
            or request.content.strip().lower() == "analyze this workspace"
        )

        if is_workspace_analysis and session.workspace_id:
            documents = crud.document.get_multi_by_workspace(
                db, workspace_id=session.workspace_id
            )
            document_ids = [doc.id for doc in documents]
            context_reason = "workspace_analysis"
        else:
            document_ids, context_reason = resolve_active_document_ids(
                prior_messages, request.document_ids
            )

        unique_doc_ids = list(dict.fromkeys(document_ids or []))
        should_create_grid = False
        if len(unique_doc_ids) > 1:
            if request.create_grid is True or getattr(request, "action", None) == "analyze":
                should_create_grid = True
            elif getattr(request, "create_grid", None) is not False and getattr(request, "action", None) != "chat":
                lower_content = request.content.strip().lower()
                if lower_content.startswith((
                    "analyze the selected document",
                    "analyze the selected documents",
                    "analyze selected document",
                    "analyze selected documents",
                )):
                    should_create_grid = True

        if should_create_grid:
            return cls._handle_multi_document_grid(
                db=db,
                current_user=current_user,
                chat_id=chat_id,
                session=session,
                request=request,
                unique_doc_ids=unique_doc_ids,
                context_reason=context_reason,
                background_tasks=background_tasks,
            )

        # Check for Automatic Trigger Agent Workflow match
        explicit_agent_id = session.agent_id or getattr(request, "agent_id", None)
        matched_agent_workflow = agent_trigger_service.match_agent_for_request(
            db,
            user_id=str(current_user.id),
            query=request.content,
            workspace_id=session.workspace_id,
            explicit_agent_id=explicit_agent_id,
        )

        if matched_agent_workflow:
            logger.info(
                "[AGENT_AUTO_TRIGGER] Routing chat message to Agent '%s' (id=%s)",
                matched_agent_workflow.name,
                matched_agent_workflow.id,
            )
            return agent_trigger_service.execute_and_stream_chat_response(
                db=db,
                current_user=current_user,
                chat_id=chat_id,
                session=session,
                request=request,
                matched_agent=matched_agent_workflow,
                unique_doc_ids=unique_doc_ids,
                background_tasks=background_tasks,
            )

        # Selected skills (manual or automatic)
        manual_skills = []
        auto_skills = []
        if request.skill_ids:
            if len(request.skill_ids) > 5:
                raise HTTPException(
                    status_code=400,
                    detail="A maximum of 5 skills can be selected for a query.",
                )

            unique_skill_ids = list(dict.fromkeys(request.skill_ids))
            if len(unique_skill_ids) > 5:
                raise HTTPException(
                    status_code=400,
                    detail="A maximum of 5 skills can be selected for a query.",
                )
            fetched_skills = crud.skill.fetch_skills_by_ids(
                db, skill_ids=unique_skill_ids, user_id=str(current_user.id)
            )

            fetched_map = {s.id: s for s in fetched_skills}
            for sid in unique_skill_ids:
                if sid not in fetched_map:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Skill '{sid}' not found, unauthorized, or not published.",
                    )

            manual_skills = [fetched_map[sid] for sid in unique_skill_ids]
        else:
            # Trigger automatic skill selection from user's published skills
            skill_selection_svc = get_skill_selection_service()
            auto_skills = skill_selection_svc.select_skills(
                db=db,
                user_id=str(current_user.id),
                query=request.content,
                max_skills=5,
            )

        active_skills = manual_skills if request.skill_ids else auto_skills
        selection_mode = "manual" if request.skill_ids else ("automatic" if auto_skills else "none")

        logger.info(
            "[SKILL_CHAT_ORCHESTRATION] User %s | selection_mode=%s | selected_skill_ids=%s | selected_skill_names=%s | count=%d",
            str(current_user.id),
            selection_mode,
            [s.id for s in active_skills],
            [s.name for s in active_skills],
            len(active_skills),
        )

        # Step 3: Load complete instructions for final selected skills (up to 5)
        skill_instruction_svc = get_skill_instruction_service()
        skills_context, skills_meta = skill_instruction_svc.load_and_format_instructions(
            skills=active_skills,
            max_skills=5,
        )
        for sk_meta in skills_meta:
            sk_meta["selection_type"] = selection_mode
            sk_meta["is_auto"] = (selection_mode != "manual")

        matched_agent = None
        if not active_skills:
            for _, agent_info in AGENT_PROMPTS.items():
                if agent_info["prompt_substring"] in request.content:
                    matched_agent = agent_info
                    break



        ws_name = "this workspace"
        if session.workspace_id:
            workspace = crud.workspace.get(db, session.workspace_id)
            if workspace:
                ws_name = workspace.name

        if is_workspace_analysis:
            stored_content = (
                f'Perform a comprehensive, structured workspace analysis for "{ws_name}".'
            )
            optimized_prompt = rag.prepare_workspace_analysis_prompt(
                db,
                client=client,
                user_id=str(current_user.id),
                workspace_id=session.workspace_id,
                ws_name=ws_name,
                current_chat_id=chat_id,
                model=settings.LLM_MODEL,
            )
            precomputed_general_intent = "Research / Deep Analysis"
            conversation_company, extracted_entities, resolved_company, resolved_ticker = "", "", "", ""
            is_sec_registrant = False
            precomputed_intent = "WORKSPACE_ANALYSIS"
            precomputed_is_document_question = True
        else:
            ws_doc_ids = []
            if session.workspace_id:
                ws_docs = crud.document.get_multi_by_workspace(db, workspace_id=session.workspace_id)
                ws_doc_ids = [str(d.id) for d in ws_docs]

            stored_content = request.content
            optimized_prompt, conversation_company, extracted_entities = rag.rewrite_query(
                client, request.content, history, model=settings.LLM_MODEL
            )
            company_to_resolve = conversation_company if conversation_company else optimized_prompt
            
            planner_exec = _planner_executor
            fut_resolve = planner_exec.submit(
                rag.resolve_company, client, company_to_resolve, model=settings.LLM_MODEL
            )
            fut_doc = planner_exec.submit(
                rag.classify_requires_document, client, question=optimized_prompt, model=settings.LLM_MODEL
            ) if (document_ids or ws_doc_ids) else None
            fut_intent = planner_exec.submit(
                rag.classify_mcp_intent, client, question=optimized_prompt, model=settings.LLM_MODEL
            )
            fut_general_intent = planner_exec.submit(
                rag.classify_intent, client, question=optimized_prompt, model=settings.LLM_MODEL
            )

            resolved_company, resolved_ticker, is_sec_registrant = fut_resolve.result()
            precomputed_is_document_question = fut_doc.result() if fut_doc else False
            precomputed_intent = fut_intent.result()
            precomputed_general_intent = fut_general_intent.result()

        sec_tickers, sec_options = [], {}
        roic_query, financial_ticker, companies_house_query = None, None, None
        requires_web, mcp_selected = False, False
        selected_provider = "None"
        routing_reason = "Intent matches provider."
        intent = precomputed_intent

        is_analyze_action = getattr(request, "action", None) == "analyze"
        if not is_workspace_analysis:
            explicit_document_request = is_explicit_document_request(optimized_prompt)
            is_document_question = precomputed_is_document_question
            live_internet_keywords = ["latest", "live", "current", "news today"]
            requires_live_internet = any(kw in optimized_prompt.lower() for kw in live_internet_keywords)

            if not document_ids and ws_doc_ids and (
                is_document_question
                or explicit_document_request
                or is_analyze_action
                or (precomputed_general_intent not in ["Greeting", "Small Talk"] and not requires_live_internet)
            ):
                document_ids = ws_doc_ids
                unique_doc_ids = list(dict.fromkeys(document_ids or []))

            if document_ids and (precomputed_general_intent not in ["Greeting", "Small Talk"] or is_analyze_action):
                is_document_intent = True
            else:
                is_document_intent = is_document_question or explicit_document_request or is_analyze_action

            if (
                document_ids
                and is_document_intent
                and not requires_live_internet
                and (precomputed_general_intent not in ["Greeting", "Small Talk"] or is_analyze_action)
            ):
                intent = "UPLOADED_DOCUMENT"
                routing_reason = "Hard Routing: Active document intent detected."
                selected_provider = "RAG"
            else:
                intent = precomputed_intent
                active_id = resolved_ticker or resolved_company
                lowered = optimized_prompt.lower()

                if precomputed_general_intent in ["Greeting", "Small Talk"]:
                    routing_reason = f"Casual conversation detected: {precomputed_general_intent}"
                elif not resolved_company and not resolved_ticker:
                    requires_web = True
                    routing_reason = "No applicable MCP provider detected, fallback to web search."
                    selected_provider = "Web Search"
                else:
                    # 1. SEC EDGAR Evaluation
                    sec_keywords = ["10-k", "10-q", "8-k", "filing", "annual report", "quarterly report", "sec submission", "company filing"]
                    if intent == "SEC_FILINGS" or any(kw in lowered for kw in sec_keywords):
                        if is_sec_registrant:
                            sec_tickers = [resolved_ticker or resolved_company]
                            sec_options = rag.sec_request_options(optimized_prompt)
                            mcp_selected = True
                            selected_provider = "SEC EDGAR"

                    # 2. Alpha Vantage Evaluation
                    if not mcp_selected:
                        alpha_keywords = ["stock price", "market price", "revenue", "eps", "financial metrics", "market data", "dividend", "financial performance"]
                        if intent == "FINANCIAL_MARKET_DATA" or any(kw in lowered for kw in alpha_keywords):
                            financial_ticker = active_id
                            mcp_selected = True
                            selected_provider = "Alpha Vantage"

                    # 3. Watchlist News Evaluation
                    if not mcp_selected:
                        if intent == "WATCHLIST_NEWS" or "watchlist" in lowered or "portfolio" in lowered:
                            source_text = extracted_entities or conversation_company or optimized_prompt
                            extracted_companies = rag.resolve_multiple_companies(client, source_text, model=settings.LLM_MODEL)
                            if extracted_companies:
                                financial_ticker = ",".join(extracted_companies)
                                mcp_selected = True
                                selected_provider = "Alpha Vantage"

                    # 4. ROIC Evaluation
                    if not mcp_selected:
                        roic_keywords = ["company fundamental", "balance sheet", "income statement", "cash flow", "valuation", "ratio", "business fundamental"]
                        if intent in ["ROIC", "COMPANY_INFORMATION"] or any(kw in lowered for kw in roic_keywords):
                            roic_query = active_id
                            mcp_selected = True
                            selected_provider = "ROIC"

                    # 5. Companies House Evaluation
                    if not mcp_selected:
                        ch_keywords = ["uk company", "uk registration", "companies house"]
                        if intent == "COMPANIES_HOUSE" or any(kw in lowered for kw in ch_keywords):
                            companies_house_query = active_id
                            mcp_selected = True
                            selected_provider = "Companies House"

                    # Default Metadata Fallback
                    if not mcp_selected:
                        if is_sec_registrant:
                            sec_tickers = [resolved_ticker or resolved_company]
                            sec_options = rag.sec_request_options(optimized_prompt)
                            mcp_selected = True
                            selected_provider = "SEC EDGAR"
                        elif resolved_ticker:
                            roic_query = active_id
                            mcp_selected = True
                            selected_provider = "ROIC"

                    if not mcp_selected:
                        requires_web = True
                        routing_reason = "No applicable MCP provider detected, fallback to web search."
                        selected_provider = "Web Search"

            if document_ids and is_document_intent and not requires_live_internet:
                requires_web = False
                selected_provider = "RAG"
        else:
            is_document_intent = True

        user_message = crud.chat_message.create(
            db=db,
            obj_in=schemas.ChatMessageCreate(
                chat_id=chat_id,
                content=stored_content,
                role="user",
                user_id=str(current_user.id),
                workspace_id=session.workspace_id,
                document_ids=document_ids or None,
            ),
        )

        used_rag, used_web = False, False
        source_documents, citations, web_sources = None, [], None
        financial_news_response, financial_context, financial_sources = None, None, []
        sec_company_facts_response, sec_context, sec_sources = None, None, []
        roic_response, roic_context, roic_sources = None, None, []
        companies_house_response, companies_house_context, companies_house_sources = None, None, []
        document_context, web_context, retrieved_chunks = None, None, None

        validation_error = None
        workspace_docs_present = bool(session.workspace_id and crud.document.get_multi_by_workspace(db, workspace_id=session.workspace_id))
        uploaded_docs_present = bool(document_ids)
        has_external_provider = bool(sec_tickers or roic_query or financial_ticker or companies_house_query or requires_web)
        mcp_data_present = has_external_provider

        user_typed_context_present = check_user_typed_context(request.content, matched_agent)

        if mcp_data_present and matched_agent and roic_query:
            if check_spurious_roic_query(roic_query, matched_agent):
                mcp_data_present = False

        any_context_present = (user_typed_context_present or uploaded_docs_present or workspace_docs_present or mcp_data_present)

        if is_workspace_analysis and not uploaded_docs_present and not workspace_docs_present:
            validation_error = "Workspace analysis requires one or more documents in the workspace. Please upload files before running this analysis."
        elif matched_agent and not any_context_present:
            validation_error = matched_agent["guidance"]
        elif is_document_intent and not any_context_present:
            validation_error = "This query requires one or more uploaded documents. Please upload files or select existing workspace files to proceed."

        selected_list = []
        if is_document_intent: selected_list.append("RAG")
        if sec_tickers: selected_list.append("SEC EDGAR")
        if roic_query: selected_list.append("ROIC AI")
        if financial_ticker: selected_list.append("Alpha Vantage")
        if companies_house_query: selected_list.append("Companies House")
        if requires_web: selected_list.append("Web Search")

        logger.info(
            build_routing_log(
                content=request.content,
                document_ids=document_ids,
                is_document_question=is_document_intent,
                selected_providers=selected_list,
                skipped_providers=[],
                rewritten_query=optimized_prompt if optimized_prompt != request.content else "",
                conversation_company=conversation_company,
                extracted_entities=extracted_entities,
                resolved_company=resolved_company,
                resolved_ticker=resolved_ticker,
                sec_registrant=is_sec_registrant,
                intent=intent,
                routing_reason=routing_reason
            )
        )

        if validation_error:
            ai_response = validation_error
            follow_up_questions, citations, source_documents, source_attribution = [], [], None, []
            intent = "Other"
        else:
            futures = {}
            user_id_str = str(current_user.id)

            executor = _retrieval_executor
            if is_document_intent:
                futures["rag"] = executor.submit(
                    rag.retrieve_documents_context, db, is_workspace_analysis, ws_name, user_id_str, document_ids, session.workspace_id, optimized_prompt, chat_id
                )
            if sec_tickers:
                futures["sec"] = executor.submit(
                    rag.retrieve_sec_company_facts_context, chat_id, user_id_str, session.workspace_id, sec_tickers, sec_options
                )
            if roic_query:
                futures["roic"] = executor.submit(
                    rag.retrieve_roic_ai_context, chat_id, user_id_str, session.workspace_id, roic_query
                )
            if financial_ticker:
                futures["news"] = executor.submit(
                    rag.retrieve_financial_news_context, chat_id, user_id_str, session.workspace_id, financial_ticker
                )
            if companies_house_query:
                futures["companies_house"] = executor.submit(
                    rag.retrieve_companies_house_context, chat_id, user_id_str, session.workspace_id, companies_house_query
                )

            if "rag" in futures:
                try:
                    source_documents, retrieved_chunks = futures["rag"].result()
                except Exception as e:
                    logger.error(f"RAG retrieval failed: {e}")
                    source_documents, retrieved_chunks = [], []
            if "sec" in futures:
                try:
                    sec_company_facts_response = futures["sec"].result()
                except Exception as e:
                    logger.error(f"SEC retrieval failed: {e}")
                    sec_company_facts_response = [{"error": {"message": str(e)}}]
            if "roic" in futures:
                try:
                    roic_response = futures["roic"].result()
                except Exception as e:
                    logger.error(f"ROIC retrieval failed: {e}")
                    roic_response = {"error": str(e)}
            if "news" in futures:
                try:
                    financial_news_response = futures["news"].result()
                except Exception as e:
                    logger.error(f"News retrieval failed: {e}")
                    financial_news_response = {"error": str(e)}
            if "companies_house" in futures:
                try:
                    companies_house_response = futures["companies_house"].result()
                except Exception as e:
                    logger.error(f"Companies House retrieval failed: {e}")
                    companies_house_response = {"error": str(e)}

            if retrieved_chunks:
                document_context = rag.build_context_with_citations(retrieved_chunks) or None
            if source_documents:
                source_documents = document_sources_only(source_documents)
            used_rag = document_context is not None

            # DEF-011: RAG Context Sufficiency & Automatic Web Search Fallback Check
            if (is_document_intent or bool(document_ids)) and not requires_web:
                is_sufficient = rag.check_document_context_sufficiency(
                    client, question=optimized_prompt, context=document_context, model=settings.LLM_MODEL
                )
                if not is_sufficient:
                    requires_web = True
                    logger.info("AUTOMATIC WEB FALLBACK TRIGGERED (DEF-011): Uploaded document context insufficient.")

            if financial_news_response:
                financial_context, financial_sources = rag.build_financial_news_context(
                    financial_news_response, start_number=len(retrieved_chunks or []) + 1
                )
                financial_context = financial_context or None
            if sec_company_facts_response:
                sec_results = sec_company_facts_response if isinstance(sec_company_facts_response, list) else [sec_company_facts_response]
                sec_context, sec_sources = rag.build_sec_contexts(
                    sec_results, start_number=len(retrieved_chunks or []) + len(financial_sources) + 1
                )
                sec_context = sec_context or None
                if sec_context is None and not used_rag:
                    error_messages = [
                        result.get("error", {}).get("message") for result in sec_results if result.get("error", {}).get("message")
                    ]
                    sec_context = "SEC EDGAR data is unavailable: " + ("; ".join(error_messages) or "no validated facts were returned.")
            if roic_response:
                roic_context, roic_sources = rag.build_roic_context(
                    roic_response, start_number=len(retrieved_chunks or []) + len(financial_sources) + len(sec_sources) + 1, query=optimized_prompt
                )
                roic_context = roic_context or None
                if roic_context and used_rag and "ROIC.ai Error" in roic_context:
                    roic_context, roic_sources = None, []

            if companies_house_response:
                companies_house_context, companies_house_sources = rag.build_companies_house_context(
                    companies_house_response, start_number=len(retrieved_chunks or []) + len(financial_sources) + len(sec_sources) + len(roic_sources) + 1
                )
                companies_house_context = companies_house_context or None
                if companies_house_context and used_rag and "No Companies House records" in companies_house_context:
                    companies_house_context, companies_house_sources = None, []

            # DEF-012: Finance MCP Data Sufficiency & Web Search Fallback Check
            mcp_executed = bool(financial_news_response or sec_company_facts_response or roic_response or companies_house_response)
            has_data = any([
                financial_news_response and rag.has_useful_data(financial_news_response),
                sec_company_facts_response and rag.has_useful_data(sec_company_facts_response),
                roic_response and rag.has_useful_data(roic_response),
                companies_house_response and rag.has_useful_data(companies_house_response),
            ])

            mcp_contexts = [
                c for c in [financial_context, sec_context, roic_context, companies_house_context]
                if c and not c.startswith("SEC EDGAR data is unavailable") and "ROIC.ai Error" not in c
            ]
            combined_mcp_context = "\n\n".join(mcp_contexts) if mcp_contexts else ""

            mcp_sufficient = True
            if mcp_executed:
                if not combined_mcp_context or not has_data:
                    mcp_sufficient = False
                else:
                    mcp_sufficient = rag.check_finance_mcp_sufficiency(
                        client, question=optimized_prompt, context=combined_mcp_context, model=settings.LLM_MODEL
                    )

            if mcp_executed and not mcp_sufficient:
                requires_web = True
                logger.info("AUTOMATIC WEB FALLBACK TRIGGERED (DEF-012): Finance MCP data insufficient.")
                if not has_data:
                    financial_context, financial_sources = None, []
                    roic_context, roic_sources = None, []
                    companies_house_context, companies_house_sources = None, []

            if requires_web:
                web_context, web_sources = rag.web_search(
                    optimized_prompt,
                    start_number=(
                        len(retrieved_chunks or [])
                        + len(financial_sources)
                        + len(sec_sources)
                        + len(roic_sources)
                        + len(companies_house_sources)
                        + 1
                    ),
                )
                web_context = web_context or None
                used_web = web_context is not None

            if is_workspace_analysis:
                intent = "Research / Deep Analysis"
            elif document_ids and (
                is_analyze_action
                or precomputed_general_intent == "Document Analysis"
                or any(kw in optimized_prompt.lower() for kw in ["analyze", "analyse", "summary", "summarize", "overview", "findings"])
            ):
                intent = "Document Analysis"
            elif intent != "UPLOADED_DOCUMENT":
                intent = precomputed_general_intent

            if document_ids and intent in ["Greeting", "Small Talk", "Simple Question"]:
                intent = "Other"

            llm_history = [] if context_reason == "new_upload" else history
            messages = rag.build_messages(
                history=llm_history,
                question=optimized_prompt,
                retrieved_context=document_context,
                web_context=web_context,
                financial_news_context=financial_context,
                sec_context=sec_context,
                roic_context=roic_context,
                companies_house_context=companies_house_context,
                intent=intent,
                skills_context=skills_context,
            )
            mode_used = "auto"
            selected_model = settings.LLM_MODEL
            final_research_context = None

            if getattr(request, "mode", None) == "think_longer":
                today_start = datetime.combine(datetime.now(timezone.utc).date(), time.min)
                daily_calls = crud.chat_message.count_user_daily_calls(
                    db=db, user_id=str(current_user.id), mode_prefix="think_longer", since=today_start
                )
                has_context = any([
                    document_context, web_context, financial_context, sec_context, roic_context, companies_house_context
                ])

                if daily_calls >= settings.THINK_LONGER_DAILY_LIMIT or not has_context:
                    raw_response = rag.generate_response(client, messages=messages, model=settings.LLM_MODEL)
                else:
                    try:
                        think_longer_messages = messages + [{"role": "system", "content": think_longer_reasoning_prompt()}]
                        selected_model = settings.THINK_LONGER_MODEL
                        raw_response = rag.generate_think_longer_response(
                            client, messages=think_longer_messages, model=settings.THINK_LONGER_MODEL, reasoning_effort=settings.THINK_LONGER_REASONING_EFFORT
                        )
                        mode_used = "think_longer"
                    except ThinkLongerUnavailable as e:
                        selected_model = settings.THINK_LONGER_FALLBACK_MODEL
                        raw_response = rag.generate_response(client, messages=messages, model=settings.THINK_LONGER_FALLBACK_MODEL)
                        mode_used = "think_longer_fallback"

            elif getattr(request, "mode", None) == "deep_research":
                orchestrator = DeepResearchOrchestrator(client, rag, db, current_user, session)
                initial_contexts = [c for c in [document_context, web_context, financial_context, sec_context, roic_context, companies_house_context] if c]
                initial_context_str = "\n\n".join(initial_contexts)
                
                final_research_context, all_chunks, all_news, all_sec, all_roic, all_ch, all_web = orchestrator.execute(
                    initial_query=optimized_prompt, chat_id=chat_id, document_ids=document_ids, history=llm_history, initial_context=initial_context_str
                )
                
                if all_chunks: retrieved_chunks = list(retrieved_chunks or []) + all_chunks
                if all_news: financial_sources.extend(all_news)
                if all_sec: sec_sources.extend(all_sec)
                if all_roic: roic_sources.extend(all_roic)
                if all_ch: companies_house_sources.extend(all_ch)
                if all_web:
                    web_sources = list(web_sources or []) + all_web
                    used_web = True
                
                if final_research_context and len(final_research_context) > 10000:
                    final_research_context = final_research_context[:10000] + "\n\n... [Truncated for context length limits]"

                report_prompt = (
                    f"You are a professional research analyst. Using the following comprehensive research context, generate a detailed research report addressing '{optimized_prompt}'.\n\n"
                    f"Context:\n{final_research_context}"
                ) + followup_instruction_block()
                
                messages = [{"role": "user", "content": report_prompt}]
                try:
                    selected_model = settings.DEEP_RESEARCH_MODEL
                    raw_response = rag.generate_think_longer_response(
                        client, messages=messages, model=settings.DEEP_RESEARCH_MODEL, reasoning_effort=settings.DEEP_RESEARCH_REASONING_EFFORT
                    )
                    mode_used = "deep_research"
                except ThinkLongerUnavailable:
                    selected_model = settings.THINK_LONGER_FALLBACK_MODEL
                    raw_response = rag.generate_response(client, messages=messages, model=settings.THINK_LONGER_FALLBACK_MODEL)
                    mode_used = "deep_research_fallback"
            else:
                selected_model = settings.LLM_MODEL
                raw_response = rag.generate_response(client, messages=messages, model=settings.LLM_MODEL)

        def event_stream():
            nonlocal ai_response, follow_up_questions, citations, source_documents, source_attribution
            
            if not validation_error:
                raw_response_str = ""
                try:
                    for token in raw_response:
                        raw_response_str += token
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    
                ai_response, follow_up_questions = rag.split_answer_and_followups(
                    raw_response_str,
                    question=(optimized_prompt if is_workspace_analysis else request.content),
                    context=(document_context or final_research_context or web_context or ""),
                )

                cited_sources: list = list(retrieved_chunks or [])
                cited_sources.extend(financial_sources)
                cited_sources.extend(sec_sources)
                cited_sources.extend(roic_sources)
                cited_sources.extend(companies_house_sources)
                if web_sources:
                    cited_sources.extend(web_sources)

                if cited_sources:
                    try:
                        ai_response, citations = citation_svc.generate_citations(ai_response, cited_sources)
                        if not citations:
                            citations = citation_svc.generate_citations_from_chunks(cited_sources)
                        if citations:
                            ai_response = citation_svc.attach_fallback_markers(ai_response, citations)
                    except Exception as _cite_exc:
                        logger.exception("Citation generation failed: %s", _cite_exc)
                        ai_response = strip_citation_markers(ai_response)
                else:
                    ai_response = strip_citation_markers(ai_response)

                source_documents = (
                    document_sources_only(cited_document_names(citations)) or None
                )
                source_attribution = source_attribution_from_citations(citations)

            assistant_message = crud.chat_message.create(
                db=db,
                obj_in=schemas.ChatMessageCreate(
                    chat_id=chat_id,
                    content=ai_response,
                    role="assistant",
                    user_id=str(current_user.id),
                    workspace_id=session.workspace_id,
                    document_ids=document_ids or None,
                    source_documents=source_documents,
                    citations=citations,
                    skills_used=skills_meta,
                    mode_used=mode_used,
                ),
            )

            AuditLogger.log(
                action="AI_RESPONSE",
                user_id=str(current_user.id),
                workspace_id=session.workspace_id,
                chat_id=chat_id,
                endpoint=f"/v1/ai/chats/{chat_id}/messages",
                http_method="POST",
                status="200",
                metadata={
                    "citations_count": len(citations) if citations else 0,
                    "follow_up_questions": follow_up_questions,
                    "mode_requested": getattr(request, "mode", None),
                    "mode_used": mode_used,
                    "skill_selection_mode": selection_mode,
                    "selected_skill_ids": [s.id for s in active_skills],
                    "selected_skill_names": [s.name for s in active_skills],
                    "applied_skills_count": len(active_skills),
                },
            )

            message_obj = {
                "id": assistant_message.id,
                "chat_id": assistant_message.chat_id,
                "role": assistant_message.role,
                "content": assistant_message.content,
                "workspace_id": assistant_message.workspace_id,
                "document_ids": assistant_message.document_ids,
                "uploaded_document_ids": (document_ids if context_reason == "new_upload" else []),
                "attachments": [],
                "source_documents": source_documents,
                "citations": citations,
                "sources": citations,
                "source_attribution": source_attribution,
                "skills_used": skills_meta if skills_meta else None,
                "selected_skills": skills_meta if skills_meta else None,
                "skills": skills_meta if skills_meta else None,
                "auto_selected_skills": skills_meta if not request.skill_ids and skills_meta else None,
                "selection_mode": selection_mode,
                "agent_id": None,
                "agent_name": None,
                "agents": None,
                "follow_up_questions": follow_up_questions,
                "created_at": assistant_message.created_at.isoformat() if assistant_message.created_at else None,
                "updated_at": assistant_message.updated_at.isoformat() if assistant_message.updated_at else None,
            }

            final_data = {
                "success": True,
                "error": None,
                "data": {
                    "chat_session_id": chat_id,
                    "user_message_id": user_message.id,
                    "assistant_message_id": assistant_message.id,
                    "response": ai_response,
                    "document_ids": document_ids,
                    "active_document_ids": document_ids,
                    "skills_used": skills_meta if skills_meta else None,
                    "selected_skills": skills_meta if skills_meta else None,
                    "skills": skills_meta if skills_meta else None,
                    "auto_selected_skills": skills_meta if not request.skill_ids and skills_meta else None,
                    "selection_mode": selection_mode,
                    "agent_id": None,
                    "agent_name": None,
                    "agent_workflow_id": None,
                    "agents": None,
                    "mode_used": mode_used,
                    "uploaded_document_ids": (
                        document_ids if context_reason == "new_upload" else []
                    ),
                    "used_documents": used_rag,
                    "used_web_search": used_web,
                    "source_documents": source_documents,
                    "citations": citations,
                    "sources": citations,
                    "source_attribution": source_attribution,
                    "web_sources": web_sources,
                    "financial_news": financial_news_response,
                    "sec_company_facts": sec_company_facts_response,
                    "roic_ai": roic_response,
                    "companies_house": companies_house_response,
                    "follow_up_questions": follow_up_questions,
                    "message": message_obj,
                },
                "message": "Message added and AI response created successfully.",
            }
            yield f"data: {json.dumps({'type': 'complete', 'data': final_data})}\n\n"

        if getattr(request, "stream", True):
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            events = list(event_stream())
            for ev_str in reversed(events):
                if "type\": \"complete" in ev_str:
                    clean_json = ev_str.replace("data: ", "").strip()
                    return json.loads(clean_json)
            return {"success": True, "data": {}}


chat_orchestration_service = ChatOrchestrationService()
