import logging
import os
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

from server import crud, schemas
from server.api_v1.deps import get_db
from server.config import settings, WORKSPACE_DOCS_DIR
from server.services.citation_service import get_citation_service
from server.services.rag_service import get_rag_service, ThinkLongerUnavailable
from datetime import datetime, time, timezone
from server.services.audit_logger import AuditLogger
from server.utils.auth import get_current_user
from server.utils.prompt import AGENT_PROMPTS
from server.utils.helpers import (
    document_sources_only,
    resolve_active_document_ids,
    resolve_or_create_session,
    strip_citation_markers,
)
from server.utils.source_attribution import (
    cited_document_names,
    source_attribution_from_citations,
)
from server.utils.chat_helpers import (
    is_explicit_document_request,
    check_user_typed_context,
    check_spurious_roic_query,
    build_routing_log,
)
from server.services.deep_research import DeepResearchOrchestrator
from server.utils.prompt import think_longer_reasoning_prompt, followup_instruction_block
import time as _time

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


ai_router = APIRouter(prefix="/ai", tags=["GenAI"])


@ai_router.post("/chats")
def create_chat_session(
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    if workspace_id:
        workspace = crud.workspace.get(db, workspace_id)
        if not workspace or workspace.user_id != str(current_user.id):
            raise HTTPException(status_code=404, detail="Workspace not found")

    session = crud.chat_session.create(
        db=db,
        obj_in=schemas.ChatSessionCreate(
            user_id=str(current_user.id), title=title, workspace_id=workspace_id
        ),
    )

    AuditLogger.log(
        action="CHAT_CREATE",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        chat_id=session.id,
        endpoint="/v1/ai/chats",
        http_method="POST",
        status="201",
        metadata={"title": session.title},
    )

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "error": None,
            "data": {
                "id": session.id,
                "title": session.title,
                "user_id": session.user_id,
                "workspace_id": session.workspace_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            },
            "message": "Chat session created successfully.",
        },
    )


@ai_router.get("/chats")
def list_chat_sessions(
    db=Depends(get_db),
    current_user=Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
):
    sessions = crud.chat_session.get_multi(
        db, user_id=str(current_user.id), skip=skip, limit=limit
    )

    AuditLogger.log(
        action="CHAT_LIST",
        user_id=str(current_user.id),
        endpoint="/v1/ai/chats",
        http_method="GET",
        status="200",
        metadata={"count": len(sessions)},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": [
                {
                    "id": item.id,
                    "title": item.title,
                    "user_id": item.user_id,
                    "workspace_id": item.workspace_id,
                    "created_at": (
                        item.created_at.isoformat() if item.created_at else None
                    ),
                    "updated_at": (
                        item.updated_at.isoformat() if item.updated_at else None
                    ),
                }
                for item in sessions
            ],
            "message": "Chat sessions fetched successfully.",
        },
    )


@ai_router.delete("/chats/{chat_id}")
def delete_chat_session(
    chat_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = crud.chat_session.get(db, chat_id)

    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    workspace_id = session.workspace_id
    crud.chat_session.remove(db, id=session.id)

    AuditLogger.log(
        action="CHAT_DELETE",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        chat_id=chat_id,
        endpoint=f"/v1/ai/chats/{chat_id}",
        http_method="DELETE",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "Chat session deleted successfully.",
        },
    )


@ai_router.get("/chats/public/{chat_id}")
def public_chat_session(
    chat_id: str,
    db=Depends(get_db),
):
    """Public, unauthenticated read-only view of a chat (for shareable links).

    Returns the chat and its messages only — no owner/user data. Declared before
    ``/chats/{chat_id}`` so the literal ``public`` segment isn't captured by the
    id parameter.
    """
    session = crud.chat_session.get(db, chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    conversation = crud.chat_message.get_visible_by_chat(db, chat_id=chat_id)
    messages = [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "source_documents": document_sources_only(
                cited_document_names(msg.citations)
            ),
            "citations": msg.citations,
            "source_attribution": source_attribution_from_citations(msg.citations),
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }
        for msg in conversation
    ]
    AuditLogger.log(
        action="CHAT_GET_PUBLIC",
        workspace_id=session.workspace_id,
        chat_id=chat_id,
        endpoint=f"/v1/ai/chats/public/{chat_id}",
        http_method="GET",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "id": session.id,
                "title": session.title,
                "messages": messages,
            },
            "message": "Public chat fetched successfully.",
        },
    )


@ai_router.get("/chats/{chat_id}")
def get_chat_session(
    chat_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = crud.chat_session.get(db, chat_id)
    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Full history (includes internal housekeeping rows) — used to build the
    # active document-id tracking state so upload-link rows are picked up.
    conversation = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)

    seen_document_ids: set = set()
    uploaded_by_message: dict = {}
    for msg in conversation:
        ids = msg.document_ids or []
        uploaded_by_message[msg.id] = [d for d in ids if d not in seen_document_ids]
        seen_document_ids.update(ids)

    # Visible history — housekeeping system rows are excluded.
    visible_conversation = crud.chat_message.get_visible_by_chat(db, chat_id=chat_id)

    doc_info_cache: dict = {}

    def _doc_info(doc_id):
        if doc_id not in doc_info_cache:
            doc = crud.document.get(db, doc_id)
            if doc:
                size = 0
                if doc.storage_path:
                    file_name_only = os.path.basename(doc.storage_path)
                    full_path = WORKSPACE_DOCS_DIR / file_name_only
                    if os.path.exists(full_path):
                        size = os.path.getsize(full_path)

                mime_type = (
                    mimetypes.guess_type(doc.file_name)[0]
                    or f"application/{doc.file_format.lower()}"
                )
                timestamp = doc.created_at.isoformat() if doc.created_at else None

                doc_info_cache[doc_id] = {
                    "id": doc.id,
                    "file_id": doc.id,
                    "file_name": doc.file_name,
                    "original_filename": doc.file_name,
                    "display_name": doc.file_name,
                    "file_format": doc.file_format,
                    "mime_type": mime_type,
                    "size": size,
                    "created_at": timestamp,
                    "upload_timestamp": timestamp,
                    "storage_path": doc.storage_path,
                }
            else:
                doc_info_cache[doc_id] = None
        return doc_info_cache[doc_id]

    messages = []
    for msg in visible_conversation:
        uploaded = uploaded_by_message.get(msg.id, [])
        attachments = [info for info in (_doc_info(d) for d in uploaded) if info]
        messages.append(
            {
                "id": msg.id,
                "chat_id": msg.chat_id,
                "role": msg.role,
                "content": msg.content,
                "workspace_id": msg.workspace_id,
                "document_ids": msg.document_ids,
                "uploaded_document_ids": uploaded,
                "attachments": attachments,
                "source_documents": document_sources_only(
                    cited_document_names(msg.citations)
                ),
                "citations": msg.citations,
                "source_attribution": source_attribution_from_citations(msg.citations),
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "updated_at": msg.updated_at.isoformat() if msg.updated_at else None,
            }
        )

    AuditLogger.log(
        action="CHAT_GET",
        user_id=str(current_user.id),
        workspace_id=session.workspace_id,
        chat_id=chat_id,
        endpoint=f"/v1/ai/chats/{chat_id}",
        http_method="GET",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "id": session.id,
                "title": session.title,
                "workspace_id": session.workspace_id,
                "document_ids": crud.chat_message.get_latest_document_ids(
                    db, chat_id=chat_id
                ),
                "source_documents": crud.chat_message.get_latest_source_documents(
                    db,
                    chat_id=chat_id,
                    document_ids=crud.chat_message.get_latest_document_ids(
                        db, chat_id=chat_id
                    ),
                ),
                "messages": messages,
            },
            "message": "Chat session details fetched successfully.",
        },
    )


@ai_router.put("/chats/{chat_id}")
def update_chat_title(
    chat_id: str,
    request: schemas.ChatSessionTitleUpdate,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = crud.chat_session.get(db, chat_id)
    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    crud.chat_session.update(db, db_obj=session, obj_in={"title": request.title})

    AuditLogger.log(
        action="CHAT_UPDATE",
        user_id=str(current_user.id),
        workspace_id=session.workspace_id,
        chat_id=chat_id,
        endpoint=f"/v1/ai/chats/{chat_id}",
        http_method="PUT",
        status="200",
        metadata={"title": session.title},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "id": session.id,
                "title": session.title,
                "user_id": session.user_id,
                "updated_at": (
                    session.updated_at.isoformat() if session.updated_at else None
                ),
            },
            "message": "Chat title updated successfully.",
        },
    )


@ai_router.post("/chats/{chat_id}/messages")
def add_chat_message(
    chat_id: str,
    request: schemas.ChatMessageRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = resolve_or_create_session(
        db,
        chat_id=chat_id,
        current_user=current_user,
        title=request.title,
        workspace_id=request.workspace_id,
    )
    chat_id = session.id

    prior_messages = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)
    history = [{"role": msg.role, "content": msg.content} for msg in prior_messages]

    is_workspace_analysis = (
        request.content.strip().startswith(
            "Perform a comprehensive, structured workspace analysis for"
        )
        or request.content.strip().lower() == "analyze this workspace"
    )

    matched_agent = None
    for agent_id, agent_info in AGENT_PROMPTS.items():
        if agent_info["prompt_substring"] in request.content:
            matched_agent = agent_info
            break

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

    logger.info(
        "chat.message: chat=%s context=%s active_docs=%s requested=%s",
        chat_id,
        context_reason,
        document_ids,
        request.document_ids,
    )
    rag = get_rag_service()
    citation_svc = get_citation_service()

    wsName = "this workspace"
    if session.workspace_id:
        workspace = crud.workspace.get(db, session.workspace_id)
        if workspace:
            wsName = workspace.name

    if is_workspace_analysis:
        stored_content = (
            f'Perform a comprehensive, structured workspace analysis for "{wsName}".'
        )
        optimized_prompt = rag.prepare_workspace_analysis_prompt(
            db,
            client=client,
            user_id=str(current_user.id),
            workspace_id=session.workspace_id,
            ws_name=wsName,
            current_chat_id=chat_id,
            model=settings.LLM_MODEL,
        )
    else:
        optimized_prompt, conversation_company, extracted_entities = rag.rewrite_query(
            client, request.content, history, model=settings.LLM_MODEL
        )
        company_to_resolve = conversation_company if conversation_company else optimized_prompt
        
        with ThreadPoolExecutor(max_workers=3) as planner_exec:
            fut_resolve = planner_exec.submit(
                rag.resolve_company, client, company_to_resolve, model=settings.LLM_MODEL
            )
            fut_doc = planner_exec.submit(
                rag.classify_requires_document, client, question=optimized_prompt, model=settings.LLM_MODEL
            ) if document_ids else None
            fut_intent = planner_exec.submit(
                rag.classify_mcp_intent, client, question=optimized_prompt, model=settings.LLM_MODEL
            )
            
            resolved_company, resolved_ticker, is_sec_registrant = fut_resolve.result()
            precomputed_is_document_question = fut_doc.result() if fut_doc else False
            precomputed_intent = fut_intent.result()

        stored_content = request.content
        
        logger.info(
            "Conversation History:\n%s\n\n"
            "Latest User Message:\n%s\n\n"
            "Resolved Company:\n%s\n\n"
            "Rewritten Query:\n%s",
            history, request.content, resolved_company, optimized_prompt
        )

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

    link_results: dict[str, bool] = {}
    if document_ids:
        link_results = crud.document_chat_link.ensure_links_for_chat(
            db,
            document_ids=document_ids,
            chat_id=chat_id,
            workspace_id=session.workspace_id,
            user_id=str(current_user.id),
        )

    AuditLogger.log(
        action="USER_MESSAGE",
        user_id=str(current_user.id),
        workspace_id=session.workspace_id,
        chat_id=chat_id,
        endpoint=f"/v1/ai/chats/{chat_id}/messages",
        http_method="POST",
        status="200",
        metadata={
            "document_ids": document_ids,
            "associations_created": {
                doc_id: created
                for doc_id, created in link_results.items()
                if created
            },
            "associations_already_existed": {
                doc_id: True
                for doc_id, created in link_results.items()
                if not created
            },
        },
    )

    used_rag = False
    used_web = False
    source_documents = None
    citations = []
    web_sources = None
    financial_news_response = None
    financial_context = None
    financial_sources: list[dict] = []
    sec_company_facts_response = None
    sec_context = None
    sec_sources: list[dict] = []
    roic_response = None
    roic_context = None
    roic_sources: list[dict] = []
    companies_house_response = None
    companies_house_context = None
    companies_house_sources: list[dict] = []
    document_context = None
    web_context = None
    retrieved_chunks = None

    if is_workspace_analysis:
        is_document_question = bool(document_ids)
        sec_tickers = []
        sec_ticker = None
        sec_options = {}
        roic_query = None
        financial_ticker = None
        companies_house_query = None
        requires_web = False
        conversation_company = ""
        extracted_entities = ""
        
        is_document_intent = False
        requires_live_internet = False
        explicit_document_request = False
        resolved_company = ""
        resolved_ticker = ""
        is_sec_registrant = False
        intent = "WORKSPACE_ANALYSIS"
        routing_reason = "Workspace Analysis Explicitly Requested"
        
        mcp_selected = False
        selected_provider = "RAG"
        fallback_triggered = False
        fallback_reason = "Not applicable (Workspace Analysis)"
        
        sec_applicable, sec_reason = False, "Bypassed (Workspace Analysis)"
        alpha_applicable, alpha_reason = False, "Bypassed (Workspace Analysis)"
        roic_applicable, roic_reason = False, "Bypassed (Workspace Analysis)"
        ch_applicable, ch_reason = False, "Bypassed (Workspace Analysis)"
    else:
        explicit_document_request = is_explicit_document_request(optimized_prompt)
        is_document_question = precomputed_is_document_question

        live_internet_keywords = ["latest", "live", "current", "news today"]
        requires_live_internet = any(kw in optimized_prompt.lower() for kw in live_internet_keywords)

        # HARD ROUTING RULE for Uploaded Documents
        is_document_intent = is_document_question or explicit_document_request
        if document_ids and is_document_intent and not requires_live_internet:
            intent = "UPLOADED_DOCUMENT"
            sec_tickers = []
            sec_ticker = None
            sec_options = {}
            roic_query = None
            financial_ticker = None
            companies_house_query = None
            requires_web = False
            routing_reason = "Hard Routing: Active document intent detected."
            mcp_selected = False
            selected_provider = "RAG"
            resolved_company = ""
            resolved_ticker = ""
            is_sec_registrant = False
            active_id = ""
            is_document_question = True
            
            sec_applicable, sec_reason = False, "Bypassed (Hard Routing)"
            alpha_applicable, alpha_reason = False, "Bypassed (Hard Routing)"
            roic_applicable, roic_reason = False, "Bypassed (Hard Routing)"
            ch_applicable, ch_reason = False, "Bypassed (Hard Routing)"
            
            fallback_triggered = False
            fallback_reason = "Bypassed (Hard Routing)"
        else:
            intent = precomputed_intent
            
            sec_tickers = []
            sec_ticker = None
            sec_options = {}
            roic_query = None
            financial_ticker = None
            companies_house_query = None
            requires_web = False
            routing_reason = "Intent matches provider."
            
            active_identifier = resolved_ticker or resolved_company
            
            # Provider-first Evaluation
            sec_applicable, sec_reason = False, "Not evaluated."
            alpha_applicable, alpha_reason = False, "Not evaluated."
            roic_applicable, roic_reason = False, "Not evaluated."
            ch_applicable, ch_reason = False, "Not evaluated."
            
            mcp_selected = False
            selected_provider = "None"
        if selected_provider != "RAG":
            if not resolved_company and not resolved_ticker:
                reason_msg = "No company resolved."
                sec_reason = reason_msg
                alpha_reason = reason_msg
                roic_reason = reason_msg
                ch_reason = reason_msg
                
                # Skip to fallback logic
                fallback_triggered = True
                fallback_reason = "No applicable MCP service detected."
                requires_web = True
                routing_reason = "No applicable MCP provider detected, fallback to web search."
                selected_provider = "Web Search"
            else:
                active_id = resolved_ticker or resolved_company
                lowered = optimized_prompt.lower()
                
                # 1. SEC EDGAR Evaluation
            sec_keywords = ["10-k", "10-q", "8-k", "filing", "annual report", "quarterly report", "sec submission", "company filing"]
            has_sec = intent == "SEC_FILINGS" or any(kw in lowered for kw in sec_keywords)
            if has_sec:
                if is_sec_registrant:
                    sec_applicable = True
                    sec_reason = "Applicable (SEC Registrant = True)"
                    sec_tickers = [resolved_ticker or resolved_company]
                    sec_ticker = sec_tickers[0]
                    sec_options = rag.sec_request_options(optimized_prompt)
                    mcp_selected = True
                    selected_provider = "SEC EDGAR"
                else:
                    sec_reason = "SEC Registrant = False"
            else:
                sec_reason = "No SEC keywords or intent."

            # 2. Alpha Vantage Evaluation
            if not mcp_selected:
                alpha_keywords = ["stock price", "market price", "revenue", "eps", "financial metrics", "market data", "dividend", "financial performance"]
                has_alpha = intent == "FINANCIAL_MARKET_DATA" or any(kw in lowered for kw in alpha_keywords)
                if has_alpha:
                    alpha_applicable = True
                    alpha_reason = "Applicable (Finance match)"
                    financial_ticker = active_id
                    mcp_selected = True
                    selected_provider = "Alpha Vantage"
                else:
                    alpha_reason = "No Finance keywords or intent."

            # Watchlist News Evaluation
            if not mcp_selected:
                has_watchlist = intent == "WATCHLIST_NEWS" or "watchlist" in lowered or "portfolio" in lowered
                if has_watchlist:
                    source_text = extracted_entities or conversation_company or optimized_prompt
                    extracted_companies = rag.resolve_multiple_companies(client, source_text, model=settings.LLM_MODEL)
                    if extracted_companies:
                        alpha_applicable = True
                        alpha_reason = "Applicable (Watchlist News match)"
                        financial_ticker = ",".join(extracted_companies)
                        mcp_selected = True
                        selected_provider = "Alpha Vantage"
                        resolved_company = financial_ticker
                        resolved_ticker = financial_ticker
                        active_id = financial_ticker

            # 3. ROIC Evaluation
            if not mcp_selected:
                roic_keywords = ["company fundamental", "balance sheet", "income statement", "cash flow", "valuation", "ratio", "business fundamental"]
                has_roic = intent in ["ROIC", "COMPANY_INFORMATION"] or any(kw in lowered for kw in roic_keywords)
                if has_roic:
                    roic_applicable = True
                    roic_reason = "Applicable (ROIC match)"
                    roic_query = active_id
                    mcp_selected = True
                    selected_provider = "ROIC"
                else:
                    roic_reason = "No ROIC keywords or intent."

            # 4. Companies House Evaluation
            if not mcp_selected:
                ch_keywords = ["uk company", "uk registration", "companies house"]
                has_ch = intent == "COMPANIES_HOUSE" or any(kw in lowered for kw in ch_keywords)
                if has_ch:
                    ch_applicable = True
                    ch_reason = "Applicable (Companies House match)"
                    companies_house_query = active_id
                    mcp_selected = True
                    selected_provider = "Companies House"
                else:
                    ch_reason = "No Companies House keywords or intent."

            # 5. Metadata Default Fallback
            if not mcp_selected:
                if is_sec_registrant:
                    sec_applicable = True
                    sec_reason = "Metadata Fallback (SEC Registrant = True)"
                    sec_tickers = [resolved_ticker or resolved_company]
                    sec_ticker = sec_tickers[0]
                    sec_options = rag.sec_request_options(optimized_prompt)
                    mcp_selected = True
                    selected_provider = "SEC EDGAR"
                elif resolved_ticker:
                    roic_applicable = True
                    roic_reason = "Metadata Fallback (Public Company)"
                    roic_query = active_id
                    mcp_selected = True
                    selected_provider = "ROIC"
                    
            # Fallback to Web Search
            fallback_triggered = False
            fallback_reason = "None"
            if not mcp_selected:
                requires_web = True
                routing_reason = "No applicable MCP provider detected, fallback to web search."
                fallback_triggered = True
                fallback_reason = "No applicable MCP service detected."
                selected_provider = "Web Search"
        # Guard: If RAG was hard-routed, make sure no MCP override happened
        if document_ids and is_document_intent and not requires_live_internet:
            requires_web = False
            selected_provider = "RAG"
            is_document_question = True

        logger.info(
            f"Planner Evaluation\n\n"
            f"Resolved Company:\n{resolved_company or 'None'}\n\n"
            f"Resolved Ticker:\n{resolved_ticker or 'None'}\n\n"
            f"SEC Registrant:\n{is_sec_registrant}\n\n"
            f"Evaluated Providers:\n\n"
            f"SEC:\nApplicable = {sec_applicable}\nReason = {sec_reason}\n\n"
            f"Alpha Vantage:\nApplicable = {alpha_applicable}\nReason = {alpha_reason}\n\n"
            f"ROIC:\nApplicable = {roic_applicable}\nReason = {roic_reason}\n\n"
            f"Companies House:\nApplicable = {ch_applicable}\nReason = {ch_reason}\n\n"
            f"Selected Provider:\n{selected_provider}\n\n"
            f"Fallback Triggered:\n{fallback_triggered}\n\n"
            f"Fallback Reason:\n{fallback_reason}\n\n"
        )

    # Apply Dynamic Source Selection priority rules
    has_external_provider = bool(
        sec_tickers
        or roic_query
        or financial_ticker
        or companies_house_query
        or requires_web
    )
    if matched_agent and bool(document_ids):
        is_document_question = True
    elif has_external_provider and not (document_ids and is_document_intent and not requires_live_internet):
        is_document_question = explicit_document_request and is_document_question
    else:
        is_document_question = bool(document_ids) and is_document_question

    # Input & Context Validation
    validation_error = None
    workspace_docs_present = False
    if session.workspace_id:
        workspace_docs = crud.document.get_multi_by_workspace(
            db, workspace_id=session.workspace_id
        )
        if workspace_docs:
            workspace_docs_present = True

    uploaded_docs_present = bool(document_ids)
    mcp_data_present = has_external_provider

    user_typed_context_present = check_user_typed_context(
        request.content, matched_agent
    )

    if mcp_data_present and matched_agent and roic_query:
        if check_spurious_roic_query(roic_query, matched_agent):
            mcp_data_present = False

    any_context_present = (
        user_typed_context_present
        or uploaded_docs_present
        or workspace_docs_present
        or mcp_data_present
    )

    if (
        is_workspace_analysis
        and not uploaded_docs_present
        and not workspace_docs_present
    ):
        validation_error = "Workspace analysis requires one or more documents in the workspace. Please upload files before running this analysis."
    elif matched_agent:
        if not any_context_present:
            validation_error = matched_agent["guidance"]
    else:
        if is_document_question and not any_context_present:
            validation_error = "This query requires one or more uploaded documents. Please upload files or select existing workspace files to proceed."

    # Log routing decision
    selected_list = []
    if is_document_question:
        selected_list.append("RAG")
    if sec_tickers:
        selected_list.append("SEC EDGAR")
    if roic_query:
        selected_list.append("ROIC AI")
    if financial_ticker:
        selected_list.append("Alpha Vantage")
    if companies_house_query:
        selected_list.append("Companies House")
    if requires_web:
        selected_list.append("Web Search")

    skipped_list = []
    if document_ids and not is_document_question:
        skipped_list.append("RAG (not relevant)")

    logger.info(
        build_routing_log(
            content=request.content,
            document_ids=document_ids,
            is_document_question=is_document_question,
            selected_providers=selected_list,
            skipped_providers=skipped_list,
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
        follow_up_questions = []
        citations = []
        source_documents = None
        source_attribution = []
        intent = "Other"
    else:
        # Context Retrieval (concurrent execution of selected providers)
        futures = {}
        user_id_str = str(current_user.id)

        # We define local helper wrappers so their names (fn.__name__) match
        # what the unit test mocks expect when inspecting the submitted tasks.
        def retrieve_documents():
            return rag.retrieve_documents_context(
                db,
                is_workspace_analysis,
                wsName,
                user_id_str,
                document_ids,
                session.workspace_id,
                optimized_prompt,
                chat_id,
            )

        def retrieve_sec_company_facts():
            return rag.retrieve_sec_company_facts_context(
                chat_id,
                user_id_str,
                session.workspace_id,
                sec_tickers,
                sec_options,
            )

        def retrieve_roic_ai():
            return rag.retrieve_roic_ai_context(
                chat_id,
                user_id_str,
                session.workspace_id,
                roic_query,
            )

        def retrieve_financial_news():
            return rag.retrieve_financial_news_context(
                chat_id,
                user_id_str,
                session.workspace_id,
                financial_ticker,
            )

        def retrieve_companies_house():
            return rag.retrieve_companies_house_context(
                chat_id,
                user_id_str,
                session.workspace_id,
                companies_house_query,
            )

        with ThreadPoolExecutor(max_workers=6) as executor:
            if is_document_question:
                futures["rag"] = executor.submit(retrieve_documents)
            if sec_tickers:
                futures["sec"] = executor.submit(retrieve_sec_company_facts)
            if roic_query:
                futures["roic"] = executor.submit(retrieve_roic_ai)
            if financial_ticker:
                futures["news"] = executor.submit(retrieve_financial_news)
            if companies_house_query:
                futures["companies_house"] = executor.submit(retrieve_companies_house)

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
            document_context = (
                rag.build_context_with_citations(retrieved_chunks) or None
            )
        if source_documents:
            source_documents = document_sources_only(source_documents)
        used_rag = document_context is not None

        if financial_news_response:
            financial_context, financial_sources = rag.build_financial_news_context(
                financial_news_response, start_number=len(retrieved_chunks or []) + 1
            )
            financial_context = financial_context or None
        if sec_company_facts_response:
            sec_results = (
                sec_company_facts_response
                if isinstance(sec_company_facts_response, list)
                else [sec_company_facts_response]
            )
            sec_context, sec_sources = rag.build_sec_contexts(
                sec_results,
                start_number=len(retrieved_chunks or []) + len(financial_sources) + 1,
            )
            sec_context = sec_context or None
            if sec_context is None and not used_rag:
                error_messages = [
                    result.get("error", {}).get("message")
                    for result in sec_results
                    if result.get("error", {}).get("message")
                ]
                sec_context = "SEC EDGAR data is unavailable: " + (
                    "; ".join(error_messages) or "no validated facts were returned."
                ) + "\n\nCRITICAL INSTRUCTION: Since the requested SEC filing data is unavailable, explicitly state that no filing was found and explain why using the error above. DO NOT fabricate or replace the answer with generic educational content."
        if roic_response:
            roic_context, roic_sources = rag.build_roic_context(
                roic_response,
                start_number=len(retrieved_chunks or [])
                + len(financial_sources)
                + len(sec_sources)
                + 1,
                query=optimized_prompt,
            )
            roic_context = roic_context or None
            if roic_context and used_rag and "ROIC.ai Error" in roic_context:
                roic_context = None
                roic_sources = []

        if companies_house_response:
            companies_house_context, companies_house_sources = (
                rag.build_companies_house_context(
                    companies_house_response,
                    start_number=len(retrieved_chunks or [])
                    + len(financial_sources)
                    + len(sec_sources)
                    + len(roic_sources)
                    + 1,
                )
            )
            companies_house_context = companies_house_context or None
            if (
                companies_house_context
                and used_rag
                and "No Companies House records" in companies_house_context
            ):
                companies_house_context = None
                companies_house_sources = []

        # Post-Provider Validation Layer
        mcp_executed = bool(financial_news_response or sec_company_facts_response or roic_response or companies_house_response)
        has_data = False
        
        if financial_news_response and rag.has_useful_data(financial_news_response):
            has_data = True
        if sec_company_facts_response and rag.has_useful_data(sec_company_facts_response):
            has_data = True
        if roic_response and rag.has_useful_data(roic_response):
            has_data = True
        if companies_house_response and rag.has_useful_data(companies_house_response):
            has_data = True
            
        fallback_triggered = False
        if mcp_executed and not has_data:
            if sec_company_facts_response is not None:
                # Per requirements, SEC EDGAR must not fallback to web search.
                # It must use the generated sec_context to explicitly state no filing was found.
                pass
            else:
                fallback_triggered = True
                requires_web = True
                fallback_reason = "Provider returned no usable business data."
                selected_prov = "Alpha Vantage" if financial_news_response else "ROIC" if roic_response else "Companies House" if companies_house_response else "Unknown"
                logger.info(
                    f"Planner Selected:\n{selected_prov}\n\n"
                    "Provider Response:\nEmpty\n\n"
                    "Fallback Triggered:\nTrue\n\n"
                    "Fallback Provider:\nWeb Search\n\n"
                    f"Search Query:\n{optimized_prompt}\n\n"
                    "Fallback Reason:\nProvider returned no usable business data."
                )
                # Clear empty/error contexts and sources so they don't corrupt citation indices
                financial_context = None
                financial_sources = []
                roic_context = None
                roic_sources = []
                companies_house_context = None
                companies_house_sources = []

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

            if fallback_triggered:
                logger.info(
                    "Fallback Triggered: True\n"
                    f"Search Query: {optimized_prompt}\n"
                    f"Search Results Count: {len(web_sources) if web_sources else 0}\n"
                    f"Generated Web Context Length: {len(web_context) if web_context else 0}\n"
                )

        if is_workspace_analysis:
            intent = "Research / Deep Analysis"
        else:
            intent = rag.classify_intent(
                client, question=optimized_prompt, model=settings.LLM_MODEL
            )

        exec_result = "Success" if has_data else "Empty" if mcp_executed else "Skipped"
        rows_returned = "N/A"
        if has_data:
            if financial_news_response: rows_returned = len(financial_news_response) if isinstance(financial_news_response, list) else 1
            if sec_company_facts_response: rows_returned = len(sec_company_facts_response) if isinstance(sec_company_facts_response, list) else 1
            if roic_response: rows_returned = 1
            if companies_house_response: rows_returned = len(companies_house_response) if isinstance(companies_house_response, list) else 1
            
        final_provider = "Web Search" if requires_web and (fallback_triggered or not mcp_executed) else selected_provider

        evaluated_providers = (
            f"SEC={sec_applicable}, AlphaVantage={alpha_applicable}, ROIC={roic_applicable}, CompaniesHouse={ch_applicable}"
        )
        
        final_sources = []
        if financial_context: final_sources.append("Alpha Vantage")
        if sec_context: final_sources.append("SEC EDGAR")
        if roic_context: final_sources.append("ROIC")
        if companies_house_context: final_sources.append("Companies House")
        if web_context: final_sources.append("Web Search")
        if document_context: final_sources.append("RAG Docs")

        if bool(document_ids) and is_document_intent and final_provider == "Web Search":
            logger.warning(
                "WARNING:\n"
                "Document exists but planner selected Web Search.\n\n"
                f"Reason:\n{routing_reason}\n\n"
                "Expected Provider:\nRAG"
            )

        logger.info(
            f"--- ROUTING PIPELINE AUDIT ---\n"
            f"Active Documents: {bool(document_ids)}\n"
            f"Document Intent: {is_document_intent}\n"
            f"RAG Classifier Result: {is_document_question}\n"
            f"Planner Document Relevant: {selected_provider == 'RAG' or bool(document_ids)}\n"
            f"Selected Provider: {selected_provider}\n"
            f"Fallback Triggered: {fallback_triggered or (not mcp_selected)}\n"
            f"Fallback Reason: {fallback_reason}\n"
            f"Chunks Retrieved: {bool(retrieved_chunks)}\n"
            f"Chunk Count: {len(retrieved_chunks) if retrieved_chunks else 0}\n"
            f"RAG Executed: {used_rag}\n"
            f"Web Search Executed: {used_web}\n"
            f"------------------------------\n"
        )

        llm_history = [] if context_reason == "new_upload" else history
        
        logger.info(
            "Prompt Context Summary\n\n"
            f"Finance Context: {bool(financial_context)}\n"
            f"SEC Context: {bool(sec_context)}\n"
            f"ROIC Context: {bool(roic_context)}\n"
            f"Companies House Context: {bool(companies_house_context)}\n"
            f"Web Context: {bool(web_context)}\n"
            f"RAG Context: {bool(document_context)}\n\n"
            f"Final Context Sources: {[s.get('source_type') if isinstance(s, dict) else getattr(s, 'source_type', 'DocumentChunk') for s in (financial_sources + sec_sources + roic_sources + companies_house_sources + (web_sources or []) + list(retrieved_chunks or []))]}"
        )
        
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
        )
        mode_used = "auto"
        start_time = _time.time()
        fallback_reason = None
        selected_model = settings.LLM_MODEL
        provider = "openai"

        if getattr(request, "mode", None) == "think_longer":
            today_start = datetime.combine(datetime.now(timezone.utc).date(), time.min)
            daily_calls = crud.chat_message.count_user_daily_calls(
                db=db,
                user_id=str(current_user.id),
                mode_prefix="think_longer",
                since=today_start
            )
            
            has_context = any([
                document_context, web_context, financial_context, 
                sec_context, roic_context, companies_house_context
            ])

            if daily_calls >= settings.THINK_LONGER_DAILY_LIMIT:
                fallback_reason = f"user {current_user.id} reached daily limit ({daily_calls}/{settings.THINK_LONGER_DAILY_LIMIT})"
                logger.info(f"Think Longer skipped: {fallback_reason}")
                selected_model = settings.LLM_MODEL
                raw_response = rag.generate_response(client, messages=messages, model=settings.LLM_MODEL)
            elif not has_context:
                fallback_reason = "no context to reason over"
                logger.info("Think Longer skipped: no context to reason over")
                selected_model = settings.LLM_MODEL
                raw_response = rag.generate_response(client, messages=messages, model=settings.LLM_MODEL)
            else:
                try:
                    think_longer_messages = messages + [{"role": "system", "content": think_longer_reasoning_prompt()}]
                    selected_model = settings.THINK_LONGER_MODEL
                    raw_response = rag.generate_think_longer_response(
                        client,
                        messages=think_longer_messages,
                        model=settings.THINK_LONGER_MODEL,
                        reasoning_effort=settings.THINK_LONGER_REASONING_EFFORT,
                    )
                    mode_used = "think_longer"
                except ThinkLongerUnavailable as e:
                    fallback_reason = str(e)
                    logger.warning("Falling back to standard model for think_longer mode")
                    logger.exception("Error during think_longer response generation: %s", fallback_reason)
                    selected_model = settings.THINK_LONGER_FALLBACK_MODEL
                    raw_response = rag.generate_response(
                        client, messages=messages, model=settings.THINK_LONGER_FALLBACK_MODEL
                    )
                    mode_used = "think_longer_fallback"

        elif getattr(request, "mode", None) == "deep_research":
            orchestrator = DeepResearchOrchestrator(client, rag, db, current_user, session)
            
            initial_contexts = []
            if document_context: initial_contexts.append(document_context)
            if web_context: initial_contexts.append(web_context)
            if financial_context: initial_contexts.append(financial_context)
            if sec_context: initial_contexts.append(sec_context)
            if roic_context: initial_contexts.append(roic_context)
            if companies_house_context: initial_contexts.append(companies_house_context)
            
            initial_context_str = "\n\n".join(initial_contexts)
            
            final_research_context, all_chunks, all_news, all_sec, all_roic, all_ch, all_web = orchestrator.execute(
                initial_query=optimized_prompt,
                chat_id=chat_id,
                document_ids=document_ids,
                history=llm_history,
                initial_context=initial_context_str
            )
            
            if all_chunks: retrieved_chunks = list(retrieved_chunks or []) + all_chunks
            if all_news: financial_sources.extend(all_news)
            if all_sec: sec_sources.extend(all_sec)
            if all_roic: roic_sources.extend(all_roic)
            if all_ch: companies_house_sources.extend(all_ch)
            if all_web:
                if web_sources is None: web_sources = []
                web_sources.extend(all_web)
            
            if final_research_context and len(final_research_context) > 10000:
                final_research_context = final_research_context[:10000] + "\n\n... [Truncated for context length limits]"

            report_prompt = (
                f"You are a professional research analyst. Using the following comprehensive and consolidated research context, generate a detailed, structured, and final research report that fully addresses the original query: '{optimized_prompt}'.\n\n"
                "Your report MUST contain the following sections:\n"
                "# Executive Summary (Summarize the research)\n"
                "# Research Objective (Clearly state what was investigated)\n"
                "# Research Methodology (Briefly explain topics researched, sources consulted, and evidence synthesis process)\n"
                "# Key Findings (Present the most important discoveries, prioritized)\n"
                "# Detailed Analysis (Explain every topic separately. Include Evidence, Interpretation, Reasoning, Implications)\n"
                "# Cross-Topic Analysis (Explain relationships, trends, and patterns between topics)\n"
                "# Trade-Off Analysis (Explain competing viewpoints, e.g., Benefits vs Risks, Short-term vs Long-term)\n"
                "# Evidence Gaps (Identify unknowns, missing information, areas for further research. Do not invent answers)\n"
                "# Recommendations (Provide evidence-based recommendations. Explain why they follow from the research)\n"
                "# Final Conclusion (Synthesize the entire report. Do not repeat earlier sections. Provide final judgment)\n"
                "# Confidence Assessment (Rate confidence as High, Moderate, or Low and explain why)\n\n"
                f"Context:\n{final_research_context}"
            ) + followup_instruction_block()
            
            messages = [{"role": "user", "content": report_prompt}]
            try:
                selected_model = settings.DEEP_RESEARCH_MODEL
                raw_response = rag.generate_think_longer_response(
                    client,
                    messages=messages,
                    model=settings.DEEP_RESEARCH_MODEL,
                    reasoning_effort=settings.DEEP_RESEARCH_REASONING_EFFORT,
                )
                mode_used = "deep_research"
            except ThinkLongerUnavailable as e:
                fallback_reason = str(e)
                logger.warning("Falling back to standard model for deep_research mode")
                logger.exception("Error during deep_research response generation: %s", fallback_reason)
                selected_model = settings.THINK_LONGER_FALLBACK_MODEL
                raw_response = rag.generate_response(
                    client, messages=messages, model=settings.THINK_LONGER_FALLBACK_MODEL
                )
                mode_used = "deep_research_fallback"
        else:
            selected_model = settings.LLM_MODEL
            raw_response = rag.generate_response(
                client, messages=messages, model=settings.LLM_MODEL
            )

        execution_time = _time.time() - start_time
        logger.info(
            f"--- LLM GENERATION AUDIT ---\n"
            f"Mode: {mode_used}\n"
            f"Model: {selected_model}\n"
            f"Provider: {provider}\n"
            f"Execution Time: {execution_time:.2f}s\n"
            f"Fallback Reason: {fallback_reason}\n"
            f"----------------------------\n"
        )

        ai_response, follow_up_questions = rag.split_answer_and_followups(raw_response)

        cited_sources: list = list(retrieved_chunks or [])
        cited_sources.extend(financial_sources)
        cited_sources.extend(sec_sources)
        cited_sources.extend(roic_sources)
        cited_sources.extend(companies_house_sources)
        if used_web:
            cited_sources.extend(web_sources or [])

        if cited_sources:
            ai_response, citations = citation_svc.generate_citations(
                ai_response, cited_sources
            )
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
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "chat_session_id": chat_id,
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "response": ai_response,
                "document_ids": document_ids,
                "active_document_ids": document_ids,
                "mode_used": mode_used,
                "uploaded_document_ids": (
                    document_ids if context_reason == "new_upload" else []
                ),
                "used_documents": used_rag,
                "used_web_search": used_web,
                "source_documents": source_documents,
                "citations": citations,
                "source_attribution": source_attribution,
                "web_sources": web_sources,
                "financial_news": financial_news_response,
                "sec_company_facts": sec_company_facts_response,
                "roic_ai": roic_response,
                "companies_house": companies_house_response,
                "follow_up_questions": follow_up_questions,
            },
            "message": "Message added and AI response created successfully.",
        },
    )
