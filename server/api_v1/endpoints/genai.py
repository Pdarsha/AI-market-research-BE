import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

from server import crud, schemas
from server.api_v1.deps import get_db
from server.config import settings
from server.services.citation_service import get_citation_service
from server.services.rag_service import get_rag_service
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
    has_roic_metric,
    is_explicit_document_request,
    is_overview_request,
    check_user_typed_context,
    check_spurious_roic_query,
    build_routing_log,
)

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1",
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

    from server.services.audit_logger import AuditLogger

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
    from server.services.audit_logger import AuditLogger

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

    from server.services.audit_logger import AuditLogger

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
    from server.services.audit_logger import AuditLogger

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
                import os
                import mimetypes
                from server.config import WORKSPACE_DOCS_DIR

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
    from server.services.audit_logger import AuditLogger

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

    from server.services.audit_logger import AuditLogger

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
        stored_content = request.content
        optimized_prompt = request.content

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

    from server.services.audit_logger import AuditLogger

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
    else:
        has_metric = has_roic_metric(request.content)
        explicit_document_request = is_explicit_document_request(request.content)

        is_document_question = False
        if document_ids:
            is_document_question = rag.classify_requires_document(
                client, question=request.content, model=settings.LLM_MODEL
            )

        sec_tickers = rag.classify_sec_edgar_tickers(
            client, question=request.content, model=settings.LLM_MODEL
        )
        sec_ticker = sec_tickers[0] if sec_tickers else None
        sec_options = rag.sec_request_options(request.content) if sec_tickers else {}

        is_overview = is_overview_request(request.content)

        # ROIC AI
        if has_metric or is_overview:
            if sec_tickers:
                roic_query = sec_tickers[0]
            else:
                roic_query = rag.classify_roic_ai_query(
                    client, question=request.content, model=settings.LLM_MODEL
                )
                if not roic_query:
                    roic_query = rag.transcript_company_query(request.content)
                if not roic_query and len(request.content.strip()) <= 100:
                    roic_query = request.content

            if is_overview:
                if not sec_tickers and roic_query:
                    sec_tickers = [roic_query]
                    sec_ticker = roic_query
                    sec_options = rag.sec_request_options(request.content)
            else:
                sec_tickers = []
                sec_ticker = None
                sec_options = {}
        else:
            roic_query = (
                None
                if sec_tickers
                else rag.classify_roic_ai_query(
                    client, question=request.content, model=settings.LLM_MODEL
                )
            )
            if (
                not roic_query
                and not sec_tickers
                and rag.is_earnings_transcript_query(request.content)
            ):
                roic_query = rag.transcript_company_query(request.content)

        # Alpha Vantage / Financial News
        financial_ticker = None
        wants_news = any(
            term in request.content.lower()
            for term in [
                "news",
                "sentiment",
                "article",
                "feed",
                "recent story",
                "recent stories",
            ]
        )
        if not sec_tickers and not roic_query or wants_news:
            financial_ticker = rag.classify_financial_news_ticker(
                client, question=request.content, model=settings.LLM_MODEL
            )
            if not financial_ticker:
                finance_words = {
                    "article",
                    "articles",
                    "earnings",
                    "financial",
                    "market",
                    "news",
                    "sentiment",
                }
                if any(word in request.content.lower() for word in finance_words):
                    for message in reversed(prior_messages):
                        for citation in message.citations or []:
                            if citation.get(
                                "source_type"
                            ) == "financial_news" and citation.get("ticker"):
                                financial_ticker = citation["ticker"]
                                break
                        if financial_ticker:
                            break

        # Companies House
        companies_house_query = rag.classify_companies_house_query(
            client, question=request.content, model=settings.LLM_MODEL
        )
        if companies_house_query:
            sec_tickers = []
            sec_ticker = None
            sec_options = {}
            roic_query = None
            financial_ticker = None

        # Web Search
        requires_web = not (
            financial_ticker or sec_ticker or roic_query or companies_house_query
        ) and rag.classify_requires_web_search(
            client, question=request.content, model=settings.LLM_MODEL
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
        elif has_external_provider:
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
            request.content,
            document_ids,
            is_document_question,
            selected_list,
            skipped_list,
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
                request.content,
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
                source_documents, retrieved_chunks = futures["rag"].result()
            if "sec" in futures:
                sec_company_facts_response = futures["sec"].result()
            if "roic" in futures:
                roic_response = futures["roic"].result()
            if "news" in futures:
                financial_news_response = futures["news"].result()
            if "companies_house" in futures:
                companies_house_response = futures["companies_house"].result()

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
                )
        if roic_response:
            roic_context, roic_sources = rag.build_roic_context(
                roic_response,
                start_number=len(retrieved_chunks or [])
                + len(financial_sources)
                + len(sec_sources)
                + 1,
                query=request.content,
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

        if requires_web:
            web_context, web_sources = rag.web_search(
                request.content,
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
        else:
            intent = rag.classify_intent(
                client, question=request.content, model=settings.LLM_MODEL
            )

        logger.info(
            "chat.message: chat=%s user=%s attached_docs=%d used_rag=%s used_web=%s finance=%s sec=%s companies_house=%s intent=%s",
            chat_id,
            user_id_str,
            len(document_ids),
            used_rag,
            used_web,
            financial_ticker,
            sec_tickers,
            companies_house_query,
            intent,
        )

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
        )
        raw_response = rag.generate_response(
            client, messages=messages, model=settings.LLM_MODEL
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
