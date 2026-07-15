import asyncio
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
from server.services.financial_news_mcp import financial_news
from server.services.rag_service import get_rag_service
from server.services.sec_edgar_mcp import sec_company_facts
from server.services.roic_ai_mcp import roic_ai
from server.utils.auth import get_current_user
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

    db.delete(session)
    db.commit()

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

    conversation = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)
    messages = [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "source_documents": document_sources_only(cited_document_names(msg.citations)),
            "citations": msg.citations,
            "source_attribution": source_attribution_from_citations(msg.citations),
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }
        for msg in conversation
    ]
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

    conversation = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)

    seen_document_ids: set = set()
    uploaded_by_message: dict = {}
    for msg in conversation:
        ids = msg.document_ids or []
        uploaded_by_message[msg.id] = [d for d in ids if d not in seen_document_ids]
        seen_document_ids.update(ids)

    doc_info_cache: dict = {}

    def _doc_info(doc_id):
        if doc_id not in doc_info_cache:
            doc = crud.document.get(db, doc_id)
            doc_info_cache[doc_id] = (
                {
                    "id": doc.id,
                    "file_name": doc.file_name,
                    "file_format": doc.file_format,
                }
                if doc
                else None
            )
        return doc_info_cache[doc_id]

    messages = []
    for msg in conversation:
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
                "source_documents": document_sources_only(cited_document_names(msg.citations)),
                "citations": msg.citations,
                "source_attribution": source_attribution_from_citations(msg.citations),
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "updated_at": msg.updated_at.isoformat() if msg.updated_at else None,
            }
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

    session.title = request.title
    db.add(session)
    db.commit()
    db.refresh(session)

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

    user_message = crud.chat_message.create(
        db=db,
        obj_in=schemas.ChatMessageCreate(
            chat_id=chat_id,
            content=request.content,
            role="user",
            user_id=str(current_user.id),
            workspace_id=session.workspace_id,
            document_ids=document_ids or None,
        ),
    )

    rag = get_rag_service()
    citation_svc = get_citation_service()

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
    document_context = None
    web_context = None
    retrieved_chunks = None

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
    roic_query = None if sec_tickers else rag.classify_roic_ai_query(
        client, question=request.content, model=settings.LLM_MODEL
    )
    if not roic_query and not sec_tickers and rag.is_earnings_transcript_query(request.content):
        # Transcript language has a deterministic meaning; active workspace
        # documents must not silently convert it into a document-only request.
        roic_query = rag.transcript_company_query(request.content)
        logger.info("chat.message: transcript intent fallback routed to Roic query=%r", roic_query)
    financial_ticker = None if (sec_tickers or roic_query) else rag.classify_financial_news_ticker(
        client, question=request.content, model=settings.LLM_MODEL
    )
    explicit_document_request = any(
        term in request.content.lower()
        for term in (
            "my document", "uploaded document", "uploaded file", "uploaded report",
            "my report", "this report", "this pdf", "this file", "annual report",
            "page ", "compare my", "compare the report", "compare with my",
        )
    )
    if (sec_tickers or roic_query or financial_ticker) and not explicit_document_request:
        # Attached files remain available as conversation state, but external
        # intents do not retrieve or cite them unless the user asks for a hybrid comparison.
        is_document_question = False
    if not financial_ticker:
        finance_words = {"article", "articles", "earnings", "financial", "market", "news", "sentiment"}
        if any(word in request.content.lower() for word in finance_words):
            for message in reversed(prior_messages):
                for citation in message.citations or []:
                    if citation.get("source_type") == "financial_news" and citation.get("ticker"):
                        financial_ticker = citation["ticker"]
                        break
                if financial_ticker:
                    break
    # Finance queries use Alpha Vantage. Other live-information questions retain
    # the existing generic web-search flow unchanged.
    requires_web = not (financial_ticker or sec_ticker or roic_query) and rag.classify_requires_web_search(
        client, question=request.content, model=settings.LLM_MODEL
    )

    def retrieve_documents():
        """Retrieve document context without allowing a retrieval failure to fail chat."""
        try:
            _, sources, chunks = rag.retrieve_context_with_chunks(
                db,
                query=request.content,
                user_id=str(current_user.id),
                document_ids=document_ids,
                top_k=3,
            )
            return sources, chunks
        except Exception:
            logger.exception(
                "chat.message: document retrieval failed; skipping RAG (chat=%s)",
                chat_id,
            )
            return None, None

    def retrieve_financial_news():
        return asyncio.run(
            financial_news(
                financial_ticker, limit=settings.ALPHA_VANTAGE_NEWS_MAX_RESULTS
            )
        )

    def retrieve_sec_company_facts():
        async def _retrieve_all():
            return await asyncio.gather(
                *(sec_company_facts(ticker=ticker, **sec_options) for ticker in sec_tickers)
            )
        return asyncio.run(_retrieve_all())

    def retrieve_roic_ai():
        return asyncio.run(roic_ai(query=roic_query, transcript=True))

    if is_document_question and (financial_ticker or sec_tickers or roic_query):
        # Hybrid queries retrieve document context and the selected MCP source
        # concurrently; neither source waits for the other.
        with ThreadPoolExecutor(max_workers=2) as executor:
            document_future = executor.submit(retrieve_documents)
            mcp_future = executor.submit(
                retrieve_sec_company_facts if sec_tickers else (retrieve_roic_ai if roic_query else retrieve_financial_news)
            )
            source_documents, retrieved_chunks = document_future.result()
            if sec_tickers:
                sec_company_facts_response = mcp_future.result()
            elif roic_query:
                roic_response = mcp_future.result()
            else:
                financial_news_response = mcp_future.result()
    elif is_document_question:
        source_documents, retrieved_chunks = retrieve_documents()
    elif financial_ticker:
        financial_news_response = retrieve_financial_news()
    elif sec_tickers:
        sec_company_facts_response = retrieve_sec_company_facts()
    elif roic_query:
        roic_response = retrieve_roic_ai()

    if retrieved_chunks:
        document_context = rag.build_context_with_citations(retrieved_chunks) or None
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
        if sec_context is None:
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
            start_number=len(retrieved_chunks or []) + len(financial_sources) + len(sec_sources) + 1,
            query=request.content,
        )
        roic_context = roic_context or None

    if requires_web:
        # Web results continue the document chunks' [n] numbering so the LLM sees
        # one unbroken citation sequence across the whole CONTEXT.
        web_context, web_sources = rag.web_search(
            request.content,
            start_number=(
                len(retrieved_chunks or []) + len(financial_sources) + len(sec_sources) + len(roic_sources) + 1
            ),
        )
        web_context = web_context or None
        used_web = web_context is not None

    logger.info(
        "chat.message: chat=%s user=%s attached_docs=%d used_rag=%s used_web=%s finance=%s sec=%s",
        chat_id,
        getattr(current_user, "id", None),
        len(document_ids),
        used_rag,
        used_web,
        financial_ticker,
        sec_tickers,
    )

    llm_history = [] if context_reason == "new_upload" else history
    messages = rag.build_messages(
        history=llm_history,
        question=request.content,
        retrieved_context=document_context,
        web_context=web_context,
        financial_news_context=financial_context,
        sec_context=sec_context,
        roic_context=roic_context,
    )
    raw_response = rag.generate_response(
        client, messages=messages, model=settings.LLM_MODEL
    )
    ai_response, follow_up_questions = rag.split_answer_and_followups(raw_response)

    # The ordered source list must mirror the numbering the LLM saw in the CONTEXT:
    # Document chunks, financial-news articles, and web results are retained in
    # the exact numbered order shown to the LLM. Index i maps marker [i + 1].
    cited_sources: list = list(retrieved_chunks or [])
    cited_sources.extend(financial_sources)
    cited_sources.extend(sec_sources)
    cited_sources.extend(roic_sources)
    if used_web:
        cited_sources.extend(web_sources or [])

    if cited_sources:
        ai_response, citations = citation_svc.generate_citations(
            ai_response, cited_sources
        )
    else:
        # Nothing grounded the answer, so any marker the model emitted is a
        # hallucination — remove it rather than link it to a source that doesn't exist.
        ai_response = strip_citation_markers(ai_response)

    source_documents = document_sources_only(cited_document_names(citations)) or None
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
                "follow_up_questions": follow_up_questions,
            },
            "message": "Message added and AI response created successfully.",
        },
    )
