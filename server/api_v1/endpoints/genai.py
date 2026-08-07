import logging
import mimetypes
import os
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from server import crud, schemas
from server.api_v1.deps import get_db
from server.config import WORKSPACE_DOCS_DIR
from server.services.audit_logger import AuditLogger
from server.services.chat_orchestration_service import chat_orchestration_service
from server.utils.auth import get_current_user
from server.utils.helpers import document_sources_only, resolve_or_create_session
from server.utils.source_attribution import (
    cited_document_names,
    source_attribution_from_citations,
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
    """Public, unauthenticated read-only view of a chat (for shareable links)."""
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
            "follow_up_questions": [],
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

    conversation = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)

    seen_document_ids: set = set()
    uploaded_by_message: dict = {}
    for msg in conversation:
        ids = msg.document_ids or []
        uploaded_by_message[msg.id] = [d for d in ids if d not in seen_document_ids]
        seen_document_ids.update(ids)

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
                "follow_up_questions": [],
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
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Add a user message and trigger AI completion orchestration."""
    session = resolve_or_create_session(
        db,
        chat_id=chat_id,
        current_user=current_user,
        title=request.title,
        workspace_id=request.workspace_id,
    )
    chat_id = session.id

    return chat_orchestration_service.process_chat_message(
        db=db,
        current_user=current_user,
        chat_id=chat_id,
        session=session,
        request=request,
        background_tasks=background_tasks,
    )
