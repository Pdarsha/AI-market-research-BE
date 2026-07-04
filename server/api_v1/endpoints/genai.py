import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

from server import crud, schemas
from server.api_v1.deps import get_db
from server.config import settings
from server.services.rag_service import get_rag_service
from server.utils.auth import get_current_user

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


ai_router = APIRouter(prefix="/ai", tags=["GenAI"])


def _resolve_or_create_session(
    db,
    *,
    chat_id: Optional[str],
    current_user,
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
):
    """Return the caller's chat session for ``chat_id``, creating it if absent.

    Shared by the chat/message and ai-agent endpoints so both handle sessions
    identically:
      * existing session -> returned only if owned by the caller (else 404);
      * missing session  -> created with the given ``chat_id`` (so the id the
        client already uses stays consistent) after validating any workspace.
    """
    if chat_id:
        session = crud.chat_session.get(db, chat_id)
        if session is not None:
            if session.user_id != str(current_user.id):
                raise HTTPException(status_code=404, detail="Chat session not found")
            return session

    if workspace_id:
        workspace = crud.workspace.get(db, workspace_id)
        if not workspace or workspace.user_id != str(current_user.id):
            raise HTTPException(status_code=404, detail="Workspace not found")

    return crud.chat_session.create_with_id(
        db,
        chat_id=chat_id,
        user_id=str(current_user.id),
        title=title,
        workspace_id=workspace_id,
    )


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
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "updated_at": item.updated_at.isoformat() if item.updated_at else None,
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
            "source_documents": msg.source_documents,
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
    messages = [
        {
            "id": msg.id,
            "chat_id": msg.chat_id,
            "role": msg.role,
            "content": msg.content,
            "workspace_id": msg.workspace_id,
            "document_ids": msg.document_ids,
            "source_documents": msg.source_documents,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "updated_at": msg.updated_at.isoformat() if msg.updated_at else None,
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
                "workspace_id": session.workspace_id,
                "document_ids": crud.chat_message.get_latest_document_ids(
                    db, chat_id=chat_id
                ),
                "source_documents": crud.chat_message.get_latest_source_documents(
                    db, chat_id=chat_id, document_ids=crud.chat_message.get_latest_document_ids(db, chat_id=chat_id)
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
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
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
    session = _resolve_or_create_session(
        db,
        chat_id=chat_id,
        current_user=current_user,
        title=request.title,
        workspace_id=request.workspace_id,
    )
    chat_id = session.id

    document_ids = request.document_ids or []

    prior_messages = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)
    history = [{"role": msg.role, "content": msg.content} for msg in prior_messages]

    user_message = crud.chat_message.create(
        db=db,
        obj_in=schemas.ChatMessageCreate(
            chat_id=chat_id,
            content=request.content,
            role="user",
            user_id=str(current_user.id),
            workspace_id=session.workspace_id,
            document_ids=request.document_ids,
        ),
    )

    used_rag = False
    source_documents = None
    retrieved_context = None

    if document_ids:
        requires_doc = get_rag_service().classify_requires_document(
            client, question=request.content, model=settings.LLM_MODEL
        )
        if requires_doc:
            try:
                retrieved_context, source_documents = get_rag_service().retrieve_context(
                    db,
                    query=request.content,
                    user_id=str(current_user.id),
                    document_ids=document_ids,
                    top_k=3,
                )
                used_rag = True
            except Exception:
                logger.exception(
                    "chat.message: document retrieval failed; falling back to "
                    "plain chat (chat=%s)",
                    chat_id,
                )
                retrieved_context = None
                source_documents = None

    logger.info(
        "chat.message: chat=%s user=%s attached_docs=%d used_rag=%s",
        chat_id,
        getattr(current_user, "id", None),
        len(document_ids),
        used_rag,
    )

    messages = get_rag_service().build_messages(
        history=history,
        question=request.content,
        retrieved_context=retrieved_context,
    )
    ai_response = get_rag_service().generate_response(client, messages=messages, model=settings.LLM_MODEL)

    assistant_message = crud.chat_message.create(
        db=db,
        obj_in=schemas.ChatMessageCreate(
            chat_id=chat_id,
            content=ai_response,
            role="assistant",
            user_id=str(current_user.id),
            workspace_id=session.workspace_id,
            document_ids=request.document_ids,
            source_documents=source_documents,
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
                "document_ids": request.document_ids,
                "used_documents": used_rag,
                "source_documents": source_documents,
            },
            "message": "Message added and AI response created successfully.",
        },
    )
