import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from groq import Groq

from server import crud, schemas
from server.api_v1.deps import get_db
from server.config import settings
from server.utils.auth import get_current_user
from server.utils.common import (
    agent_search,
    ingest_file_into_workspace,
    search_similarity,
)
from server.utils.prompt import (
    ai_assistent,
    doc_agent_system_agent,
    doom_agent_system_prompt,
)

client = Groq(api_key=settings.GROQ_API_KEY)

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
    session = crud.chat_session.get(db, chat_id)
    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    prior_messages = crud.chat_message.get_multi_by_chat(db, chat_id=chat_id)
    message_history = [
        {"role": msg.role, "content": msg.content} for msg in prior_messages
    ]
    message_history.append({"role": "user", "content": request.content})

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

    ai_response = ""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message_history,
        temperature=0,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if content:
                ai_response += content

    assistant_message = crud.chat_message.create(
        db=db,
        obj_in=schemas.ChatMessageCreate(
            chat_id=chat_id,
            content=ai_response,
            role="assistant",
            user_id=str(current_user.id),
            workspace_id=session.workspace_id,
            document_ids=request.document_ids,
        ),
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "response": ai_response,
                "document_ids": request.document_ids,
            },
            "message": "Message added and AI response created successfully.",
        },
    )


@ai_router.post("/ai-agent")
def ai_agent(
    request: schemas.GenAIBase,
    workspace_id: Optional[str] = None,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = crud.chat_session.create(
        db=db,
        obj_in=schemas.ChatSessionCreate(
            user_id=str(current_user.id), title=request.title, workspace_id=workspace_id
        ),
    )

    file_names = None

    if "@doom" in request.user_query:
        logger.info("This is the agentic flow. User query contains '@doom' trigger.")

        organic_results = agent_search(request.user_query.replace("@doom", "").strip())

        message = [
            {"role": "system", "content": doom_agent_system_prompt()},
            {
                "role": "system",
                "content": f"Search Results:\n\n{json.dumps(organic_results, indent=2)}",
            },
            {"role": "user", "content": request.user_query},
        ]

    elif "@doc" in request.user_query:

        logger.info(
            "grok_ai: user=%s starting AI request; query_len=%d",
            getattr(current_user, "id", None),
            len(request.user_query) if getattr(request, "user_query", None) else 0,
        )

        if workspace_id:
            workspace = crud.workspace._get_owned_workspace(
                db, workspace_id, str(current_user.id)
            )
            if workspace is None:
                raise HTTPException(status_code=404, detail="Workspace not found")

        similarity_results, file_names = search_similarity(
            request.user_query, current_user, top_k=3, workspace_id=workspace_id
        )
        logger.debug(
            "grok_ai: user=%s similarity results=%d",
            getattr(current_user, "id", None),
            len(similarity_results) if similarity_results is not None else 0,
        )
        message = [
            {"role": "system", "content": doc_agent_system_agent(similarity_results)},
            {"role": "user", "content": request.user_query},
        ]

    else:
        logger.info("This is the AI assistent flow.")
        message = [
            {"role": "system", "content": ai_assistent()},
            {"role": "user", "content": request.user_query},
        ]

    ai_response = ""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message,
        temperature=0,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if content:
                ai_response += content

    logger.info(
        "grok_ai: user=%s AI response generated; response_len=%d",
        getattr(current_user, "id", None),
        len(ai_response),
    )

    if session is not None:
        crud.chat_message.create(
            db=db,
            obj_in=schemas.ChatMessageCreate(
                chat_id=session.id,
                content=request.user_query,
                role="user",
                user_id=str(current_user.id),
                workspace_id=workspace_id or session.workspace_id,
                document_ids=request.document_ids,
            ),
        )
        crud.chat_message.create(
            db=db,
            obj_in=schemas.ChatMessageCreate(
                chat_id=session.id,
                content=ai_response,
                role="assistant",
                user_id=str(current_user.id),
                workspace_id=workspace_id or session.workspace_id,
                document_ids=request.document_ids,
                source_documents=file_names,
            ),
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "prompt": request.user_query,
                "response": ai_response,
                "file_names": file_names if "@doc" in request.user_query else None,
                "chat_session_id": session.id if session else None,
            },
            "message": "AI response generated successfully.",
        },
    )


# @ai_router.post("/upload-docs")
# async def upload_docs(
#     workspace_id: str,
#     file: UploadFile = File(...),
#     db=Depends(get_db),
#     current_user=Depends(get_current_user),
# ):
#     """Upload a document into a workspace.
#
#     Files always belong to a workspace: the caller must own ``workspace_id``.
#     The file's text is chunked, embedded and stored (tagged with the workspace)
#     so a chat inside the workspace can retrieve it via the ``@doc`` agent.
#     """
#     workspace = crud.workspace._get_owned_workspace(
#         db, workspace_id, str(current_user.id)
#     )
#     if workspace is None:
#         raise HTTPException(status_code=404, detail="Workspace not found")
#
#     logger.info(
#         "create_embedding: user=%s workspace=%s uploaded file=%s",
#         getattr(current_user, "id", None),
#         workspace_id,
#         file.filename,
#     )
#
#     await ingest_file_into_workspace(
#         db, file=file, user_id=current_user.id, workspace_id=workspace_id
#     )
#
#     return JSONResponse(
#         status_code=200,
#         content={
#             "success": True,
#             "error": None,
#             "data": None,
#             "message": "File uploaded successfully.",
#         },
#     )
