import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse

from server import crud, schemas
from server.api_v1.deps import get_db
from server.models.genai import ChatSession
from server.services.document_processor import get_document_processor
from server.utils.auth import get_current_user
from server.utils.helpers import serialize_chat, serialize_document, serialize_workspace
from server.utils.source_attribution import (
    cited_document_names,
    source_attribution_from_citations,
)
from server.models.genai import ChatMessage
import uuid as _uuid
from server.services.audit_logger import AuditLogger
from server.services.document_takeaways_service import document_takeaways_service


logger = logging.getLogger(__name__)

workspace_router = APIRouter(prefix="/workspaces", tags=["Workspace"])


@workspace_router.post("/create-workspace")
def create_workspace(
    request: schemas.WorkspaceCreate,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    workspace = crud.workspace.create(
        db=db,
        obj_in=schemas.WorkspaceCreate(
            name=request.name,
            description=request.description,
            user_id=str(current_user.id),
        ),
    )

    AuditLogger.log(
        action="WORKSPACE_CREATE",
        user_id=str(current_user.id),
        workspace_id=workspace.id,
        endpoint="/v1/workspaces/create-workspace",
        http_method="POST",
        status="201",
        metadata={"name": workspace.name},
    )

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "error": None,
            "data": serialize_workspace(workspace, chat_count=0),
            "message": "Workspace created successfully.",
        },
    )


@workspace_router.get("/list-workspaces")
def list_workspaces(
    db=Depends(get_db),
    current_user=Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
):
    workspaces = crud.workspace.get_multi(
        db, user_id=str(current_user.id), skip=skip, limit=limit
    )

    AuditLogger.log(
        action="WORKSPACE_LIST",
        user_id=str(current_user.id),
        endpoint="/v1/workspaces/list-workspaces",
        http_method="GET",
        status="200",
        metadata={"count": len(workspaces)},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": [
                serialize_workspace(ws, chat_count=len(ws.chats)) for ws in workspaces
            ],
            "message": "Workspaces fetched successfully.",
        },
    )


@workspace_router.get("/public/{workspace_id}")
def public_workspace(
    workspace_id: str,
    db=Depends(get_db),
):
    """Public, unauthenticated view of a workspace."""
    workspace = crud.workspace.get(db, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    files = crud.document.get_multi_by_workspace(db, workspace_id=workspace_id)
    chats = sorted(
        workspace.chats,
        key=lambda c: c.updated_at or c.created_at,
        reverse=True,
    )

    raw_grids = crud.document_takeaways.get_grids_by_workspace(db, workspace_id=workspace_id)
    serialized_grids = [
        document_takeaways_service.get_cached_serialized_grid(g, db)
        for g in raw_grids
    ]

    data = {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "created_at": (
            workspace.created_at.isoformat() if workspace.created_at else None
        ),
        "files": [serialize_document(f) for f in files],
        "chats": [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "source_documents": cited_document_names(m.citations),
                        "citations": m.citations,
                        "source_attribution": source_attribution_from_citations(
                            m.citations
                        ),
                        "created_at": (
                            m.created_at.isoformat() if m.created_at else None
                        ),
                    }
                    for m in crud.chat_message.get_visible_by_chat(db, chat_id=c.id)
                ],
            }
            for c in chats
        ],
        "grids": serialized_grids,
    }

    AuditLogger.log(
        action="WORKSPACE_GET_PUBLIC",
        workspace_id=workspace_id,
        endpoint=f"/v1/workspaces/public/{workspace_id}",
        http_method="GET",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": data,
            "message": "Public workspace fetched successfully.",
        },
    )


@workspace_router.get("/{workspace_id}")
def get_workspace(
    workspace_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    chats = sorted(
        workspace.chats,
        key=lambda c: c.updated_at or c.created_at,
        reverse=True,
    )
    data = serialize_workspace(workspace, chat_count=len(chats))
    data["chats"] = [serialize_chat(chat) for chat in chats]
    workspace_files = crud.document.get_multi_by_workspace(
        db, workspace_id=workspace_id
    )
    files_serialized = []
    for f in workspace_files:
        chat_count = crud.document.get_chat_count_for_document(db, doc_id=f.id)
        chat_ids = crud.document.get_chat_ids_for_document(db, doc_id=f.id)
        logger.info(
            "TEMPORARY DEBUG - get_workspace - doc_id: %s, workspace_id: %s, associated_chat_ids_from_db: %s, returned_chat_count: %d, returned_chat_ids: %s",
            f.id,
            workspace_id,
            chat_ids,
            chat_count,
            chat_ids,
        )
        files_serialized.append(
            serialize_document(
                f,
                chat_count=chat_count,
                chat_ids=chat_ids,
            )
        )
    data["files"] = files_serialized

    AuditLogger.log(
        action="WORKSPACE_GET",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        endpoint=f"/v1/workspaces/{workspace_id}",
        http_method="GET",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": data,
            "message": "Workspace details fetched successfully.",
        },
    )


@workspace_router.put("/{workspace_id}")
def update_workspace(
    workspace_id: str,
    request: schemas.WorkspaceUpdate,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    workspace = crud.workspace.update(db, db_obj=workspace, obj_in=request)

    AuditLogger.log(
        action="WORKSPACE_UPDATE",
        user_id=str(current_user.id),
        workspace_id=workspace.id,
        endpoint=f"/v1/workspaces/{workspace_id}",
        http_method="PUT",
        status="200",
        metadata={"name": workspace.name},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialize_workspace(workspace),
            "message": "Workspace updated successfully.",
        },
    )


@workspace_router.delete("/{workspace_id}")
def delete_workspace(
    workspace_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    crud.workspace.delete_workspace(db, workspace=workspace)

    AuditLogger.log(
        action="WORKSPACE_DELETE",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        endpoint=f"/v1/workspaces/{workspace_id}",
        http_method="DELETE",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "Workspace deleted successfully.",
        },
    )


@workspace_router.post("/upload-docs")
async def upload_docs(
    workspace_id: str,
    request: Request,
    doc: UploadFile = File(...),
    chat_id: Optional[str] = None,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload a document into a workspace.

    Files always belong to a workspace: the caller must own ``workspace_id``.
    The file's text is chunked, embedded and stored (tagged with the workspace)
    so a chat inside the workspace can retrieve it via the ``@doc`` agent.
    """
    if doc.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

    # Fallback to form data if parameters are None/empty
    form_data = await request.form()
    resolved_workspace_id = workspace_id or form_data.get("workspace_id")
    resolved_chat_id = chat_id or form_data.get("chat_id")

    if resolved_workspace_id and not isinstance(resolved_workspace_id, str):
        resolved_workspace_id = str(resolved_workspace_id)
    if resolved_chat_id and not isinstance(resolved_chat_id, str):
        resolved_chat_id = str(resolved_chat_id)

    if not resolved_workspace_id:
        raise HTTPException(status_code=400, detail="Workspace ID is required.")

    workspace = crud.workspace._get_owned_workspace(
        db, resolved_workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Validate chat ownership when a chat_id is supplied.
    chat_session = None
    if resolved_chat_id:
        chat_session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == resolved_chat_id,
                ChatSession.user_id == str(current_user.id),
            )
            .first()
        )
        if chat_session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")

    logger.info(
        "upload_docs: user=%s workspace=%s chat=%s uploaded file=%s",
        getattr(current_user, "id", None),
        resolved_workspace_id,
        resolved_chat_id,
        doc.filename,
    )

    document = await get_document_processor().process(
        db, file=doc, user_id=current_user.id, workspace_id=resolved_workspace_id
    )

    link_created = False
    if chat_session is not None:
        # 1. Create entry in DocumentChatLink (clean/single source of truth)
        _, link_created = crud.document_chat_link.ensure_link(
            db,
            document_id=document.id,
            chat_id=chat_session.id,
            workspace_id=resolved_workspace_id,
            user_id=str(current_user.id),
        )

        # 2. Create system ChatMessage housekeeping row for chat view attachments (backward compatibility/tests)
        existing_sys_msg = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.chat_id == chat_session.id,
                ChatMessage.role == "system",
                ChatMessage.content == f"[document_upload] {document.file_name}",
            )
            .first()
        )
        if not existing_sys_msg:
            link_msg = ChatMessage(
                id=str(_uuid.uuid4()),
                chat_id=chat_session.id,
                user_id=str(current_user.id),
                workspace_id=resolved_workspace_id,
                role="system",
                content=f"[document_upload] {document.file_name}",
                document_ids=[document.id],
            )
            db.add(link_msg)
            db.commit()
            db.refresh(link_msg)

    AuditLogger.log(
        action="DOCUMENT_UPLOAD",
        user_id=str(current_user.id),
        workspace_id=resolved_workspace_id,
        document_id=document.id,
        endpoint="/v1/workspaces/upload-docs",
        http_method="POST",
        status="200",
        metadata={
            "filename": doc.filename,
            "chat_id": resolved_chat_id,
            "association_created": link_created,
            "association_already_existed": (
                chat_session is not None and not link_created
            ),
        },
    )

    # Compute the chat_count and chat_ids for the newly uploaded document.
    chat_count = crud.document.get_chat_count_for_document(db, doc_id=document.id)
    doc_chat_ids = crud.document.get_chat_ids_for_document(db, doc_id=document.id)

    all_workspace_docs = crud.document.get_multi_by_workspace(
        db, workspace_id=resolved_workspace_id
    )
    workspace_docs_serialized = []
    for d in all_workspace_docs:
        d_chat_count = crud.document.get_chat_count_for_document(db, doc_id=d.id)
        d_chat_ids = crud.document.get_chat_ids_for_document(db, doc_id=d.id)
        workspace_docs_serialized.append(
            serialize_document(d, chat_count=d_chat_count, chat_ids=d_chat_ids)
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "document": serialize_document(
                    document, chat_count=chat_count, chat_ids=doc_chat_ids
                ),
                "workspace_documents": workspace_docs_serialized,
                "chat_id": resolved_chat_id,
            },
            "message": "File uploaded successfully.",
        },
    )


@workspace_router.get("/{workspace_id}/docs")
def list_workspace_docs(
    workspace_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    files = crud.document.get_multi_by_workspace(db, workspace_id=workspace_id)

    AuditLogger.log(
        action="DOCUMENT_LIST",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        endpoint=f"/v1/workspaces/{workspace_id}/docs",
        http_method="GET",
        status="200",
        metadata={"count": len(files)},
    )

    serialized_files = []
    for f in files:
        chat_count = crud.document.get_chat_count_for_document(db, doc_id=f.id)
        chat_ids = crud.document.get_chat_ids_for_document(db, doc_id=f.id)
        logger.info(
            "TEMPORARY DEBUG - list_workspace_docs - doc_id: %s, workspace_id: %s, associated_chat_ids_from_db: %s, returned_chat_count: %d, returned_chat_ids: %s",
            f.id,
            workspace_id,
            chat_ids,
            chat_count,
            chat_ids,
        )
        serialized_files.append(
            serialize_document(
                f,
                chat_count=chat_count,
                chat_ids=chat_ids,
            )
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialized_files,
            "message": "Workspace files fetched successfully.",
        },
    )


@workspace_router.get("/{workspace_id}/docs/{doc_id}")
async def get_workspace_document(
    workspace_id: str,
    doc_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    document = crud.document.get_document(db, doc_id, str(current_user.id))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    AuditLogger.log(
        action="DOCUMENT_GET",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        document_id=doc_id,
        endpoint=f"/v1/workspaces/{workspace_id}/docs/{doc_id}",
        http_method="GET",
        status="200",
    )

    chat_count = crud.document.get_chat_count_for_document(db, doc_id=document.id)
    chat_ids = crud.document.get_chat_ids_for_document(db, doc_id=document.id)
    logger.info(
        "TEMPORARY DEBUG - get_workspace_document - doc_id: %s, workspace_id: %s, associated_chat_ids_from_db: %s, returned_chat_count: %d, returned_chat_ids: %s",
        document.id,
        workspace_id,
        chat_ids,
        chat_count,
        chat_ids,
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialize_document(
                document,
                chat_count=chat_count,
                chat_ids=chat_ids,
            ),
            "message": "Document fetched successfully.",
        },
    )


@workspace_router.delete("/{workspace_id}/docs/{doc_id}")
def delete_workspace_doc(
    workspace_id: str,
    doc_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    file = crud.document.get(db, doc_id)
    if (
        not file
        or file.workspace_id != workspace_id
        or file.user_id != str(current_user.id)
    ):
        raise HTTPException(status_code=404, detail="File not found")

    chats = crud.workspace.get_sessions_by_workspace_id(
        db, workspace_id=workspace_id, user_id=str(current_user.id)
    )
    crud.document.delete_document_cleanup_chats(
        db, doc_id=doc_id, file_name=file.file_name, chats=chats
    )
    db.commit()

    # Removing the document row cascades (ON DELETE CASCADE) to its
    # ``document_chunks`` rows, purging the file's embeddings from pgvector.
    crud.document.remove(db, id=doc_id)

    AuditLogger.log(
        action="DOCUMENT_DELETE",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        document_id=doc_id,
        endpoint=f"/v1/workspaces/{workspace_id}/docs/{doc_id}",
        http_method="DELETE",
        status="200",
        metadata={"filename": file.file_name},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "File deleted successfully.",
        },
    )
