import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server import crud, schemas
from server.api_v1.deps import get_db
from server.models.genai import ChatSession
from server.services.document_processor import get_document_processor
from server.utils.auth import get_current_user
from server.utils.helpers import serialize_chat, serialize_document, serialize_workspace

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
                        "source_documents": m.source_documents,
                        "created_at": (
                            m.created_at.isoformat() if m.created_at else None
                        ),
                    }
                    for m in crud.chat_message.get_multi_by_chat(db, chat_id=c.id)
                ],
            }
            for c in chats
        ],
    }
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
    data["files"] = [
        serialize_document(f)
        for f in crud.document.get_multi_by_workspace(db, workspace_id=workspace_id)
    ]
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
    doc: UploadFile = File(...),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload a document into a workspace.

    Files always belong to a workspace: the caller must own ``workspace_id``.
    The file's text is chunked, embedded and stored (tagged with the workspace)
    so a chat inside the workspace can retrieve it via the ``@doc`` agent.
    """
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    logger.info(
        "upload_docs: user=%s workspace=%s uploaded file=%s",
        getattr(current_user, "id", None),
        workspace_id,
        doc.filename,
    )

    document = await get_document_processor().process(
        db, file=doc, user_id=current_user.id, workspace_id=workspace_id
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialize_document(document),
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
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": [serialize_document(f) for f in files],
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
    document = crud.document.get_document(
        db, workspace_id, doc_id, str(current_user.id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialize_document(document),
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

    chats = (
        db.query(ChatSession)
        .filter(
            ChatSession.workspace_id == workspace_id,
            ChatSession.user_id == str(current_user.id),
        )
        .all()
    )
    for chat in chats:
        messages = crud.chat_message.get_multi_by_chat(db, chat_id=chat.id)
        doc_ids = set()
        for m in messages:
            if m.document_ids:
                doc_ids.update(m.document_ids)
        if doc_id not in doc_ids:
            continue
        if doc_ids == {doc_id}:
            db.delete(chat)  # cascades to its messages
        else:
            for m in messages:
                if m.document_ids and doc_id in m.document_ids:
                    m.document_ids = [d for d in m.document_ids if d != doc_id]
                    db.add(m)
                if m.source_documents and file.file_name in m.source_documents:
                    m.source_documents = [
                        n for n in m.source_documents if n != file.file_name
                    ]
                    db.add(m)

    # Removing the document row cascades (ON DELETE CASCADE) to its
    # ``document_chunks`` rows, purging the file's embeddings from pgvector.
    crud.document.remove(db, id=doc_id)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "File deleted successfully.",
        },
    )
