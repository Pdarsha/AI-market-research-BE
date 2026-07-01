import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server import crud, schemas
from server.api_v1.deps import get_db
from server.utils.auth import get_current_user
from server.utils.common import delete_chroma_by, ingest_file_into_workspace

logger = logging.getLogger(__name__)

workspace_router = APIRouter(prefix="/workspaces", tags=["Workspace"])


def _serialize_workspace(workspace, chat_count: Optional[int] = None) -> dict:
    data = {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "user_id": workspace.user_id,
        "created_at": (
            workspace.created_at.isoformat() if workspace.created_at else None
        ),
        "updated_at": (
            workspace.updated_at.isoformat() if workspace.updated_at else None
        ),
    }
    if chat_count is not None:
        data["chat_count"] = chat_count
    return data


def _serialize_chat(chat) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "user_id": chat.user_id,
        "workspace_id": chat.workspace_id,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
    }


def _serialize_file(file) -> dict:
    return {
        "id": file.id,
        "file_name": file.file_name,
        "file_format": file.file_format,
        "workspace_id": file.workspace_id,
        "created_at": file.created_at.isoformat() if file.created_at else None,
    }


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
            "data": _serialize_workspace(workspace, chat_count=0),
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
                _serialize_workspace(ws, chat_count=len(ws.chats)) for ws in workspaces
            ],
            "message": "Workspaces fetched successfully.",
        },
    )


@workspace_router.get("/public/{workspace_id}")
def public_workspace(
        workspace_id: str,
        db=Depends(get_db),
):
    """Public, unauthenticated view of a workspace.

    Exposes ONLY the workspace metadata and its uploaded files. Chats and
    messages are never included, so a shared link cannot leak conversation
    history. Declared before ``/{workspace_id}`` so the literal ``public``
    segment is not captured by the id parameter.
    """
    workspace = crud.workspace.get(db, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    files = crud.enbedding.get_multi_by_workspace(db, workspace_id=workspace_id)
    data = {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "created_at": (
            workspace.created_at.isoformat() if workspace.created_at else None
        ),
        "files": [_serialize_file(f) for f in files],
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
    data = _serialize_workspace(workspace, chat_count=len(chats))
    data["chats"] = [_serialize_chat(chat) for chat in chats]
    data["files"] = [
        _serialize_file(f)
        for f in crud.enbedding.get_multi_by_workspace(db, workspace_id=workspace_id)
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
            "data": _serialize_workspace(workspace),
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
        "create_embedding: user=%s workspace=%s uploaded file=%s",
        getattr(current_user, "id", None),
        workspace_id,
        doc.filename,
    )

    await ingest_file_into_workspace(
        db, file=doc, user_id=current_user.id, workspace_id=workspace_id
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
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
            "data": [_serialize_file(f) for f in files],
            "message": "Workspace files fetched successfully.",
        },
    )


@workspace_router.post("/{workspace_id}/docs/{doc_id}")
async def get_workspace_document(
        workspace_id: str,
        doc: UploadFile = File(...),
        db=Depends(get_db),
        current_user=Depends(get_current_user),
):
    workspace = crud.workspace._get_owned_workspace(
        db, workspace_id, str(current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    embedding = await ingest_file_into_workspace(
        db, file=doc, user_id=current_user.id, workspace_id=workspace_id
    )
    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "error": None,
            "data": _serialize_file(embedding),
            "message": "File uploaded successfully.",
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

    # Remove the file's vector chunks from ChromaDB, then the relational row.
    delete_chroma_by(
        {
            "$and": [
                {"user_id": {"$eq": str(current_user.id)}},
                {"workspace_id": {"$eq": workspace_id}},
                {"file_name": {"$eq": file.file_name}},
            ]
        }
    )
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
