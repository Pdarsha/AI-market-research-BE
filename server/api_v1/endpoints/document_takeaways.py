import logging
import time
from server.services.audit_logger import audit_context
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from server.api_v1.deps import get_db
from server.crud.crud_document_takeaways import crud_document_takeaways
from server.crud.crud_document import document as crud_document
from server.schemas.document_takeaways import (
    CreateDocumentTakeawayGridRequest,
    AddCustomColumnRequest,
    AddDocumentsToGridRequest,
    RenameGridRequest,
    DocumentTakeawayGridListItem,
)
from server.services.document_takeaways_service import document_takeaways_service
from server.utils.auth import get_current_user

logger = logging.getLogger(__name__)

document_takeaways_router = APIRouter(
    prefix="/workspaces/{workspace_id}/document-takeaways", tags=["Document Takeaways"]
)
global_takeaways_router = APIRouter(
    prefix="/document-takeaways", tags=["Document Takeaways"]
)


def serialize_grid(grid, db: Session):
    return document_takeaways_service.get_cached_serialized_grid(grid, db)


@document_takeaways_router.post("/grids")
def create_grid(
    workspace_id: str,
    payload: CreateDocumentTakeawayGridRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        grid = document_takeaways_service.create_grid(
            db,
            background_tasks,
            workspace_id,
            payload.name,
            payload.document_ids,
            str(current_user.id),
        )
        return serialize_grid(grid, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error creating grid")
        raise HTTPException(status_code=500, detail="Internal server error")


@document_takeaways_router.get("/grids/{grid_id}")
def get_grid(
    workspace_id: str,
    grid_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    with audit_context(
        action="FETCH_GRID_BY_ID",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        endpoint=f"/v1/workspaces/{workspace_id}/document-takeaways/grids/{grid_id}",
        http_method="GET",
        metadata={"grid_id": grid_id},
    ) as meta:
        t0 = time.perf_counter()
        grid = crud_document_takeaways.get_grid(db, grid_id)
        db_fetch_ms = round((time.perf_counter() - t0) * 1000, 2)
        if not grid or grid.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Grid not found")

        t1 = time.perf_counter()
        serialized = serialize_grid(grid, db)
        serialization_ms = round((time.perf_counter() - t1) * 1000, 2)

        meta.update(
            {
                "grid_id": grid.id,
                "grid_status": grid.status,
                "db_fetch_time_ms": db_fetch_ms,
                "serialization_time_ms": serialization_ms,
                "total_backend_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "rows_count": len(serialized.get("rows", [])),
                "columns_count": len(serialized.get("columns", [])),
            }
        )

        return serialized


@document_takeaways_router.get("/grid")
def get_current_grid(
    workspace_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    with audit_context(
        action="FETCH_CURRENT_GRID",
        user_id=str(current_user.id),
        workspace_id=workspace_id,
        endpoint=f"/v1/workspaces/{workspace_id}/document-takeaways/grid",
        http_method="GET",
    ) as meta:
        t0 = time.perf_counter()
        grid = crud_document_takeaways.get_latest_grid_by_workspace(db, workspace_id)
        db_fetch_ms = round((time.perf_counter() - t0) * 1000, 2)
        if not grid:
            meta["grid_found"] = False
            return None

        t1 = time.perf_counter()
        serialized = serialize_grid(grid, db)
        serialization_ms = round((time.perf_counter() - t1) * 1000, 2)

        meta.update(
            {
                "grid_id": grid.id,
                "grid_status": grid.status,
                "db_fetch_time_ms": db_fetch_ms,
                "serialization_time_ms": serialization_ms,
                "total_backend_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "rows_count": len(serialized.get("rows", [])),
                "columns_count": len(serialized.get("columns", [])),
            }
        )

        return serialized


@document_takeaways_router.post("/grids/{grid_id}/columns")
def add_custom_column(
    workspace_id: str,
    grid_id: str,
    payload: AddCustomColumnRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        grid = document_takeaways_service.add_custom_column(
            db,
            background_tasks,
            grid_id,
            payload.name,
            payload.prompt,
            str(current_user.id),
        )
        return serialize_grid(grid, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error adding column")
        raise HTTPException(status_code=500, detail="Internal server error")


@document_takeaways_router.post("/grids/{grid_id}/cells/{cell_id}/retry")
def retry_cell(
    workspace_id: str,
    grid_id: str,
    cell_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        cell = document_takeaways_service.retry_cell(
            db, background_tasks, cell_id, str(current_user.id)
        )
        return {
            "id": cell.id,
            "grid_column_id": cell.grid_column_id,
            "value": cell.value,
            "status": cell.status,
            "error_message": cell.error_message,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error retrying cell")
        raise HTTPException(status_code=500, detail="Internal server error")


@document_takeaways_router.post("/grids/{grid_id}/documents")
def add_documents_to_grid(
    workspace_id: str,
    grid_id: str,
    payload: AddDocumentsToGridRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        grid = document_takeaways_service.add_documents_to_grid(
            db,
            background_tasks,
            grid_id,
            payload.document_ids,
            str(current_user.id),
        )
        return serialize_grid(grid, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error adding documents to grid")
        raise HTTPException(status_code=500, detail="Internal server error")

@document_takeaways_router.get("/grids", response_model=list[DocumentTakeawayGridListItem])
def list_grids(
    workspace_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    grids = crud_document_takeaways.get_grids_by_workspace(db, workspace_id)
    return [
        DocumentTakeawayGridListItem(
            id=g.id,
            name=g.name,
            status=g.status,
            document_count=len(g.rows),
            column_count=len(g.columns),
            created_at=g.created_at,
            updated_at=g.updated_at
        ) for g in grids
    ]

@document_takeaways_router.put("/grids/{grid_id}/rename")
def rename_grid(
    workspace_id: str,
    grid_id: str,
    payload: RenameGridRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    grid = crud_document_takeaways.get_grid(db, grid_id)
    if not grid or grid.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Grid not found")
    crud_document_takeaways.update(db, db_obj=grid, obj_in={"name": payload.name})
    return {"message": "Grid renamed successfully", "name": grid.name}

@document_takeaways_router.delete("/grids/{grid_id}")
def delete_grid(
    workspace_id: str,
    grid_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    grid = crud_document_takeaways.get_grid(db, grid_id)
    if not grid or grid.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Grid not found")
    crud_document_takeaways.remove(db, id=grid.id)
    return {"message": "Grid deleted successfully"}

@document_takeaways_router.post("/grids/{grid_id}/duplicate")
def duplicate_grid(
    workspace_id: str,
    grid_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        new_grid = document_takeaways_service.duplicate_grid(
            db,
            grid_id,
            str(current_user.id),
        )
        return serialize_grid(new_grid, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error duplicating grid")
        raise HTTPException(status_code=500, detail="Internal server error")
