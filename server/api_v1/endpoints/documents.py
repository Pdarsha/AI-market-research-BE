"""Citation-view & document-viewer endpoints (read-only, additive).

These power the Citation View Page without touching the chat/RAG flow:

  * ``GET /v1/citations/{citation_id}``            — citation details (citation_id
                                                     is a ``DocumentChunk.id``).
  * ``GET /v1/documents/{document_id}/viewer``     — metadata + signed viewer URL.
  * ``GET /v1/documents/{document_id}/highlight``  — highlight rectangles for text.
  * ``GET /v1/documents/{document_id}/file``       — signed, streamed file serving.

Page numbers and highlight rectangles are derived at read time from the PDF
(PyMuPDF) — nothing new is stored, and ingestion/retrieval are untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from server import crud
from server.api_v1.deps import get_db
from server.models.workspace import DocumentChunk
from server.schemas.documents import ConversationExportRequest
from server.services.pdf_export_service import conversation_to_pdf
from server.utils.auth import get_current_user
from server.utils.common import slugify
from server.utils.helpers import (
    _abs_path,
    _document_meta,
    find_highlights,
    get_owned_document,
    _mime_type,
    signed_file_url,
)
from server.utils.signed_url import verify_document_token

logger = logging.getLogger(__name__)

documents_router = APIRouter(prefix="/documents", tags=["Documents"])
citations_router = APIRouter(prefix="/citations", tags=["Citations"])

# --------------------------------------------------------------------------- #
# Citation details                                                            #
# --------------------------------------------------------------------------- #


@citations_router.get("/{citation_id}")
def get_citation(
    citation_id: str,
    request: Request,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Citation details for a chunk (``citation_id`` == ``DocumentChunk.id``)."""
    chunk = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.id == citation_id,
            DocumentChunk.user_id == str(current_user.id),
        )
        .first()
    )
    if not chunk:
        raise HTTPException(status_code=404, detail="Citation not found")

    doc = get_owned_document(db, chunk.document_id, current_user)
    abs_path = _abs_path(doc.storage_path)
    is_pdf = (doc.file_format or "").lower() == "pdf"
    page, _rects = find_highlights(abs_path, chunk.content) if is_pdf else (None, [])
    meta = _document_meta(doc)

    logger.info(
        "citation opened: user=%s citation=%s document=%s page=%s",
        getattr(current_user, "id", None),
        citation_id,
        doc.id,
        page,
    )
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "citation_id": citation_id,
                "document": {
                    "id": doc.id,
                    "name": doc.file_name,
                    "type": doc.file_format,
                    "size": meta["file_size"],
                    "uploaded_at": meta["uploaded_at"],
                    "url": signed_file_url(request, doc.id, str(current_user.id)),
                },
                "page": page,
                "highlight_text": chunk.content,
                "chunk_text": chunk.content,
                "chunk_id": chunk.id,
                "document_id": doc.id,
                "workspace_id": doc.workspace_id,
                "metadata": {
                    "author": meta["author"],
                    "uploaded_at": meta["uploaded_at"],
                    "file_size": meta["file_size"],
                    "mime_type": meta["mime_type"],
                    "page_count": meta["page_count"],
                    "chunk_number": chunk.chunk_number,
                    "total_chunks": chunk.total_chunks,
                },
            },
            "message": "Citation fetched successfully.",
        },
    )


# --------------------------------------------------------------------------- #
# Document viewer                                                             #
# --------------------------------------------------------------------------- #


@documents_router.get("/{document_id}/viewer")
def get_document_viewer(
    document_id: str,
    request: Request,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Secure viewer URL + metadata + page count + document type."""
    doc = get_owned_document(db, document_id, current_user)
    meta = _document_meta(doc)

    logger.info(
        "document viewed: user=%s document=%s",
        getattr(current_user, "id", None),
        document_id,
    )
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "id": doc.id,
                "name": doc.file_name,
                "type": doc.file_format,
                "mime_type": meta["mime_type"],
                "page_count": meta["page_count"],
                "size": meta["file_size"],
                "uploaded_at": meta["uploaded_at"],
                "workspace_id": doc.workspace_id,
                "url": signed_file_url(request, doc.id, str(current_user.id)),
            },
            "message": "Document viewer prepared successfully.",
        },
    )


# --------------------------------------------------------------------------- #
# Highlight rectangles                                                        #
# --------------------------------------------------------------------------- #


@documents_router.get("/{document_id}/highlight")
def get_document_highlight(
    document_id: str,
    text: Optional[str] = Query(None, description="Text to locate/highlight"),
    chunk_id: Optional[str] = Query(None, description="Chunk whose text to highlight"),
    page: Optional[int] = Query(None, description="0-based page to search (optional)"),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return highlight rectangles for text on a page (for overlay drawing)."""
    doc = get_owned_document(db, document_id, current_user)

    search_text = text
    if not search_text and chunk_id:
        chunk = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.id == chunk_id,
                DocumentChunk.document_id == document_id,
                DocumentChunk.user_id == str(current_user.id),
            )
            .first()
        )
        search_text = chunk.content if chunk else None
    if not search_text:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'chunk_id'.")

    found_page, rects = find_highlights(_abs_path(doc.storage_path), search_text, page)
    logger.info(
        "page viewed: user=%s document=%s page=%s highlights=%d",
        getattr(current_user, "id", None),
        document_id,
        found_page,
        len(rects),
    )
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {"page": found_page, "highlight": rects},
            "message": "Highlights computed successfully.",
        },
    )


# --------------------------------------------------------------------------- #
# Signed file serving (streamed)                                             #
# --------------------------------------------------------------------------- #


@documents_router.get("/{document_id}/file")
def get_document_file(
    document_id: str,
    token: str = Query(..., description="Signed access token"),
    db=Depends(get_db),
):
    """Stream the original file, authorised by a signed, expiring token."""
    payload = verify_document_token(token)
    if not payload or payload["document_id"] != document_id:
        raise HTTPException(status_code=403, detail="Invalid or expired document link")

    doc = crud.document.get(db, document_id)
    if not doc or doc.user_id != payload["user_id"]:
        raise HTTPException(status_code=404, detail="Document not found")

    abs_path = _abs_path(doc.storage_path)
    if not abs_path or not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found on server")

    # FileResponse streams the file in chunks (handles large documents).
    return FileResponse(
        path=str(abs_path),
        media_type=_mime_type(doc.file_format),
        filename=doc.file_name,
        headers={"Content-Disposition": f'inline; filename="{doc.file_name}"'},
    )


# --------------------------------------------------------------------------- #
# Conversation PDF Export                                                    #
# --------------------------------------------------------------------------- #


@documents_router.post("/export-pdf")
def export_conversation_pdf(
    payload: ConversationExportRequest,
    current_user=Depends(get_current_user),
):
    """Render the posted conversation to a PDF file and return it."""
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Nothing to export")

    title = payload.title or "CapitalEdge conversation"
    try:
        pdf_bytes = conversation_to_pdf(
            title, [m.model_dump() for m in payload.messages]
        )
    except Exception as exc:  # pragma: no cover - layout engine failure
        raise HTTPException(status_code=500, detail="PDF generation failed") from exc

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{slugify(title)}.pdf"'
        },
    )
