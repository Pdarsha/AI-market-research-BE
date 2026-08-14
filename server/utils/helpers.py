from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

from server import crud
from server.utils.markdown import tidy_markdown_spacing
from server.config import PROJECT_ROOT, settings
from server.utils.signed_url import create_document_token

logger = logging.getLogger(__name__)

_MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "md": "text/markdown",
    "txt": "text/plain",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "webp": "image/webp",
}

_IMAGE_SOURCE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _abs_path(storage_path: Optional[str]) -> Optional[Path]:
    if not storage_path:
        return None
    path = Path(storage_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / storage_path
    return path


def _mime_type(file_format: Optional[str]) -> str:
    return _MIME_TYPES.get((file_format or "").lower(), "application/octet-stream")


def get_owned_document(db, document_id: str, current_user=None):
    """Return the document if caller owns it or if it belongs to an accessible workspace."""
    doc = crud.document.get(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if current_user and getattr(current_user, "id", None):
        user_id_str = str(current_user.id)
        if doc.user_id == user_id_str:
            return doc
        if doc.workspace_id:
            workspace = crud.workspace.get(db, doc.workspace_id)
            if workspace and (workspace.user_id == user_id_str or workspace.is_public):
                return doc
        raise HTTPException(status_code=403, detail="Access denied")
    if doc.workspace_id:
        workspace = crud.workspace.get(db, doc.workspace_id)
        if workspace and workspace.is_public:
            return doc
    return doc


@lru_cache(maxsize=256)
def _pdf_meta_cached(abs_path: str, mtime: float) -> dict:
    """Page count + author for a PDF, cached by (path, mtime). Avoids re-opening
    the same unchanged PDF on every viewer/citation request."""
    try:
        import fitz

        with fitz.open(abs_path) as doc:
            meta = doc.metadata or {}
            return {"page_count": doc.page_count, "author": meta.get("author") or None}
    except Exception:
        logger.debug("pdf meta failed for %s", abs_path, exc_info=True)
        return {"page_count": None, "author": None}


def _document_meta(doc) -> dict:
    """File-system + PDF metadata that isn't persisted on the row."""
    abs_path = _abs_path(doc.storage_path)
    size = None
    page_count = None
    author = None
    if abs_path and abs_path.exists():
        try:
            stat = abs_path.stat()
            size = stat.st_size
            if (doc.file_format or "").lower() == "pdf":
                pdf = _pdf_meta_cached(str(abs_path), stat.st_mtime)
                page_count = pdf["page_count"]
                author = pdf["author"]
        except Exception:
            logger.debug("document meta failed for %s", doc.id, exc_info=True)
    return {
        "file_size": size,
        "mime_type": _mime_type(doc.file_format),
        "page_count": page_count,
        "author": author,
        "uploaded_at": doc.created_at.isoformat() if doc.created_at else None,
    }


def _search_snippet(text: str) -> str:
    """A short, single-line phrase suitable for PyMuPDF ``search_for`` (which
    matches exact substrings on one line)."""
    if not text:
        return ""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())
    return line[:80]


def find_highlights(abs_path: Optional[Path], text: str, page: Optional[int] = None):
    """Return ``(page_number, [rects])`` for ``text`` in a PDF via PyMuPDF.

    Best-effort: searches the given page, or all pages, for a representative
    snippet. Non-PDF or not-found -> ``(page or 0, [])``.
    """
    snippet = _search_snippet(text)
    if not abs_path or not abs_path.exists() or not snippet:
        return (page or 0), []
    try:
        import fitz

        with fitz.open(str(abs_path)) as doc:
            if page is not None and 0 <= page < doc.page_count:
                candidates = [page]
            else:
                candidates = range(doc.page_count)
            for index in candidates:
                rects = doc.load_page(index).search_for(snippet)
                if rects:
                    return index, [
                        {
                            "x": round(r.x0, 1),
                            "y": round(r.y0, 1),
                            "width": round(r.x1 - r.x0, 1),
                            "height": round(r.y1 - r.y0, 1),
                        }
                        for r in rects
                    ]
    except Exception:
        logger.debug("highlight search failed for %s", abs_path, exc_info=True)
    return (page or 0), []


def signed_file_url(request: Request, document_id: str, user_id: str) -> str:
    token = create_document_token(document_id, user_id, settings.SIGNED_URL_TTL)
    base = str(request.base_url).rstrip("/")
    return f"{base}/v1/documents/{document_id}/file?token={token}"


def serialize_workspace(workspace, chat_count: Optional[int] = None) -> dict:
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


def serialize_chat(chat) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "user_id": chat.user_id,
        "workspace_id": chat.workspace_id,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
    }


def serialize_document(
    doc, chat_count: Optional[int] = None, chat_ids: Optional[list] = None
) -> dict:
    data = {
        "id": doc.id,
        "workspace_id": doc.workspace_id,
        "file_name": doc.file_name,
        "file_format": doc.file_format,
        "storage_path": doc.storage_path,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }
    if chat_count is not None:
        data["chat_count"] = chat_count
    if chat_ids is not None:
        data["chat_ids"] = chat_ids
    return data




def resolve_or_create_session(
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




def document_sources_only(names):
    """Drop image files from a list of source file names (keep only documents)."""
    if not names:
        return names
    return [n for n in names if not str(n).lower().endswith(_IMAGE_SOURCE_EXTS)]


def strip_citation_markers(answer):
    """Remove stray citation markers from non-document-grounded answers."""
    rewritten = re.sub(r"\[(\d+)\]", "", answer or "")
    return tidy_markdown_spacing(rewritten)


def resolve_active_document_ids(prior_messages, requested):
    """Resolve the active document context for the current message.

    File-context rules (backend-managed, independent of what the frontend sends):
      * Any document in ``requested`` that has NOT appeared earlier in this chat is
        treated as a NEW upload — the new document(s) become the ONLY active
        context, replacing whatever was active before (Scenario 1).
      * If the request introduces no new documents, the most recent active set is
        reused so follow-up questions keep working on the same file(s) (Scenario 2).
      * With no prior context, the requested ids are used as-is.

    Returns ``(active_ids, reason)`` where ``reason`` is one of
    ``"new_upload" | "reuse_active" | "follow_up" | "none"`` (for logging).
    """
    requested = list(dict.fromkeys(requested or []))

    seen: set[str] = set()
    last_active: list[str] = []
    for msg in prior_messages:
        if msg.document_ids:
            seen.update(msg.document_ids)
            last_active = list(msg.document_ids)

    new_docs = [doc_id for doc_id in requested if doc_id not in seen]
    if new_docs:
        return new_docs, "new_upload"
    if requested:
        return (last_active or requested), "reuse_active"
    if last_active:
        return last_active, "follow_up"
    return [], "none"

