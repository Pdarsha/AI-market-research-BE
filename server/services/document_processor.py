"""Document processor.

Orchestrates the ingestion pipeline end to end:

    file bytes -> save original -> extract text -> clean -> chunk
               -> embed (OpenAI) -> store chunks + metadata in pgvector

It coordinates the single-responsibility services and owns persistence; the
individual services stay free of orchestration and storage concerns.
"""

from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, UploadFile

from server import crud, schemas
from server.config import WORKSPACE_DOCS_DIR, create_storage_directories
from server.models.workspace import Document, DocumentChunk
from server.services.chunking_service import ChunkingService, get_chunking_service
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.services.ocr_service import OCRService, get_ocr_service

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalise extracted text: collapse whitespace, strip control chars."""
    # Drop non-printable control characters (keep tab/newline).
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Collapse runs of blank lines and trailing whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentProcessor:
    """Coordinate extraction, chunking, embedding and storage."""

    def __init__(
        self,
        ocr_service: OCRService,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._ocr = ocr_service
        self._chunker = chunking_service
        self._embedder = embedding_service

    async def process(
        self,
        db,
        *,
        file: UploadFile,
        user_id: str,
        workspace_id: str,
    ) -> Document:
        """Ingest an uploaded file and return the persisted ``Document`` row."""
        # 1. Validate file type up front.
        try:
            file_format = self._ocr.file_format(file.filename or "")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        # 2. Persist the original file to disk.
        storage_path = self._save_original(file.filename or "", content)

        # 3. Extract text.
        try:
            raw_text = self._ocr.extract_text(file.filename or "", content)
        except Exception as exc:
            logger.exception("Text extraction failed for %s", file.filename)
            raise HTTPException(
                status_code=400, detail=f"Failed to read file: {exc}"
            ) from exc

        # 4. Clean.
        text = clean_text(raw_text)
        if not text:
            raise HTTPException(
                status_code=400, detail="No extractable text found in file"
            )

        # 5. Chunk.
        chunks = self._chunker.chunk_text(text, source=file.filename)
        if not chunks:
            raise HTTPException(
                status_code=400, detail="No extractable text found in file"
            )

        # 6. Embed (OpenAI, async + batched).
        embeddings = await self._embedder.aembed_texts([c.text for c in chunks])

        # 7. Store the document, then its chunks (with embeddings) in pgvector.
        document = crud.document.create(
            db=db,
            obj_in=schemas.DocumentResponse(
                user_id=str(user_id),
                workspace_id=str(workspace_id),
                file_name=file.filename,
                file_format=file_format,
                storage_path=storage_path,
            ),
        )

        chunk_rows: list[DocumentChunk] = [
            DocumentChunk(
                document_id=document.id,
                user_id=str(user_id),
                workspace_id=str(workspace_id),
                file_name=file.filename,
                chunk_number=chunk.chunk_index,
                total_chunks=len(chunks),
                content=chunk.text,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        db.add_all(chunk_rows)
        db.commit()

        logger.info(
            "Ingested file=%s user=%s workspace=%s chunks=%d",
            file.filename,
            user_id,
            workspace_id,
            len(chunk_rows),
        )
        return document

    @staticmethod
    def _save_original(filename: str, content: bytes) -> str:
        create_storage_directories()
        ext = Path(filename).suffix.lower()
        unique_name = f"{uuid.uuid4()}{ext}"
        file_path = WORKSPACE_DOCS_DIR / unique_name
        with open(file_path, "wb") as fh:
            fh.write(content)
        return f"static/workspace_docs/{unique_name}"


@lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    """Return the shared :class:`DocumentProcessor` singleton."""
    return DocumentProcessor(
        ocr_service=get_ocr_service(),
        chunking_service=get_chunking_service(),
        embedding_service=get_embedding_service(),
    )
