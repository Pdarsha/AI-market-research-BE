"""Chunking service.

Single responsibility: split cleaned text into overlapping, metadata-carrying
chunks. No extraction, no embedding, no persistence.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel

from server.config import settings


class Chunk(BaseModel):
    """A single unit of text produced by the chunker."""

    text: str
    chunk_index: int
    page_number: int | None = None
    source: str | None = None
    # Text offsets within the original document for citation highlighting
    start_offset: int | None = None
    end_offset: int | None = None
    section_title: str | None = None


class ChunkingService:
    """Split text into fixed-size chunks with a configurable overlap.

    Overlap keeps context from spilling across chunk boundaries, which improves
    retrieval quality. Sizes are character-based, which is dependency-free and
    predictable; swap in a token-aware splitter here if needed without touching
    callers.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    def chunk_text(
        self,
        text: str,
        *,
        source: str | None = None,
        page_number: int | None = None,
    ) -> list[Chunk]:
        """Return structured :class:`Chunk` objects for ``text``."""
        if not text or not text.strip():
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        index = 0
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size]
            if piece.strip():
                chunks.append(
                    Chunk(
                        text=piece,
                        chunk_index=index,
                        page_number=page_number,
                        source=source,
                        start_offset=start,
                        end_offset=start + len(piece),
                    )
                )
                index += 1
            if start + self.chunk_size >= len(text):
                break
        return chunks


@lru_cache(maxsize=1)
def get_chunking_service() -> ChunkingService:
    """Return the shared :class:`ChunkingService` singleton."""
    return ChunkingService()
