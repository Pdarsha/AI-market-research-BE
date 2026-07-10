"""Citation service for mapping answer text to retrieved document chunks.

Orchestrates the citation generation pipeline:
  1. Extract citation markers ([1], [2], ...) from the LLM-generated answer.
  2. Map each marker to a retrieved chunk.
  3. Deduplicate: reuse citation numbers when the same chunk is cited multiple times.
  4. Renumber to a clean 1..k sequence in order of appearance.
  5. Return structured citation metadata for the Citation View.

The service is backend-only and contains no frontend-specific logic.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from server.models.workspace import DocumentChunk

logger = logging.getLogger(__name__)


class CitationData:
    """Structured citation metadata for a single retrieved chunk.

    Contains all information needed by the Citation View to display and
    highlight the original document.
    """

    def __init__(
        self,
        citation_id: int,
        chunk: DocumentChunk,
    ):
        self.id = citation_id
        self.document_id = chunk.document_id
        self.document_name = chunk.file_name
        self.document_url = chunk.document_url or chunk.document_path
        self.page_number = chunk.page_number
        self.chunk_id = chunk.id
        self.highlight_text = chunk.content
        self.section_title = chunk.section_title
        self.start_offset = chunk.start_offset
        self.end_offset = chunk.end_offset

    def to_dict(self) -> dict:
        """Export as a dictionary for API responses."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_url": self.document_url,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "highlight_text": self.highlight_text,
            "section_title": self.section_title,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


class CitationService:
    """Map LLM-generated answers to retrieved chunks and produce citations."""

    @staticmethod
    def generate_citations(
        answer: str,
        retrieved_chunks: list[DocumentChunk],
    ) -> tuple[str, list[dict]]:
        """Generate structured citations from answer text and retrieved chunks.

        Citation generation rules:
          1. Extract markers ([1], [2], ...) from the answer.
          2. Map each marker to a chunk index in ``retrieved_chunks``.
          3. Deduplicate: if chunk 2 is cited multiple times, reuse its number.
          4. Renumber to 1..k in order of appearance in the answer.
          5. Rewrite the answer text with clean numbering.
          6. Return (rewritten_answer, structured_citations).

        Out-of-range markers are stripped; no chunks are cited if no markers appear.

        Args:
            answer: The LLM-generated answer text containing [n] markers.
            retrieved_chunks: The list of retrieved DocumentChunk objects,
                            ordered by retrieval relevance.

        Returns:
            (rewritten_answer, citations_list) where:
            - rewritten_answer: The answer with cleaned-up citation markers.
            - citations_list: A list of citation dicts with metadata.
        """
        marker_pattern = r"\[(\d+)\]"
        markers = []
        for match in re.finditer(marker_pattern, answer):
            marker_number = int(match.group(1))
            markers.append({
                "position": match.start(),
                "marker_number": marker_number,
                "end_position": match.end(),
            })

        if not markers:
            return answer, []

        # Map original markers to chunk indices (1-indexed markers -> 0-indexed chunks)
        # Keep track of which marker refers to which chunk
        marker_to_chunk: dict[int, int] = {}  # marker_number -> chunk_idx
        for marker in markers:
            chunk_idx = marker["marker_number"] - 1
            if 0 <= chunk_idx < len(retrieved_chunks):
                marker_to_chunk[marker["marker_number"]] = chunk_idx

        if not marker_to_chunk:
            # Strip all invalid markers
            rewritten = re.sub(marker_pattern, "", answer)
            rewritten = CitationService._cleanup_spacing(rewritten)
            return rewritten, []

        # Track unique chunks in order of first appearance in markers
        seen_chunks: dict[int, int] = {}  # chunk_idx -> citation_number
        citation_order: list[int] = []

        for marker_number in [m["marker_number"] for m in markers]:
            if marker_number in marker_to_chunk:
                chunk_idx = marker_to_chunk[marker_number]
                if chunk_idx not in seen_chunks:
                    seen_chunks[chunk_idx] = len(citation_order) + 1
                    citation_order.append(chunk_idx)

        # Build old-to-new marker mapping (from original marker numbers to renumbered ones)
        old_to_new: dict[int, int] = {}
        for marker_number, chunk_idx in marker_to_chunk.items():
            new_number = seen_chunks[chunk_idx]
            old_to_new[marker_number] = new_number

        # Rewrite answer with new numbering
        def _replace_marker(match):
            old_number = int(match.group(1))
            return f"[{old_to_new[old_number]}]" if old_number in old_to_new else ""

        rewritten = re.sub(marker_pattern, _replace_marker, answer)
        rewritten = CitationService._cleanup_spacing(rewritten)

        # Build citation metadata for unique chunks in citation order
        citations = []
        for idx, chunk_idx in enumerate(citation_order, start=1):
            chunk = retrieved_chunks[chunk_idx]
            citation_data = CitationData(citation_id=idx, chunk=chunk)
            citations.append(citation_data.to_dict())

        logger.info(
            "citations.generate: answer_len=%d markers_found=%d unique_chunks=%d",
            len(answer),
            len(markers),
            len(citations),
        )

        return rewritten, citations

    @staticmethod
    def _cleanup_spacing(text: str) -> str:
        """Remove extra whitespace left by stripped citation markers.

        Ensures the text reads naturally after markers are removed or renumbered.
        """
        # Move closing punctuation back to before extra spaces
        text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
        # Collapse multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


@lru_cache(maxsize=1)
def get_citation_service() -> CitationService:
    """Return the shared :class:`CitationService` singleton."""
    return CitationService()
