"""Citation service for mapping answer text to the sources that grounded it.

Orchestrates the citation generation pipeline:
  1. Extract citation markers ([1], [2], ...) from the LLM-generated answer.
  2. Map each marker to a retrieved source.
  3. Deduplicate: reuse citation numbers when the same source is cited twice.
  4. Renumber to a clean 1..k sequence in order of appearance.
  5. Return structured citation metadata for the Citation View.

A "source" is either a retrieved ``DocumentChunk`` (from an uploaded document) or
a web search result (a dict from ``RAGService.web_search``). Both are numbered in
ONE shared sequence, because the LLM sees one combined CONTEXT block and can only
emit one set of [n] markers. Document chunks always occupy the low numbers, and
web results continue from there — see ``RAGService.web_search(start_number=...)``.

The service is backend-only and contains no frontend-specific logic.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from urllib.parse import urlparse

from server.models.workspace import DocumentChunk
from server.utils.markdown import tidy_markdown_spacing

logger = logging.getLogger(__name__)


def _host_of(url: str | None) -> str | None:
    """``https://www.example.com/a/b`` -> ``example.com``."""
    if not url:
        return None
    try:
        host = urlparse(url).netloc
    except ValueError:
        return None
    return host[4:] if host.startswith("www.") else host or None


class CitationData:
    """Structured citation metadata for a single cited source.

    Contains everything the Citation View needs to display the source and, for
    uploaded documents, to highlight the passage in the original file.

    The key names are the contract with the frontend (``CitationViewer.jsx``):
    ``title``, ``source_type``, ``source_name``, ``page`` and ``excerpt`` are the
    canonical fields. Legacy aliases (``document_name``, ``page_number``,
    ``highlight_text``) are also emitted so existing consumers keep working.
    """

    def __init__(self, citation_id: int, *, source: DocumentChunk | dict):
        self.id = citation_id
        if isinstance(source, dict):
            self._init_web(source)
        else:
            self._init_document(source)

    def _init_document(self, chunk: DocumentChunk) -> None:
        self.source_type = "document"
        self.title = chunk.file_name
        self.source_name = chunk.file_name
        self.document_id = chunk.document_id
        self.document_url = chunk.document_url or chunk.document_path
        self.url = None
        self.page = chunk.page_number
        self.chunk_id = chunk.id
        self.excerpt = chunk.content
        self.section = chunk.section_title
        self.published_date = None
        self.start_offset = chunk.start_offset
        self.end_offset = chunk.end_offset

    def _init_web(self, result: dict) -> None:
        link = result.get("link") or result.get("url")
        self.source_type = "web"
        self.title = result.get("title") or link
        self.source_name = _host_of(link)
        # No document_id: the Citation View keys off its absence to decide that a
        # citation is a web page rather than an uploaded document.
        self.document_id = None
        self.document_url = None
        self.url = link
        self.page = None
        self.chunk_id = None
        self.excerpt = result.get("snippet")
        self.section = None
        self.published_date = result.get("date")
        self.start_offset = None
        self.end_offset = None

    def to_dict(self) -> dict:
        """Export as a dictionary for API responses."""
        return {
            "id": self.id,
            "title": self.title,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "document_id": self.document_id,
            "document_url": self.document_url,
            "url": self.url,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "section": self.section,
            "published_date": self.published_date,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            # Legacy aliases — kept so older clients don't break.
            "document_name": self.title if self.source_type == "document" else None,
            "page_number": self.page,
            "highlight_text": self.excerpt,
            "section_title": self.section,
        }


class CitationService:
    """Map LLM-generated answers to their sources and produce citations."""

    @staticmethod
    def generate_citations(
        answer: str,
        sources: list[DocumentChunk | dict],
    ) -> tuple[str, list[dict]]:
        """Generate structured citations from answer text and its sources.

        Citation generation rules:
          1. Extract markers ([1], [2], ...) from the answer.
          2. Map each marker to a source index in ``sources``.
          3. Deduplicate: if source 2 is cited multiple times, reuse its number.
          4. Renumber to 1..k in order of appearance in the answer.
          5. Rewrite the answer text with clean numbering.
          6. Return (rewritten_answer, structured_citations).

        Out-of-range markers are stripped; nothing is cited if no markers appear.

        Args:
            answer: The LLM-generated answer text containing [n] markers.
            sources: Ordered sources the LLM was shown, in the SAME order they were
                numbered in the prompt context. Each is either a ``DocumentChunk``
                (uploaded document) or a web-result ``dict``. ``sources[0]`` is the
                source the model saw as ``[1]``.

        Returns:
            (rewritten_answer, citations_list) where:
            - rewritten_answer: The answer with cleaned-up, sequential markers.
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

        # Map original markers to source indices (1-indexed markers -> 0-indexed).
        marker_to_source: dict[int, int] = {}  # marker_number -> source_idx
        for marker in markers:
            source_idx = marker["marker_number"] - 1
            if 0 <= source_idx < len(sources):
                marker_to_source[marker["marker_number"]] = source_idx

        if not marker_to_source:
            # Every marker was out of range (hallucinated) — strip them all.
            rewritten = re.sub(marker_pattern, "", answer)
            rewritten = CitationService._cleanup_spacing(rewritten)
            return rewritten, []

        # Track unique sources in order of first appearance in the answer. This is
        # what makes the final numbering sequential: the first cited source becomes
        # [1] no matter which context slot the model originally referenced.
        seen_sources: dict[int, int] = {}  # source_idx -> citation_number
        citation_order: list[int] = []

        for marker_number in [m["marker_number"] for m in markers]:
            if marker_number in marker_to_source:
                source_idx = marker_to_source[marker_number]
                if source_idx not in seen_sources:
                    seen_sources[source_idx] = len(citation_order) + 1
                    citation_order.append(source_idx)

        # Build old-to-new marker mapping.
        old_to_new: dict[int, int] = {}
        for marker_number, source_idx in marker_to_source.items():
            old_to_new[marker_number] = seen_sources[source_idx]

        # Rewrite answer with new numbering; drop markers that pointed nowhere.
        def _replace_marker(match):
            old_number = int(match.group(1))
            return f"[{old_to_new[old_number]}]" if old_number in old_to_new else ""

        rewritten = re.sub(marker_pattern, _replace_marker, answer)
        rewritten = CitationService._cleanup_spacing(rewritten)

        # Build citation metadata for unique sources, in citation order.
        citations = []
        for idx, source_idx in enumerate(citation_order, start=1):
            citation_data = CitationData(citation_id=idx, source=sources[source_idx])
            citations.append(citation_data.to_dict())

        logger.info(
            "citations.generate: answer_len=%d markers_found=%d unique_sources=%d",
            len(answer),
            len(markers),
            len(citations),
        )

        return rewritten, citations

    @staticmethod
    def _cleanup_spacing(text: str) -> str:
        """Remove extra whitespace left by stripped/renumbered citation markers.

        Delegates to the shared Markdown-safe tidier: collapsing whitespace across
        the whole answer would flatten nested lists and break code blocks, turning
        a formatted answer into a wall of raw text.
        """
        return tidy_markdown_spacing(text)


@lru_cache(maxsize=1)
def get_citation_service() -> CitationService:
    """Return the shared :class:`CitationService` singleton."""
    return CitationService()
