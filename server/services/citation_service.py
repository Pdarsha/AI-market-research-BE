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
    """Provider-independent citation record with compatibility aliases.

    Source modules supply ordinary dictionaries; the formatter does not need to
    know which provider produced them. A document chunk is simply adapted to the
    same generic source shape. New providers can add arbitrary JSON-safe fields
    under ``metadata`` without a citation-service change.
    """

    def __init__(self, citation_id: int, *, source: DocumentChunk | dict):
        self.id = f"citation_{citation_id}"
        raw = self._document_source(source) if not isinstance(source, dict) else dict(source)
        self.source_type = raw.get("source_type") or raw.get("type") or "external"
        self.type = raw.get("type") or ("document" if self.source_type == "document" else "external")
        self.url = raw.get("url") or raw.get("link") or raw.get("document_url") or raw.get("document_path")
        self.provider = raw.get("provider") or raw.get("source_type") or raw.get("publisher") or _host_of(self.url) or "unknown"
        self.title = raw.get("title") or raw.get("document_name") or self.url or "Untitled source"
        self.reference_index = citation_id
        self.description = raw.get("description") or raw.get("excerpt") or raw.get("snippet") or raw.get("highlight_text")
        self.document_id = raw.get("document_id")
        self.chunk_id = (
            raw.get("chunk_id") or raw.get("id")
            if self.type == "document"
            else raw.get("chunk_id")
        )
        self.document_url = raw.get("document_url") or raw.get("document_path")
        self.page = raw.get("page") if raw.get("page") is not None else raw.get("page_number")
        self.excerpt = raw.get("excerpt") or raw.get("snippet") or raw.get("highlight_text")
        self.section = raw.get("section") or raw.get("section_title")
        self.published_date = raw.get("published_date") or raw.get("date")
        self.ticker = raw.get("ticker")
        self.start_offset = raw.get("start_offset")
        self.end_offset = raw.get("end_offset")
        self.metadata = {
            **(raw.get("metadata") or {}),
            "openable": bool(self.url),
            **{
                key: value
                for key, value in raw.items()
                if key not in {"metadata", "content", "snippet", "excerpt", "highlight_text"}
                and value is not None
            },
        }
        self.navigation_type, self.navigation_target, self.navigation_reason = self._navigation(raw)
        self.navigable = self.navigation_target is not None

    def _navigation(self, raw: dict) -> tuple[str | None, str | None, str | None]:
        """Produce one validated navigation contract for all source providers."""
        navigation_type = raw.get("navigation_type")
        target = raw.get("navigation_target")
        if not navigation_type and self.type == "document" and self.document_id:
            navigation_type = "document"
            target = f"/v1/documents/{self.document_id}/viewer"
        elif not navigation_type and self.url and self.url.startswith(("https://", "http://")):
            navigation_type = "external_url"
            target = self.url
        allowed = {"document", "external_url", "internal_page", "api"}
        if navigation_type in allowed and isinstance(target, str) and target.strip():
            return navigation_type, target, None
        return None, None, raw.get("navigation_reason") or "Navigation target unavailable."

    @staticmethod
    def _document_source(chunk: DocumentChunk) -> dict:
        """Adapt a retrieved chunk to the generic source-dictionary contract."""
        return {
            "type": "document",
            "provider": "uploaded_document",
            "title": chunk.file_name,
            "document_id": chunk.document_id,
            "document_url": chunk.document_url or chunk.document_path,
            "page": chunk.page_number,
            "chunk_id": chunk.id,
            "excerpt": chunk.content,
            "section": chunk.section_title,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
        }

    def to_dict(self) -> dict:
        """Return the universal citation schema plus legacy frontend aliases."""
        return {
            "id": self.id,
            "reference_index": self.reference_index,
            "provider": self.provider,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "navigable": self.navigable,
            "navigation_type": self.navigation_type,
            "navigation_target": self.navigation_target,
            "navigation_reason": self.navigation_reason,
            "url": self.url,
            "metadata": self.metadata,
            # Existing citation contract (deprecated aliases retained).
            "source_type": self.source_type,
            "source_name": self.provider or _host_of(self.url),
            "document_id": self.document_id,
            "document_url": self.document_url,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "section": self.section,
            "published_date": self.published_date,
            "ticker": self.ticker,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "document_name": self.title if self.type == "document" else None,
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
