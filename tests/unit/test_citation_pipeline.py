"""
Regression test for DEF-015 / TC-AD-03:
"Analyze 1 Source" document analysis must always return citations
even when the LLM response does not include explicit [1][2] markers.
"""

import pytest
from unittest.mock import MagicMock
from server.services.citation_service import CitationService
from server.models.workspace import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    doc_id="doc-001",
    chunk_id="chunk-abc",
    file_name="annual_report.pdf",
    content="Revenue grew to $1.2B in FY2025.",
    page_number=3,
    section_title="Financial Highlights",
    document_url="/static/workspace_docs/abc.pdf",
) -> DocumentChunk:
    chunk = MagicMock(spec=DocumentChunk)
    chunk.id = chunk_id
    chunk.document_id = doc_id
    chunk.file_name = file_name
    chunk.content = content
    chunk.page_number = page_number
    chunk.section_title = section_title
    chunk.document_url = document_url
    chunk.document_path = document_url
    chunk.start_offset = 0
    chunk.end_offset = len(content)
    return chunk


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestCitationServiceMarkerBased:
    """Standard marker-based citation generation (LLM uses [1], [2] markers)."""

    def test_markers_generate_citations(self):
        chunk = _make_chunk()
        answer = "Revenue grew significantly [1]. This indicates strong performance."
        svc = CitationService()
        rewritten, citations = svc.generate_citations(answer, [chunk])

        assert len(citations) == 1
        c = citations[0]
        assert c["document_id"] == "doc-001"
        assert c["chunk_id"] == "chunk-abc"
        assert c["page_number"] == 3
        assert c["section_title"] == "Financial Highlights"
        assert c["type"] == "document"
        assert "[1]" in rewritten

    def test_out_of_range_markers_stripped(self):
        chunk = _make_chunk()
        answer = "Revenue [1] also showed growth [99]."
        svc = CitationService()
        rewritten, citations = svc.generate_citations(answer, [chunk])

        assert len(citations) == 1
        assert "[99]" not in rewritten

    def test_no_markers_returns_empty_citations(self):
        """Confirms that without markers, generate_citations() returns empty list —
        this is the root cause of DEF-015 and the trigger for the fallback."""
        chunk = _make_chunk()
        answer = "The company performed well. Revenue increased substantially. EBITDA improved."
        svc = CitationService()
        _, citations = svc.generate_citations(answer, [chunk])
        assert citations == []


class TestCitationServiceChunkFallback:
    """Fallback path: generate_citations_from_chunks() builds citations from
    retrieved chunks when the LLM emits no [n] markers."""

    def test_single_chunk_produces_one_citation(self):
        chunk = _make_chunk()
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([chunk])

        assert len(citations) == 1
        c = citations[0]
        assert c["document_id"] == "doc-001"
        assert c["chunk_id"] == "chunk-abc"
        assert c["page_number"] == 3
        assert c["section_title"] == "Financial Highlights"
        assert c["type"] == "document"
        assert "Revenue" in (c["excerpt"] or "")

    def test_multiple_chunks_all_returned(self):
        chunks = [
            _make_chunk(chunk_id="chunk-1", page_number=1, section_title="Intro"),
            _make_chunk(chunk_id="chunk-2", page_number=5, section_title="Risk Factors"),
            _make_chunk(chunk_id="chunk-3", page_number=9, section_title="Financials"),
        ]
        svc = CitationService()
        citations = svc.generate_citations_from_chunks(chunks)
        assert len(citations) == 3

    def test_duplicate_chunk_ids_deduplicated(self):
        chunks = [
            _make_chunk(chunk_id="same-id", page_number=1),
            _make_chunk(chunk_id="same-id", page_number=1),  # duplicate
            _make_chunk(chunk_id="other-id", page_number=2),
        ]
        svc = CitationService()
        citations = svc.generate_citations_from_chunks(chunks)
        assert len(citations) == 2

    def test_empty_content_chunks_excluded_by_default(self):
        chunk = _make_chunk(content="")
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([chunk])
        assert len(citations) == 0

    def test_empty_content_chunks_included_when_allowed(self):
        chunk = _make_chunk(content="")
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([chunk], allow_empty_excerpt=True)
        assert len(citations) == 1

    def test_non_document_dict_with_url_included(self):
        web_source = {
            "type": "web",
            "source_type": "web",
            "url": "https://example.com/article",
            "title": "Web Article",
            "excerpt": "This is a snippet.",
        }
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([web_source])
        assert len(citations) == 1

    def test_non_document_dict_without_url_excluded(self):
        source = {"type": "web", "source_type": "web", "title": "No URL"}
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([source])
        assert len(citations) == 0

    def test_metadata_fully_mapped(self):
        chunk = _make_chunk(
            doc_id="doc-xyz",
            chunk_id="chunk-xyz",
            file_name="Q4_results.pdf",
            page_number=12,
            section_title="CEO Commentary",
            document_url="/static/workspace_docs/Q4.pdf",
        )
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([chunk])
        c = citations[0]

        assert c["document_id"] == "doc-xyz"
        assert c["chunk_id"] == "chunk-xyz"
        assert c["document_name"] == "Q4_results.pdf"
        assert c["page_number"] == 12
        assert c["section_title"] == "CEO Commentary"
        assert c["navigation_type"] == "document"
        assert "/v1/documents/doc-xyz/viewer" in c["navigation_target"]


class TestCitationFallbackIntegration:
    """Simulates the full genai.py citation path:
    1. LLM generates a response with NO markers.
    2. generate_citations() returns empty.
    3. Fallback triggers generate_citations_from_chunks().
    4. Final citations are non-empty.
    """

    def _run_citation_pipeline(self, answer: str, chunks: list) -> list:
        """Mirrors the genai.py event_stream() citation logic exactly."""
        svc = CitationService()
        cited_sources = list(chunks)

        # Step 1: marker-based
        _, citations = svc.generate_citations(answer, cited_sources)

        # Step 2: fallback if no markers
        if not citations and chunks:
            citations = svc.generate_citations_from_chunks(cited_sources)

        return citations

    def test_narrative_response_still_gets_citations(self):
        """Core regression for DEF-015: a narrative analysis with no markers must
        still produce citations from the retrieved chunks."""
        chunk = _make_chunk(
            chunk_id="chunk-narr",
            page_number=4,
            section_title="Market Overview",
            content="The company operates in 50 countries with a market share of 32%.",
        )
        answer = (
            "The company shows strong global presence across multiple regions. "
            "Its market penetration strategy has been particularly effective. "
            "The risk profile remains moderate given the diversified portfolio."
        )
        citations = self._run_citation_pipeline(answer, [chunk])
        assert len(citations) == 1, "Expected chunk-based fallback citations"
        assert citations[0]["chunk_id"] == "chunk-narr"
        assert citations[0]["page_number"] == 4
        assert citations[0]["section_title"] == "Market Overview"

    def test_marker_response_uses_standard_citations(self):
        """When markers ARE present, the standard path is used (not the fallback)."""
        chunk = _make_chunk(chunk_id="chunk-mark")
        answer = "Revenue grew 20% [1]. This reflects solid execution."
        citations = self._run_citation_pipeline(answer, [chunk])
        assert len(citations) == 1
        assert citations[0]["reference_index"] == 1  # marker-based

    def test_no_chunks_no_citations(self):
        """If no chunks were retrieved, no citations are generated."""
        citations = self._run_citation_pipeline("Some response.", [])
        assert citations == []

    def test_multiple_chunks_all_cited_in_fallback(self):
        chunks = [
            _make_chunk(chunk_id="c1", page_number=1, section_title="Overview"),
            _make_chunk(chunk_id="c2", page_number=3, section_title="Revenue"),
            _make_chunk(chunk_id="c3", page_number=7, section_title="Risks"),
        ]
        answer = "The company has strong fundamentals and manages risk effectively."
        citations = self._run_citation_pipeline(answer, chunks)
        assert len(citations) == 3
        cited_ids = {c["chunk_id"] for c in citations}
        assert cited_ids == {"c1", "c2", "c3"}

    def test_citation_saved_to_db_reflects_correct_fields(self):
        """Verify the citation dict matches what the API serializes and what the
        frontend citation viewer expects."""
        chunk = _make_chunk(
            doc_id="d-001",
            chunk_id="ck-001",
            file_name="report.pdf",
            page_number=10,
            section_title="Summary",
            content="Net income was $500M.",
        )
        answer = "No markers here."
        citations = self._run_citation_pipeline(answer, [chunk])
        c = citations[0]

        # Fields required by the frontend Citation View
        required_fields = [
            "id", "document_id", "chunk_id", "page_number", "section_title",
            "excerpt", "document_name", "navigation_type", "navigation_target",
            "type", "provider",
        ]
        for field in required_fields:
            assert field in c, f"Missing field: {field}"

        assert c["document_id"] == "d-001"
        assert c["chunk_id"] == "ck-001"
        assert c["page_number"] == 10
        assert c["section_title"] == "Summary"
        assert "Net income" in (c["excerpt"] or "")
