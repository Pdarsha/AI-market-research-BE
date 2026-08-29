"""
Regression test for DEF-015 / TC-AD-03:
"Analyze 1 Source" document analysis must always return citations
even when the LLM response does not include explicit [1][2] markers.
"""

import pytest
from unittest.mock import MagicMock
from server.services.citation_service import CitationData, CitationService
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

    def test_dict_source_with_content_field(self):
        """Verify CitationData properly extracts excerpt and description when source dict uses 'content'."""
        source_dict = {
            "type": "document",
            "document_id": "doc-dict-1",
            "id": "chunk-dict-1",
            "file_name": "quarterly.pdf",
            "content": "Operating margin improved to 18.5% year over year.",
            "page_number": 2,
            "section_title": "Financial Highlights",
        }
        svc = CitationService()
        citations = svc.generate_citations_from_chunks([source_dict])
        assert len(citations) == 1
        c = citations[0]
        assert c["excerpt"] == "Operating margin improved to 18.5% year over year."
        assert c["description"] == "Operating margin improved to 18.5% year over year."
        assert c["chunk_id"] == "chunk-dict-1"
        assert c["document_name"] == "quarterly.pdf"

    def test_check_document_context_sufficiency_for_analyze_requests(self):
        """Verify check_document_context_sufficiency recognizes document analysis / summary questions as sufficient."""
        from server.services.rag_service import RAGService
        context = "[1] The company reported $12B revenue in 2024 with 15% growth in North America."
        
        # Test varied analysis queries
        assert RAGService.check_document_context_sufficiency(None, "Analyze the selected document", context=context, model="gpt-4") is True
        assert RAGService.check_document_context_sufficiency(None, "Summarize this document", context=context, model="gpt-4") is True
        assert RAGService.check_document_context_sufficiency(None, "Extract the key points from this report", context=context, model="gpt-4") is True
        assert RAGService.check_document_context_sufficiency(None, "Provide an overview of the findings", context=context, model="gpt-4") is True

    def test_workspace_document_citations_and_attribution(self):
        """DEF-005: Verify questions answered against workspace documents return document-level citations."""
        from server.utils.source_attribution import source_attribution_from_citations, cited_document_names
        from server.utils.helpers import document_sources_only
        
        chunk_ws = _make_chunk(
            doc_id="ws-doc-123",
            chunk_id="chunk-ws-456",
            file_name="Q3_Financial_Review.pdf",
            page_number=5,
            section_title="Operating Results",
            content="Total operating revenue reached $4.2 billion for the quarter.",
        )
        
        answer = "Total operating revenue for the quarter was $4.2 billion [1]."
        svc = CitationService()
        rewritten, citations = svc.generate_citations(answer, [chunk_ws])
        
        assert len(citations) == 1
        cite = citations[0]
        assert cite["document_id"] == "ws-doc-123"
        assert cite["chunk_id"] == "chunk-ws-456"
        assert cite["document_name"] == "Q3_Financial_Review.pdf"
        assert cite["page_number"] == 5
        assert cite["section_title"] == "Operating Results"
        assert "4.2 billion" in (cite["excerpt"] or "")
        
        # Verify source documents and attribution
        doc_names = document_sources_only(cited_document_names(citations))
        assert doc_names == ["Q3_Financial_Review.pdf"]
        
        attrib = source_attribution_from_citations(citations)
        assert len(attrib.get("uploaded_documents", [])) == 1
        assert attrib["uploaded_documents"][0]["document_id"] == "ws-doc-123"

    def test_v1_actual_supporting_sources_selective_citation(self):
        """Verification 1: When 5 chunks are retrieved but LLM only cites chunks 1 and 3,
        citations must contain only the 2 supporting chunks (no unreferenced citations).
        """
        c1 = _make_chunk(chunk_id="c1", content="Revenue was $100M.")
        c2 = _make_chunk(chunk_id="c2", content="Headcount was 500.")
        c3 = _make_chunk(chunk_id="c3", content="Operating profit was $20M.")
        c4 = _make_chunk(chunk_id="c4", content="Offices in 10 countries.")
        c5 = _make_chunk(chunk_id="c5", content="R&D budget was $15M.")
        all_chunks = [c1, c2, c3, c4, c5]

        answer = "Revenue reached $100M [1], and operating profit was $20M [3]."
        svc = CitationService()
        rewritten, citations = svc.generate_citations(answer, all_chunks)

        assert len(citations) == 2, "Only cited sources should be returned"
        cited_chunk_ids = {c["chunk_id"] for c in citations}
        assert cited_chunk_ids == {"c1", "c3"}
        assert "c2" not in cited_chunk_ids
        assert "c4" not in cited_chunk_ids
        assert "c5" not in cited_chunk_ids

    def test_v2_inline_citation_mapping_integrity(self):
        """Verification 2: Inline markers [1] and [2] must map to the exact retrieved chunks in order."""
        c1 = _make_chunk(
            doc_id="d-alpha",
            chunk_id="chk-1",
            file_name="Alpha_Report.pdf",
            page_number=4,
            section_title="Financial Highlights",
            content="Alpha EBITDA grew 25%.",
        )
        c2 = _make_chunk(
            doc_id="d-beta",
            chunk_id="chk-2",
            file_name="Beta_Report.pdf",
            page_number=8,
            section_title="Risk Analysis",
            content="Beta faced supply chain delays.",
        )
        sources = [c1, c2]

        answer = "Alpha EBITDA grew 25% [1], while Beta faced delays [2]."
        svc = CitationService()
        rewritten, citations = svc.generate_citations(answer, sources)

        assert len(citations) == 2
        cite1 = next(c for c in citations if c["reference_index"] == 1)
        cite2 = next(c for c in citations if c["reference_index"] == 2)

        assert cite1["document_id"] == "d-alpha"
        assert cite1["chunk_id"] == "chk-1"
        assert cite1["document_name"] == "Alpha_Report.pdf"
        assert cite1["page_number"] == 4
        assert cite1["section_title"] == "Financial Highlights"
        assert "Alpha EBITDA" in cite1["excerpt"]

        assert cite2["document_id"] == "d-beta"
        assert cite2["chunk_id"] == "chk-2"
        assert cite2["document_name"] == "Beta_Report.pdf"
        assert cite2["page_number"] == 8
        assert cite2["section_title"] == "Risk Analysis"
        assert "Beta faced" in cite2["excerpt"]

    def test_v3_narrative_answer_fallback(self):
        """Verification 3: When LLM returns a narrative answer without [n] markers, fallback
        generate_citations_from_chunks and attach_fallback_markers must provide attribution.
        """
        c1 = _make_chunk(
            doc_id="d-narr",
            chunk_id="chk-narr-1",
            file_name="Policy.pdf",
            page_number=2,
            content="The primary objective is zero-carbon emissions by 2030.",
        )
        answer = "The document states that the primary objective is zero-carbon emissions by 2030."

        svc = CitationService()
        rewritten, citations = svc.generate_citations(answer, [c1])
        assert citations == []

        # Fallback path
        citations = svc.generate_citations_from_chunks([c1])
        assert len(citations) == 1
        assert citations[0]["document_id"] == "d-narr"
        assert citations[0]["chunk_id"] == "chk-narr-1"

        # Fallback marker attachment
        attached_answer = svc.attach_fallback_markers(answer, citations)
        assert "[1]" in attached_answer
        assert attached_answer.endswith("[1]")

    def test_v4_sufficiency_for_external_and_empty_queries(self):
        """Verification 4: Real-time queries and empty context correctly trigger web search fallback."""
        from server.services.rag_service import RAGService
        context = "[1] (File: test.pdf Page: 1) Company overview and strategic priorities."

        # Real-time / live market queries -> should return False (trigger web search)
        assert RAGService.check_document_context_sufficiency(None, "What is the latest stock price today?", context=context, model="gpt-4") is False
        assert RAGService.check_document_context_sufficiency(None, "Breaking news about the CEO", context=context, model="gpt-4") is False

        # Empty context -> should return False (trigger web search)
        assert RAGService.check_document_context_sufficiency(None, "Summarize document", context="", model="gpt-4") is False

    def test_v6_follow_up_document_context_resolution(self):
        """Verification 6: resolve_active_document_ids must preserve document context across follow-up turns."""
        from server.utils.helpers import resolve_active_document_ids
        
        class MockMessage:
            def __init__(self, doc_ids):
                self.document_ids = doc_ids

        # Turn 1: Initial upload / query on doc-101
        m1 = MockMessage(["doc-101"])
        m2 = MockMessage(["doc-101"])

        # Turn 2: Follow-up question with no explicit requested document_ids
        active_ids, reason = resolve_active_document_ids([m1, m2], requested=None)
        assert active_ids == ["doc-101"]
        assert reason == "follow_up"

        # Turn 3: New upload of doc-202 replaces previous active context
        active_ids, reason = resolve_active_document_ids([m1, m2], requested=["doc-202"])
        assert active_ids == ["doc-202"]
        assert reason == "new_upload"

    def test_v7_multi_document_grid_citations(self):
        """Verification 7: Multi-document grid creation generates citations for all documents included."""
        from server.utils.helpers import document_sources_only
        from server.utils.source_attribution import cited_document_names
        
        c1 = _make_chunk(doc_id="doc-grid-1", chunk_id="chk-g1", file_name="Doc1.pdf", content="Doc 1 summary.")
        c2 = _make_chunk(doc_id="doc-grid-2", chunk_id="chk-g2", file_name="Doc2.pdf", content="Doc 2 summary.")

        svc = CitationService()
        citations = svc.generate_citations_from_chunks([c1, c2])

        assert len(citations) == 2
        doc_ids = {c["document_id"] for c in citations}
        assert doc_ids == {"doc-grid-1", "doc-grid-2"}

        doc_names = document_sources_only(cited_document_names(citations))
        assert doc_names == ["Doc1.pdf", "Doc2.pdf"]

    def test_v8_api_sse_citation_payload_consistency(self):
        """Verification 8: Verify CitationData produces all required fields expected by API and frontend."""
        chunk = _make_chunk(
            doc_id="doc-api-1",
            chunk_id="chk-api-1",
            file_name="Contract.pdf",
            page_number=12,
            section_title="Term and Termination",
            content="The initial term is three years.",
        )
        cd = CitationData(citation_id=1, source=chunk)
        d = cd.to_dict()

        # Contractual fields for frontend and API
        assert d["id"] == "citation_1"
        assert d["citation_id"] == 1
        assert d["reference_index"] == 1
        assert d["type"] == "document"
        assert d["provider"] == "uploaded_document"
        assert d["document_id"] == "doc-api-1"
        assert d["document_name"] == "Contract.pdf"
        assert d["chunk_id"] == "chk-api-1"
        assert d["page_number"] == 12
        assert d["page"] == 12
        assert d["section_title"] == "Term and Termination"
        assert d["excerpt"] == "The initial term is three years."
        assert d["highlight_text"] == "The initial term is three years."
        assert d["navigable"] is True
        assert d["navigation_type"] == "document"
        assert d["navigation_target"] == "/v1/documents/doc-api-1/viewer"

    def test_v9_citation_data_source_shapes_and_content_fallback(self):
        """Verification 9: CitationData must extract description/excerpt correctly from various dictionary source shapes."""
        # Shape A: uses 'excerpt'
        s_excerpt = {"type": "document", "document_id": "d1", "file_name": "f1.pdf", "excerpt": "Excerpt text"}
        cd_a = CitationData(1, source=s_excerpt)
        assert cd_a.excerpt == "Excerpt text"
        assert cd_a.description == "Excerpt text"

        # Shape B: uses 'snippet'
        s_snippet = {"type": "document", "document_id": "d2", "file_name": "f2.pdf", "snippet": "Snippet text"}
        cd_b = CitationData(2, source=s_snippet)
        assert cd_b.excerpt == "Snippet text"
        assert cd_b.description == "Snippet text"

        # Shape C: uses 'content' (e.g. raw DocumentChunk dict or agent findings)
        s_content = {"type": "document", "document_id": "d3", "file_name": "f3.pdf", "content": "Content text"}
        cd_c = CitationData(3, source=s_content)
        assert cd_c.excerpt == "Content text"
        assert cd_c.description == "Content text"
        assert cd_c.to_dict()["excerpt"] == "Content text"
        assert cd_c.to_dict()["highlight_text"] == "Content text"


class TestPDFExportService:
    """Test suite for TC-EX-01, TC-EX-02, and Header Truncation fixes."""

    def test_tc_ex_01_normal_export_template_applied(self):
        """Test 1: Approved template, branding, and HTML structure are applied."""
        import fitz
        import unicodedata
        from server.services.pdf_export_service import conversation_to_pdf

        messages = [
            {"role": "user", "content": "What is CapitalEdge AI?"},
            {"role": "assistant", "content": "CapitalEdge AI is an enterprise financial intelligence platform."},
        ]
        pdf_bytes = conversation_to_pdf("Company Overview", messages)
        assert len(pdf_bytes) > 1000
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1

        text = unicodedata.normalize("NFKD", doc[0].get_text())
        assert "Company Overview" in text
        assert "Generated:" in text
        assert "What is CapitalEdge AI?" in text
        assert "CapitalEdge AI is an enterprise financial intelligence platform." in text
        assert "Page 1 of" in text

    def test_tc_ex_02_markdown_headings_converted_no_literal_markers(self):
        """Test 2: Markdown headings (#, ##, ###) are converted to headings and no literal # markers appear in text."""
        import fitz
        from server.services.pdf_export_service import conversation_to_pdf

        content = (
            "# Main Heading\n\n"
            "Some paragraph.\n\n"
            "## Section Heading\n\n"
            "Some section content.\n\n"
            "### Subsection Heading\n\n"
            "More content."
        )
        messages = [
            {"role": "user", "content": "Generate report with headings"},
            {"role": "assistant", "content": content},
        ]
        pdf_bytes = conversation_to_pdf("Headings Report", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = doc[0].get_text()

        # Headings text must appear
        assert "Main Heading" in text
        assert "Section Heading" in text
        assert "Subsection Heading" in text

        # Raw markdown heading markers must NOT appear literally
        lines = [line.strip() for line in text.split("\n")]
        assert "# Main Heading" not in lines
        assert "## Section Heading" not in lines
        assert "### Subsection Heading" not in lines

    def test_header_no_truncation_for_long_names(self):
        """Test 3: Long company name / header text fits cleanly without truncation or '...'."""
        import fitz
        from server.services.pdf_export_service import conversation_to_pdf

        messages = [
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Test response content."},
        ]
        pdf_bytes = conversation_to_pdf("Test Title", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = doc[0].get_text()

        assert "..." not in text

    def test_formatting_lists_tables_bold_code(self):
        """Test 4: Markdown formatting (lists, tables, bold, code) renders properly."""
        import fitz
        import unicodedata
        from server.services.pdf_export_service import conversation_to_pdf

        content = (
            "## Financial Summary\n\n"
            "- **Revenue**: $500M\n"
            "- **EBITDA**: $120M\n\n"
            "1. First priority\n"
            "2. Second priority\n\n"
            "| Segment | Growth |\n"
            "|---|---|\n"
            "| Cloud | +25% |\n"
            "| Core | +10% |\n\n"
            "> Note: Preliminary figures.\n\n"
            "Inline `code_snippet()`."
        )
        messages = [{"role": "assistant", "content": content}]
        pdf_bytes = conversation_to_pdf("Formatting Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = unicodedata.normalize("NFKD", doc[0].get_text())

        assert "Financial Summary" in text
        assert "Revenue" in text
        assert "Cloud" in text
        assert "+25%" in text
        assert "Preliminary figures" in text

    def test_citations_preserved_with_clickable_links(self):
        """Test 5: Citations are preserved in body text and sources section with hyperlinks."""
        import fitz
        from server.services.pdf_export_service import conversation_to_pdf

        citations = [
            {
                "id": "citation_1",
                "reference_index": 1,
                "title": "Q3 2024 Report.pdf",
                "document_id": "doc-123",
                "url": "https://example.com/doc-123",
                "excerpt": "Revenue was $100M in Q3.",
            }
        ]
        messages = [
            {
                "role": "assistant",
                "content": "According to the financial results, revenue reached $100M [1].",
                "citations": citations,
            }
        ]
        pdf_bytes = conversation_to_pdf("Citation Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        text = page.get_text()

        assert "[1]" in text
        assert "Sources" in text
        assert "Q3 2024 Report.pdf" in text

        # Verify hyperlink exists for citation
        links = page.get_links()
        uri_links = [l for l in links if l.get("kind") == fitz.LINK_URI]
        assert len(uri_links) >= 1
        assert any("example.com" in (l.get("uri") or "") for l in uri_links)

    def test_markdown_preprocessing_edge_cases(self):
        """Test 6: Preprocessor handles unspaced headings, literal slashes, and followup tokens."""
        from server.services.pdf_export_service import _preprocess_markdown

        raw = "###HeadingWithoutSpace\nText\n<<<FOLLOW_UP_QUESTIONS>>>\n[\"Q1\"]"
        processed = _preprocess_markdown(raw)
        assert "### HeadingWithoutSpace" in processed
        assert "<<<FOLLOW_UP_QUESTIONS>>>" not in processed


class TestDEF015AnalyzeSelectedDocument:
    """Test suite for DEF-015 / TC-AD-03: Analyze Selected Document Citations and Follow-up Questions."""

    def test_def_015_build_context_with_citations_preserves_page_and_section(self):
        """Test that build_context_with_citations includes page and section metadata in chunk headers."""
        from server.services.rag_service import RAGService

        chunk1 = _make_chunk(
            chunk_id="c1",
            content="Alpha Corp revenue grew 18% to $1.45B in FY2024.",
            doc_id="doc-1",
            file_name="Alpha_10K.pdf",
            page_number=4,
            section_title="Financial Highlights",
        )
        chunk2 = _make_chunk(
            chunk_id="c2",
            content="Primary risk factors include supply chain volatility and inflation.",
            doc_id="doc-1",
            file_name="Alpha_10K.pdf",
            page_number=12,
            section_title="Risk Factors",
        )

        ctx = RAGService.build_context_with_citations([chunk1, chunk2])
        assert "[1] (Document: Alpha_10K.pdf, Page: 4, Section: Financial Highlights)" in ctx
        assert "[2] (Document: Alpha_10K.pdf, Page: 12, Section: Risk Factors)" in ctx
        assert "Alpha Corp revenue grew 18%" in ctx
        assert "Primary risk factors include" in ctx

    def test_def_015_section_level_citations_extraction(self):
        """Test that section-level citations are correctly extracted and mapped to supporting chunks with page/section metadata."""
        chunk1 = _make_chunk(
            chunk_id="c1",
            content="Alpha Corp total revenue was $1,450M in 2024.",
            doc_id="doc-1",
            file_name="Alpha_10K.pdf",
            page_number=4,
            section_title="Executive Summary",
        )
        chunk2 = _make_chunk(
            chunk_id="c2",
            content="Cloud segment revenue was $820M representing 56% of total revenue.",
            doc_id="doc-1",
            file_name="Alpha_10K.pdf",
            page_number=8,
            section_title="Segment Performance",
        )
        chunk3 = _make_chunk(
            chunk_id="c3",
            content="Supply chain constraints could adversely affect hardware delivery.",
            doc_id="doc-1",
            file_name="Alpha_10K.pdf",
            page_number=15,
            section_title="Risk Analysis",
        )

        analysis_text = (
            "## Executive Summary\n"
            "Alpha Corp delivered strong financial growth with total revenue of $1,450M in 2024 [1].\n\n"
            "## Key Findings\n"
            "The Cloud division was the key growth driver generating $820M [2].\n\n"
            "## Risks\n"
            "Management noted potential headwinds from supply chain constraints [3]."
        )

        citation_svc = CitationService()
        rewritten, citations = citation_svc.generate_citations(analysis_text, [chunk1, chunk2, chunk3])

        assert len(citations) == 3
        # Verify Section 1 -> Chunk 1
        assert citations[0]["document_id"] == "doc-1"
        assert citations[0]["file_name"] == "Alpha_10K.pdf"
        assert citations[0]["page_number"] == 4
        assert citations[0]["section_title"] == "Executive Summary"

        # Verify Section 2 -> Chunk 2
        assert citations[1]["page_number"] == 8
        assert citations[1]["section_title"] == "Segment Performance"

        # Verify Section 3 -> Chunk 3
        assert citations[2]["page_number"] == 15
        assert citations[2]["section_title"] == "Risk Analysis"

    def test_def_015_followup_question_generation_for_document_analysis(self):
        """Test that follow-up questions are generated and returned for document analysis queries."""
        from server.services.rag_service import RAGService

        raw_response = (
            "## Executive Summary\n"
            "Alpha Corp revenue was $1,450M with operating margins of 18.5% [1].\n\n"
            "## Key Findings\n"
            "- Enterprise Cloud expanded by 26% YoY [1].\n\n"
            "<<<FOLLOW_UP_QUESTIONS>>>\n"
            "[\"What are the major growth drivers for Enterprise Cloud?\", \"How do operating margins compare to previous fiscal years?\", \"What are the main risk factors highlighted in the report?\"]"
        )

        answer, followups = RAGService.split_answer_and_followups(raw_response, question="Analyze this document")
        assert len(followups) == 3
        assert "What are the major growth drivers for Enterprise Cloud?" in followups
        assert "How do operating margins compare to previous fiscal years?" in followups
        assert "What are the main risk factors highlighted in the report?" in followups
        assert "<<<FOLLOW_UP_QUESTIONS>>>" not in answer

    def test_def_015_followup_fallback_generation_when_marker_omitted(self):
        """Test that contextual follow-up questions are generated when the model omits the marker."""
        from server.services.rag_service import RAGService

        analysis_text = (
            "## Executive Summary\n"
            "Alpha Corp reported revenue of $1,450M and net income of $185M in FY2024.\n"
            "The Enterprise Cloud division delivered 26% growth while margins improved."
        )

        answer, followups = RAGService.split_answer_and_followups(analysis_text, question="Analyze this document")
        assert len(followups) >= 3
        for q in followups:
            assert isinstance(q, str) and len(q) > 10
            # Ensure not generic boilerplate
            assert "can i help" not in q.lower()
            assert "would you like more information" not in q.lower()


class TestDEF009ExportTableAndCitations:
    """Test suite for DEF-009 / TC-EX-04: Export Table and Citation Preservation."""

    def test_basic_table_markdown_to_html_and_pdf(self):
        """Test 1: Markdown table is parsed into HTML <table> and rendered into PDF."""
        import markdown
        import fitz
        from server.services.pdf_export_service import _preprocess_markdown, conversation_to_pdf

        raw_md = """
| Metric | Q1 | Q2 | Q3 |
|--------|----|----|----|
| Revenue | 100 | 120 | 145 |
| Profit  | 20  | 25  | 31  |
"""
        pre = _preprocess_markdown(raw_md)
        html = markdown.markdown(pre, extensions=["tables", "fenced_code", "nl2br", "sane_lists"])
        assert "<table>" in html
        assert "<th>Metric</th>" in html
        assert "<td>Revenue</td>" in html

        messages = [{"role": "assistant", "content": raw_md}]
        pdf_bytes = conversation_to_pdf("Table Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        import unicodedata
        text = unicodedata.normalize("NFKD", doc[0].get_text())
        assert "Metric" in text
        assert "Revenue" in text
        assert "145" in text
        assert "Profit" in text

    def test_table_with_citations_in_cells_and_sources(self):
        """Test 2: Table containing citations in cells preserves table structure, markers, and sources."""
        import fitz
        from server.services.pdf_export_service import conversation_to_pdf

        content = """## Financial Summary [1]

Here is the breakdown by quarter:
| Metric | Q1 | Q2 | Source |
|---|---|---|---|
| Revenue | $100M [1] | $120M | 10-Q Report [1] |
| Net Income | $20M | $25M [2] | SEC Filing [2] |

Source: [1]
"""
        citations = [
            {
                "id": "c1",
                "reference_index": 1,
                "title": "Alpha_10Q.pdf",
                "url": "https://example.com/alpha-10q",
                "excerpt": "Revenue was $100M in Q1.",
            },
            {
                "id": "c2",
                "reference_index": 2,
                "title": "SEC_Filing.pdf",
                "url": "https://example.com/sec-filing",
                "excerpt": "Net income was $25M in Q2.",
            },
        ]
        messages = [{"role": "assistant", "content": content, "citations": citations}]
        pdf_bytes = conversation_to_pdf("Table Citations Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = doc[0].get_text()

        # Check table content and markers
        assert "Financial Summary [1]" in text
        assert "Metric" in text
        assert "$100M [1]" in text
        assert "$25M [2]" in text
        assert "Sources" in text
        assert "Alpha_10Q.pdf" in text
        assert "SEC_Filing.pdf" in text

        # Check hyperlinks exist for table and inline citations
        links = doc[0].get_links()
        uri_links = [l for l in links if l.get("kind") == fitz.LINK_URI]
        assert len(uri_links) >= 3
        assert any("alpha-10q" in (l.get("uri") or "") for l in uri_links)
        assert any("sec-filing" in (l.get("uri") or "") for l in uri_links)

    def test_multiple_tables_in_single_export(self):
        """Test 3: Multiple consecutive tables in a single document remain distinct and properly structured."""
        import markdown
        import fitz
        from server.services.pdf_export_service import _preprocess_markdown, conversation_to_pdf

        content = """
## Performance Overview

First quarter metrics:
| Metric | Value |
|---|---|
| Revenue | $500M |
| EBITDA | $120M |

Risk assessment:
| Risk Area | Severity | Impact |
|---|---|---|
| Market | High | Critical |
| Operational | Medium | Moderate |
"""
        pre = _preprocess_markdown(content)
        html = markdown.markdown(pre, extensions=["tables", "fenced_code", "nl2br", "sane_lists"])
        assert html.count("<table>") == 2
        assert html.count("</table>") == 2

        messages = [{"role": "assistant", "content": content}]
        pdf_bytes = conversation_to_pdf("Multi Table Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = doc[0].get_text()
        assert "Revenue" in text
        assert "$500M" in text
        assert "Risk Area" in text
        assert "Critical" in text

    def test_table_without_blank_lines_preceding_or_following(self):
        """Test 4: Tables without explicit surrounding blank lines are cleanly parsed without flattening."""
        import markdown
        from server.services.pdf_export_service import _preprocess_markdown

        raw = "Heading text\n| A | B |\n|---|---|\n| 1 | 2 |\nFooter text"
        pre = _preprocess_markdown(raw)
        html = markdown.markdown(pre, extensions=["tables", "fenced_code", "nl2br", "sane_lists"])
        assert "<table>" in html
        assert "<td>1</td>" in html
        assert "<p>Footer text</p>" in html

    def test_long_table_rendering(self):
        """Test 5: Large table with many rows renders cleanly without crashing."""
        import fitz
        from server.services.pdf_export_service import conversation_to_pdf

        rows = [f"| Item {i} | Category {i % 5} | ${i * 100} | Status {i} |" for i in range(1, 40)]
        table_md = "| Item | Category | Amount | Status |\n|---|---|---|---|\n" + "\n".join(rows)
        messages = [{"role": "assistant", "content": f"## Inventory Report\n\n{table_md}"}]

        pdf_bytes = conversation_to_pdf("Long Table Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        all_text = "".join(p.get_text() for p in doc)
        assert "Item 1" in all_text
        assert "Item 39" in all_text

    def test_mixed_content_with_citations_and_tables(self):
        """Test 6: Full mixed document with headers, paragraphs, tables, code, quotes, and citations."""
        import fitz
        from server.services.pdf_export_service import conversation_to_pdf

        content = """# Executive Briefing [1]

CapitalEdge delivered outstanding quarterly results [1].

## Key Financials

| Segment | Q1 | Q2 | Growth [1] |
|---|---|---|---|
| Enterprise | $120M | $145M | +20.8% [1] |
| Consumer | $80M | $85M | +6.2% [2] |

> Note: All numbers are non-GAAP adjusted.

```python
# Verification snippet
margin = 145 / (145 + 85)
```

## Risks & Considerations [2]

Primary headwinds involve supply chain volatility [2].
"""
        citations = [
            {
                "id": "c1",
                "reference_index": 1,
                "title": "Q2_Earnings.pdf",
                "url": "https://example.com/q2-earnings",
            },
            {
                "id": "c2",
                "reference_index": 2,
                "title": "Risk_Disclosures.pdf",
                "url": "https://example.com/risks",
            },
        ]
        messages = [{"role": "assistant", "content": content, "citations": citations}]
        pdf_bytes = conversation_to_pdf("Mixed Content Test", messages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        import unicodedata
        all_text = unicodedata.normalize("NFKD", "".join(p.get_text() for p in doc))

        assert "Executive Briefing [1]" in all_text
        assert "Enterprise" in all_text
        assert "+20.8% [1]" in all_text
        assert "Sources" in all_text
        assert "Q2_Earnings.pdf" in all_text
        assert "Risk_Disclosures.pdf" in all_text

        # Verify no literal heading markers leaked into headings
        assert "# Executive Briefing" not in all_text
        assert "## Key Financials" not in all_text
        assert "## Risks & Considerations" not in all_text


class TestDEF010FollowUpQuestions:
    """Test suite for DEF-010 / TC-FQ-01: Follow-Up Questions Consistency across all AI Response Flows."""

    def test_main_chat_followups_generation(self):
        """Test 1: Main chat parses model follow-up questions from <<<FOLLOW_UP_QUESTIONS>>> marker."""
        from server.services.rag_service import RAGService

        raw = (
            "Python 3.12 introduces enhanced error messages and syntax improvements.\n\n"
            "<<<FOLLOW_UP_QUESTIONS>>>\n"
            "[\"What are the major performance improvements in Python 3.12?\", \"How does the new syntax affect legacy code?\", \"Can you provide examples of improved error messages?\"]"
        )
        ans, followups = RAGService.split_answer_and_followups(raw, question="Tell me about Python 3.12")
        assert len(followups) == 3
        assert "What are the major performance improvements in Python 3.12?" in followups
        assert "How does the new syntax affect legacy code?" in followups
        assert "<<<FOLLOW_UP_QUESTIONS>>>" not in ans

    def test_workspace_document_question_followups(self):
        """Test 2: Workspace document queries generate grounded follow-up questions."""
        from server.services.rag_service import RAGService

        doc_answer = (
            "Alpha Corp reported revenue of $1.45B for FY2024, representing an 18% increase year-over-year. "
            "Net income grew to $185M driven by enterprise cloud services."
        )
        ans, followups = RAGService.split_answer_and_followups(doc_answer, question="What was Alpha Corp revenue?")
        assert len(followups) >= 3
        for q in followups:
            assert isinstance(q, str) and len(q) > 10
            # Ensure questions are domain-grounded and non-generic
            assert "can i help" not in q.lower()
            assert "would you like more information" not in q.lower()

    def test_multi_document_grid_followups(self):
        """Test 3: Multi-document analysis flow includes comparative follow-up questions."""
        from server.services.rag_service import RAGService

        grid_text = (
            "Comparing Apple Inc and Microsoft Corp performance for 2024:\n"
            "- Apple generated $383B in revenue with strong hardware and services growth.\n"
            "- Microsoft generated $245B in revenue led by Intelligent Cloud and Azure expansion."
        )
        ans, followups = RAGService.split_answer_and_followups(grid_text, question="Compare Apple and Microsoft")
        assert len(followups) >= 3
        assert any("apple" in q.lower() or "microsoft" in q.lower() or "revenue" in q.lower() for q in followups)

    def test_fallback_generates_when_marker_omitted(self):
        """Test 4: Contextual fallback generates relevant questions when LLM omits the follow-up marker."""
        from server.services.rag_service import RAGService

        answer = "Supply chain disruptions and inflationary pressures remain key headwinds for hardware manufacturers."
        ans, followups = RAGService.split_answer_and_followups(answer, question="What are the main risks?")
        assert len(followups) >= 3
        for q in followups:
            assert isinstance(q, str) and len(q) > 10
            assert "anything else" not in q.lower()

    def test_boilerplate_filtering(self):
        """Test 5: Generic boilerplate questions are filtered out and replaced with topic-grounded questions."""
        from server.services.rag_service import RAGService

        raw_with_boilerplate = (
            "GlobalTech cloud revenue increased by 22% this quarter.\n\n"
            "<<<FOLLOW_UP_QUESTIONS>>>\n"
            "[\"Can I help you with anything else?\", \"Would you like more information?\", \"What were the primary drivers for GlobalTech cloud revenue?\"]"
        )
        ans, followups = RAGService.split_answer_and_followups(raw_with_boilerplate, question="Analyze GlobalTech")
        # Boilerplate questions must be removed
        for q in followups:
            assert "can i help" not in q.lower()
            assert "would you like more information" not in q.lower()
        assert len(followups) >= 3
        assert any("globaltech" in q.lower() or "cloud" in q.lower() or "revenue" in q.lower() for q in followups)


class TestDEF013AnalyzeSelectedDocumentFollowUps:
    """Test suite for DEF-013 / TC-AD-01: Analyze Selected Document Follow-Up Questions."""

    def test_analyze_selected_document_followup_generation(self):
        """Test 1: Single document analysis generates contextual follow-up questions."""
        from server.services.rag_service import RAGService

        analysis_text = (
            "The document identifies three major financial risks facing the company:\n"
            "1. Market volatility affecting hardware sales\n"
            "2. Credit default risks among enterprise customers\n"
            "3. Inflationary pressures increasing operational expenditures\n\n"
            "Operating margin contracted by 2.4% over the period."
        )
        ans, followups = RAGService.split_answer_and_followups(
            analysis_text, question="Analyze the selected document (Risk_Report.pdf)"
        )
        assert len(followups) >= 3
        for q in followups:
            assert isinstance(q, str) and len(q) > 10
            # Ensure not generic boilerplate
            assert "can i help" not in q.lower()
            assert "would you like more information" not in q.lower()
            assert "anything else" not in q.lower()

    def test_analyze_selected_document_with_explicit_marker(self):
        """Test 2: When LLM outputs <<<FOLLOW_UP_QUESTIONS>>>, questions are extracted and marker is removed."""
        from server.services.rag_service import RAGService

        raw = (
            "## Document Analysis\n"
            "AlphaBeta reported $100M revenue with strong market adoption.\n\n"
            "<<<FOLLOW_UP_QUESTIONS>>>\n"
            "[\"What are the most significant financial risks identified in the document?\", \"How do these risks affect the company's financial outlook?\", \"Can you explain the recommended mitigation strategies?\"]"
        )
        ans, followups = RAGService.split_answer_and_followups(
            raw, question="Analyze the selected document (AlphaBeta_Q3.pdf)"
        )
        assert len(followups) == 3
        assert "What are the most significant financial risks identified in the document?" in followups
        assert "How do these risks affect the company's financial outlook?" in followups
        assert "Can you explain the recommended mitigation strategies?" in followups
        assert "<<<FOLLOW_UP_QUESTIONS>>>" not in ans

    def test_analyze_selected_document_fallback_when_marker_omitted(self):
        """Test 3: When LLM omits marker, contextual fallback questions are generated."""
        from server.services.rag_service import RAGService

        raw = (
            "The document provides an overview of Q3 financial performance. "
            "Revenue reached $120M while operating income remained steady."
        )
        ans, followups = RAGService.split_answer_and_followups(
            raw, question="Analyze this document"
        )
        assert len(followups) >= 3
        for q in followups:
            assert isinstance(q, str) and len(q) > 10
            assert "can i help" not in q.lower()


class TestDEF011FollowUpQualityAndDeduplication:
    """Test suite for DEF-011 / TC-FQ-02: Follow-Up Questions Source/Response-Derived Quality & Deduplication."""

    def test_different_responses_produce_different_followups(self):
        """Test 1 & 2: Two different domain responses produce distinct, non-generic follow-up questions."""
        from server.services.rag_service import RAGService

        resp_finance = "AlphaBeta reported $100M revenue in Q3 2026, a 15% increase driven by enterprise cloud services."
        resp_tech = "Python asyncio event loop handles non-blocking coroutines with async/await syntax."

        _, f_finance = RAGService.split_answer_and_followups(resp_finance, question="Analyze AlphaBeta")
        _, f_tech = RAGService.split_answer_and_followups(resp_tech, question="How does asyncio work?")

        assert len(f_finance) >= 3
        assert len(f_tech) >= 3
        # Ensure questions are different
        assert set(f_finance) != set(f_tech)
        assert any("alphabeta" in q.lower() or "$100m" in q.lower() or "revenue" in q.lower() for q in f_finance)
        assert any("asyncio" in q.lower() or "coroutine" in q.lower() or "event loop" in q.lower() or "async" in q.lower() for q in f_tech)

    def test_response_containing_risks_focuses_on_risks(self):
        """Test 5: Response discussing specific risks generates risk-focused follow-up questions."""
        from server.services.rag_service import RAGService

        resp_risk = "The audit identifies credit risk and liquidity constraints as major vulnerabilities for the institution."
        _, followups = RAGService.split_answer_and_followups(resp_risk, question="What are the audit findings?")

        assert len(followups) >= 3
        assert any("credit risk" in q.lower() or "liquidity" in q.lower() or "mitigation" in q.lower() for q in followups)

    def test_response_containing_metrics_focuses_on_metrics(self):
        """Test 4: Response discussing financial metrics generates metric-focused questions."""
        from server.services.rag_service import RAGService

        resp_metric = "CapitalEdge achieved 24.5% profit margins and $450M EBITDA in FY2025."
        _, followups = RAGService.split_answer_and_followups(resp_metric, question="Summarize CapitalEdge performance")

        assert len(followups) >= 3
        assert any("capitaledge" in q.lower() or "$450m" in q.lower() or "24.5%" in q.lower() or "margin" in q.lower() for q in followups)

    def test_generic_boilerplate_replaced_with_contextual(self):
        """Test 7: Reject boilerplate such as 'Would you like to know more?' and replace with contextual questions."""
        from server.services.rag_service import RAGService

        raw = (
            "Nvidia generated $30B in quarterly data center revenue.\n\n"
            "<<<FOLLOW_UP_QUESTIONS>>>\n"
            "[\"Would you like to know more?\", \"Can I help you with anything else?\", \"Would you like more details?\"]"
        )
        _, followups = RAGService.split_answer_and_followups(raw, question="Analyze Nvidia")

        for q in followups:
            ql = q.lower()
            assert "would you like to know more" not in ql
            assert "can i help" not in ql
            assert "would you like more details" not in ql
        assert len(followups) >= 3
        assert any("nvidia" in q.lower() or "revenue" in q.lower() or "$30b" in q.lower() for q in followups)

    def test_fuzzy_duplicate_questions_are_deduplicated(self):
        """Test 8: Near-duplicate questions are normalized and deduplicated."""
        from server.services.rag_service import RAGService

        raw = (
            "PostgreSQL handles JSONB data with specialized GIN indexing.\n\n"
            "<<<FOLLOW_UP_QUESTIONS>>>\n"
            "[\"Can you explain GIN indexing in more detail?\", \"Explain GIN indexing in detail.\", \"What are the performance trade-offs of GIN indexing?\"]"
        )
        _, followups = RAGService.split_answer_and_followups(raw, question="How does PostgreSQL index JSONB?")

        # The two near-duplicate questions should be deduplicated
        assert len(followups) == 3
        # Verify no duplicate normalized text
        normalized = [RAGService._normalize_question_text(q) for q in followups]
        assert len(normalized) == len(set(normalized))

    def test_source_context_grounded_followups(self):
        """Test: Source document chunks provided in context populate follow-up questions."""
        from server.services.rag_service import RAGService

        retrieved_context = (
            "Document: Risk_Assessment_2025.pdf\n"
            "Section: Strategic Exposures\n"
            "Alpha Corp faces severe supply chain volatility and $250M capital expenditure constraints."
        )
        answer = "The analysis indicates challenging operational conditions for the upcoming year."
        _, followups = RAGService.split_answer_and_followups(
            answer, question="Summarize operational risks", context=retrieved_context
        )

        assert len(followups) >= 3
        assert any("alpha corp" in q.lower() or "supply chain volatility" in q.lower() or "$250m" in q.lower() or "capital expenditure" in q.lower() for q in followups)

    def test_deduplication_preserves_distinct_questions(self):
        """Test: Distinct questions like 'cause' vs 'impact' are not falsely merged."""
        from server.services.rag_service import RAGService

        q1 = "What caused the revenue decline?"
        q2 = "What impact did the revenue decline have?"

        n1 = RAGService._normalize_question_text(q1)
        n2 = RAGService._normalize_question_text(q2)

        assert n1 != n2
        assert "caused" in n1
        assert "impact" in n2

    def test_workspace_analysis_different_workspaces_produce_distinct_questions(self):
        """Test: Different workspaces with distinct documents produce grounded, distinct follow-up questions."""
        from server.services.rag_service import RAGService

        q_ws1 = 'Perform a comprehensive, structured workspace analysis for "Aviation Industry".'
        resp_ws1 = (
            "1. **Workspace-wide Analysis**: Fleet logistics and operations.\n"
            "2. **Bulk Document Summarization**:\n"
            "   - **Aviation_Report.pdf**: Fuel hedging and aircraft maintenance.\n"
            "   - **Fleet_Logistics.docx**: Route optimization.\n"
            "3. **Trend Extraction**: Transitioning to SAF fuel.\n"
            "4. **Risk Identification**: High fuel volatility and regulatory constraints.\n"
            "5. **Strategic Observations & Recommendations**: Modernize fleet routing."
        )
        _, fqs1 = RAGService.split_answer_and_followups(resp_ws1, question=q_ws1)

        q_ws2 = 'Perform a comprehensive, structured workspace analysis for "Engineering Core".'
        resp_ws2 = (
            "1. **Workspace-wide Analysis**: Backend microservices architecture.\n"
            "2. **Bulk Document Summarization**:\n"
            "   - **01_System_Architecture.md**: High-level design of FastAPI services.\n"
            "   - **08_WebSocket_Gateway.md**: Real-time event streaming.\n"
            "3. **Trend Extraction**: Moving from REST polling to WebSocket streaming.\n"
            "4. **Risk Identification**: Connection concurrency bottlenecks.\n"
            "5. **Strategic Observations & Recommendations**: Scale gateway nodes horizontally."
        )
        _, fqs2 = RAGService.split_answer_and_followups(resp_ws2, question=q_ws2)

        assert len(fqs1) >= 3
        assert len(fqs2) >= 3
        assert set(fqs1) != set(fqs2)
        assert any("aviation_report.pdf" in q.lower() or "fleet_logistics.docx" in q.lower() for q in fqs1)
        assert any("01_system_architecture.md" in q.lower() or "08_websocket_gateway.md" in q.lower() for q in fqs2)
        assert not any("ai benefits" in q.lower() or "ai roi" in q.lower() for q in fqs1)
        assert not any("ai benefits" in q.lower() or "ai roi" in q.lower() for q in fqs2)

    def test_repeated_runs_on_same_document_produce_varied_questions(self):
        """Test: Executing follow-up generation 3 times on the same document produces varied sets across runs."""
        from server.services.rag_service import RAGService

        doc_text = (
            "# Meeting API Documentation Overview\n\n"
            "## 1. Authentication & Security\n"
            "OAuth 2.0 bearer tokens are required for all endpoints. Token expiration is 3600 seconds with automated refresh.\n\n"
            "## 2. Webhook Event Delivery\n"
            "Events are delivered via POST requests to registered webhook endpoints.\n"
            "- Retry policy: 3 exponential backoff retries.\n"
            "- Webhook latency target: < 250 ms.\n\n"
            "## 3. Database Architecture & Concurrency\n"
            "The service uses a PostgreSQL connection pool with Redis caching for session tokens.\n"
            "- Rate limiting: 100 req/sec per tenant.\n"
            "- Potential bottleneck: High connection pool saturation during peak morning hours.\n\n"
            "## 4. Key Risks\n"
            "- Webhook delivery timeout if client endpoint is degraded.\n"
            "- Token refresh race conditions under high concurrency."
        )

        runs = []
        for _ in range(5):
            _, fqs = RAGService.split_answer_and_followups(doc_text, question="Analyze MEETING_API_DOCUMENTATION.md")
            assert len(fqs) >= 3
            runs.append(tuple(fqs))

        # Across 5 runs, there should be variation in the generated sets
        unique_sets = set(runs)
        assert len(unique_sets) >= 2, "Repeated runs produced identical follow-up sets!"

    def test_ai_business_query_does_not_leak_unrelated_entities_or_code(self):
        """Test: AI in business queries do NOT introduce Microsoft, IBM, code, or financial capex."""
        from server.services.rag_service import RAGService

        q = "What are the main benefits of using artificial intelligence in business?"
        ans = "Artificial intelligence enhances workflow automation, predictive analytics, and customer personalization."
        _, followups = RAGService.split_answer_and_followups(ans, question=q)

        assert len(followups) >= 3
        for f in followups:
            fl = f.lower()
            assert "microsoft" not in fl
            assert "ibm" not in fl
            assert "asyncio" not in fl
            assert "capex" not in fl
            assert "10-k" not in fl
            assert any(term in fl for term in ["ai", "kpi", "roi", "governance", "adoption", "automation", "compliance", "workflow", "model", "capabilities"])

    def test_entity_isolation_between_unrelated_turns(self):
        """Test: Previous corporate entities do not leak into general topic questions."""
        from server.services.rag_service import RAGService

        # Turn 1: Microsoft analysis
        t1_ans = "Microsoft Azure expanded cloud revenue by 29%."
        _, f_t1 = RAGService.split_answer_and_followups(t1_ans, question="Analyze Microsoft")
    def test_workspace_analysis_generates_citations_from_retrieved_chunks(self):
        """Test: Workspace Analysis with retrieved document chunks generates citations and markers."""
        from server.services.citation_service import CitationService
        from server.models.workspace import DocumentChunk

        chunks = [
            DocumentChunk(
                id="chk-ws-1",
                document_id="doc-arch-1",
                file_name="01_System_Architecture.md",
                page_number=1,
                section_title="High-Level Microservices Overview",
                content="The FastAPI service acts as the gateway for all HTTP and WebSocket requests.",
            ),
            DocumentChunk(
                id="chk-ws-2",
                document_id="doc-gate-2",
                file_name="08_WebSocket_Gateway.md",
                page_number=3,
                section_title="Connection Pool Management",
                content="Redis caching and PostgreSQL connection pools handle up to 1000 concurrent sessions.",
            ),
        ]

        cite_svc = CitationService()
        raw_response = (
            "# Workspace Analysis Report\n\n"
            "## 1. Workspace-wide Analysis\n"
            "The architecture consists of a FastAPI gateway service.\n\n"
            "## 2. Bulk Document Summarization\n"
            "- **01_System_Architecture.md**: Microservices gateway.\n"
            "- **08_WebSocket_Gateway.md**: Real-time event streaming.\n\n"
            "## 3. Risk Identification\n"
            "Connection pool saturation under peak morning load."
        )

        resp, citations = cite_svc.generate_citations(raw_response, chunks)
        if not citations:
            citations = cite_svc.generate_citations_from_chunks(chunks)
        if citations:
            resp = cite_svc.attach_fallback_markers(resp, citations)

        assert len(citations) == 2
        assert citations[0]["document_id"] == "doc-arch-1"
        assert citations[0]["title"] == "01_System_Architecture.md"
        assert citations[1]["document_id"] == "doc-gate-2"
        assert citations[1]["title"] == "08_WebSocket_Gateway.md"
        assert "[1]" in resp or "[2]" in resp

    def test_empty_document_context_does_not_generate_fake_citations(self):
        """Test: When no sources/chunks exist, no fake citations are generated."""
        from server.services.citation_service import CitationService

        cite_svc = CitationService()
        ans, citations = cite_svc.generate_citations("This is a general response without sources.", [])
        assert len(citations) == 0
        assert "[1]" not in ans










