"""Final-answer source attribution helpers, independent of retrieval backends."""


def source_attribution_from_citations(citations: list[dict] | None) -> dict:
    """Group only sources whose citations survived into the final answer."""
    groups = {
        "uploaded_documents": [],
        "sec_edgar": [],
        "alpha_vantage": [],
        "roic_ai": [],
        "web": [],
        "companies_house": [],
        "other_mcp": [],
    }
    seen: dict[str, set] = {name: set() for name in groups}
    for citation in citations or []:
        source_type = (citation.get("source_type") or citation.get("type") or "").lower()
        if not source_type and citation.get("document_id"):
            source_type = "document"
        elif not source_type:
            source_type = "web" if citation.get("url") else "other_mcp"

        if source_type in ("document", "uploaded_document", "workspace_documents", "rag") or citation.get("document_id"):
            group = "uploaded_documents"
        elif source_type in ("sec_edgar", "sec"):
            group = "sec_edgar"
        elif source_type in ("financial_news", "alpha_vantage", "news"):
            group = "alpha_vantage"
        elif source_type in ("roic_ai", "roic", "roic.ai"):
            group = "roic_ai"
        elif source_type in ("web", "web_search", "google", "bing", "external"):
            group = "web"
        elif source_type in ("companies_house", "companies_house_uk"):
            group = "companies_house"
        else:
            group = "other_mcp"

        key = str(citation.get("chunk_id") or citation.get("url") or citation.get("title") or citation.get("id"))
        if key not in seen[group]:
            groups[group].append(citation)
            seen[group].add(key)
    return groups


def cited_document_names(citations: list[dict] | None) -> list[str]:
    """Return document names represented by final citations, order-stably."""
    names: list[str] = []
    for citation in source_attribution_from_citations(citations)["uploaded_documents"]:
        name = citation.get("document_name") or citation.get("title")
        if name and name not in names:
            names.append(name)
    return names
