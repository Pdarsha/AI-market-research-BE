"""Final-answer source attribution helpers, independent of retrieval backends."""


def source_attribution_from_citations(citations: list[dict] | None) -> dict:
    """Group only sources whose citations survived into the final answer."""
    groups = {
        "uploaded_documents": [],
        "sec_edgar": [],
        "alpha_vantage": [],
        "roic_ai": [],
        "web": [],
        "other_mcp": [],
    }
    seen: dict[str, set] = {name: set() for name in groups}
    for citation in citations or []:
        source_type = citation.get("source_type") or (
            "document" if citation.get("document_id") else "web"
        )
        group = {
            "document": "uploaded_documents",
            "sec_edgar": "sec_edgar",
            "financial_news": "alpha_vantage",
            "roic_ai": "roic_ai",
            "web": "web",
        }.get(source_type, "other_mcp")
        key = citation.get("chunk_id") or citation.get("url") or citation.get("title")
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
