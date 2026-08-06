"""Server-side conversation PDF export.

Renders a chat transcript to PDF with PyMuPDF's Story engine (already a
project dependency — no headless browser or extra PDF library needed).
The layout mirrors a professional report template: a title header,
rendered markdown including tables and citations, and a footer with page numbers.
"""

import html
import io
import re
import markdown
from datetime import datetime, timezone

import fitz  # PyMuPDF


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _normalize_url(raw_url: str) -> str | None:
    """Normalize URLs for PDF annotation. Return valid HTTP/HTTPS URL or None."""
    if not raw_url or not isinstance(raw_url, str):
        return None
    url = raw_url.strip()
    if not url:
        return None

    if url.startswith("http://") or url.startswith("https://"):
        return url

    if url.startswith("www.") or any(
        ext in url.lower()
        for ext in [
            ".com",
            ".org",
            ".net",
            ".gov",
            ".edu",
            ".io",
            ".ai",
            ".co",
            ".uk",
        ]
    ):
        return "https://" + url.lstrip("/")

    return None


def _extract_citation_url(cit: dict) -> str | None:
    """Extract and normalize external source URL from citation object across all schema representations."""
    meta_dict = cit.get("metadata") if isinstance(cit.get("metadata"), dict) else {}
    raw_url = (
        cit.get("url")
        or cit.get("link")
        or cit.get("uri")
        or cit.get("source_url")
        or cit.get("web_url")
        or cit.get("href")
        or cit.get("target_url")
        or meta_dict.get("url")
        or meta_dict.get("link")
        or meta_dict.get("uri")
        or meta_dict.get("source_url")
        or meta_dict.get("web_url")
    )

    # Check if cit.get('source') is a URL string
    if not raw_url and isinstance(cit.get("source"), str):
        if cit["source"].startswith(("http://", "https://", "www.")):
            raw_url = cit["source"]

    # Check navigation_target / document_url
    if not raw_url:
        target = (
            cit.get("navigation_target")
            or cit.get("document_url")
            or meta_dict.get("navigation_target")
        )
        if target and isinstance(target, str):
            if target.startswith(("http://", "https://", "www.")):
                raw_url = target
            elif target.startswith("/"):
                raw_url = f"https://app.capitaledge.ai{target}"

    # Fallback for uploaded document IDs
    if not raw_url and cit.get("document_id"):
        raw_url = f"https://app.capitaledge.ai/v1/documents/{cit['document_id']}/viewer"

    # Fallback to search query for title-only citations to guarantee clickable link
    if not raw_url:
        title = (
            cit.get("title")
            or meta_dict.get("title")
            or cit.get("document_name")
            or cit.get("source")
        )
        if (
            title
            and isinstance(title, str)
            and title.strip()
            and title.strip() != "Untitled source"
        ):
            from urllib.parse import quote

            raw_url = f"https://www.google.com/search?q={quote(title.strip())}"

    # Regex extraction fallback if URL string is embedded in dict representation
    if not raw_url:
        cit_str = str(cit)
        url_match = re.search(r"https?://[^\s<'\"]+", cit_str)
        if url_match:
            raw_url = url_match.group(0)

    return _normalize_url(raw_url)


_CSS = """
body { font-family: 'Helvetica', sans-serif; font-size: 10.5pt; line-height: 1.6; color: #2d3748; }
h1 { font-size: 20pt; color: #1a202c; margin-bottom: 8pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 4pt; }
h2 { font-size: 16pt; color: #2d3748; margin-top: 16pt; margin-bottom: 8pt; }
h3 { font-size: 13pt; color: #4a5568; margin-top: 12pt; margin-bottom: 6pt; }
p { margin-bottom: 10pt; }
ul, ol { margin-bottom: 10pt; padding-left: 20pt; }
li { margin-bottom: 4pt; }
table { width: 100%; border-collapse: collapse; margin-bottom: 12pt; }
th, td { border: 1px solid #cbd5e0; padding: 6pt; text-align: left; }
th { background-color: #edf2f7; font-weight: bold; }
code { font-family: monospace; background-color: #edf2f7; padding: 2pt 4pt; border-radius: 3pt; font-size: 9pt; }
pre { background-color: #edf2f7; padding: 10pt; border-radius: 4pt; overflow-x: auto; margin-bottom: 10pt; }
pre code { padding: 0; background-color: transparent; }
blockquote { border-left: 3px solid #a0aec0; margin-left: 0; padding-left: 10pt; color: #718096; font-style: italic; }
.meta { color: #718096; font-size: 8.5pt; margin-bottom: 20pt; }
.logo { font-size: 16pt; font-weight: bold; color: #2b6cb0; margin-bottom: 10pt; }
.user-query { font-size: 13pt; font-weight: bold; color: #2b6cb0; margin-top: 20pt; margin-bottom: 10pt; }
.references-section { margin-top: 30pt; border-top: 1px solid #e2e8f0; padding-top: 15pt; }
.ref-title { font-size: 14pt; font-weight: bold; margin-bottom: 10pt; }
.ref-item { font-size: 9pt; margin-bottom: 6pt; }
a { color: #3182ce; text-decoration: none; }
"""


def conversation_to_pdf(title: str, messages: list[dict]) -> bytes:
    """Render the transcript to a professional PDF report and return the bytes.

    Each assistant message owns its own citation map so that [1] in message-1
    always links to message-1's source, and [1] in message-2 links to
    message-2's source — completely independently.
    """
    from server.services.audit_logger import audit_context

    with audit_context(
        action="PDF_EXPORT",
        metadata={
            "title": title,
            "messages_count": len(messages),
            "pdf_size_bytes": 0,
        },
    ) as meta:
        exported = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        body_parts: list[str] = []

        annotation_plan: list[tuple[str, str]] = []

        # Per-message sources section parts (appended right after the message body)
        for m in messages:
            if m.get("error") or m.get("pending"):
                continue

            role = m.get("role", "assistant")
            content = m.get("content") or ""

            if role == "user":
                body_parts.append(f'<div class="user-query">{_esc(content)}</div>')
                continue

            # ── Build THIS message's citation map ─────────────────────────────
            # key: marker number string ("1", "2", …)
            # value: resolved absolute URL (str) or None
            msg_citations: list[dict] = m.get("citations") or []

            # Build index: reference_index (or positional) -> citation dict
            msg_cit_map: dict[str, dict] = {}
            for pos, cit in enumerate(msg_citations, start=1):
                idx_raw = (
                    cit.get("reference_index")
                    or cit.get("id")
                    or cit.get("citation_id")
                )
                if idx_raw is not None:
                    num_match = re.search(r"\d+", str(idx_raw))
                    idx = num_match.group(0) if num_match else str(pos)
                else:
                    idx = str(pos)
                msg_cit_map[idx] = cit

            # Resolve URLs for this message
            msg_url_map: dict[str, str] = {}
            for idx, cit in msg_cit_map.items():
                url = _extract_citation_url(cit)
                if url:
                    msg_url_map[idx] = url

            # ── Replace [N] markers in content with <a href="…"> ─────────────
            def _replace(match, _url_map=msg_url_map):
                n = match.group(1)
                url = _url_map.get(n)
                if url:
                    return (
                        f'<a href="{_esc(url)}" target="_blank" '
                        f'rel="noopener noreferrer">[{n}]</a>'
                    )
                return f"[{n}]"

            linked_content = re.sub(r"\[(\d+)\]", _replace, content)
            html_content = markdown.markdown(
                linked_content, extensions=["tables", "fenced_code", "nl2br"]
            )
            body_parts.append(f"<div>{html_content}</div>")

            # ── Per-message Sources section ───────────────────────────────────
            if msg_cit_map:
                ref_items: list[str] = []
                for idx in sorted(
                    msg_cit_map, key=lambda x: int(x) if x.isdigit() else x
                ):
                    cit = msg_cit_map[idx]
                    meta_dict = (
                        cit.get("metadata")
                        if isinstance(cit.get("metadata"), dict)
                        else {}
                    )
                    label = (
                        cit.get("title")
                        or meta_dict.get("title")
                        or cit.get("source")
                        or cit.get("document_name")
                        or f"Source {idx}"
                    )
                    url = msg_url_map.get(idx)
                    if url:
                        ref_html = (
                            f'<a href="{_esc(url)}" target="_blank" '
                            f'rel="noopener noreferrer">{_esc(label)}</a>'
                        )
                    else:
                        ref_html = _esc(label)

                    ref_items.append(
                        f'<div class="ref-item" id="ref-{_esc(idx)}">'
                        f"<b>[{_esc(idx)}]</b> {ref_html}"
                        f"</div>"
                    )

                body_parts.append(
                    '<div class="references-section">'
                    '<div class="ref-title">Sources</div>'
                    + "".join(ref_items)
                    + "</div>"
                )

            # ── Record annotation plan (in render order) ──────────────────────
            # Each (marker, url) pair will be used in the bounding-box injection
            # pass. We record them in the order markers appear in the content so
            # page.search_for hits can be matched correctly later.
            for idx in sorted(msg_url_map, key=lambda x: int(x) if x.isdigit() else x):
                annotation_plan.append((f"[{idx}]", msg_url_map[idx]))

        # ── Assemble & render the PDF ─────────────────────────────────────────
        doc_html = f"""
        <html>
        <head>
            <style>{_CSS}</style>
        </head>
        <body>
            <div class="logo">CapitalEdge AI</div>
            <h1>{_esc(title)}</h1>
            <div class="meta">Generated: {exported}</div>
            {''.join(body_parts)}
        </body>
        </html>
        """

        story = fitz.Story(html=doc_html, user_css=_CSS)
        mediabox = fitz.paper_rect("a4")
        content_box = mediabox + (40, 44, -40, -44)

        def rectfn(rect_num, filled):
            return mediabox, content_box, None

        doc = story.write_with_links(rectfn=rectfn)

        # ── Pass 1: page numbers + normalise existing links ───────────────────
        for page_num in range(len(doc)):
            page = doc[page_num]
            page.insert_text(
                fitz.Point(mediabox.width - 60, mediabox.height - 25),
                f"Page {page_num + 1}",
                fontsize=9,
                color=(0.5, 0.5, 0.5),
            )
            for link in page.get_links():
                kind = link.get("kind")
                if kind == fitz.LINK_URI:
                    normalized = _normalize_url(link.get("uri", ""))
                    if normalized and normalized != link.get("uri"):
                        link["uri"] = normalized
                        page.update_link(link)
                elif kind in (fitz.LINK_GOTOR, fitz.LINK_LAUNCH, fitz.LINK_NAMED):
                    normalized = _normalize_url(link.get("file", "") or "")
                    if normalized:
                        link["kind"] = fitz.LINK_URI
                        link["uri"] = normalized
                        link["file"] = None
                        page.update_link(link)
                    else:
                        page.delete_link(link)

        # ── Pass 2: explicit bounding-box annotation injection ────────────────
        # For each (marker, url) pair in annotation_plan we search every page
        # for that marker text.  We track which rects have already been
        # annotated so that the FIRST match (= first message that uses this
        # marker number) wins and subsequent messages with the same marker
        # number get their own distinct hits.
        #
        # Because annotation_plan is ordered by message render order and
        # page.search_for returns rects top-to-bottom, the Nth occurrence of
        # "[1]" on the full document corresponds to message N's [1].
        # We use already_annotated (rounded rect coords) to consume each rect
        # exactly once in the correct annotation_plan order.

        already_annotated: set[tuple[int, int, int, int]] = set()

        for marker_text, url in annotation_plan:
            placed = False
            for page_num in range(len(doc)):
                if placed:
                    break
                page = doc[page_num]

                for rect in page.search_for(marker_text):
                    key = (
                        round(rect.x0),
                        round(rect.y0),
                        round(rect.x1),
                        round(rect.y1),
                    )

                    # Skip rects already claimed by a previous (marker, url) pair
                    if key in already_annotated:
                        continue

                    # Remove any existing link that overlaps this rect (avoids xref errors)
                    for el in list(page.get_links()):
                        el_rect = el.get("from")
                        if (
                            el_rect
                            and abs(el_rect.x0 - rect.x0) < 5
                            and abs(el_rect.y0 - rect.y0) < 5
                        ):
                            try:
                                page.delete_link(el)
                            except Exception:
                                pass

                    # Insert a fresh LINK_URI annotation on the exact bounding box
                    page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": url})

                    already_annotated.add(key)
                    placed = True
                    break  # move to next (marker_text, url) in annotation_plan

        # ── Audit metadata ────────────────────────────────────────────────────
        total_links_count = 0
        uri_links_count = 0
        for page in doc:
            page_links = page.get_links()
            total_links_count += len(page_links)
            uri_links_count += len(
                [l for l in page_links if l.get("kind") == fitz.LINK_URI]
            )

        final_bytes = doc.write()
        meta["pdf_size_bytes"] = len(final_bytes)
        meta["has_hyperlinks"] = uri_links_count > 0
        meta["uri_links_count"] = uri_links_count
        meta["total_links_count"] = total_links_count

        return final_bytes
