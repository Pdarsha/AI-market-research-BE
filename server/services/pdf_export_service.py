"""Server-side conversation PDF export — professional CapitalEdge AI branded reports.

Renders a chat transcript to a multi-page PDF via PyMuPDF's Story engine.
Every page receives:
  • High-resolution company logo (top-left) + centered company name in a consistent header.
  • Semi-transparent 45° watermark behind content.
  • Timestamp + page-numbering footer.

All branding values are read from ``server.config.Settings`` so they can be
overridden via environment variables without touching source code.
"""

import html
import logging
import os
import re
from datetime import datetime, timezone

import fitz  # PyMuPDF
import markdown
from PIL import Image

from server.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Logo loading & caching
# ─────────────────────────────────────────────────────────────────────────────


def _load_logo_pixmap(logo_path: str, dpi: int = 300) -> fitz.Pixmap | None:
    """Return a maximum-quality fitz.Pixmap for the company logo.

    Supports both SVG and PNG/JPG logo assets:
    - If SVG: renders at high-resolution matrix scale (300+ DPI).
    - If PNG: loads full RGBA pixel dimensions via PIL -> fitz.Pixmap.
    - If logo fails to load: logs a warning and returns None (PDF export continues).
    """
    if not logo_path or not isinstance(logo_path, str):
        return None

    paths_to_try = [logo_path]
    if logo_path.lower().endswith(".svg"):
        paths_to_try.append(logo_path[:-4] + ".png")
        paths_to_try.append(logo_path[:-4] + ".jpg")
    elif logo_path.lower().endswith(".png") or logo_path.lower().endswith(".jpg"):
        paths_to_try.append(logo_path[:-4] + ".svg")

    for path in paths_to_try:
        if not os.path.exists(path):
            continue
        try:
            if path.lower().endswith(".svg"):
                doc = fitz.open(path)
                pix = doc[0].get_pixmap(
                    matrix=fitz.Matrix(25, 25), alpha=True, colorspace=fitz.csRGB
                )
                doc.close()
                return pix
            else:
                img = Image.open(path).convert("RGBA")
                data = img.tobytes("raw", "RGBA")
                return fitz.Pixmap(fitz.csRGB, img.width, img.height, data, True)
        except Exception as exc:
            logger.warning("Failed to load PDF logo from %s: %s", path, exc)

    logger.warning("PDF logo file not found or failed to load: %s", logo_path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _normalize_url(raw_url: str) -> str | None:
    """Return a valid http(s) URL string or None."""
    if not raw_url or not isinstance(raw_url, str):
        return None
    url = raw_url.strip()
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("www.") or any(
        ext in url.lower()
        for ext in [".com", ".org", ".net", ".gov", ".edu", ".io", ".ai", ".co", ".uk"]
    ):
        return "https://" + url.lstrip("/")
    return None


def _extract_citation_url(cit: dict) -> str | None:
    """Extract and normalise a source URL from any citation schema variant."""
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
    if not raw_url and isinstance(cit.get("source"), str):
        if cit["source"].startswith(("http://", "https://", "www.")):
            raw_url = cit["source"]
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
    if not raw_url and cit.get("document_id"):
        raw_url = f"https://app.capitaledge.ai/v1/documents/{cit['document_id']}/viewer"
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
    if not raw_url:
        url_match = re.search(r"https?://[^\s<'\"]+", str(cit))
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
.user-query { font-size: 13pt; font-weight: bold; color: #2b6cb0; margin-top: 20pt; margin-bottom: 10pt; }
.references-section { margin-top: 30pt; border-top: 1px solid #e2e8f0; padding-top: 15pt; }
.ref-title { font-size: 14pt; font-weight: bold; margin-bottom: 10pt; }
.ref-item { font-size: 9pt; margin-bottom: 6pt; }
a { color: #3182ce; text-decoration: none; }
"""


def conversation_to_pdf(title: str, messages: list[dict]) -> bytes:
    """Render the transcript to a professional branded PDF and return the bytes.

    Each assistant message owns its own citation map so that [1] in message-1
    always links to message-1's source, and [1] in message-2 links to
    message-2's source — completely independently.
    """
    from server.services.audit_logger import audit_context

    with audit_context(
        action="PDF_EXPORT",
        metadata={"title": title, "messages_count": len(messages), "pdf_size_bytes": 0},
    ) as meta:

        exported = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Branding Configuration (driven by settings / env vars) ─────────
        company_name: str = getattr(settings, "PDF_COMPANY_NAME", "CapitalEdge AI")
        logo_path: str = getattr(settings, "PDF_LOGO_PATH", "")
        watermark_text: str = getattr(settings, "PDF_WATERMARK_TEXT", "CapitalEdge AI")
        header_height: float = float(getattr(settings, "PDF_HEADER_HEIGHT", 60.0))  # pt
        footer_height: float = float(getattr(settings, "PDF_FOOTER_HEIGHT", 35.0))  # pt
        watermark_opacity: float = float(
            getattr(settings, "PDF_WATERMARK_OPACITY", 0.08)
        )
        logo_height_pt: float = float(getattr(settings, "PDF_LOGO_HEIGHT", 22.0))  # pt
        logo_margin: float = float(
            getattr(settings, "PDF_LOGO_MARGIN", 40.0)
        )  # pt left margin
        logo_dpi: int = int(getattr(settings, "PDF_LOGO_DPI", 300))
        pdf_margin: float = float(getattr(settings, "PDF_MARGIN", 40.0))  # pt margin

        # Load logo once per PDF export session (None on failure -> graceful fallback)
        logo_pixmap: fitz.Pixmap | None = _load_logo_pixmap(logo_path, dpi=logo_dpi)

        # ── Build per-message HTML + citation maps ───────────────────────────
        body_parts: list[str] = []
        annotation_plan: list[tuple[str, str]] = []  # (marker_text, url)

        for m in messages:
            if m.get("error") or m.get("pending"):
                continue

            role = m.get("role", "assistant")
            content = m.get("content") or ""

            if role == "user":
                body_parts.append(f'<div class="user-query">{_esc(content)}</div>')
                continue

            # Build this message's citation index
            msg_citations: list[dict] = m.get("citations") or []
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

            # Replace [N] markers with <a href="…">
            def _replace(match, _url_map=msg_url_map):
                n = match.group(1)
                url = _url_map.get(n)
                if url:
                    return f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">[{n}]</a>'
                return f"[{n}]"

            linked_content = re.sub(r"\[(\d+)\]", _replace, content)
            html_content = markdown.markdown(
                linked_content, extensions=["tables", "fenced_code", "nl2br"]
            )
            body_parts.append(f"<div>{html_content}</div>")

            # Per-message Sources section
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
                    ref_html = (
                        f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">{_esc(label)}</a>'
                        if url
                        else _esc(label)
                    )
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

            # Accumulate annotation plan in render order
            for idx in sorted(msg_url_map, key=lambda x: int(x) if x.isdigit() else x):
                annotation_plan.append((f"[{idx}]", msg_url_map[idx]))

        # ── Assemble HTML document ────────────────────────────────────────────
        doc_html = f"""
        <html>
        <head><style>{_CSS}</style></head>
        <body>
            <h1>{_esc(title)}</h1>
            <div class="meta">Generated: {exported}</div>
            {''.join(body_parts)}
        </body>
        </html>
        """

        # ── Render Story ──────────────────────────────────────────────────────
        story = fitz.Story(html=doc_html, user_css=_CSS)
        mediabox = fitz.paper_rect("a4")  # 595.32 × 841.89 pt

        # Reserve vertical space for header and footer
        top_offset = max(header_height + 12.0, 72.0)
        bottom_offset = footer_height + 10.0
        content_box = mediabox + (pdf_margin, top_offset, -pdf_margin, -bottom_offset)

        def rectfn(rect_num, filled):
            return mediabox, content_box, None

        doc = story.write_with_links(rectfn=rectfn)
        total_pages = len(doc)

        # Shared layout dimensions and fonts
        helv_font = fitz.Font("helv")
        pw = mediabox.width
        ph = mediabox.height
        header_yc = header_height / 2.0  # vertical centre of header band
        divider_y = header_height - 2.0  # line just below header band
        footer_div_y = ph - footer_height  # top of footer band
        footer_text_y = ph - footer_height + 14.0  # text baseline inside footer
        footer_color = (0.44, 0.50, 0.59)  # #718096 slate

        # ── Pass 1: Header, Watermark & Footer injection on every page ───────
        for page_num in range(total_pages):
            page = doc[page_num]

            # 1a. Watermark — behind content, 45° rotated, low opacity
            if watermark_text:
                wm_size = 70
                wm_w = helv_font.text_length(watermark_text, fontsize=wm_size)
                wm_pt = fitz.Point(pw / 2 - wm_w / 2, ph / 2 + 20)
                page.insert_text(
                    wm_pt,
                    watermark_text,
                    fontsize=wm_size,
                    fontname="helv",
                    color=(0.70, 0.75, 0.82),
                    morph=(fitz.Point(pw / 2, ph / 2), fitz.Matrix(-45)),
                    fill_opacity=watermark_opacity,
                )

            # 1b. Header — Top-left Logo + Centered Company Name
            if logo_pixmap:
                aspect = logo_pixmap.width / max(logo_pixmap.height, 1)
                logo_w = logo_height_pt * aspect
                logo_rect = fitz.Rect(
                    logo_margin,
                    header_yc - logo_height_pt / 2,
                    logo_margin + logo_w,
                    header_yc + logo_height_pt / 2,
                )
                page.insert_image(logo_rect, pixmap=logo_pixmap)

            if company_name:
                cn_size = 15.0
                cn_w = helv_font.text_length(company_name, fontsize=cn_size)
                page.insert_text(
                    fitz.Point((pw - cn_w) / 2, header_yc + 5),
                    company_name,
                    fontsize=cn_size,
                    fontname="helv",
                    color=(0.17, 0.42, 0.69),  # #2b6cb0 — CapitalEdge brand blue
                )

            # Thin header divider line
            page.draw_line(
                fitz.Point(pdf_margin, divider_y),
                fitz.Point(pw - pdf_margin, divider_y),
                color=(0.88, 0.91, 0.94),
                width=0.75,
            )

            # 1c. Footer — thin divider + timestamp (left) + page numbers (right)
            page.draw_line(
                fitz.Point(pdf_margin, footer_div_y),
                fitz.Point(pw - pdf_margin, footer_div_y),
                color=(0.88, 0.91, 0.94),
                width=0.75,
            )
            page.insert_text(
                fitz.Point(pdf_margin, footer_text_y),
                f"Generated: {exported}",
                fontsize=8.5,
                fontname="helv",
                color=footer_color,
            )
            page_str = f"Page {page_num + 1} of {total_pages}"
            p_w = helv_font.text_length(page_str, fontsize=8.5)
            page.insert_text(
                fitz.Point(pw - pdf_margin - p_w, footer_text_y),
                page_str,
                fontsize=8.5,
                fontname="helv",
                color=footer_color,
            )

            # Normalise existing URI links emitted by the Story engine
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

        # ── Pass 2: Bounding-box annotation injection for citation markers ────
        already_annotated: set[tuple[int, int, int, int]] = set()

        for marker_text, url in annotation_plan:
            placed = False
            for page_num in range(total_pages):
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
                    if key in already_annotated:
                        continue
                    # Remove any overlapping link before inserting a clean one
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
                    page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": url})
                    already_annotated.add(key)
                    placed = True
                    break

        # ── Audit metadata ────────────────────────────────────────────────────
        total_links = uri_links = 0
        for page in doc:
            page_links = page.get_links()
            total_links += len(page_links)
            uri_links += sum(1 for l in page_links if l.get("kind") == fitz.LINK_URI)

        final_bytes = doc.write()
        meta["pdf_size_bytes"] = len(final_bytes)
        meta["has_hyperlinks"] = uri_links > 0
        meta["uri_links_count"] = uri_links
        meta["total_links_count"] = total_links

        return final_bytes
