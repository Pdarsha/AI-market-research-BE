"""Server-side conversation PDF export.

Renders a chat transcript to PDF with PyMuPDF's Story engine (already a
project dependency — no headless browser or extra PDF library needed).
The layout mirrors the frontend's previous print stylesheet: a title header,
one bordered block per message, attachment chips and a sources list.
"""

import html
import io
import re
from datetime import datetime, timezone

import fitz  # PyMuPDF


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _inline_markup(escaped: str) -> str:
    """Re-introduce a conservative subset of Markdown on escaped text."""
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    out = re.sub(r"(^|[^*])\*(?!\s)(.+?)\*", r"\1<i>\2</i>", out)
    out = re.sub(r"`([^`]+?)`", r"<code>\1</code>", out)
    return out


_BULLET_RE = re.compile(r"^\s*[-*]\s+")


def _content_html(text: str) -> str:
    """Markdown-ish message content -> safe HTML blocks (escape-first)."""
    blocks = re.split(r"\n{2,}", str(text or "").strip())
    rendered = []
    for block in blocks:
        if not block.strip():
            continue
        lines = block.split("\n")
        if all(_BULLET_RE.match(line) for line in lines):
            items = "".join(
                f"<li>{_inline_markup(_esc(_BULLET_RE.sub('', line)))}</li>"
                for line in lines
            )
            rendered.append(f"<ul>{items}</ul>")
            continue
        safe = _inline_markup(_esc(block)).replace("\n", "<br/>")
        rendered.append(f"<p>{safe}</p>")
    return "\n".join(rendered) or "<p></p>"


def _message_html(message: dict) -> str:
    role = message.get("role") or "assistant"
    who = "You" if role == "user" else "CapitalEdge"
    cls = "user" if role == "user" else "assistant"

    attachments = message.get("attachments") or []
    files = ""
    if attachments:
        chips = " ".join(
            f'<span class="file">[file] {_esc(a.get("name") or "attachment")}</span>'
            for a in attachments
        )
        files = f'<p class="files">{chips}</p>'

    citations = message.get("citations") or []
    sources = ""
    if citations:
        items = []
        for i, c in enumerate(citations):
            label = (
                c.get("title")
                or c.get("source")
                or c.get("document_name")
                or f"Source {i + 1}"
            )
            url = c.get("url")
            items.append(f"<li>{_esc(label)}{f' — {_esc(url)}' if url else ''}</li>")
        sources = (
            '<p class="sources-title">Sources</p>'
            f'<ol class="sources">{"".join(items)}</ol>'
        )

    return (
        f'<div class="msg {cls}">'
        f'<p class="who">{who.upper()}</p>'
        f"{files}"
        f"{_content_html(message.get('content'))}"
        f"{sources}"
        "</div>"
    )


_CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; }
p.meta { color: #888888; font-size: 8pt; margin: 0 0 14pt 0; }
div.msg { margin: 0 0 10pt 0; padding: 9pt 11pt; background-color: #ffffff; border: 0.7pt solid #e0e0e0; }
div.msg.user { background-color: #f5f6f8; }
p.who { font-weight: bold; font-size: 7.5pt; color: #555555; letter-spacing: 0.5pt; margin: 0 0 4pt 0; }
p { margin: 0 0 6pt 0; }
ul, ol { margin: 0 0 6pt 0; }
code { font-family: monospace; font-size: 9pt; background-color: #eeeeee; }
p.files { font-size: 8.5pt; color: #444444; }
span.file { border: 0.5pt solid #d8d8d8; background-color: #f0f1f4; }
p.sources-title { font-weight: bold; font-size: 8pt; color: #555555; margin: 6pt 0 2pt 0; }
ol.sources { font-size: 8.5pt; color: #555555; }
"""


def conversation_to_pdf(title: str, messages: list[dict]) -> bytes:
    """Render the transcript and return the PDF file bytes."""
    from server.services.audit_logger import audit_context

    with audit_context(
        action="PDF_EXPORT",
        metadata={
            "title": title,
            "messages_count": len(messages),
            "pdf_size_bytes": 0,
        }
    ) as meta:
        exported = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body = "\n".join(
            _message_html(m)
            for m in messages
            if not m.get("pending") and not m.get("error")
        )
        doc_html = (
            "<body>"
            f"<h1>{_esc(title)}</h1>"
            f'<p class="meta">Exported {exported} · CapitalEdge</p>'
            f"{body or '<p>No messages.</p>'}"
            "</body>"
        )

        story = fitz.Story(html=doc_html, user_css=_CSS)
        buffer = io.BytesIO()
        writer = fitz.DocumentWriter(buffer)
        mediabox = fitz.paper_rect("a4")
        content_box = mediabox + (40, 44, -40, -44)
        more = 1
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(content_box)
            story.draw(device)
            writer.end_page()
        writer.close()
        pdf_bytes = buffer.getvalue()
        meta["pdf_size_bytes"] = len(pdf_bytes)
        return pdf_bytes
