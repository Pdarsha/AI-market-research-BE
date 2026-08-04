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
    """Render the transcript to a professional PDF report and return the bytes."""
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
        
        body_parts = []
        all_citations = {}
        
        # First pass: collect all valid citations across messages
        global_cit_counter = 1
        for m in messages:
            if m.get("error") or m.get("pending"): continue
            for cit in m.get("citations") or []:
                idx = str(cit.get("reference_index") or cit.get("id") or cit.get("citation_id") or "")
                
                if not idx:
                    idx = str(global_cit_counter)
                else:
                    # If idx is a string like "citation_1" or "cit-1", extract the number to match "[1]"
                    num_match = re.search(r'\d+', idx)
                    if num_match:
                        idx = num_match.group(0)
                
                all_citations[str(idx)] = cit
                global_cit_counter += 1

        # Second pass: render messages
        linked_internal_refs = set()
        for m in messages:
            if m.get("error") or m.get("pending"): continue
            role = m.get("role", "assistant")
            content = m.get("content") or ""
            
            if role == "user":
                body_parts.append(f'<div class="user-query">{_esc(content)}</div>')
            else:
                def replace_citation(match):
                    n = match.group(1)
                    cit = all_citations.get(str(n))
                    if cit:
                        meta = cit.get("metadata") or {}
                        url = (
                            cit.get("url") or 
                            cit.get("link") or 
                            cit.get("navigation_target") or 
                            cit.get("document_url") or
                            meta.get("url") or
                            meta.get("link")
                        )
                        
                        if url and (url.startswith("http://") or url.startswith("https://")):
                            return f'<a href="{url}" target="_blank" rel="noopener noreferrer">[{n}]</a>'
                            
                        # If no valid external URL, fallback to internal anchor
                        linked_internal_refs.add(str(n))
                        return f'<a href="#ref-{n}">[{n}]</a>'
                    
                    return f'[{n}]'
                
                content = re.sub(r'\[(\d+)\]', replace_citation, content)
                html_content = markdown.markdown(content, extensions=['tables', 'fenced_code', 'nl2br'])
                body_parts.append(f'<div>{html_content}</div>')
                        
        ref_parts = []
        # Combine all-referenced and actually-linked citations
        refs_to_render = sorted(set(list(all_citations.keys()) + list(linked_internal_refs)), key=lambda x: int(x) if str(x).isdigit() else x)
        
        if refs_to_render:
            ref_parts.append('<div class="references-section">')
            ref_parts.append('<div class="ref-title">References</div>')
            for idx in refs_to_render:
                if idx in all_citations:
                    cit = all_citations[idx]
                    meta = cit.get("metadata") or {}
                    label = cit.get("title") or meta.get("title") or cit.get("source") or cit.get("document_name") or f"Source {idx}"
                    url = (
                        cit.get("url") or 
                        cit.get("link") or 
                        cit.get("navigation_target") or 
                        cit.get("document_url") or
                        meta.get("url") or
                        meta.get("link")
                    )
                    
                    link_html = ''
                    if url and (url.startswith("http://") or url.startswith("https://")):
                        link_html = f' &mdash; <a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">{_esc(url)}</a>'
                        
                    ref_parts.append(f'<div class="ref-item" id="ref-{_esc(idx)}"><b>[{_esc(idx)}]</b> {_esc(label)}{link_html}</div>')
                else:
                    ref_parts.append(f'<div id="ref-{_esc(idx)}"></div>')
            ref_parts.append('</div>')
            
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
            {''.join(ref_parts)}
        </body>
        </html>
        """

        story = fitz.Story(html=doc_html, user_css=_CSS)
        mediabox = fitz.paper_rect("a4")
        content_box = mediabox + (40, 44, -40, -44)
        
        def rectfn(rect_num, filled):
            return mediabox, content_box, None
            
        doc = story.write_with_links(rectfn=rectfn)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = f"Page {page_num + 1}"
            page.insert_text(fitz.Point(mediabox.width - 60, mediabox.height - 25), text, fontsize=9, color=(0.5, 0.5, 0.5))
            
        final_bytes = doc.write()
        meta["pdf_size_bytes"] = len(final_bytes)
        
        return final_bytes
