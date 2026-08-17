"""Markdown-safe text utilities.

Dependency-free on purpose: imported by both ``utils.helpers`` and
``services.citation_service`` without dragging in CRUD or the DB engine.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?])")
_RUN_OF_SPACES_RE = re.compile(r"[ \t]{2,}")


def tidy_markdown_spacing(text: str) -> str:
    """Clean whitespace left by removed/renumbered citation markers, WITHOUT
    damaging the answer's Markdown.

    The naive version of this — ``re.sub(r"[ \\t]{2,}", " ", text)`` across the
    whole string — collapses every run of spaces, which silently destroys
    Markdown structure: nested bullets flatten to a single level, ordered-list
    continuation lines lose their indent, indented code blocks stop being code,
    and the two-space hard line break disappears. The answer then renders as one
    wall of raw text.

    So clean each line's *interior* only:
      * leading indentation is preserved (list nesting, code indentation);
      * trailing whitespace is preserved (Markdown hard line breaks);
      * content inside fenced code blocks is never touched.
    """
    if not text:
        return ""

    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        lead = re.match(r"[ \t]*", line).group(0)
        rest = line[len(lead) :]
        trail = re.search(r"[ \t]*$", rest).group(0)
        body = rest[: len(rest) - len(trail)] if trail else rest

        body = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", body)
        body = _RUN_OF_SPACES_RE.sub(" ", body)
        out.append(lead + body + trail)

    # Trim only blank lines at the very start/end; interior blank lines separate
    # Markdown blocks and must survive.
    return "\n".join(out).strip("\n")


def render_rich_text(text: str | None) -> str:
    """Convert text with Markdown or rich-text syntax to clean, renderable HTML.

    Supports:
    - Bold (**text**)
    - Italic (*text*)
    - Underline (<u>text</u>)
    - Strikethrough (~~text~~ or ~text~)
    - Headings (# Heading)
    - Ordered/Unordered Lists (1., -, *)
    - Blockquotes (> text)
    - Code blocks & inline code (`code` / ```code```)
    - Links ([text](url))
    """
    if text is None:
        return ""
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        import markdown
        # Pre-process strikethrough syntax before markdown conversion
        processed = re.sub(r"~~(.*?)~~", r"<del>\1</del>", text)
        processed = re.sub(r"(?<!~)~(?!~)(.*?)(?<!~)~(?!~)", r"<del>\1</del>", processed)
        
        html = markdown.markdown(
            processed,
            extensions=["fenced_code", "tables", "nl2br", "sane_lists", "extra"]
        )
        return html
    except Exception:
        return text
