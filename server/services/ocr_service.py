"""OCR / text-extraction service.

Single responsibility: turn raw file bytes into plain text. It knows nothing
about chunking, embeddings, or storage. Supported inputs:

* Documents: ``.txt``, ``.docx``, ``.doc``, ``.pdf``, ``.csv``, ``.xlsx``, ``.xls``
* Images: ``.png``, ``.jpg``, ``.jpeg``, ``.tiff``, ``.bmp`` (via Tesseract OCR)

PDF extraction reads the embedded text layer; scanned/image-only PDFs will yield
little or no text (image-PDF OCR is intentionally out of scope here).
"""

from __future__ import annotations

import io
import logging
import tempfile
from functools import lru_cache
from io import BytesIO, StringIO
from typing import Callable, Dict, Tuple

import openpyxl  # noqa: F401  (ensures the xlsx engine is importable)
import pandas as pd
import textract
from docx import Document as DocxDocument
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Extensions handled by OCR (image) rather than a document parser.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")


def _extract_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def _extract_docx(content: bytes) -> str:
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs)


def _extract_doc(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=True) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        text = textract.process(temp_file.name)
    return text.decode("utf-8", errors="ignore")


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)
    return "\n".join(parts)


def _extract_csv(content: bytes) -> str:
    df = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")))
    return df.to_string(index=False)


def _extract_excel(content: bytes, engine: str) -> str:
    sheets = pd.read_excel(BytesIO(content), sheet_name=None, engine=engine)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"Sheet: {sheet_name}")
        parts.append(df.fillna("").to_string(index=False))
        parts.append("")
    return "\n".join(parts)


def _extract_xlsx(content: bytes) -> str:
    return _extract_excel(content, engine="openpyxl")


def _extract_xls(content: bytes) -> str:
    return _extract_excel(content, engine="xlrd")


def _extract_image(content: bytes) -> str:
    """Run OCR on an image. Requires ``pytesseract`` + a Tesseract binary."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Image OCR requires 'pytesseract' and 'Pillow' (and a Tesseract "
            "install). Install them to upload images."
        ) from exc

    image = Image.open(BytesIO(content))
    return pytesseract.image_to_string(image)


class OCRService:
    """Extract plain text from uploaded file bytes based on file extension."""

    # extension -> (stored file_format, extractor). Single source of truth for
    # which upload types are accepted across the whole pipeline.
    _EXTRACTORS: Dict[str, Tuple[str, Callable[[bytes], str]]] = {
        ".txt": ("txt", _extract_txt),
        ".docx": ("docx", _extract_docx),
        ".doc": ("doc", _extract_doc),
        ".pdf": ("pdf", _extract_pdf),
        ".csv": ("csv", _extract_csv),
        ".xlsx": ("xlsx", _extract_xlsx),
        ".xls": ("xls", _extract_xls),
        ".png": ("png", _extract_image),
        ".jpg": ("jpg", _extract_image),
        ".jpeg": ("jpeg", _extract_image),
        ".tiff": ("tiff", _extract_image),
        ".tif": ("tif", _extract_image),
        ".bmp": ("bmp", _extract_image),
    }

    @property
    def supported_extensions(self) -> Tuple[str, ...]:
        return tuple(self._EXTRACTORS)

    def detect_extension(self, filename: str) -> str:
        """Return the matching supported extension for ``filename``.

        Raises :class:`ValueError` if the file type is unsupported.
        """
        name = (filename or "").lower()
        ext = next((e for e in self._EXTRACTORS if name.endswith(e)), None)
        if ext is None:
            allowed = ", ".join(self._EXTRACTORS)
            raise ValueError(f"Unsupported file type. Allowed: {allowed}")
        return ext

    def file_format(self, filename: str) -> str:
        """Return the canonical stored ``file_format`` for ``filename``."""
        return self._EXTRACTORS[self.detect_extension(filename)][0]

    def extract_text(self, filename: str, content: bytes) -> str:
        """Extract and return plain text from ``content`` for ``filename``."""
        ext = self.detect_extension(filename)
        _, extractor = self._EXTRACTORS[ext]
        text = extractor(content)
        logger.debug("Extracted %d chars from %s (%s)", len(text), filename, ext)
        return text


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    """Return the shared :class:`OCRService` singleton."""
    return OCRService()