"""OCR / text-extraction service.

Single responsibility: turn raw file bytes into plain text. It knows nothing
about chunking, embeddings, or storage. Supported inputs:

* Documents: ``.txt``, ``.docx``, ``.doc``, ``.pdf``, ``.csv``, ``.xlsx``, ``.xls``
* Images: ``.png``, ``.jpg``, ``.jpeg``, ``.tiff``, ``.tif``, ``.bmp``, ``.webp``

Image OCR runs an OpenCV preprocessing pipeline (grayscale, denoise, threshold,
optional upscale) before the configured engine when RapidOCR is selected.

Extraction is **hybrid**: for documents it reads the native text layer *and* runs
OCR on any embedded images, merging both in reading order. Scanned PDF pages that
have no text layer are rasterised and OCR'd automatically. OCR is best-effort — a
failure on one image never aborts the rest of the document.

OCR engine is pluggable and cross-platform via ``settings.OCR_ENGINE``:
    "tesseract" (default) -> pytesseract + a Tesseract binary (install per OS)
    "rapidocr"            -> rapidocr-onnxruntime (pure pip, no system binary)
    "easyocr"             -> easyocr (pure pip, Torch-based)

Optional dependencies (lazy-imported, gracefully degraded if absent):
    Pillow                                       -> decode images (required for OCR)
    the chosen OCR engine (see above)            -> image OCR
    PyMuPDF (``fitz``)                           -> PDF embedded-image extraction
                                                    and scanned-page rasterisation
"""

from __future__ import annotations

import io
import logging
import os
import platform
import re
import shutil
import tempfile
from functools import lru_cache
from io import BytesIO, StringIO
from typing import Callable

import openpyxl  # noqa: F401  (ensures the xlsx engine is importable)
import pandas as pd
import textract
from docx import Document as DocxDocument
from pypdf import PdfReader

from server.config import settings

logger = logging.getLogger(__name__)

# Extensions handled by OCR (image) rather than a document parser.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# OCR primitives (optional dependencies, lazily imported)                     #
# --------------------------------------------------------------------------- #

def _tesseract_install_hint() -> str:
    """Return the correct Tesseract install command for the current OS."""
    system = platform.system()
    if system == "Darwin":
        return "macOS: 'brew install tesseract'"
    if system == "Windows":
        return (
            "Windows: install from "
            "https://github.com/UB-Mannheim/tesseract/wiki (or 'choco install "
            "tesseract')"
        )
    # Linux — pick the hint matching the available package manager.
    if shutil.which("apt-get"):
        return "Debian/Ubuntu: 'sudo apt-get install -y tesseract-ocr'"
    if shutil.which("dnf"):
        return "Fedora/RHEL: 'sudo dnf install -y tesseract'"
    if shutil.which("pacman"):
        return "Arch: 'sudo pacman -S tesseract'"
    if shutil.which("zypper"):
        return "openSUSE: 'sudo zypper install -y tesseract-ocr'"
    return "install the Tesseract OCR engine for your OS"


def _ocr_image_bytes(content: bytes) -> str:
    """Best-effort OCR of raw image bytes; returns ``""`` on any problem.

    Dispatches to the configured engine (``settings.OCR_ENGINE``):
      * ``tesseract`` — needs the system binary (cross-OS, installed per OS);
      * ``rapidocr`` / ``easyocr`` — pure-pip, no system binary, identical on every OS.

    Used by standalone image uploads and embedded/scanned document images, so a
    single blank/corrupt/undecodable image is skipped rather than failing the
    whole ingestion (blank images naturally yield an empty string).
    """
    if not settings.OCR_ENABLED:
        return ""

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("OCR unavailable: install 'Pillow' to read images.")
        return ""
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        logger.warning("OCR: could not decode image (corrupted/unsupported); skipped")
        return ""

    engine = settings.OCR_ENGINE or "tesseract"
    logging.info("OCR: using engine '%s' for image bytes", engine)
    if engine == "rapidocr":
        return _ocr_with_rapidocr(image)
    if engine == "easyocr":
        return _ocr_with_easyocr(image)
    return _ocr_with_tesseract(image)


def _ocr_with_tesseract(image) -> str:
    try:
        import pytesseract
    except ImportError:
        logger.warning(
            "OCR_ENGINE=tesseract but 'pytesseract' is not installed "
            "('pip install pytesseract')."
        )
        return ""
    try:
        return (pytesseract.image_to_string(image) or "").strip()
    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "OCR skipped: the Tesseract engine binary is not installed or not on "
            "PATH. %s. Alternatively use a pure-pip engine that needs no system "
            "install: set OCR_ENGINE=rapidocr and 'pip install "
            "rapidocr-onnxruntime' (works on all OS).",
            _tesseract_install_hint(),
        )
        return ""
    except Exception:
        logger.exception("OCR: Tesseract failed on an image; skipped")
        return ""


@lru_cache(maxsize=1)
def _get_rapidocr():
    """Load the RapidOCR engine once (ONNX-based, cross-platform, pip-only)."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _preprocess_for_ocr(image):
    """OpenCV preprocessing to improve OCR accuracy before RapidOCR.

    Pipeline: read image -> grayscale -> optional upscale for small images ->
    Gaussian blur (denoise) -> threshold (Otsu for high-contrast/bimodal inputs,
    adaptive Gaussian for uneven lighting). Returns a numpy image for RapidOCR.

    Best-effort: if OpenCV isn't installed, preprocessing is disabled, or anything
    fails, it falls back to the plain RGB array so OCR still runs.
    """
    import numpy as np

    rgb = np.array(image.convert("RGB"))
    if not settings.OCR_PREPROCESS:
        return rgb

    try:
        import cv2
    except ImportError:
        logger.warning(
            "OCR preprocessing skipped: OpenCV not installed "
            "('pip install opencv-python-headless')."
        )
        return rgb

    try:
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        min_side = min(height, width)
        if 0 < min_side < 1000:
            scale = min(3.0, 1000.0 / min_side)
            gray = cv2.resize(
                gray,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if float(gray.std()) > 60.0:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )

        logger.info(
            "OCR preprocess: %dx%d -> %s (threshold=%s)",
            width,
            height,
            binary.shape,
            "otsu" if float(gray.std()) > 60.0 else "adaptive",
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    except Exception:
        logger.exception("OCR preprocessing failed; using original image")
        return rgb


def _ocr_with_rapidocr(image) -> str:
    try:
        engine = _get_rapidocr()
    except ImportError:
        logger.warning(
            "OCR_ENGINE=rapidocr but 'rapidocr-onnxruntime' is not installed "
            "('pip install rapidocr-onnxruntime')."
        )
        return ""
    try:
        processed = _preprocess_for_ocr(image)
        logger.info("OCR: running RapidOCR on preprocessed image")
        result, _ = engine(processed)
        if not result:
            logger.info("OCR: RapidOCR found no text")
            return ""
        text = "\n".join(line[1] for line in result).strip()
        logger.info("OCR: RapidOCR extracted %d chars", len(text))
        return text
    except Exception:
        logger.exception("OCR: rapidocr failed on an image; skipped")
        return ""


@lru_cache(maxsize=1)
def _get_easyocr():
    """Load the EasyOCR reader once (Torch-based, cross-platform, pip-only)."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


def _ocr_with_easyocr(image) -> str:
    try:
        reader = _get_easyocr()
    except ImportError:
        logger.warning(
            "OCR_ENGINE=easyocr but 'easyocr' is not installed "
            "('pip install easyocr')."
        )
        return ""
    try:
        import numpy as np

        lines = reader.readtext(np.array(image.convert("RGB")), detail=0)
        return "\n".join(lines).strip()
    except Exception:
        logger.exception("OCR: easyocr failed on an image; skipped")
        return ""


def _try_import_fitz():
    """Return the PyMuPDF module (``fitz``) or ``None`` if not installed."""
    try:
        import fitz

        return fitz
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# Plain-text document extractors                                              #
# --------------------------------------------------------------------------- #

def _extract_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def _docx_paragraph_image_rids(paragraph) -> list[str]:
    """Relationship ids of images embedded in a DOCX paragraph, in order."""
    try:
        from docx.oxml.ns import qn

        rids = []
        for blip in paragraph._p.findall(".//" + qn("a:blip")):
            rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
            if rid:
                rids.append(rid)
        return rids
    except Exception:
        return []


def _extract_docx(content: bytes) -> str:
    """DOCX native text + OCR of embedded images, merged in reading order.

    Each image's OCR text is inserted immediately after the paragraph that
    contains it (not appended at the end). Images not anchored to a body
    paragraph (e.g. inside tables/headers) are OCR'd in a final pass so none are
    dropped. OCR is best-effort: one failed image never aborts the document.
    """
    doc = DocxDocument(io.BytesIO(content))

    rid_to_blob: dict[str, bytes] = {}
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.reltype:
            try:
                rid_to_blob[rel_id] = rel.target_part.blob
            except Exception:
                continue

    parts: list[str] = []
    ocr_done: set[str] = set()
    images_detected = 0

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
        for rid in _docx_paragraph_image_rids(para):
            blob = rid_to_blob.get(rid)
            if blob is None or rid in ocr_done:
                continue
            ocr_done.add(rid)
            images_detected += 1
            text = _ocr_image_bytes(blob)
            if text:
                parts.append(text)

    for rid, blob in rid_to_blob.items():
        if rid in ocr_done:
            continue
        images_detected += 1
        text = _ocr_image_bytes(blob)
        if text:
            parts.append(text)

    merged = "\n".join(parts)
    logger.info(
        "DOCX: %d embedded image(s) detected; merged text length=%d chars",
        images_detected,
        len(merged),
    )
    return merged


def _clean_doc_text(text: str) -> str:
    """Normalise legacy-Word control marks into plain text."""
    text = text.replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
    text = text.replace("\x07", "\t")
    return re.sub(r"[\x00-\x08\x0e-\x1f]", "", text)


def _extract_doc_via_ole(content: bytes) -> str:
    """Pure-Python legacy ``.doc`` extraction via the OLE piece table.

    No external binary — works on any OS where ``olefile`` is installed. Parses
    the FIB in the ``WordDocument`` stream to locate the piece table (Clx) in the
    Table stream, then reads each piece (CP1252 or UTF-16LE) to reconstruct the
    main document text.
    """
    import struct

    import olefile

    bio = BytesIO(content)
    if not olefile.isOleFile(bio):
        raise ValueError("Not an OLE2 .doc file")

    ole = olefile.OleFileIO(bio)
    try:
        if not ole.exists("WordDocument"):
            raise ValueError("No WordDocument stream")
        wd = ole.openstream("WordDocument").read()
        if len(wd) < 0x01A6 + 4:
            raise ValueError("WordDocument stream too small")

        flags = struct.unpack_from("<H", wd, 0x0A)[0]
        table_name = "1Table" if (flags & 0x0200) else "0Table"
        if not ole.exists(table_name):
            table_name = "0Table" if table_name == "1Table" else "1Table"
        if not ole.exists(table_name):
            raise ValueError("No table stream")
        table = ole.openstream(table_name).read()

        fc_clx = struct.unpack_from("<I", wd, 0x01A2)[0]
        lcb_clx = struct.unpack_from("<I", wd, 0x01A6)[0]
        clx = table[fc_clx:fc_clx + lcb_clx]

        # Find the Pcdt (clxt == 0x02), skipping any leading Prc blocks (0x01).
        i, pcdt = 0, None
        while i < len(clx):
            if clx[i] == 0x01:
                cb = struct.unpack_from("<H", clx, i + 1)[0]
                i += 3 + cb
            elif clx[i] == 0x02:
                lcb = struct.unpack_from("<I", clx, i + 1)[0]
                pcdt = clx[i + 5:i + 5 + lcb]
                break
            else:
                break
        if not pcdt:
            raise ValueError("No piece table (Pcdt) found")

        piece_count = (len(pcdt) - 4) // 12
        cps = [struct.unpack_from("<I", pcdt, 4 * j)[0] for j in range(piece_count + 1)]
        pcd_base = 4 * (piece_count + 1)

        parts = []
        for j in range(piece_count):
            fc = struct.unpack_from("<I", pcdt, pcd_base + 8 * j + 2)[0]
            compressed = bool(fc & 0x40000000)
            fc &= 0x3FFFFFFF
            n_chars = cps[j + 1] - cps[j]
            if n_chars <= 0:
                continue
            if compressed:
                start = fc // 2
                parts.append(wd[start:start + n_chars].decode("cp1252", errors="ignore"))
            else:
                parts.append(wd[fc:fc + 2 * n_chars].decode("utf-16-le", errors="ignore"))
        text = "".join(parts)
    finally:
        ole.close()

    return _clean_doc_text(text)


def _extract_doc_via_soffice(content: bytes) -> str:
    """Convert ``.doc`` to text with LibreOffice headless, if it is installed."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("LibreOffice (soffice) is not installed")

    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.doc")
        with open(src, "wb") as fh:
            fh.write(content)
        subprocess.run(
            [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, src],
            check=True,
            capture_output=True,
            timeout=120,
        )
        out = os.path.join(tmp, "input.txt")
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
    raise RuntimeError("LibreOffice conversion produced no output")


def _extract_doc(content: bytes) -> str:
    """Legacy ``.doc`` extraction — cross-platform, no *required* system binary.

    Tries, in order: pure-Python OLE piece-table parsing (``olefile``, works on
    every OS), then LibreOffice headless conversion (if installed), then
    ``textract``/antiword (if installed). Raises a clear error only if all fail.
    """
    try:
        text = _extract_doc_via_ole(content)
        if text and text.strip():
            logger.info("DOC: extracted %d chars via olefile (pure-python)", len(text))
            return text
        logger.info("DOC: olefile returned empty text; trying fallbacks")
    except Exception:
        logger.warning("DOC: olefile extraction failed; trying fallbacks", exc_info=True)

    try:
        text = _extract_doc_via_soffice(content)
        if text and text.strip():
            logger.info("DOC: extracted %d chars via LibreOffice", len(text))
            return text
    except Exception:
        logger.warning("DOC: LibreOffice conversion unavailable/failed", exc_info=True)

    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=True) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            return textract.process(temp_file.name).decode("utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(
            "Could not extract text from the .doc file. Convert it to .docx, or "
            "install LibreOffice (soffice) or antiword on the server."
        ) from exc


def _extract_pdf(content: bytes) -> str:
    """Hybrid PDF extraction.

    Prefers PyMuPDF (native text + embedded images + scanned-page rasterisation);
    falls back to pypdf text-only when PyMuPDF isn't installed.
    """
    fitz = _try_import_fitz()
    if fitz is not None:
        try:
            return _extract_pdf_with_fitz(fitz, content)
        except _PasswordProtected:
            raise
        except Exception:
            logger.exception("PDF: PyMuPDF extraction failed; falling back to pypdf")
    return _extract_pdf_text_only(content)


class _PasswordProtected(RuntimeError):
    """Raised for encrypted PDFs so the caller can report a clear error."""


def _merge_texts(primary: str, secondary: str) -> str:
    """Append lines from ``secondary`` not already present in ``primary``.

    Keeps the primary (native text-layer) reading order first, then adds only the
    genuinely new lines OCR found in embedded images / rendered pages — so body
    text isn't duplicated while image-only text is preserved.
    """
    if not secondary.strip():
        return primary
    if not primary.strip():
        return secondary
    existing = {ln.strip().lower() for ln in primary.splitlines() if ln.strip()}
    new_lines = [
        ln for ln in secondary.splitlines()
        if ln.strip() and ln.strip().lower() not in existing
    ]
    if not new_lines:
        return primary
    return primary + "\n" + "\n".join(new_lines)


def _extract_pdf_with_fitz(fitz, content: bytes) -> str:
    """Hybrid per-page extraction: native text layer merged with OCR of the
    rendered page.

    For each page we take the selectable text, then rasterise the page and OCR it
    (this captures embedded images, charts, and scanned pages — anything that is
    visually text but not in the text layer). The two are merged per page,
    preserving reading order. Pages that have a text layer and no images are not
    rendered, to avoid needless OCR work.
    """
    parts: list[str] = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        if getattr(doc, "needs_pass", False):
            raise _PasswordProtected("Password-protected PDF is not supported.")

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = (page.get_text() or "").strip()

            # OCR the rendered page when it has no text layer (scanned) or carries
            # embedded images/graphics whose text the layer won't include.
            try:
                has_images = bool(page.get_images(full=True))
            except Exception:
                has_images = False

            page_text = text
            if not text or has_images:
                try:
                    pix = page.get_pixmap(dpi=settings.OCR_PDF_RENDER_DPI)
                    ocr_text = _ocr_image_bytes(pix.tobytes("png"))
                except Exception:
                    logger.exception(
                        "PDF: render/OCR failed on page %d; continuing", page_index
                    )
                    ocr_text = ""
                if ocr_text:
                    page_text = _merge_texts(text, ocr_text) if text else ocr_text

            if page_text.strip():
                parts.append(page_text)

    return "\n".join(parts)


def _extract_pdf_text_only(content: bytes) -> str:
    """Fallback: read only the PDF text layer (no OCR)."""
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        # pypdf can sometimes open with an empty password; try, else report.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise _PasswordProtected(
                "Password-protected PDF is not supported."
            ) from exc
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
    """Standalone image OCR. Empty result -> caller reports 'no extractable text'."""
    return _ocr_image_bytes(content)


class OCRService:
    """Extract plain text from uploaded file bytes based on file extension."""

    _EXTRACTORS: dict[str, tuple[str, Callable[[bytes], str]]] = {
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
        ".webp": ("webp", _extract_image),
    }

    @property
    def supported_extensions(self) -> tuple[str, ...]:
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
        try:
            text = extractor(content)
        except _PasswordProtected as exc:
            raise ValueError(str(exc)) from exc
        logger.debug("Extracted %d chars from %s (%s)", len(text), filename, ext)
        return text


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    """Return the shared :class:`OCRService` singleton."""
    return OCRService()
