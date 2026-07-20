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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

    return _ocr_image_obj(image)


def _ocr_image_obj(image) -> str:
    """Dispatch an image (raw bytes, PIL image, or RGB ndarray) to the engine.

    Separated from :func:`_ocr_image_bytes` so in-memory renders (PDF pages) skip
    the encode-to-PNG -> decode-from-PNG round trip entirely. Raw bytes (e.g.
    DOCX image blobs) are decoded here, inside the calling (worker) thread, so
    decoding is parallelised along with the OCR itself.
    """
    if isinstance(image, (bytes, bytearray)):
        return _ocr_image_bytes(bytes(image))

    if not settings.OCR_ENABLED:
        return ""

    engine = settings.OCR_ENGINE or "tesseract"
    logger.debug("OCR: using engine '%s'", engine)
    if engine == "rapidocr":
        return _ocr_with_rapidocr(image)
    if engine == "easyocr":
        return _ocr_with_easyocr(image)
    # Tesseract's API wants a PIL image; convert only if given an array.
    if not hasattr(image, "convert"):
        from PIL import Image

        image = Image.fromarray(image)
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


# Per-thread OCR engines. Each worker thread owns its RapidOCR instance (ONNX
# sessions are only a few tens of MB), which avoids any shared-state races in
# the RapidOCR wrapper. The pool is persistent (module lifetime), so each
# worker loads its engine exactly once per process — not once per document.
_ocr_thread_state = threading.local()
_engine_patch_lock = threading.Lock()


def _build_rapidocr(intra_threads=None):
    """Construct a tuned RapidOCR engine.

    Two session-level tunings are injected through the ``SessionOptions``
    factory the library's ``utils`` module calls during construction (swapped
    under a lock, then restored):

    * ``intra_op_num_threads`` cap for pool workers (workers x intra ~= cores).
      Without it every session threads across ALL cores, so N parallel
      inferences spawn N x cores threads and thrash the CPU — measured *slower*
      than sequential.
    * keep the CPU **memory arena** enabled. The library hard-disables it
      (``enable_cpu_mem_arena = False``), forcing fresh allocations on every
      inference; pooled buffers are pure throughput with zero effect on
      results. Implemented as a property override that ignores the library's
      ``False`` assignment. Toggle with ``OCR_MEM_ARENA=false`` to trade the
      speed back for lower memory.

    After construction, the recognition/classification batch sizes are raised
    (``OCR_REC_BATCH``/``OCR_CLS_BATCH``): text-dense pages produce dozens of
    line crops, and larger batches mean fewer ONNX calls for the same crops.
    """
    from rapidocr_onnxruntime import RapidOCR

    keep_arena = getattr(settings, "OCR_MEM_ARENA", True)
    engine = None
    if intra_threads or keep_arena:
        try:
            import onnxruntime as ort
            from rapidocr_onnxruntime import utils as rapid_utils

            base_property = type(ort.SessionOptions()).enable_cpu_mem_arena

            class _TunedSessionOptions(ort.SessionOptions):
                if keep_arena:
                    @property
                    def enable_cpu_mem_arena(self):
                        return base_property.__get__(self)

                    @enable_cpu_mem_arena.setter
                    def enable_cpu_mem_arena(self, value):
                        base_property.__set__(self, True)  # ignore disable

            def _tuned_session_options():
                options = _TunedSessionOptions()
                if intra_threads:
                    options.intra_op_num_threads = int(intra_threads)
                return options

            with _engine_patch_lock:
                original = rapid_utils.SessionOptions
                rapid_utils.SessionOptions = _tuned_session_options
                try:
                    engine = RapidOCR()
                finally:
                    rapid_utils.SessionOptions = original
        except Exception:
            logger.warning(
                "OCR: session tuning unavailable; using library defaults",
                exc_info=True,
            )
    if engine is None:
        engine = RapidOCR()

    # Batch-size tuning (per-thread engine, so plain attribute writes are safe).
    try:
        rec_batch = int(getattr(settings, "OCR_REC_BATCH", 0) or 0)
        if rec_batch > 0 and hasattr(engine, "text_recognizer"):
            engine.text_recognizer.rec_batch_num = rec_batch
        cls_batch = int(getattr(settings, "OCR_CLS_BATCH", 0) or 0)
        if cls_batch > 0 and getattr(engine, "text_cls", None) is not None:
            engine.text_cls.cls_batch_num = cls_batch
    except Exception:
        logger.debug("OCR: batch tuning skipped", exc_info=True)
    return engine


def _get_rapidocr():
    """Return this thread's RapidOCR engine, creating it on first use.

    Pool worker threads (stamped by the pool initializer) get intra-op-capped
    sessions; the main thread keeps onnxruntime's default thread count so
    single-image latency is unchanged.
    """
    engine = getattr(_ocr_thread_state, "rapidocr", None)
    if engine is None:
        engine = _build_rapidocr(
            getattr(_ocr_thread_state, "intra_op_threads", None)
        )
        _ocr_thread_state.rapidocr = engine
    return engine


def warm_up_ocr() -> None:
    """Pre-load the OCR engines so the first document doesn't pay model-load
    latency (one engine per pool worker + the main thread's).

    Intended to be called from app startup in a background thread; entirely
    best-effort — any failure just means the first document loads lazily as
    before.
    """
    try:
        if not settings.OCR_ENABLED or settings.OCR_ENGINE != "rapidocr":
            return
        pool = _get_ocr_pool()
        size = pool._max_workers
        barrier = threading.Barrier(size, timeout=60)

        def _load_engine():
            _get_rapidocr()
            try:
                barrier.wait()  # hold the thread so every worker builds its own
            except Exception:
                pass

        for future in [pool.submit(_load_engine) for _ in range(size)]:
            future.result(timeout=120)
        _get_rapidocr()  # main-thread engine
        logger.info("OCR: %d worker engine(s) + main engine warmed up", size)
    except Exception:
        logger.debug("OCR warm-up skipped", exc_info=True)


def _auto_pool_size() -> int:
    """Auto worker count: min(4, cores/2), bounded by available memory.

    Each worker thread owns a RapidOCR engine, so on memory-constrained hosts
    spawning the full CPU-based worker count can OOM the process. Best-effort
    read of Linux MemAvailable caps the pool; the per-worker budget is larger
    when the ONNX memory arena is enabled (pooled buffers raise steady-state
    RSS per session — measured OOM at 4 arena workers in ~2GB free). On other
    platforms (or any error) the CPU-based size is used unchanged.
    """
    cpu = os.cpu_count() or 4
    size = max(1, min(4, cpu // 2 or 1))
    per_worker_mb = 700 if getattr(settings, "OCR_MEM_ARENA", True) else 400
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    available_mb = int(line.split()[1]) // 1024
                    size = max(1, min(size, available_mb // per_worker_mb))
                    break
    except Exception:
        pass
    return size


def _ocr_worker_count(task_count: int) -> int:
    """Concurrency for a batch of OCR tasks.

    OCR is CPU-bound, but every heavy step (onnxruntime inference, OpenCV, PIL
    decode) releases the GIL, so a *thread* pool gives real parallelism without
    the pickling/startup cost of processes. Capped at 4 by default: each
    worker's inference is itself multi-threaded (intra-op), so workers x intra
    is sized to ~= the core count.
    """
    if task_count <= 1:
        return 1
    configured = getattr(settings, "OCR_MAX_WORKERS", 0)
    if configured and configured > 0:
        return max(1, min(configured, task_count))
    return min(task_count, _auto_pool_size())


@lru_cache(maxsize=1)
def _get_ocr_pool() -> ThreadPoolExecutor:
    """Persistent OCR worker pool (created once, reused for every document).

    The initializer stamps each worker thread with its intra-op thread budget
    *before* the thread ever builds an engine, so workers x intra ~= cpu cores.
    A persistent pool also means worker engines are loaded once per process
    instead of once per uploaded document.
    """
    cpu = os.cpu_count() or 4
    configured = getattr(settings, "OCR_MAX_WORKERS", 0)
    size = configured if configured > 0 else _auto_pool_size()
    intra = max(1, cpu // size)

    def _init_worker():
        _ocr_thread_state.intra_op_threads = intra

    return ThreadPoolExecutor(
        max_workers=size, thread_name_prefix="ocr", initializer=_init_worker
    )


def _run_parallel_ocr(items: list[tuple]) -> dict:
    """OCR ``[(key, image/array/bytes), ...]`` concurrently; return ``{key: text}``.

    Results are keyed (not ordered) so callers reassemble in their own document
    order regardless of completion order. Each task is best-effort: a failure
    yields ``""`` for that key and never aborts the batch.
    """
    if not items:
        return {}

    if _ocr_worker_count(len(items)) == 1:
        return {key: _ocr_image_obj(payload) for key, payload in items}

    pool = _get_ocr_pool()
    futures = {key: pool.submit(_ocr_image_obj, payload) for key, payload in items}
    results: dict = {}
    for key, future in futures.items():
        try:
            results[key] = future.result()
        except Exception:
            logger.exception("OCR: parallel task failed for %s; skipped", key)
            results[key] = ""
    return results


def _preprocess_for_ocr(image):
    """OpenCV preprocessing to improve OCR accuracy before RapidOCR.

    Pipeline: read image -> grayscale -> optional upscale for small images ->
    Gaussian blur (denoise) -> threshold (Otsu for high-contrast/bimodal inputs,
    adaptive Gaussian for uneven lighting). Returns a numpy image for RapidOCR.

    Best-effort: if OpenCV isn't installed, preprocessing is disabled, or anything
    fails, it falls back to the plain RGB array so OCR still runs.
    """
    import numpy as np

    # Accept a PIL image OR a ready RGB ndarray (PDF pages arrive as raw arrays
    # straight from the renderer — no PNG encode/decode round-trip).
    if hasattr(image, "convert"):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image
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

        std = float(gray.std())
        if std > 60.0:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )

        logger.debug(
            "OCR preprocess: %dx%d -> %s (threshold=%s)",
            width,
            height,
            binary.shape,
            "otsu" if std > 60.0 else "adaptive",
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
        started = time.perf_counter()
        processed = _preprocess_for_ocr(image)
        preprocess_ms = (time.perf_counter() - started) * 1000

        infer_started = time.perf_counter()
        result, elapse = engine(processed)
        infer_ms = (time.perf_counter() - infer_started) * 1000

        # RapidOCR reports per-stage seconds [det, cls, rec] — logged so real
        # documents show which stage dominates (guides further tuning).
        det_ms = cls_ms = rec_ms = 0.0
        if isinstance(elapse, (list, tuple)) and len(elapse) == 3:
            det_ms, cls_ms, rec_ms = (float(v) * 1000 for v in elapse)

        text = (
            "\n".join(line[1] for line in result).strip() if result else ""
        )
        logger.info(
            "OCR: rapidocr preprocess=%.0fms infer=%.0fms "
            "(det=%.0f cls=%.0f rec=%.0f) lines=%d chars=%d [%s]",
            preprocess_ms,
            infer_ms,
            det_ms,
            cls_ms,
            rec_ms,
            len(result) if result else 0,
            len(text),
            threading.current_thread().name,
        )
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

        array = (
            np.array(image.convert("RGB")) if hasattr(image, "convert") else image
        )
        lines = reader.readtext(array, detail=0)
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

    # Phase 1: walk the document once, recording reading order as a sequence of
    # text/image entries (no OCR yet).
    entries: list[tuple[str, str]] = []  # ("text", paragraph) | ("image", rid)
    ocr_seen: set[str] = set()
    for para in doc.paragraphs:
        if para.text.strip():
            entries.append(("text", para.text))
        for rid in _docx_paragraph_image_rids(para):
            if rid in rid_to_blob and rid not in ocr_seen:
                ocr_seen.add(rid)
                entries.append(("image", rid))
    for rid in rid_to_blob:  # images not anchored to a body paragraph
        if rid not in ocr_seen:
            ocr_seen.add(rid)
            entries.append(("image", rid))

    # Phase 2: OCR every embedded image concurrently (keyed by rid), then
    # Phase 3: assemble in the recorded order — output identical to sequential.
    started = time.perf_counter()
    ocr_texts = _run_parallel_ocr(
        [(rid, rid_to_blob[rid]) for kind, rid in entries if kind == "image"]
    )
    parts = [
        value if kind == "text" else ocr_texts.get(value, "")
        for kind, value in entries
    ]
    merged = "\n".join(part for part in parts if part)

    logger.info(
        "DOCX: %d embedded image(s) OCR'd in %.0fms (workers=%d); merged=%d chars",
        len(ocr_texts),
        (time.perf_counter() - started) * 1000,
        _ocr_worker_count(len(ocr_texts)),
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


def _pixmap_to_rgb_array(pix):
    """Convert a PyMuPDF pixmap to an RGB ndarray **in memory**.

    Replaces the old ``pix.tobytes("png")`` -> ``PIL.Image.open`` round trip,
    which zlib-compressed every rendered page to PNG only to immediately decode
    it back to pixels (~0.3-0.8s and ~2x memory per page, pure waste).
    """
    import numpy as np

    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n >= 4:
        array = array[:, :, :3]
    elif pix.n == 1:
        array = np.repeat(array, 3, axis=2)
    return array.copy()


def _extract_pdf_with_fitz(fitz, content: bytes) -> str:
    """Hybrid per-page extraction: native text layer merged with OCR of the
    rendered page.

    Optimised two-phase pipeline:

    * **Phase 1 (sequential, cheap):** read each page's text layer and, only for
      pages that need OCR (no text layer, or embedded images), render the page
      straight to an in-memory RGB array. Sequential because PyMuPDF documents
      are not thread-safe; rendering is ~100ms/page vs ~10s/page for OCR.
      Pages with selectable text and no images are never rendered or OCR'd.
    * **Phase 2 (parallel):** OCR all rendered pages concurrently (thread pool;
      onnxruntime/OpenCV/PIL release the GIL). Results are keyed by page index.
    * **Phase 3:** merge text + OCR per page and join in page order, so the
      output is byte-identical in structure to the sequential version.
    """
    total_started = time.perf_counter()
    pages: list[tuple[int, str]] = []  # (page_index, native_text)
    ocr_jobs: list[tuple[int, object]] = []  # (page_index, rgb_array)

    with fitz.open(stream=content, filetype="pdf") as doc:
        if getattr(doc, "needs_pass", False):
            raise _PasswordProtected("Password-protected PDF is not supported.")

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = (page.get_text() or "").strip()

            try:
                has_images = bool(page.get_images(full=True))
            except Exception:
                has_images = False

            pages.append((page_index, text))

            # OCR only image-bearing or scanned pages — and skip rendering
            # altogether when OCR is disabled (output is unchanged either way).
            if (not text or has_images) and settings.OCR_ENABLED:
                try:
                    pix = page.get_pixmap(dpi=settings.OCR_PDF_RENDER_DPI)
                    ocr_jobs.append((page_index, _pixmap_to_rgb_array(pix)))
                    pix = None  # release pixmap buffer promptly
                except Exception:
                    logger.exception(
                        "PDF: render failed on page %d; continuing", page_index
                    )

    render_ms = (time.perf_counter() - total_started) * 1000

    ocr_started = time.perf_counter()
    ocr_results = _run_parallel_ocr(ocr_jobs)
    ocr_ms = (time.perf_counter() - ocr_started) * 1000

    merge_started = time.perf_counter()
    parts: list[str] = []
    for page_index, text in pages:
        ocr_text = ocr_results.get(page_index, "")
        page_text = text
        if ocr_text:
            page_text = _merge_texts(text, ocr_text) if text else ocr_text
        if page_text.strip():
            parts.append(page_text)
    merged = "\n".join(parts)
    merge_ms = (time.perf_counter() - merge_started) * 1000

    logger.info(
        "PDF extract: pages=%d ocr_pages=%d workers=%d | load+render=%.0fms "
        "ocr=%.0fms merge=%.0fms total=%.0fms",
        len(pages),
        len(ocr_jobs),
        _ocr_worker_count(len(ocr_jobs)),
        render_ms,
        ocr_ms,
        merge_ms,
        (time.perf_counter() - total_started) * 1000,
    )
    return merged


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

    _CACHE_MAX_ENTRIES = 32
    _extract_cache: "dict[tuple, str]" = {}
    _extract_cache_lock = threading.Lock()

    @staticmethod
    def _cache_key(ext: str, content: bytes) -> tuple:
        import hashlib

        return (
            hashlib.sha256(content).hexdigest(),
            ext,
            settings.OCR_ENABLED,
            settings.OCR_ENGINE,
            settings.OCR_PREPROCESS,
            settings.OCR_PDF_RENDER_DPI,
            settings.OCR_REC_BATCH,
        )

    def extract_text(self, filename: str, content: bytes) -> str:
        """Extract and return plain text from ``content`` for ``filename``."""
        from server.services.audit_logger import audit_context
        ext = self.detect_extension(filename)

        with audit_context(
            action="OCR_TEXT_EXTRACTION",
            metadata={
                "filename": filename,
                "extension": ext,
                "cache_hit": False,
            }
        ) as meta:
            key = self._cache_key(ext, content)
            with self._extract_cache_lock:
                cached = self._extract_cache.get(key)
            if cached is not None:
                logger.info(
                    "Extraction cache HIT for %s (%d chars) — OCR skipped",
                    filename,
                    len(cached),
                )
                meta["character_count"] = len(cached)
                meta["cache_hit"] = True
                return cached

            _, extractor = self._EXTRACTORS[ext]
            try:
                text = extractor(content)
                meta["character_count"] = len(text)
            except _PasswordProtected as exc:
                raise ValueError(str(exc)) from exc

            with self._extract_cache_lock:
                if len(self._extract_cache) >= self._CACHE_MAX_ENTRIES:
                    self._extract_cache.pop(next(iter(self._extract_cache)))
                self._extract_cache[key] = text

            logger.debug("Extracted %d chars from %s (%s)", len(text), filename, ext)
            return text


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    """Return the shared :class:`OCRService` singleton."""
    return OCRService()
