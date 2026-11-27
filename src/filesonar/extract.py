"""Turn files into plain-text segments that can be indexed.

Each extractor returns an :class:`Extraction`: a list of segments (text plus the page it
came from and how it was obtained) and a short metadata summary. Extractors never write
to disk and never execute anything embedded in a file.
"""

from __future__ import annotations

import errno
import io
import logging
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pymupdf
import pytesseract
from docx import Document
from PIL import ExifTags, Image, ImageFile, ImageOps, UnidentifiedImageError

from .config import IndexConfig
from .models import Captioner, ModelError

log = logging.getLogger(__name__)

# Refuse images that would decode to more than ~90 megapixels (decompression bombs).
Image.MAX_IMAGE_PIXELS = 90_000_000
# Index the readable part of an incomplete download, as image viewers display it.
ImageFile.LOAD_TRUNCATED_IMAGES = True

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".org",
    ".csv",
    ".tsv",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".xml",
    ".html",
    ".htm",
    ".tex",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS

# Below this many characters of real text on a PDF page we try OCR instead.
_MIN_PAGE_TEXT = 25
_OCR_DPI = 200
_CAPTION_MAX_SIDE = 1024
_OCR_MAX_SIDE = 2000
# Tesseract can spend minutes on a noisy photo that holds no text at all.
_OCR_TIMEOUT = 15
_OCR_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")


@dataclass
class Segment:
    text: str
    page: int | None = None
    source: str = "text"  # text | ocr | caption | metadata


@dataclass
class Extraction:
    kind: str
    segments: list[Segment] = field(default_factory=list)
    details: str | None = None
    captioned: bool = False


def kind_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in DOCX_EXTENSIONS:
        return "document"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _clean(text: str) -> str:
    # Drop NULs and control characters that confuse tokenizers, keep newlines.
    text = text.replace("\x00", "")
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _ocr(image: Image.Image, cfg: IndexConfig) -> str:
    # Text stays legible at this size; full-resolution camera photos only cost time.
    if max(image.size) > _OCR_MAX_SIDE:
        image = image.copy()
        image.thumbnail((_OCR_MAX_SIDE, _OCR_MAX_SIDE))
    try:
        return _clean(
            pytesseract.image_to_string(image, lang=cfg.ocr_languages, timeout=_OCR_TIMEOUT)
        )
    except RuntimeError as exc:  # pytesseract signals its timeout this way
        log.info("OCR gave up after %ds: %s", _OCR_TIMEOUT, exc)
        return ""
    except pytesseract.TesseractError as exc:
        log.warning("OCR failed: %s", exc)
        return ""


# -- plain text ------------------------------------------------------------


def extract_text(path: Path, data: bytes, cfg: IndexConfig) -> Extraction:
    raw = data
    if b"\x00" in raw[:8192]:
        return Extraction(kind="text")  # binary file with a text-like extension
    # utf-8-sig drops a byte-order mark (Windows editors, Google Docs exports).
    text = raw.decode("utf-8-sig", errors="replace")
    return Extraction(kind="text", segments=[Segment(_clean(text))])


# -- docx ------------------------------------------------------------------


def extract_docx(path: Path, data: bytes, cfg: IndexConfig) -> Extraction:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    props = doc.core_properties
    details = _join_details(
        title=props.title,
        author=props.author,
        created=props.created.date().isoformat() if props.created else None,
    )
    segments = [Segment(_clean("\n".join(parts)))]
    if details:
        segments.append(Segment(details, source="metadata"))
    return Extraction(kind="document", segments=segments, details=details)


# -- pdf -------------------------------------------------------------------


def extract_pdf(path: Path, data: bytes, cfg: IndexConfig) -> Extraction:
    segments: list[Segment] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            return Extraction(kind="pdf", details="password protected")
        meta = doc.metadata or {}
        details = _join_details(
            title=meta.get("title"),
            author=meta.get("author"),
            pages=str(doc.page_count),
        )
        can_ocr = cfg.ocr and tesseract_available()
        # No point reading (or OCRing) more than the indexer will keep.
        budget = cfg.max_chunks_per_file * max(cfg.chunk_chars - cfg.chunk_overlap, 1)
        collected = 0
        for number, page in enumerate(doc, start=1):
            if number > cfg.max_pdf_pages or collected >= budget:
                log.info("%s: stopped after %d pages", path, number - 1)
                break
            text = _clean(page.get_text("text"))
            if len(text) >= _MIN_PAGE_TEXT:
                segments.append(Segment(text, page=number))
                collected += len(text)
            elif can_ocr:
                pix = page.get_pixmap(dpi=_OCR_DPI)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = _ocr(image, cfg)
                if ocr_text:
                    segments.append(Segment(ocr_text, page=number, source="ocr"))
                    collected += len(ocr_text)
    if details:
        segments.append(Segment(details, source="metadata"))
    return Extraction(kind="pdf", segments=segments, details=details)


# -- images ----------------------------------------------------------------

_EXIF_DATE = ExifTags.Base.DateTimeOriginal
_EXIF_MAKE = ExifTags.Base.Make
_EXIF_MODEL = ExifTags.Base.Model


def _exif_summary(image: Image.Image) -> str | None:
    try:
        exif = image.getexif()
        sub = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:  # malformed EXIF is common and never fatal
        return None
    taken = sub.get(_EXIF_DATE) or exif.get(ExifTags.Base.DateTime)
    when = None
    if isinstance(taken, str):
        try:
            when = datetime.strptime(taken.strip(), "%Y:%m:%d %H:%M:%S").date().isoformat()
        except ValueError:
            when = None
    camera = " ".join(
        str(v).strip("\x00 ") for v in (exif.get(_EXIF_MAKE), exif.get(_EXIF_MODEL)) if v
    )
    # GPS coordinates are deliberately not stored: they reveal where people live.
    return _join_details(taken=when, camera=camera or None)


def extract_image(
    path: Path, data: bytes, cfg: IndexConfig, captioner: Captioner | None
) -> Extraction:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            # Pillow only raises above 2x the limit; between 1x and 2x it just warns.
            if opened.width * opened.height > Image.MAX_IMAGE_PIXELS:
                raise ExtractionError("image is too large to index safely")
            opened.seek(0)
            exif_details = _exif_summary(opened)
            # Some encoders write a transparency entry Pillow cannot convert; we only
            # need the RGB pixels, so drop it rather than fail on the whole image.
            opened.info.pop("transparency", None)
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ExtractionError(f"unreadable image: {exc}") from exc

    width, height = image.size
    details = _join_details(size=f"{width}x{height}", extra=exif_details)
    segments: list[Segment] = []
    captioned = False

    # OCR (CPU) runs while the vision model (GPU) describes the image.
    ocr_job = None
    if cfg.ocr and tesseract_available() and min(width, height) >= 32:
        ocr_job = _OCR_POOL.submit(_ocr, image, cfg)

    if captioner is not None and captioner.available():
        thumb = image.copy()
        thumb.thumbnail((_CAPTION_MAX_SIDE, _CAPTION_MAX_SIDE))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        try:
            caption = captioner.caption(buf.getvalue())
        except ModelError as exc:
            # Leave the image uncaptioned; the next index run retries it.
            log.warning("%s: caption failed: %s", path, exc)
            caption = ""
        if caption:
            segments.append(Segment(caption, source="caption"))
            captioned = True

    if ocr_job is not None:
        text = ocr_job.result()
        # Tesseract emits noise on photos; keep only output that looks like words.
        if sum(ch.isalpha() for ch in text) >= 12:
            segments.append(Segment(text, source="ocr"))

    if details:
        segments.append(Segment(details, source="metadata"))
    return Extraction(kind="image", segments=segments, details=details, captioned=captioned)


# -- helpers ---------------------------------------------------------------


class ExtractionError(RuntimeError):
    pass


def _join_details(extra: str | None = None, **fields: str | None) -> str | None:
    parts = [f"{k}: {v.strip()}" for k, v in fields.items() if v and str(v).strip()]
    if extra:
        parts.append(extra)
    return "; ".join(parts) or None


def read_regular_file(path: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    """Read a file in one go without following a symlink at the last path component.

    Opening once and checking the open descriptor (rather than the name) means the file
    cannot be swapped for a link to something else between the check and the read.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ExtractionError("refusing to follow a symbolic link") from None
        raise
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ExtractionError("not a regular file")
        if st.st_size > max_bytes:
            raise FileTooLarge(st.st_size)
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FileTooLarge(len(data))
    return data, st


class FileTooLarge(Exception):
    pass


def extract(
    path: Path,
    cfg: IndexConfig,
    captioner: Captioner | None = None,
    data: bytes | None = None,
) -> Extraction:
    kind = kind_for(path)
    if kind is None:
        raise ExtractionError(f"unsupported file type: {path.suffix}")
    if data is None:
        data, _ = read_regular_file(path, int(cfg.max_file_size_mb * 1024 * 1024))
    try:
        if kind == "pdf":
            return extract_pdf(path, data, cfg)
        if kind == "document":
            return extract_docx(path, data, cfg)
        if kind == "image":
            return extract_image(path, data, cfg, captioner)
        return extract_text(path, data, cfg)
    except ExtractionError:
        raise
    except Exception as exc:
        # Third-party parsers raise many exception types on corrupt input.
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
