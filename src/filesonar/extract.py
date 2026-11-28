"""Turn files into plain-text segments that can be indexed.

Each extractor returns an :class:`Extraction`: a list of segments (text plus the page it
came from and how it was obtained) and a short metadata summary. Extractors never write
to disk and never execute anything embedded in a file.
"""

from __future__ import annotations

import errno
import hashlib
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

from . import media, office
from .config import IndexConfig, MediaConfig
from .models import VIDEO_PROMPT, Captioner, ModelError
from .office import html_to_text

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
# ZIP-based formats read by office.py: slides, spreadsheets, OpenDocument, books.
OFFICE_EXTENSIONS = {".pptx", ".pptm", ".xlsx", ".xlsm", ".odt", ".ods", ".odp", ".epub"}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | PDF_EXTENSIONS
    | DOCX_EXTENSIONS
    | OFFICE_EXTENSIONS
    | media.AUDIO_EXTENSIONS
    | media.VIDEO_EXTENSIONS
)
MEDIA_KINDS = ("audio", "video")

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
    if ext in DOCX_EXTENSIONS or ext in OFFICE_EXTENSIONS:
        return "document"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in media.AUDIO_EXTENSIONS:
        return "audio"
    if ext in media.VIDEO_EXTENSIONS:
        return "video"
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
    if path.suffix.lower() in HTML_EXTENSIONS:
        # A saved web page is mostly markup, scripts and styles: keep what a reader sees.
        title, text = html_to_text(text)
        details = join_details(title=title)
        segments = [Segment(_clean(text))]
        if details:
            segments.append(Segment(details, source="metadata"))
        return Extraction(kind="text", segments=segments, details=details)
    return Extraction(kind="text", segments=[Segment(_clean(text))])


# -- slides, spreadsheets, OpenDocument, EPUB ---------------------------------


def extract_office(path: Path, data: bytes, cfg: IndexConfig) -> Extraction:
    ext = path.suffix.lower()
    arc = office.Archive(data)
    segments: list[Segment] = []
    if ext in (".pptx", ".pptm"):
        title, author = office.ooxml_properties(arc)
        slides = office.pptx_slides(arc)
        segments = [Segment(_clean(t), page=n) for n, t in enumerate(slides, 1) if t.strip()]
        details = join_details(title=title, author=author, slides=str(len(slides)))
    elif ext in (".xlsx", ".xlsm"):
        title, author = office.ooxml_properties(arc)
        sheets = office.xlsx_sheets(arc)
        segments = [Segment(_clean(f"Sheet {name}\n{rows}")) for name, rows in sheets]
        details = join_details(title=title, author=author, sheets=str(len(sheets)))
    elif ext == ".epub":
        title, author, chapters = office.epub_book(arc)
        segments = [Segment(_clean(t)) for t in chapters]
        details = join_details(title=title, author=author)
    else:  # OpenDocument
        title, author = office.odf_properties(arc)
        parts = office.odf_parts(arc)
        if ext == ".odp":
            segments = [Segment(_clean(t), page=n) for n, (_, t) in enumerate(parts, 1) if t]
            details = join_details(title=title, author=author, slides=str(len(parts)))
        elif ext == ".ods":
            segments = [Segment(_clean(f"Sheet {name}\n{t}")) for name, t in parts if t]
            details = join_details(title=title, author=author, sheets=str(len(parts)))
        else:
            segments = [Segment(_clean(t)) for _, t in parts if t]
            details = join_details(title=title, author=author)
    if details:
        segments.append(Segment(details, source="metadata"))
    return Extraction(kind="document", segments=segments, details=details)


# -- docx ------------------------------------------------------------------


def extract_docx(path: Path, data: bytes, cfg: IndexConfig) -> Extraction:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    props = doc.core_properties
    details = join_details(
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
        details = join_details(
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


def camera_name(make: object, model: object) -> str:
    """ "motorola" + "motorola edge 2024" -> "motorola edge 2024" (many phones repeat it)."""
    make_s = str(make or "").strip("\x00 ")
    model_s = str(model or "").strip("\x00 ")
    if make_s and model_s.lower().startswith(make_s.lower()):
        return model_s
    return " ".join(x for x in (make_s, model_s) if x)


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
    camera = camera_name(exif.get(_EXIF_MAKE), exif.get(_EXIF_MODEL))
    # GPS coordinates are deliberately not stored: they reveal where people live.
    return join_details(taken=when, camera=camera or None)


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
    details = join_details(size=f"{width}x{height}", extra=exif_details)
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


# What tools write when there is no title or author; not worth indexing.
_PLACEHOLDERS = {"unknown", "unknown title", "unknown author", "untitled", "none", "null", "-"}


def join_details(extra: str | None = None, **fields: str | None) -> str | None:
    parts = [
        f"{k}: {str(v).strip()}"
        for k, v in fields.items()
        if v and str(v).strip() and str(v).strip().lower() not in _PLACEHOLDERS
    ]
    if extra:
        parts.append(extra)
    return "; ".join(parts) or None


# -- audio and video -------------------------------------------------------------


def extract_media(
    path: Path,
    fd: int,
    cfg: MediaConfig,
    captioner: Captioner | None = None,
    transcriber: media.Transcriber | None = None,
    tagger: media.SoundTagger | None = None,
) -> Extraction:
    """What is said, heard and shown in a recording, read from an open file descriptor."""
    kind = kind_for(path) or "video"
    try:
        info = media.probe(fd)
    except media.MediaError as exc:
        raise ExtractionError(str(exc)) from exc
    segments: list[Segment] = []
    # "captioned" means "nothing left to describe". Only a vision model that is
    # configured but not there yet (or failed) leaves it False, to be retried later.
    captioned = not (info.has_video and cfg.frames and captioner is not None)
    language = None
    speech = cfg.transcribe and transcriber is not None and transcriber.available()
    sounds = cfg.sounds and tagger is not None and tagger.available()
    samples = None
    if info.has_audio and (speech or sounds):
        try:
            samples = media.decode_audio(fd, cfg.max_minutes * 60)
        except Exception as exc:  # unreadable sound: the rest of the file still counts
            log.warning("%s: could not read the sound: %s", path, exc)
    if samples is not None and speech:
        try:
            transcript = transcriber.transcribe(samples)
        except Exception as exc:  # the speech library: lose this transcript only
            log.warning("%s: could not read the sound: %s", path, exc)
        else:
            text = media.transcript_text(transcript)
            if text:
                segments.append(Segment(_clean(text), source="speech"))
                language = transcript.language
    if samples is not None and sounds:
        try:
            labels = tagger.tag(samples)
        except Exception as exc:  # the sound model: lose these labels only
            log.warning("%s: could not name the sounds: %s", path, exc)
        else:
            if labels:
                segments.append(Segment(media.sounds_text(labels), source="sound"))
    if not captioned and captioner.available():
        times = media.frame_times(info.duration, cfg.frames)
        sheet = media.contact_sheet(fd, times)
        if sheet is None:
            captioned = True  # no frame could be read; trying again would not help
        else:
            try:
                caption = captioner.caption(sheet, prompt=VIDEO_PROMPT.format(n=len(times)))
            except ModelError as exc:
                log.warning("%s: frames not described: %s", path, exc)  # retried later
                caption = ""
            if caption:
                segments.append(Segment(caption, source="caption"))
                captioned = True
    tags = info.tags
    details = join_details(
        title=tags.get("title"),
        artist=tags.get("artist") or tags.get("album_artist"),
        album=tags.get("album"),
        duration=media.duration_text(info.duration) if info.duration else None,
        recorded=info.created,
        size=f"{info.width}x{info.height}" if info.has_video and info.width else None,
        language=language,
        extra="no sound" if info.has_video and not info.has_audio else None,
    )
    if details:
        segments.append(Segment(details, source="metadata"))
    return Extraction(kind=kind, segments=segments, details=details, captioned=captioned)


def open_regular_file(path: Path, max_bytes: int) -> tuple[int, os.stat_result]:
    """Open a file for reading without following a symlink at the last path component.

    Opening once and checking the open descriptor (rather than the name) means the file
    cannot be swapped for a link to something else between the check and the read.
    The caller closes the descriptor.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ExtractionError("refusing to follow a symbolic link") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ExtractionError("not a regular file")
        if st.st_size > max_bytes:
            raise FileTooLarge(st.st_size)
    except BaseException:
        os.close(fd)
        raise
    return fd, st


def hash_open_file(fd: int) -> tuple[str, str]:
    """SHA-256 and MD5 of an open file, read in pieces (recordings can be gigabytes)."""
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    offset = 0
    while chunk := os.pread(fd, 1024 * 1024, offset):
        sha.update(chunk)
        md5.update(chunk)
        offset += len(chunk)
    return sha.hexdigest(), md5.hexdigest()


def read_regular_file(path: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    """Read a whole file at once (see ``open_regular_file``)."""
    fd, st = open_regular_file(path, max_bytes)
    with os.fdopen(fd, "rb") as fh:
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
            if path.suffix.lower() in OFFICE_EXTENSIONS:
                return extract_office(path, data, cfg)
            return extract_docx(path, data, cfg)
        if kind == "image":
            return extract_image(path, data, cfg, captioner)
        return extract_text(path, data, cfg)
    except ExtractionError:
        raise
    except office.OfficeError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        # Third-party parsers raise many exception types on corrupt input.
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
