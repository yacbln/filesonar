"""Generate realistic sample files on the fly, so the repo ships no binary fixtures."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def text_image(path: Path, lines: list[str], size=(1200, 500)) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = _font(44)
    for n, line in enumerate(lines):
        draw.text((40, 40 + n * 70), line, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def photo(path: Path, exif: dict | None = None) -> Path:
    image = Image.new("RGB", (640, 480), (200, 30, 30))
    ImageDraw.Draw(image).ellipse((200, 150, 440, 330), fill=(20, 20, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    if exif:
        data = Image.Exif()
        for tag, value in exif.items():
            data[tag] = value
        image.save(path, exif=data)
    else:
        image.save(path)
    return path


def text_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), body, fontsize=12)
    if title:
        doc.set_metadata({"title": title})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def scanned_pdf(path: Path, image_path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=str(image_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def encrypted_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret contents")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    return path


def docx(
    path: Path,
    paragraphs: list[str],
    table: list[list[str]] | None = None,
    title: str | None = None,
) -> Path:
    document = Document()
    for p in paragraphs:
        document.add_paragraph(p)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    if title:
        document.core_properties.title = title
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    return path
