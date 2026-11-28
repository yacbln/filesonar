"""File writers for the benchmark: one function per format, all deterministic."""

from __future__ import annotations

import io
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from docx import Document
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, mono: bool = False, bold: bool = False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold" if bold else ""
    path = FONTS / f"{name}.ttf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


# -- writers ------------------------------------------------------------------------------


def pdf(path: Path, pages: list[str], title: str | None = None) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 550, 790), text, fontsize=10.5)
    if title:
        doc.set_metadata({"title": title, "author": "Sam Rivera"})
    doc.save(path)


def text_image(
    lines: list[str], width=1100, mono=False, size=26, bg="white", pad=50
) -> Image.Image:
    line_h = int(size * 1.45)
    img = Image.new("RGB", (width, pad * 2 + line_h * len(lines)), bg)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        bold = line.startswith("**")
        d.text((pad, pad + i * line_h), line.strip("*"), fill="black",
               font=font(size, mono=mono, bold=bold))  # fmt: skip
    return img


def photo_of(img: Image.Image, seed: int) -> Image.Image:
    """Make a clean render look like a phone photo of paper: tilt, shadow, noise."""
    rnd = random.Random(seed)
    canvas = Image.new("RGB", (img.width + 200, img.height + 200), (92, 84, 74))
    canvas.paste(img, (100, 100))
    canvas = canvas.rotate(rnd.uniform(-3, 3), expand=False, fillcolor=(92, 84, 74))
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.6))
    px = canvas.load()
    for _ in range(canvas.width * canvas.height // 40):
        x, y = rnd.randrange(canvas.width), rnd.randrange(canvas.height)
        r, g, b = px[x, y]
        n = rnd.randint(-18, 18)
        px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return canvas


def scanned_pdf(path: Path, pages: list[list[str]], seed: int) -> None:
    """A PDF that is only pictures of pages (needs OCR), like a scanner makes."""
    doc = pymupdf.open()
    for i, lines in enumerate(pages):
        img = photo_of(text_image(lines, width=1240, size=24), seed + i).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page = doc.new_page(width=img.width * 0.6, height=img.height * 0.6)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)


def docx(path: Path, paragraphs: list[str], table: list[list[str]] | None = None, title=None):
    d = Document()
    if title:
        d.core_properties.title = title
    d.core_properties.author = "Sam Rivera"
    for p in paragraphs:
        if p.startswith("# "):
            d.add_heading(p[2:], level=1)
        else:
            d.add_paragraph(p)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                t.cell(r, c).text = cell
    d.save(path)


def _zip(path: Path, parts: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)


def xlsx(path: Path, sheets: dict[str, list[list[str]]]) -> None:
    """A minimal but valid .xlsx (inline strings)."""
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="rels" ContentType="application/'
            'vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType='
            '"application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/'
                'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>"
        ),
        "_rels/.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            f'<workbook {ns} xmlns:r="{rel}"><sheets>'
            + "".join(
                f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>'
                for i, n in enumerate(sheets, 1)
            )
            + "</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Relationships>"
        ),
    }
    for i, rows in enumerate(sheets.values(), 1):
        body = "".join(
            "<row>"
            + "".join(f'<c t="inlineStr"><is><t>{escape(str(v))}</t></is></c>' for v in row)
            + "</row>"
            for row in rows
        )
        parts[f"xl/worksheets/sheet{i}.xml"] = (
            f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>"
        )
    _zip(path, parts)


def pptx(path: Path, slides: list[list[str]], notes: dict[int, str] | None = None, title=None):
    """Slides as PowerPoint stores them (text parts only; enough for indexing)."""
    a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    p = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
    parts = {}
    for n, lines in enumerate(slides, 1):
        body = "".join(f"<a:p><a:r><a:t>{escape(x)}</a:t></a:r></a:p>" for x in lines)
        parts[f"ppt/slides/slide{n}.xml"] = (
            f"<p:sld {a} {p}><p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>"
        )
    for n, text in (notes or {}).items():
        parts[f"ppt/notesSlides/notesSlide{n}.xml"] = (
            f"<p:notes {a} {p}><p:cSld><p:spTree><a:p><a:r><a:t>{escape(text)}</a:t></a:r></a:p>"
            "</p:spTree></p:cSld></p:notes>"
        )
    if title:
        parts["docProps/core.xml"] = (
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
            'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            f"<dc:title>{escape(title)}</dc:title><dc:creator>Sam Rivera</dc:creator>"
            "</cp:coreProperties>"
        )
    _zip(path, parts)


ODF = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
)


def odt(path: Path, paragraphs: list[str]) -> None:
    body = "".join(f"<text:p>{escape(x)}</text:p>" for x in paragraphs)
    _zip(path, {
        "mimetype": "application/vnd.oasis.opendocument.text",
        "content.xml": f"<office:document-content {ODF}><office:body><office:text>{body}"
                       "</office:text></office:body></office:document-content>",
    })  # fmt: skip


def ods(path: Path, sheets: dict[str, list[list[str]]]) -> None:
    tables = ""
    for name, rows in sheets.items():
        tables += (
            f'<table:table table:name="{escape(name)}">'
            + "".join(
                "<table:table-row>"
                + "".join(
                    f"<table:table-cell><text:p>{escape(str(v))}</text:p></table:table-cell>"
                    for v in row
                )
                + "</table:table-row>"
                for row in rows
            )
            + "</table:table>"
        )
    _zip(path, {
        "mimetype": "application/vnd.oasis.opendocument.spreadsheet",
        "content.xml": f"<office:document-content {ODF}><office:body><office:spreadsheet>{tables}"
                       "</office:spreadsheet></office:body></office:document-content>",
    })  # fmt: skip


def epub(path: Path, title: str, author: str, chapters: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>'
        for i in range(len(chapters))
    )
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    parts = {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": (
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/book.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>"
        ),
        "OEBPS/book.opf": (
            '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" '
            f'version="3.0"><metadata><dc:title>{escape(title)}</dc:title><dc:creator>{escape(author)}'
            f"</dc:creator></metadata><manifest>{items}</manifest><spine>{spine}</spine></package>"
        ),
    }
    for i, (head, body) in enumerate(chapters):
        paras = "".join(f"<p>{escape(x)}</p>" for x in body.split("\n\n"))
        parts[f"OEBPS/c{i}.xhtml"] = f"<html><body><h1>{escape(head)}</h1>{paras}</body></html>"
    _zip(path, parts)


def has_flite() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return " flite " in out.stdout


def speech(path: Path, text: str, picture: Image.Image | None = None) -> None:
    """A spoken recording (and, with ``picture``, a video showing it)."""
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "say.txt"
        txt.write_text(text)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        if picture is not None:
            png = Path(tmp) / "slide.png"
            picture.save(png)
            cmd += ["-loop", "1", "-framerate", "2", "-i", str(png)]
        cmd += ["-f", "lavfi", "-i", f"flite=textfile={txt}:voice=slt"]
        if picture is not None:
            cmd += ["-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1280:-2", "-shortest"]  # fmt: skip
        cmd += ["-c:a", "aac", "-b:a", "64k", str(path)]
        subprocess.run(cmd, check=True)


def slide(title: str, bullets: list[str]) -> Image.Image:
    img = Image.new("RGB", (1280, 720), (24, 48, 89))
    d = ImageDraw.Draw(img)
    d.text((70, 60), title, fill="white", font=font(54, bold=True))
    for i, b in enumerate(bullets):
        d.text((90, 190 + i * 80), "• " + b, fill=(225, 232, 245), font=font(36))
    return img


def screenshot(title: str, lines: list[str], accent=(0, 122, 255)) -> Image.Image:
    img = Image.new("RGB", (900, 1600), (245, 246, 248))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 900, 140), fill=accent)
    d.text((40, 50), title, fill="white", font=font(42, bold=True))
    y = 200
    for line in lines:
        d.rounded_rectangle((30, y - 15, 870, y + 75), radius=20, fill="white")
        d.text((55, y), line, fill=(30, 30, 30), font=font(30))
        y += 120
    return img
