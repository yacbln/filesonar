"""Text from ZIP-based document formats, with the standard library only.

PowerPoint and Excel (.pptx, .xlsx), OpenDocument (.odt, .ods, .odp) and EPUB books
are ZIP archives of XML files. Reading them is mostly a matter of finding the right
parts and collecting their text. These files come from anywhere, so:

- an archive may not unpack to more than ``MAX_UNPACKED`` bytes in total, no matter
  what sizes it claims (a "zip bomb" is a small file that expands to gigabytes);
- XML is parsed with defusedxml, and anything with a document type declaration is
  refused before that: these formats never use one, and it is how entity-expansion
  attacks ("billion laughs") work;
- nothing outside the archive is ever read: no external entities, no links followed.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

import defusedxml.ElementTree as SafeET
from defusedxml import DefusedXmlException

MAX_UNPACKED = 200 * 1024 * 1024  # bytes read from one archive, in total
MAX_MEMBERS = 10_000
MAX_SHEET_ROWS = 20_000  # rows read per spreadsheet; the indexer keeps fewer anyway

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "meta": "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
    "opf": "http://www.idpf.org/2007/opf",
    "cn": "urn:oasis:names:tc:opendocument:xmlns:container",
}


class OfficeError(ValueError):
    pass


class Archive:
    """A ZIP archive read with a budget on how much it may unpack."""

    def __init__(self, data: bytes):
        try:
            self.zip = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise OfficeError("not a valid document (damaged ZIP container)") from exc
        infos = self.zip.infolist()
        if len(infos) > MAX_MEMBERS:
            raise OfficeError("too many parts in the document")
        self.names = {i.filename for i in infos}
        self.left = MAX_UNPACKED

    def has(self, name: str) -> bool:
        return name in self.names

    def read(self, name: str) -> bytes:
        # The sizes in a ZIP header can lie, so count what really comes out.
        out = bytearray()
        with self.zip.open(name) as fh:
            while chunk := fh.read(1024 * 1024):
                out += chunk
                if len(out) > self.left:
                    raise OfficeError("document unpacks to more than allowed (zip bomb?)")
        self.left -= len(out)
        return bytes(out)

    def xml(self, name: str) -> ET.Element:
        return parse_xml(self.read(name))


def parse_xml(raw: bytes) -> ET.Element:
    if b"<!DOCTYPE" in raw[:4096].upper() or b"<!ENTITY" in raw.upper():
        raise OfficeError("XML with a document type declaration is not read")
    try:
        # defusedxml refuses entity tricks, external references and DTDs outright.
        return SafeET.fromstring(raw)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise OfficeError(f"damaged or unsafe XML: {exc}") from exc


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _natural(name: str) -> tuple:
    """slide2.xml before slide10.xml."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


# -- Office Open XML (.pptx, .xlsx) ----------------------------------------------


def ooxml_properties(arc: Archive) -> tuple[str | None, str | None]:
    if not arc.has("docProps/core.xml"):
        return None, None
    core = arc.xml("docProps/core.xml")
    return (
        _text(core.find("dc:title", NS)) or None,
        _text(core.find("dc:creator", NS)) or None,
    )


def pptx_slides(arc: Archive) -> list[str]:
    """Text of each slide in order, with its speaker notes."""
    slides = sorted(
        (n for n in arc.names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=_natural
    )
    out = []
    for name in slides:
        root = arc.xml(name)
        paragraphs = [_text(p) for p in root.iter(f"{{{NS['a']}}}p")]
        notes_name = name.replace("slides/slide", "notesSlides/notesSlide")
        if arc.has(notes_name):
            notes = [_text(p) for p in arc.xml(notes_name).iter(f"{{{NS['a']}}}p")]
            # Notes repeat the slide number as text; skip lines that are only digits.
            notes = [n for n in notes if n and not n.isdigit()]
            if notes:
                paragraphs += ["Notes:", *notes]
        out.append("\n".join(p for p in paragraphs if p))
    return out


def xlsx_sheets(arc: Archive) -> list[tuple[str, str]]:
    """(sheet name, rows as "a | b | c" lines) for each worksheet."""
    shared: list[str] = []
    if arc.has("xl/sharedStrings.xml"):
        root = arc.xml("xl/sharedStrings.xml")
        shared = [
            "".join(t.text or "" for t in si.iter(f"{{{NS['x']}}}t"))
            for si in root.findall("x:si", NS)
        ]
    # Sheet names and their files come from the workbook and its relationships.
    targets: dict[str, str] = {}
    rels = "xl/_rels/workbook.xml.rels"
    if arc.has(rels):
        for rel in arc.xml(rels).findall("rel:Relationship", NS):
            target = rel.get("Target", "")
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            targets[rel.get("Id", "")] = posixpath.normpath(target)
    sheets: list[tuple[str, str]] = []
    if arc.has("xl/workbook.xml"):
        for sheet in arc.xml("xl/workbook.xml").iter(f"{{{NS['x']}}}sheet"):
            rid = sheet.get(f"{{{NS['r']}}}id", "")
            if rid in targets:
                sheets.append((sheet.get("name", ""), targets[rid]))
    if not sheets:  # no workbook index: take the sheet files in order
        found = (n for n in arc.names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        sheets = [(posixpath.basename(n)[:-4], n) for n in sorted(found, key=_natural)]
    out = []
    for name, part in sheets:
        if not arc.has(part):
            continue
        lines = []
        for row in arc.xml(part).iter(f"{{{NS['x']}}}row"):
            cells = []
            for c in row.findall("x:c", NS):
                kind = c.get("t")
                if kind == "s":
                    v = c.findtext("x:v", default="", namespaces=NS)
                    value = shared[int(v)] if v.isdigit() and int(v) < len(shared) else ""
                elif kind == "inlineStr":
                    value = _text(c.find("x:is", NS))
                else:
                    value = c.findtext("x:v", default="", namespaces=NS)
                if value.strip():
                    cells.append(value.strip())
            if cells:
                lines.append(" | ".join(cells))
            if len(lines) >= MAX_SHEET_ROWS:
                break
        if lines:
            out.append((name, "\n".join(lines)))
    return out


# -- OpenDocument (.odt, .ods, .odp) ---------------------------------------------


def odf_properties(arc: Archive) -> tuple[str | None, str | None]:
    if not arc.has("meta.xml"):
        return None, None
    meta = arc.xml("meta.xml")
    creator = meta.find(".//meta:initial-creator", NS)
    if creator is None:
        creator = meta.find(".//dc:creator", NS)
    return _text(meta.find(".//dc:title", NS)) or None, _text(creator) or None


def _odf_lines(el: ET.Element) -> list[str]:
    """Paragraphs and headings in document order; table rows as "a | b" lines."""
    lines: list[str] = []
    for child in el:
        tag = child.tag
        if tag in (f"{{{NS['text']}}}p", f"{{{NS['text']}}}h"):
            if t := _text(child):
                lines.append(t)
        elif tag == f"{{{NS['table']}}}table-row":
            cells = [_text(c) for c in child if c.tag == f"{{{NS['table']}}}table-cell"]
            if any(cells):
                lines.append(" | ".join(c for c in cells if c))
        else:
            lines += _odf_lines(child)
    return lines


def odf_parts(arc: Archive) -> list[tuple[str | None, str]]:
    """(page or sheet name, text). Text documents give one part; presentations one
    per slide; spreadsheets one per sheet."""
    if not arc.has("content.xml"):
        raise OfficeError("not an OpenDocument file (no content.xml)")
    body = arc.xml("content.xml").find("office:body", NS)
    if body is None:
        return []
    pres = body.find("office:presentation", NS)
    if pres is not None:
        return [(None, "\n".join(_odf_lines(page))) for page in pres.findall("draw:page", NS)]
    sheet = body.find("office:spreadsheet", NS)
    if sheet is not None:
        return [
            (table.get(f"{{{NS['table']}}}name"), "\n".join(_odf_lines(table)[:MAX_SHEET_ROWS]))
            for table in sheet.findall("table:table", NS)
        ]
    return [(None, "\n".join(_odf_lines(body)))]


# -- EPUB --------------------------------------------------------------------------


def epub_book(arc: Archive) -> tuple[str | None, str | None, list[str]]:
    """Title, author and the text of each chapter in reading order."""
    if not arc.has("META-INF/container.xml"):
        raise OfficeError("not an EPUB book (no META-INF/container.xml)")
    rootfile = arc.xml("META-INF/container.xml").find(".//cn:rootfile", NS)
    opf_path = rootfile.get("full-path", "") if rootfile is not None else ""
    if not arc.has(opf_path):
        raise OfficeError("EPUB package file is missing")
    opf = arc.xml(opf_path)
    base = posixpath.dirname(opf_path)
    title = _text(opf.find(".//dc:title", NS)) or None
    author = _text(opf.find(".//dc:creator", NS)) or None
    items = {
        item.get("id"): item.get("href", "")
        for item in opf.iterfind(".//opf:manifest/opf:item", NS)
        if "html" in (item.get("media-type") or "")
    }
    chapters = []
    for ref in opf.iterfind(".//opf:spine/opf:itemref", NS):
        href = items.get(ref.get("idref"))
        if not href:
            continue
        name = posixpath.normpath(posixpath.join(base, href.split("#")[0]))
        if arc.has(name):
            text = html_to_text(arc.read(name).decode("utf-8", errors="replace"))[1]
            if text:
                chapters.append(text)
    return title, author, chapters


# -- HTML ----------------------------------------------------------------------------


class _HTMLText(HTMLParser):
    SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head"})
    BLOCK = frozenset({
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
        "article", "header", "footer", "blockquote", "pre", "table", "ul", "ol", "dt", "dd",
    })  # fmt: skip

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str | None, str]:
    """(title, readable text) of an HTML page, without scripts, styles or tags."""
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line.strip(" |") for line in lines if line.strip(" |"))
    title = " ".join("".join(parser.title).split()) or None
    return title, text
