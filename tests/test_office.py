"""Slides, spreadsheets, OpenDocument, EPUB and HTML.

The files are built here part by part, the way the real formats are laid out, so the
tests need no office suite. The readers were also checked against real files made by
PowerPoint, Excel, LibreOffice, Google Docs and e-book tools.
"""

import io
import zipfile

import pytest
from conftest import FakeCaptioner, write

from filesonar import office
from filesonar.config import IndexConfig
from filesonar.extract import ExtractionError, extract, kind_for

IX = IndexConfig()
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
X = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
CORE = (
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/'
    'core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">'
    "<dc:title>{title}</dc:title><dc:creator>{author}</dc:creator></cp:coreProperties>"
)
ODF = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"'
)


def zip_file(path, parts: dict[str, str | bytes]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return path


def slide(*paragraphs):
    body = "".join(f"<a:p><a:r><a:t>{p}</a:t></a:r></a:p>" for p in paragraphs)
    return f"<p:sld {A} xmlns:p='urn:p'><p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>"


def pptx(path, slides, notes=None, title="Quarterly review", author="Sam"):
    parts = {"docProps/core.xml": CORE.format(title=title, author=author)}
    for n, paragraphs in enumerate(slides, 1):
        parts[f"ppt/slides/slide{n}.xml"] = slide(*paragraphs)
    for n, text in (notes or {}).items():
        parts[f"ppt/notesSlides/notesSlide{n}.xml"] = slide(text, str(n))
    return zip_file(path, parts)


def text_of(out):
    return "\n".join(s.text for s in out.segments if s.source != "metadata")


def test_kinds():
    for name in ("a.pptx", "a.xlsx", "a.odt", "a.ods", "a.odp", "a.epub", "a.PPTX"):
        assert kind_for(__import__("pathlib").Path(name)) == "document"


def test_pptx_slides_in_order_with_notes_and_properties(tmp_path):
    slides = [[f"Slide {n} heading", f"point for slide {n}"] for n in range(1, 12)]
    p = pptx(tmp_path / "deck.pptx", slides, notes={2: "Mention the budget overrun"})
    out = extract(p, IX)
    pages = [s.page for s in out.segments if s.source != "metadata"]
    assert pages == list(range(1, 12))  # slide 10 after slide 9, not after slide 1
    second = next(s.text for s in out.segments if s.page == 2)
    assert "point for slide 2" in second and "Notes:" in second
    assert "Mention the budget overrun" in second and "\n2\n" not in second + "\n"
    assert out.details == "title: Quarterly review; author: Sam; slides: 11"
    assert out.kind == "document"


def test_xlsx_shared_inline_and_numeric_cells_by_sheet_name(tmp_path):
    shared = (
        f"<sst {X}><si><t>Invoice</t></si><si><r><t>Blue </t></r><r><t>Heron</t></r></si></sst>"
    )
    workbook = (
        f'<workbook {X} {R}><sheets><sheet name="Expenses 2024" r:id="rId1"/>'
        '<sheet name="Notes" r:id="rId2"/></sheets></workbook>'
    )
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml"/></Relationships>'
    )
    sheet1 = (
        f'<worksheet {X}><sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c>'
        "<c><v>312.5</v></c></row><row><c t='inlineStr'><is><t>paid in March</t></is></c>"
        '</row><row><c t="s"><v>99</v></c></row></sheetData></worksheet>'
    )
    sheet2 = (
        f"<worksheet {X}><sheetData><row><c t='inlineStr'><is><t>call the plumber</t></is>"
        "</c></row></sheetData></worksheet>"
    )
    p = zip_file(
        tmp_path / "book.xlsx",
        {
            "xl/sharedStrings.xml": shared,
            "xl/workbook.xml": workbook,
            "xl/_rels/workbook.xml.rels": rels,
            "xl/worksheets/sheet1.xml": sheet1,
            "xl/worksheets/sheet2.xml": sheet2,
        },
    )
    out = extract(p, IX)
    first, second = [s.text for s in out.segments if s.source != "metadata"]
    assert first == "Sheet Expenses 2024\nInvoice | Blue Heron | 312.5\npaid in March"
    assert second == "Sheet Notes\ncall the plumber"  # an out-of-range string is skipped
    assert out.details == "sheets: 2"


def test_odt_headings_paragraphs_and_tables(tmp_path):
    content = (
        f"<office:document-content {ODF}><office:body><office:text>"
        "<text:h>Lease agreement</text:h><text:p>The tenant pays <text:span>1800</text:span>"
        " dollars.</text:p><table:table><table:table-row><table:table-cell><text:p>Rent"
        "</text:p></table:table-cell><table:table-cell><text:p>monthly</text:p>"
        "</table:table-cell></table:table-row></table:table></office:text></office:body>"
        "</office:document-content>"
    )
    meta = (
        '<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/"><office:meta><dc:title>Lease</dc:title>'
        "<meta:initial-creator>Sam</meta:initial-creator></office:meta></office:document-meta>"
    )
    p = zip_file(tmp_path / "lease.odt", {"content.xml": content, "meta.xml": meta})
    out = extract(p, IX)
    assert text_of(out) == "Lease agreement\nThe tenant pays 1800 dollars.\nRent | monthly"
    assert out.details == "title: Lease; author: Sam"


def test_ods_sheets_and_odp_pages(tmp_path):
    ods = (
        f"<office:document-content {ODF}><office:body><office:spreadsheet>"
        '<table:table table:name="Budget"><table:table-row><table:table-cell><text:p>Seeds'
        "</text:p></table:table-cell><table:table-cell><text:p>40</text:p></table:table-cell>"
        "</table:table-row></table:table></office:spreadsheet></office:body>"
        "</office:document-content>"
    )
    out = extract(zip_file(tmp_path / "b.ods", {"content.xml": ods}), IX)
    assert text_of(out) == "Sheet Budget\nSeeds | 40" and out.details == "sheets: 1"

    odp = (
        f"<office:document-content {ODF}><office:body><office:presentation>"
        "<draw:page><draw:frame><draw:text-box><text:p>Welcome</text:p></draw:text-box>"
        "</draw:frame></draw:page><draw:page/><draw:page><draw:frame><draw:text-box>"
        "<text:p>Thanks</text:p></draw:text-box></draw:frame></draw:page>"
        "</office:presentation></office:body></office:document-content>"
    )
    out = extract(zip_file(tmp_path / "talk.odp", {"content.xml": odp}), IX)
    assert [(s.page, s.text) for s in out.segments if s.source != "metadata"] == [
        (1, "Welcome"),
        (3, "Thanks"),  # an empty slide keeps the numbering
    ]


def epub(path, chapters, title="The Garden Book", author="A. Writer"):
    manifest = "".join(
        f'<item id="c{n}" href="text/ch{n}.xhtml" media-type="application/xhtml+xml"/>'
        for n in range(len(chapters))
    )
    spine = "".join(f'<itemref idref="c{n}"/>' for n in reversed(range(len(chapters))))
    opf = (
        '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<metadata><dc:title>{title}</dc:title><dc:creator>{author}</dc:creator></metadata>"
        f'<manifest>{manifest}<item id="css" href="s.css" media-type="text/css"/></manifest>'
        f"<spine>{spine}</spine></package>"
    )
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="OEBPS/book.opf"/></rootfiles></container>'
    )
    parts = {"META-INF/container.xml": container, "OEBPS/book.opf": opf}
    for n, body in enumerate(chapters):
        parts[f"OEBPS/text/ch{n}.xhtml"] = f"<html><body>{body}</body></html>"
    return zip_file(path, parts)


def test_epub_chapters_follow_the_reading_order(tmp_path):
    p = epub(tmp_path / "b.epub", ["<p>written first</p>", "<h1>Read first</h1><p>soil</p>"])
    out = extract(p, IX)
    # The spine lists chapter 1 before chapter 0, and that is the order to read.
    assert [s.text for s in out.segments if s.source != "metadata"] == [
        "Read first\nsoil",
        "written first",
    ]
    assert out.details == "title: The Garden Book; author: A. Writer"


def test_html_keeps_what_a_reader_sees(tmp_path):
    page = write(
        tmp_path / "saved.html",
        "<html><head><title>Visa  appointment</title><style>p{color:red}</style>"
        "<script>var token='abc';</script></head><body><nav>Menu</nav><p>Your appointment "
        "is on <b>May 3</b>.</p><table><tr><td>Office</td><td>Chicago</td></tr></table>"
        "<noscript>enable JS</noscript></body></html>",
    )
    out = extract(page, IX)
    assert text_of(out) == "Menu\nYour appointment is on May 3.\nOffice | Chicago"
    assert out.details == "title: Visa appointment" and out.kind == "text"
    assert "token" not in text_of(out) and "color" not in text_of(out)


# -- hostile and damaged files ---------------------------------------------------


def test_zip_bomb_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "MAX_UNPACKED", 1_000_000)
    huge = slide("x" * 2_000_000)  # compresses to a few KB
    p = zip_file(tmp_path / "bomb.pptx", {"ppt/slides/slide1.xml": huge})
    assert p.stat().st_size < 50_000
    with pytest.raises(ExtractionError, match="zip bomb"):
        extract(p, IX)


def test_entity_expansion_is_refused(tmp_path):
    laughs = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>'
        f"<office:document-content {ODF}><office:body><office:text><text:p>&lol2;</text:p>"
        "</office:text></office:body></office:document-content>"
    )
    p = zip_file(tmp_path / "laughs.odt", {"content.xml": laughs})
    with pytest.raises(ExtractionError, match="document type declaration"):
        extract(p, IX)


def test_entity_hidden_past_the_first_kilobytes_is_still_refused(tmp_path):
    padding = "<!--" + "x" * 10_000 + "-->"
    xml = (
        f'<?xml version="1.0"?>{padding}'
        '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><r>&e;</r>'
    )
    with pytest.raises(office.OfficeError):
        office.parse_xml(xml.encode())


@pytest.mark.parametrize(
    ("name", "parts", "message"),
    [
        ("broken.pptx", None, "damaged ZIP"),
        ("plain.odt", {"other.xml": "<a/>"}, "no content.xml"),
        ("plain.epub", {"x.html": "<p/>"}, "not an EPUB"),
        ("bad.odt", {"content.xml": "<unclosed>"}, "damaged"),
    ],
)
def test_damaged_documents_raise_extraction_error(tmp_path, name, parts, message):
    p = tmp_path / name
    if parts is None:
        p.write_bytes(b"PK\x03\x04 this is not really a zip")
    else:
        zip_file(p, parts)
    with pytest.raises(ExtractionError, match=message):
        extract(p, IX, FakeCaptioner())


def test_too_many_parts_is_refused(monkeypatch):
    monkeypatch.setattr(office, "MAX_MEMBERS", 5)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in range(6):
            z.writestr(f"p{n}.xml", "<a/>")
    with pytest.raises(office.OfficeError, match="too many parts"):
        office.Archive(buf.getvalue())


def test_slides_are_searchable_with_their_slide_number(home, indexer, db):
    from filesonar.search import Searcher

    pptx(home / "deck.pptx", [["Welcome"], ["Roadmap"], ["Hiring two firmware engineers"]])
    assert indexer.run().added == 1
    hits = Searcher(db, indexer.embedder, indexer.cfg.search).search("firmware engineers")
    assert hits and hits[0].name == "deck.pptx" and hits[0].page == 3


def test_html_indexed_by_an_older_version_is_read_again_once(home, indexer, db):
    write(home / "page.html", "<html><body><p>visa appointment</p></body></html>")
    write(home / "notes.txt", "plain notes")
    assert indexer.run().added == 2
    assert db.get_meta("reader_version") == "3"  # a new index needs no upgrade
    assert indexer.run().unchanged == 2

    db.set_meta("reader_version", "1")  # as left by 0.1.x, which kept the markup
    report = indexer.run()
    assert (report.updated, report.unchanged) == (1, 1)  # the page, not the text file
    assert indexer.run().unchanged == 2  # once only
    assert db.get_file(str(home / "page.html"))["sha256"]  # a real checksum again


def test_reader_upgrade_matches_file_endings_literally(home, indexer, db):
    write(home / "page.htm", "<p>hello</p>")
    write(home / "aXbYc.txt", "would match if _ and % were SQL wildcards")
    write(home / "a_b%c.txt", "the literal name")
    indexer.run()
    db.conn.execute("DELETE FROM meta WHERE key = 'reader_version'")
    assert db.upgrade_readers(2, {2: (("name", ".htm"), ("name", "_b%c.txt"))}) == 2
    assert db.get_file(str(home / "aXbYc.txt"))["sha256"]  # untouched


def test_placeholder_titles_are_not_indexed(tmp_path):
    p = epub(tmp_path / "b.epub", ["<p>soil</p>"], title="Unknown Title", author="Unknown Author")
    assert extract(p, IX).details is None
    p = pptx(tmp_path / "d.pptx", [["hi"]], title="Untitled", author="Sam")
    assert extract(p, IX).details == "author: Sam; slides: 1"


@pytest.mark.parametrize("depth", [500, 50_000])
def test_deeply_nested_document_reads_or_fails_cleanly(tmp_path, depth):
    """Hostile nesting must not crash the indexer (recursion limits, parser limits)."""
    nested = "<text:section>" * depth + "<text:p>deep</text:p>" + "</text:section>" * depth
    content = (
        f"<office:document-content {ODF}><office:body><office:text>{nested}"
        "</office:text></office:body></office:document-content>"
    )
    p = zip_file(tmp_path / "deep.odt", {"content.xml": content})
    try:
        out = extract(p, IX)
    except ExtractionError:
        return  # refused: fine
    assert text_of(out) == "deep"
