import pytest
from conftest import FakeCaptioner, needs_tesseract
from PIL import ExifTags, Image
from samples import docx, encrypted_pdf, photo, scanned_pdf, text_image, text_pdf

from filesonar.config import IndexConfig
from filesonar.extract import ExtractionError, extract, kind_for

IX = IndexConfig()


def sources(extraction):
    return {s.source for s in extraction.segments}


def full_text(extraction):
    return "\n".join(s.text for s in extraction.segments)


@pytest.mark.parametrize(
    "name, kind",
    [
        ("a.PDF", "pdf"),
        ("b.docx", "document"),
        ("c.JPeG", "image"),
        ("d.md", "text"),
        ("e.exe", None),
        ("f", None),
        ("g.doc", None),
    ],
)
def test_kind_for(tmp_path, name, kind):
    assert kind_for(tmp_path / name) == kind


def test_plain_text(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Trip\n\nBook   the ferry to Oran\n\n\n")
    out = extract(p, IX)
    assert out.kind == "text"
    assert full_text(out) == "# Trip\nBook the ferry to Oran"


def test_binary_file_with_text_extension_is_skipped(tmp_path):
    p = tmp_path / "weird.txt"
    p.write_bytes(b"\x7fELF\x00\x00\x01binary")
    assert extract(p, IX).segments == []


def test_invalid_utf8_does_not_crash(tmp_path):
    p = tmp_path / "latin.txt"
    p.write_bytes("café crème".encode("latin-1"))
    assert "caf" in full_text(extract(p, IX))


def test_docx_paragraphs_tables_and_properties(tmp_path):
    p = docx(
        tmp_path / "lease.docx",
        ["Residential lease agreement", "Rent is due monthly"],
        table=[["Tenant", "Landlord"], ["Sam", "Acme Rentals"]],
        title="Lease 2024",
    )
    out = extract(p, IX)
    text = full_text(out)
    assert out.kind == "document"
    assert "Residential lease agreement" in text
    assert "Sam | Acme Rentals" in text
    assert "title: Lease 2024" in out.details


def test_pdf_text_by_page(tmp_path):
    p = text_pdf(
        tmp_path / "policy.pdf",
        [
            "Car insurance policy for a 2019 Toyota Corolla.",
            "Claims must be filed within thirty days of the incident.",
        ],
        title="Auto Policy",
    )
    out = extract(p, IX)
    pages = {s.page: s.text for s in out.segments if s.source == "text"}
    assert "Toyota" in pages[1]
    assert "thirty days" in pages[2]
    assert "title: Auto Policy" in out.details
    assert "pages: 2" in out.details


def test_pdf_page_limit(tmp_path):
    p = text_pdf(
        tmp_path / "long.pdf", [f"This is page number {i} of the book." for i in range(1, 11)]
    )
    out = extract(p, IndexConfig(max_pdf_pages=3))
    assert max(s.page for s in out.segments if s.page) == 3


@needs_tesseract
def test_scanned_pdf_falls_back_to_ocr(tmp_path):
    img = text_image(tmp_path / "scan.png", ["INVOICE 4471", "Total due 312.50 EUR"])
    p = scanned_pdf(tmp_path / "scan.pdf", img)
    out = extract(p, IX)
    ocr = [s for s in out.segments if s.source == "ocr"]
    assert ocr and ocr[0].page == 1
    assert "4471" in ocr[0].text


def test_scanned_pdf_without_ocr_is_empty_not_an_error(tmp_path):
    img = text_image(tmp_path / "scan.png", ["INVOICE 4471"])
    p = scanned_pdf(tmp_path / "scan.pdf", img)
    out = extract(p, IndexConfig(ocr=False))
    assert [s for s in out.segments if s.source != "metadata"] == []


def test_encrypted_pdf_is_not_read(tmp_path):
    out = extract(encrypted_pdf(tmp_path / "locked.pdf"), IX)
    assert out.details == "password protected"
    assert "secret" not in full_text(out)


def test_corrupt_files_raise_extraction_error(tmp_path):
    for name in ("broken.pdf", "broken.docx", "broken.png"):
        p = tmp_path / name
        p.write_bytes(b"this is not really a " + name.encode())
        with pytest.raises(ExtractionError):
            extract(p, IX)


@needs_tesseract
def test_image_gets_caption_ocr_and_metadata(tmp_path):
    p = text_image(tmp_path / "shot.png", ["Boarding pass", "Flight AH1052 to Algiers"])
    cap = FakeCaptioner("a screenshot of an airline boarding pass")
    out = extract(p, IX, cap)
    assert out.captioned and cap.calls == 1
    assert sources(out) == {"caption", "ocr", "metadata"}
    assert "AH1052" in full_text(out)
    assert "size: 1200x500" in out.details


def test_image_without_captioner_or_ocr(tmp_path):
    out = extract(photo(tmp_path / "p.jpg"), IndexConfig(ocr=False), None)
    assert not out.captioned
    assert sources(out) == {"metadata"}


def test_unavailable_captioner_is_not_called(tmp_path):
    cap = FakeCaptioner(available=False)
    out = extract(photo(tmp_path / "p.jpg"), IndexConfig(ocr=False), cap)
    assert cap.calls == 0 and not out.captioned


def test_caption_failure_does_not_fail_the_file(tmp_path):
    cap = FakeCaptioner()
    cap.fail = True
    out = extract(photo(tmp_path / "p.jpg"), IndexConfig(ocr=False), cap)
    assert not out.captioned
    assert sources(out) == {"metadata"}


def test_exif_date_and_camera_kept_but_gps_dropped(tmp_path):
    p = photo(
        tmp_path / "IMG_1.jpg",
        exif={
            ExifTags.Base.Make: "Canon",
            ExifTags.Base.Model: "EOS 80D",
            ExifTags.Base.DateTime: "2023:07:14 18:30:00",
        },
    )
    # add GPS info
    with Image.open(p) as im:
        exif = im.getexif()
        exif[ExifTags.IFD.GPSInfo] = {1: "N", 2: (36.0, 45.0, 0.0), 3: "E", 4: (3.0, 3.0, 0.0)}
        im.save(p, exif=exif)
    out = extract(p, IndexConfig(ocr=False))
    assert "taken: 2023-07-14" in out.details
    assert "Canon EOS 80D" in out.details
    assert "36" not in out.details and "gps" not in out.details.lower()


def test_decompression_bomb_is_refused(tmp_path, monkeypatch):
    p = photo(tmp_path / "big.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(ExtractionError, match=r"too large|decompression|exceeds"):
        extract(p, IX)


def test_huge_caption_is_never_sent_full_size(tmp_path):
    seen = {}

    class Recorder(FakeCaptioner):
        def caption(self, image_png):
            import io

            seen["size"] = Image.open(io.BytesIO(image_png)).size
            return "x"

    big = Image.new("RGB", (4000, 3000), "green")
    p = tmp_path / "big.jpg"
    big.save(p)
    extract(p, IndexConfig(ocr=False), Recorder())
    assert max(seen["size"]) == 1024


def test_pdf_reading_stops_once_the_chunk_budget_is_full(tmp_path):
    pages = [f"Page {i}. " + "lorem ipsum dolor sit amet " * 60 for i in range(1, 41)]
    p = text_pdf(tmp_path / "book.pdf", pages)
    out = extract(p, IndexConfig(max_chunks_per_file=2, chunk_chars=1200, chunk_overlap=200))
    read = [s.page for s in out.segments if s.source == "text"]
    assert 1 in read and max(read) < 10


def test_ocr_timeout_keeps_the_rest_of_the_image(tmp_path, monkeypatch):
    import pytesseract

    def slow(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_string", slow)
    out = extract(photo(tmp_path / "noisy.jpg"), IX, FakeCaptioner())
    assert out.captioned
    assert {s.source for s in out.segments} == {"caption", "metadata"}


def test_truncated_image_is_indexed_from_what_is_readable(tmp_path):
    full = photo(tmp_path / "full.png")
    cut = tmp_path / "cut.png"
    cut.write_bytes(full.read_bytes()[: int(full.stat().st_size * 0.6)])
    out = extract(cut, IndexConfig(ocr=False), FakeCaptioner())
    assert out.captioned and "size: 640x480" in out.details


def test_rgb_png_with_odd_transparency_entry(tmp_path):
    import zlib

    # A valid 2x2 RGB PNG whose tRNS chunk is malformed for its colour type.
    def chunk(kind, data):
        import struct

        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    import struct

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 2 for _ in range(2))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"tRNS", b"\x00\x00")
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    p = tmp_path / "AIM.png"
    p.write_bytes(png)
    out = extract(p, IndexConfig(ocr=False), FakeCaptioner())
    assert out.captioned
