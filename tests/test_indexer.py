import os
import stat

import pytest
from conftest import FakeCaptioner, FakeEmbedder, needs_tesseract, write
from samples import photo

from filesonar.db import Database
from filesonar.indexer import Indexer
from filesonar.models import ModelError


def paths(db):
    return sorted(db.all_paths())


def test_index_add_unchanged_update_remove(home, indexer, db, embedder):
    a = write(home / "a.txt", "alpha content")
    b = write(home / "sub/b.md", "beta content")
    r = indexer.run()
    assert (r.added, r.failed) == (2, 0)
    assert paths(db) == [str(a), str(b)]

    calls = embedder.calls
    r = indexer.run()
    assert r.unchanged == 2 and embedder.calls == calls  # nothing re-embedded

    b.write_text("beta content, revised")
    r = indexer.run()
    assert r.updated == 1 and r.unchanged == 1

    a.unlink()
    r = indexer.run()
    assert r.removed == 1
    assert paths(db) == [str(b)]


def test_touch_without_content_change_is_cheap(home, indexer, embedder):
    f = write(home / "a.txt", "same")
    indexer.run()
    calls = embedder.calls
    os.utime(f, (1, 1))
    r = indexer.run()
    assert r.unchanged == 1 and embedder.calls == calls


def test_move_is_detected_without_reprocessing(home, indexer, db, embedder):
    f = write(home / "inbox/report.txt", "quarterly revenue report")
    indexer.run()
    calls = embedder.calls
    target = home / "archive/2024/report-final.txt"
    target.parent.mkdir(parents=True)
    f.rename(target)
    r = indexer.run()
    assert r.moved == 1 and r.added == 0 and r.removed == 0
    assert embedder.calls == calls
    assert paths(db) == [str(target)]
    # the new name is searchable
    hits = db.fts_search('"final"', 5)
    assert hits


def test_duplicate_copies_are_both_indexed(home, indexer, db):
    write(home / "one.txt", "identical")
    write(home / "two.txt", "identical")
    r = indexer.run()
    assert r.added == 2 and len(paths(db)) == 2


def test_sensitive_files_never_indexed_even_with_empty_exclude(home, make_cfg, db, embedder):
    # Worst case: no user excludes and hidden files allowed.
    cfg = make_cfg(exclude=[], index={"include_hidden": True})
    write(home / ".ssh/id_rsa.txt", "PRIVATE KEY")
    write(home / "project/.env", "TOKEN=abc")
    write(home / "notes/id_ed25519.txt", "PRIVATE KEY")
    write(home / ".aws/credentials.txt", "TOKEN=aws")
    write(home / "ok.txt", "hello")
    Indexer(cfg, db, embedder).run()
    stored = paths(db)
    assert str(home / "ok.txt") in stored
    assert not any(".ssh" in p or p.endswith(".env") or p.endswith(".key") for p in stored)
    assert not any("PRIVATE KEY" in t or "TOKEN=" in t for t in embedder.texts)


def test_hidden_files_and_folders_skipped_by_default(home, indexer, db):
    write(home / ".hidden.txt", "x")
    write(home / ".config/app/settings.txt", "x")
    write(home / "visible.txt", "x")
    indexer.run()
    assert paths(db) == [str(home / "visible.txt")]


def test_user_excludes_are_relative_to_the_indexed_folder(home, make_cfg, db, embedder):
    # The test folder itself lives under /tmp: "**/tmp/**" must only match inside it.
    cfg = make_cfg(exclude=["**/tmp/**", "*.log", "/drafts/**"])
    write(home / "tmp/a.txt", "x")
    write(home / "b.log", "x")
    write(home / "drafts/d.txt", "x")
    write(home / "keep/drafts/e.txt", "x")
    write(home / "c.txt", "x")
    Indexer(cfg, db, embedder).run()
    assert paths(db) == [str(home / "c.txt"), str(home / "keep/drafts/e.txt")]


def test_root_inside_sensitive_folder_indexes_nothing(tmp_path, db, embedder):
    from filesonar.config import config_from_dict

    secret = tmp_path / ".ssh"
    write(secret / "notes.txt", "x")
    cfg = config_from_dict({"roots": [str(secret)], "data_dir": str(tmp_path / "d")}, base=tmp_path)
    Indexer(cfg, db, embedder).run()
    assert paths(db) == []


def test_symlinks_are_never_followed(home, tmp_path, indexer, db, embedder):
    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside / "secret.txt", "outside the indexed folder")
    (home / "link.txt").symlink_to(outside / "secret.txt")
    (home / "linkdir").symlink_to(outside, target_is_directory=True)
    write(home / "real.txt", "inside")
    indexer.run()
    assert paths(db) == [str(home / "real.txt")]
    assert not any("outside the indexed" in t for t in embedder.texts)


def test_unsupported_large_and_special_files_skipped(home, make_cfg, db, embedder):
    cfg = make_cfg(index={"max_file_size_mb": 0.001})
    write(home / "program.exe", "MZ")
    write(home / "big.txt", "x" * 5000)
    write(home / "small.txt", "x")
    os.mkfifo(home / "pipe.txt")
    r = Indexer(cfg, db, embedder).run()
    assert paths(db) == [str(home / "small.txt")]
    assert r.skipped == 1


def test_unreadable_file_is_reported_not_fatal(home, indexer, db):
    if os.geteuid() == 0:
        pytest.skip("root can read everything")
    f = write(home / "locked.txt", "x")
    f.chmod(0)
    write(home / "fine.txt", "y")
    try:
        r = indexer.run()
    finally:
        f.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert r.failed == 1 and r.added == 1
    assert r.errors[0][0] == str(f)


def test_corrupt_file_is_recorded_once(home, indexer, db, embedder):
    write(home / "broken.pdf", "not a pdf")
    r = indexer.run()
    assert r.failed == 1
    row = db.get_file(str(home / "broken.pdf"))
    assert row["status"] == "error" and row["error"]
    r = indexer.run()
    assert r.failed == 0 and r.unchanged == 1  # not retried until it changes


def test_empty_file_still_findable_by_name(home, indexer, db):
    write(home / "tax_return_2023.txt", "")
    indexer.run()
    row = db.get_file(str(home / "tax_return_2023.txt"))
    assert row["status"] == "empty"
    assert db.fts_search('"tax"', 5)


def test_model_server_down_aborts_without_corrupting(home, indexer, db, embedder):
    write(home / "a.txt", "alpha")
    embedder.fail = True
    with pytest.raises(ModelError):
        indexer.run()
    assert paths(db) == []
    embedder.fail = False
    assert indexer.run().added == 1


def test_images_are_recaptioned_when_vision_model_appears(home, cfg, db, embedder):
    photo(home / "bike.jpg")
    cap = FakeCaptioner(available=False)
    Indexer(cfg, db, embedder, cap).run()
    assert not db.get_file(str(home / "bike.jpg"))["captioned"]
    cap.is_available = True
    r = Indexer(cfg, db, embedder, cap).run()
    assert r.updated == 1
    assert db.get_file(str(home / "bike.jpg"))["captioned"]
    assert Indexer(cfg, db, embedder, cap).run().unchanged == 1


def test_captions_disabled_in_config(home, make_cfg, db, embedder):
    cfg = make_cfg(models={"caption_images": False})
    photo(home / "bike.jpg")
    cap = FakeCaptioner()
    Indexer(cfg, db, embedder, cap).run()
    assert cap.calls == 0


def test_changing_embedding_model_requires_reembedding(home, cfg, db, embedder):
    write(home / "a.txt", "alpha")
    Indexer(cfg, db, embedder).run()
    write(home / "b.txt", "beta")
    other = FakeEmbedder()
    other.name = "different-model"
    with pytest.raises(RuntimeError, match="reembed"):
        Indexer(cfg, Database(cfg.db_path), other).run()
    db.reset()
    assert Indexer(cfg, db, other).run().added == 2


def test_reembed_switches_model_without_reading_files_again(home, cfg, db, embedder):
    from filesonar.search import Searcher

    write(home / "a.txt", "alpha lease")
    photo(home / "p.jpg")
    cap = FakeCaptioner()
    Indexer(cfg, db, embedder, cap).run()
    chunks = db.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    other = FakeEmbedder()
    other.name = "different-model"
    cap2 = FakeCaptioner()
    indexer = Indexer(cfg, db, other, cap2)
    assert indexer.reembed() == chunks
    assert cap2.calls == 0  # nothing was read or described again
    assert db.get_meta("embedding_model") == "different-model"
    assert db.conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == chunks
    assert indexer.run().unchanged == 2
    searcher = Searcher(db, other)
    assert searcher.search("alpha lease")[0].name == "a.txt"
    assert searcher.last_warning is None  # meaning search works with the new model


def test_interrupted_reembedding_finishes_on_the_next_run(home, cfg, db, embedder):
    write(home / "a.txt", "alpha")
    Indexer(cfg, db, embedder).run()
    other = FakeEmbedder()
    other.name = "different-model"
    db.set_meta("reembedding", "different-model")
    db.drop_vectors()  # as if stopped half way
    Indexer(cfg, db, other).run()
    assert db.get_meta("reembedding") is None
    assert db.get_meta("embedding_model") == "different-model"


def test_update_paths_handles_files_and_folders(home, indexer, db):
    a = write(home / "docs/a.txt", "alpha")
    b = write(home / "docs/b.txt", "beta")
    indexer.run()

    c = write(home / "docs/c.txt", "gamma")
    r = indexer.update_paths([c])
    assert r.added == 1

    # rename the whole folder: files move, nothing is re-read
    (home / "docs").rename(home / "papers")
    r = indexer.update_paths([home / "docs", home / "papers"])
    assert r.moved == 3 and r.removed == 0
    assert paths(db) == sorted(str(home / "papers" / p.name) for p in (a, b, c))

    # delete the folder
    for p in (home / "papers").iterdir():
        p.unlink()
    (home / "papers").rmdir()
    r = indexer.update_paths([home / "papers"])
    assert r.removed == 3 and paths(db) == []


def test_update_paths_ignores_paths_outside_roots(home, tmp_path, indexer, db):
    outside = write(tmp_path / "elsewhere.txt", "x")
    r = indexer.update_paths([outside, tmp_path])
    assert r.added == 0 and paths(db) == []


def test_database_files_are_private(cfg, db):
    assert stat.S_IMODE(os.stat(cfg.db_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(cfg.data_dir).st_mode) == 0o700


def test_file_swapped_for_symlink_after_the_check_is_not_read(
    home, tmp_path, indexer, db, embedder
):
    """Simulate the race: the walk saw a regular file, then it became a symlink."""
    from filesonar.indexer import IndexReport

    secret = write(tmp_path / "outside/secret.txt", "TOP SECRET")
    target = home / "innocent.txt"
    target.symlink_to(secret)
    report = IndexReport()
    indexer.index_file(target, report)  # called directly, bypassing eligible()
    assert db.all_paths() == []
    assert not any("TOP SECRET" in t for t in embedder.texts)


def test_symlinked_parent_folder_is_not_read(home, tmp_path, indexer, db, embedder):
    from filesonar.indexer import IndexReport

    write(tmp_path / "outside/secret.txt", "TOP SECRET")
    (home / "sub").symlink_to(tmp_path / "outside", target_is_directory=True)
    indexer.index_file(home / "sub/secret.txt", IndexReport())
    assert db.all_paths() == []
    assert not any("TOP SECRET" in t for t in embedder.texts)


def test_read_regular_file_refuses_links_fifos_and_oversized(tmp_path):
    from filesonar.extract import ExtractionError, FileTooLarge, read_regular_file

    real = write(tmp_path / "real.txt", "hello")
    (tmp_path / "link.txt").symlink_to(real)
    os.mkfifo(tmp_path / "fifo.txt")  # a naive open() would block forever
    assert read_regular_file(real, 100)[0] == b"hello"
    with pytest.raises(ExtractionError, match="symbolic link"):
        read_regular_file(tmp_path / "link.txt", 100)
    with pytest.raises(ExtractionError, match="regular file"):
        read_regular_file(tmp_path / "fifo.txt", 100)
    with pytest.raises(FileTooLarge):
        read_regular_file(real, 3)


def test_default_excludes_apply_on_top_of_user_excludes(home, make_cfg, db, embedder):
    cfg = make_cfg(exclude=["*.log"])
    write(home / "proj/node_modules/lib/readme.md", "x")
    write(home / "proj/build/lib/site-packages/a.txt", "x")
    write(home / "proj/vendor/pkg/notes.md", "x")
    write(home / "proj/app.log", "x")
    write(home / "proj/README.md", "my project")
    Indexer(cfg, db, embedder).run()
    assert paths(db) == [str(home / "proj/README.md")]


def test_default_excludes_can_be_turned_off(home, make_cfg, db, embedder):
    cfg = make_cfg(index={"default_excludes": False})
    write(home / "proj/vendor/notes.md", "x")
    Indexer(cfg, db, embedder).run()
    assert paths(db) == [str(home / "proj/vendor/notes.md")]


def test_python_package_install_folders_are_skipped(home, make_cfg, indexer, db):
    # e.g. an AWS Lambda bundle created with `pip install -t lambda_src`
    write(home / "proj/lambda_src/botocore/data/examples.json", '{"x": 1}')
    (home / "proj/lambda_src/botocore-1.31.0.dist-info").mkdir()
    write(home / "proj/lambda_src/handler.txt", "x")
    write(home / "proj/notes.md", "project notes")
    indexer.run()
    assert paths(db) == [str(home / "proj/notes.md")]
    # also when the watcher reports a single file inside such a folder
    r = indexer.update_paths([home / "proj/lambda_src/botocore/data/examples.json"])
    assert r.added == 0

    cfg = make_cfg(index={"skip_dependency_folders": False})
    Indexer(cfg, db, FakeEmbedder()).run()
    assert str(home / "proj/lambda_src/handler.txt") in paths(db)


def test_newly_excluded_files_are_removed_from_the_index(home, make_cfg, db, embedder):
    write(home / "proj/vendor/notes.md", "x")
    Indexer(make_cfg(index={"default_excludes": False}), db, embedder).run()
    assert len(paths(db)) == 1
    r = Indexer(make_cfg(), db, embedder).run()
    assert r.removed == 1 and paths(db) == []


def test_very_long_files_are_capped_but_keep_metadata(home, make_cfg, db, embedder):
    from samples import docx

    cfg = make_cfg(index={"max_chunks_per_file": 5})
    docx(
        home / "book.docx", [f"Paragraph {n} " + "word " * 300 for n in range(40)], title="Big Book"
    )
    Indexer(cfg, db, embedder).run()
    rows = db.conn.execute("SELECT source FROM chunks").fetchall()
    assert len([r for r in rows if r[0] == "text"]) == 5
    assert any(r[0] == "metadata" for r in rows)  # title still searchable


def test_identical_copy_reuses_the_existing_entry(home, indexer, db, embedder, captioner):
    photo(home / "Downloads/beach.jpg")
    indexer.run()
    calls_embed, calls_caption = embedder.calls, captioner.calls
    import shutil

    shutil.copy(home / "Downloads/beach.jpg", home / "Downloads/beach (1).jpg")
    r = indexer.run()
    assert r.added == 1
    assert (embedder.calls, captioner.calls) == (calls_embed, calls_caption)  # no rework
    a = db.get_file(str(home / "Downloads/beach.jpg"))
    b = db.get_file(str(home / "Downloads/beach (1).jpg"))
    assert b["captioned"] and b["sha256"] == a["sha256"]
    n = lambda fid: db.conn.execute(  # noqa: E731
        "SELECT COUNT(*) FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE file_id=?)",
        (fid,),
    ).fetchone()[0]
    assert n(a["id"]) == n(b["id"]) > 0
    # deleting the original leaves the copy fully searchable
    (home / "Downloads/beach.jpg").unlink()
    indexer.run()
    assert db.fts_search('"bicycle"', 5)


def test_office_lock_files_are_skipped(home, cfg, db, embedder):
    write(home / "~$report.docx", "owner file")
    write(home / ".~lock.report.docx#", "lock")
    write(home / "price$5 x~$y.txt", "a real file with odd characters")
    Indexer(cfg, db, embedder).run()
    assert [p.rsplit("/", 1)[-1] for p in db.all_paths()] == ["price$5 x~$y.txt"]


# -- OCR that becomes available later ------------------------------------------
# Found on a fresh install: setup suggests installing Tesseract, but scans indexed
# before that were never read again, because their files had not changed.


def _set_ocr(monkeypatch, available: bool) -> None:
    from filesonar import extract as extract_module

    monkeypatch.setattr(extract_module, "tesseract_available", lambda: available)


@needs_tesseract
def test_scans_are_read_again_once_ocr_is_installed(home, indexer, db, monkeypatch):
    from samples import scanned_pdf, text_image

    text_image(home / "invoice.png", ["INVOICE 4471", "BLUE HERON PLUMBING"])
    scanned_pdf(home / "invoice.pdf", text_image(home / ".x/p.png", ["RECEIPT 9921"]))
    write(home / "notes.txt", "plain text needs no OCR")

    _set_ocr(monkeypatch, False)
    assert indexer.run().added == 3
    assert {r["name"]: r["ocr"] for r in db.conn.execute("SELECT name, ocr FROM files")} == {
        "invoice.png": 0,
        "invoice.pdf": 0,
        "notes.txt": 0,  # nothing to re-read for text files, whatever this says
    }
    assert indexer.run().unchanged == 3  # still no OCR: nothing to do

    _set_ocr(monkeypatch, True)
    report = indexer.run()
    assert (report.updated, report.unchanged) == (2, 1)  # the scans, not the text file
    text = " ".join(r["text"] for r in db.conn.execute("SELECT text FROM chunks"))
    assert "4471" in text and "9921" in text
    assert indexer.run().unchanged == 3  # read once, then settled


def test_ocr_turned_off_in_the_config_never_rereads(home, make_cfg, db, embedder, monkeypatch):
    from samples import text_image

    text_image(home / "invoice.png", ["INVOICE 4471"])
    _set_ocr(monkeypatch, True)
    off = Indexer(make_cfg(index={"ocr": False}), db, embedder, FakeCaptioner())
    assert off.run().added == 1
    assert off.run().unchanged == 1


def test_identical_copy_of_an_unread_scan_is_not_copied(home, indexer, db, monkeypatch):
    """A twin indexed without OCR must not hand its empty result to a new copy."""
    from samples import text_image

    text_image(home / "a/invoice.png", ["INVOICE 4471"])
    _set_ocr(monkeypatch, False)
    indexer.run()
    (home / "b").mkdir()
    (home / "b/invoice.png").write_bytes((home / "a/invoice.png").read_bytes())
    _set_ocr(monkeypatch, True)
    indexer.run()
    rows = db.conn.execute("SELECT path, ocr FROM files ORDER BY path").fetchall()
    assert [r["ocr"] for r in rows] == [1, 1]


def test_upgrade_marks_only_textless_scans_for_ocr(tmp_path):
    """A 0.1.0 index has no OCR flag. Scans and photos that gave no text at all get
    another try once OCR is available; everything else stays as it is."""
    path = tmp_path / "index.sqlite3"
    db = Database(path)
    rows = [
        ("scan.png", "image", "empty"),
        ("scan.pdf", "pdf", "empty"),
        ("photo.jpg", "image", "ok"),  # has a description already
        ("notes.txt", "text", "empty"),  # OCR cannot help a text file
    ]
    for n, (name, kind, status) in enumerate(rows):
        db.conn.execute(
            "INSERT INTO files(path, name, kind, size, mtime, sha256, status, indexed_at) "
            "VALUES (?, ?, ?, 1, 1.0, ?, ?, 1.0)",
            (f"/x/{name}", name, kind, f"sha{n}", status),
        )
    db.conn.execute("ALTER TABLE files DROP COLUMN ocr")
    db.set_meta("schema_version", "2")
    db.conn.commit()
    db.close()

    db = Database(path)
    flags = dict(db.conn.execute("SELECT name, ocr FROM files"))
    assert flags == {"scan.png": 0, "scan.pdf": 0, "photo.jpg": 1, "notes.txt": 1}
    assert db.get_meta("schema_version") == "3"
    db.close()
