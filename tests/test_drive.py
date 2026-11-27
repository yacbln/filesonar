import asyncio
import json
import os
import sqlite3
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import httpx
import pytest
from conftest import write
from fakedrive import FakeDrive
from mcp.server.mcpserver.exceptions import ToolError
from samples import photo, text_pdf

from filesonar import drive as drive_mod
from filesonar.db import Database
from filesonar.drive import (
    DriveClient,
    DriveError,
    NotSignedIn,
    Token,
    is_thumbnail_url,
    load_token,
    read_client_secret,
    save_token,
)
from filesonar.drive_index import DriveIndexer, drive_key
from filesonar.search import Searcher
from filesonar.server import build_server


@pytest.fixture
def fake():
    return FakeDrive()


@pytest.fixture
def dcfg(make_cfg):
    return make_cfg(drive={"enabled": True})


@pytest.fixture
def ddb(dcfg):
    database = Database(dcfg.db_path)
    yield database
    database.close()


@pytest.fixture
def dindexer(dcfg, ddb, embedder, captioner, fake):
    return DriveIndexer(dcfg, ddb, embedder, fake.client(), captioner)


def pdf_bytes(tmp_path, pages, name="x.pdf"):
    return text_pdf(tmp_path / "src" / name, pages).read_bytes()


def photo_bytes(tmp_path, name="p.jpg"):
    return photo(tmp_path / "src" / name).read_bytes()


def call(server, name, args):
    result = asyncio.run(server.call_tool(name, args))
    return json.loads(result.content[0].text)


# -- indexing ----------------------------------------------------------------


def test_full_scan_indexes_supported_files(fake, dindexer, ddb, tmp_path):
    taxes = fake.folder("Taxes")
    fake.add("return-2023.pdf", pdf_bytes(tmp_path, ["IRS form 1040 refund 412"]), parent=taxes.id)
    fake.add(
        "Trip plan",
        mime="application/vnd.google-apps.document",
        export={"text/plain": b"Itinerary for Lisbon in June: tram 28 and pastel de nata"},
    )
    fake.add("budget", mime="application/vnd.google-apps.spreadsheet",
             export={"text/csv": b"rent,1200\ngroceries,400"})  # fmt: skip
    fake.add("archive.zip", b"PK\x03\x04")  # unsupported
    fake.add("notes.txt", b"old notes", trashed=True)
    fake.add("shared.txt", b"someone else's file", owned=False)

    report = dindexer.sync()

    assert report.added == 3 and report.failed == 0
    rows = {r["name"]: r for r in ddb.conn.execute("SELECT * FROM files WHERE origin='drive'")}
    assert set(rows) == {"return-2023.pdf", "Trip plan", "budget"}
    pdf = rows["return-2023.pdf"]
    assert pdf["location"] == "My Drive/Taxes"
    assert pdf["link"].startswith("https://drive.google.com/")
    assert pdf["md5"] and pdf["path"] == drive_key(pdf["remote_id"])
    assert rows["Trip plan"]["kind"] == "document"
    assert rows["budget"]["kind"] == "text"
    assert ddb.get_meta("drive_page_token") == str(len(fake.changes))


def test_sensitive_and_excluded_drive_files_are_skipped(fake, make_cfg, embedder, tmp_path):
    cfg = make_cfg(drive={"enabled": True, "exclude": ["/Old/**"]})
    db = Database(cfg.db_path)
    keys = fake.folder(".ssh")
    old = fake.folder("Old")
    fake.add("id_rsa.txt", b"-----BEGIN KEY-----", parent=keys.id)
    fake.add(".env", b"SECRET=1")
    fake.add("server.pem", b"-----BEGIN CERTIFICATE-----")
    fake.add("letter.txt", b"dear landlord", parent=old.id)
    fake.add("keep.txt", b"keep me")
    DriveIndexer(cfg, db, embedder, fake.client()).sync()
    assert [r["name"] for r in db.conn.execute("SELECT name FROM files")] == ["keep.txt"]
    db.close()


def test_folders_option_limits_scope(fake, make_cfg, embedder):
    cfg = make_cfg(drive={"enabled": True, "folders": ["Work/Reports"], "include_shared": True})
    db = Database(cfg.db_path)
    work = fake.folder("Work")
    reports = fake.folder("Reports", parent=work.id)
    fake.add("q1.txt", b"quarterly report", parent=reports.id)
    fake.add("todo.txt", b"buy milk", parent=work.id)
    fake.add("shared.txt", b"shared with me", owned=False, parent="someone-elses-folder")
    DriveIndexer(cfg, db, embedder, fake.client()).sync()
    assert [r["name"] for r in db.conn.execute("SELECT name FROM files")] == ["q1.txt"]
    db.close()


def test_photos_use_drive_preview_and_drive_metadata(fake, dindexer, ddb, tmp_path, captioner):
    fake.add(
        "IMG_0001.jpg",
        b"\xff\xd8" + b"x" * 5000,  # the full file is never read
        photo={"time": "2023:07:14 18:02:11", "cameraMake": "Pixel", "cameraModel": "7"},
        thumbnail=photo_bytes(tmp_path),
    )
    dindexer.sync()
    row = ddb.conn.execute("SELECT * FROM files").fetchone()
    assert fake.downloads == [] and len(fake.thumbnails) == 1
    assert row["captioned"] == 1 and captioner.calls == 1
    assert "taken: 2023-07-14" in row["details"] and "camera: Pixel 7" in row["details"]


def test_preview_with_its_own_exif_is_not_described_twice(fake, dindexer, ddb, tmp_path):
    """Found on a real Drive: previews can keep EXIF, which Drive also reports."""
    from samples import photo as make_photo

    preview = make_photo(
        tmp_path / "src" / "exif.jpg", exif={306: "2026:11:26 22:22:53"}
    )  # DateTime
    fake.add(
        "IMG_1.jpg",
        b"x",
        photo={"time": "2026:11:26 22:22:53", "cameraMake": "motorola", "cameraModel": "edge"},
        thumbnail=preview.read_bytes(),
    )
    dindexer.sync()
    details = ddb.conn.execute("SELECT details FROM files").fetchone()[0]
    assert details.count("taken:") == 1 and "camera: motorola edge" in details


def test_redirected_preview_falls_back_to_download(fake, dindexer, ddb, tmp_path, monkeypatch):
    f = fake.add("pic.jpg", photo_bytes(tmp_path), thumbnail=b"")
    original = fake.handle

    def redirecting(request):
        if request.url.host == "lh3.googleusercontent.com":
            return httpx.Response(302, headers={"Location": "https://example.com/"})
        return original(request)

    monkeypatch.setattr(fake, "handle", redirecting)
    dindexer.client.http = httpx.Client(transport=httpx.MockTransport(redirecting))
    dindexer.sync()
    assert fake.downloads == [f.id]
    assert ddb.conn.execute("SELECT status FROM files").fetchone()[0] == "ok"


def test_photo_without_preview_is_downloaded(fake, dindexer, ddb, tmp_path):
    f = fake.add("pic.jpg", photo_bytes(tmp_path))
    dindexer.sync()
    assert fake.downloads == [f.id]
    assert ddb.conn.execute("SELECT status FROM files").fetchone()[0] == "ok"


def test_file_already_on_this_computer_is_not_downloaded(
    fake, dcfg, ddb, embedder, captioner, home, tmp_path
):
    from filesonar.indexer import Indexer

    content = pdf_bytes(tmp_path, ["Lease agreement for apartment 4B"])
    (home / "lease.pdf").write_bytes(content)
    Indexer(dcfg, ddb, embedder, captioner).run()
    fake.add("lease (scan).pdf", content)
    calls = embedder.calls

    report = DriveIndexer(dcfg, ddb, embedder, fake.client(), captioner).sync()

    assert report.added == 1 and fake.downloads == [] and embedder.calls == calls
    hits = Searcher(ddb, embedder).search("lease apartment")
    assert len(hits) == 1  # one result for both copies
    assert hits[0].source == "local"
    assert hits[0].other_copies == ["Google Drive: My Drive/lease (scan).pdf"]


def test_second_sync_without_changes_does_nothing(fake, dindexer):
    fake.add("a.txt", b"alpha")
    dindexer.sync()
    fake.calls.clear()
    fake.downloads.clear()
    report = dindexer.sync()
    assert report.scanned == 0 and fake.downloads == []
    assert ("GET", "/drive/v3/changes") in fake.calls
    assert ("GET", "/drive/v3/files") not in fake.calls  # no re-listing


def test_changes_are_applied(fake, dindexer, ddb, tmp_path):
    a = fake.add("a.txt", b"alpha version one")
    b = fake.add("b.txt", b"bravo")
    c = fake.add("c.txt", b"charlie")
    dindexer.sync()
    fake.downloads.clear()

    fake.touch(a.id, content=b"alpha version two", modified="2024-06-01T00:00:00Z")
    fake.touch(b.id, name="bravo-renamed.txt")
    fake.touch(c.id, trashed=True)
    report = dindexer.sync()

    assert fake.downloads == [a.id]  # the rename needed no download
    assert report.updated == 1 and report.moved == 1 and report.removed == 1
    names = {r["name"] for r in ddb.conn.execute("SELECT name FROM files")}
    assert names == {"a.txt", "bravo-renamed.txt"}
    hits = Searcher(ddb, dindexer.local.embedder).search("alpha two")
    assert "version two" in hits[0].snippet


def test_deleted_file_is_removed(fake, dindexer, ddb):
    a = fake.add("a.txt", b"alpha")
    dindexer.sync()
    fake.delete(a.id)
    assert dindexer.sync().removed == 1
    assert ddb.all_paths("drive") == []


def test_folder_rename_updates_locations(fake, dindexer, ddb):
    folder = fake.folder("Taxes")
    fake.add("a.txt", b"alpha", parent=folder.id)
    dindexer.sync()
    fake.downloads.clear()
    fake.touch(folder.id, name="Taxes 2024")
    dindexer.sync()
    assert ddb.conn.execute("SELECT location FROM files").fetchone()[0] == "My Drive/Taxes 2024"
    assert fake.downloads == []


def test_download_failure_is_reported_and_retried(fake, dindexer, ddb):
    f = fake.add("a.txt", b"alpha")
    fake.broken.add(f.id)
    report = dindexer.sync()
    assert report.failed == 1 and ddb.all_paths("drive") == []
    fake.broken.clear()
    assert dindexer.sync(full=True).added == 1


def test_local_scan_keeps_drive_entries(fake, dcfg, ddb, embedder, dindexer, home):
    from filesonar.indexer import Indexer

    fake.add("a.txt", b"alpha")
    dindexer.sync()
    write(home / "b.txt", "bravo")
    Indexer(dcfg, ddb, embedder).run()
    assert len(ddb.all_paths("drive")) == 1 and len(ddb.all_paths()) == 1


def test_search_filters_and_drive_fields(fake, dindexer, ddb, dcfg, embedder, home):
    from filesonar.indexer import Indexer

    folder = fake.folder("Recipes")
    fake.add("soup.txt", b"tomato soup with basil", parent=folder.id)
    write(home / "soup-local.txt", "tomato soup without basil")
    dindexer.sync()
    Indexer(dcfg, ddb, embedder).run()
    searcher = Searcher(ddb, embedder)
    drive_hits = searcher.search("tomato soup", source="drive")
    assert [h.name for h in drive_hits] == ["soup.txt"]
    data = drive_hits[0].to_dict()
    assert data["source"] == "drive" and data["location"] == "My Drive/Recipes"
    assert data["link"].startswith("https://drive.google.com/")
    assert [h.name for h in searcher.search("tomato soup", source="local")] == ["soup-local.txt"]
    assert [h.name for h in searcher.search("soup", folder="Recipes")] == ["soup.txt"]
    with pytest.raises(ValueError, match="source"):
        searcher.search("soup", source="dropbox")


# -- MCP tools ---------------------------------------------------------------


@pytest.fixture
def dserver(dcfg, ddb, embedder, fake):
    client = fake.client()
    return build_server(dcfg, ddb, embedder, drive=lambda: client)


def test_share_file_is_listed_only_with_drive(dserver, cfg, db, embedder):
    tools = {t.name: t for t in asyncio.run(dserver.list_tools())}
    share = tools["share_file"].annotations
    assert share.open_world_hint is True and share.destructive_hint is False
    plain = build_server(cfg, db, embedder)
    assert "share_file" not in {t.name for t in asyncio.run(plain.list_tools())}


def test_prepare_drive_file(dserver, dindexer, fake, dcfg):
    f = fake.add("a.txt", b"alpha")
    dindexer.sync()
    out = call(dserver, "prepare_file", {"path": drive_key(f.id)})
    staged = Path(out["path"])
    assert staged.is_relative_to(dcfg.outbox_dir) and staged.read_bytes() == b"alpha"


def test_prepare_google_doc_as_pdf(dserver, dindexer, fake):
    f = fake.add(
        "Trip plan",
        mime="application/vnd.google-apps.document",
        export={"text/plain": b"Lisbon", "application/pdf": b"%PDF-1.4 trip"},
    )
    dindexer.sync()
    out = call(dserver, "prepare_file", {"path": drive_key(f.id)})
    assert out["name"] == "Trip plan.pdf"
    assert Path(out["path"]).read_bytes() == b"%PDF-1.4 trip"


def test_prepare_drive_file_prefers_local_copy(dserver, dindexer, fake, dcfg, ddb, embedder, home):
    from filesonar.indexer import Indexer

    (home / "a.txt").write_bytes(b"alpha")
    Indexer(dcfg, ddb, embedder).run()
    f = fake.add("a.txt", b"alpha")
    dindexer.sync()
    fake.downloads.clear()
    call(dserver, "prepare_file", {"path": drive_key(f.id)})
    assert fake.downloads == []


def test_share_file_reuses_existing_drive_copy(dserver, dindexer, fake, dcfg, ddb, embedder, home):
    from filesonar.indexer import Indexer

    (home / "lease.txt").write_bytes(b"lease for apartment 4B")
    Indexer(dcfg, ddb, embedder).run()
    f = fake.add("lease.txt", b"lease for apartment 4B")
    dindexer.sync()
    out = call(dserver, "share_file", {"path": str(home / "lease.txt")})
    assert out["uploaded"] is False and f.id in out["link"]
    assert fake.uploads == []


def test_share_file_uploads_once_into_its_folder(dserver, fake, dcfg, ddb, embedder, home):
    from filesonar.indexer import Indexer

    (home / "report.txt").write_bytes(b"annual report")
    (home / "other.txt").write_bytes(b"other report")
    Indexer(dcfg, ddb, embedder).run()
    out = call(dserver, "share_file", {"path": str(home / "report.txt")})
    assert out["uploaded"] is True and len(fake.uploads) == 1
    uploaded = fake.files[fake.uploads[0]]
    folder = fake.files[uploaded.parents[0]]
    assert folder.name == "filesonar" and uploaded.content == b"annual report"
    call(dserver, "share_file", {"path": str(home / "other.txt")})
    assert fake.files[fake.uploads[1]].parents == [folder.id]  # same folder reused
    # The folder was deleted in Drive: a new one is made.
    fake.touch(folder.id, trashed=True)
    (home / "third.txt").write_bytes(b"third")
    Indexer(dcfg, ddb, embedder).run()
    call(dserver, "share_file", {"path": str(home / "third.txt")})
    assert fake.files[fake.uploads[2]].parents != [folder.id]


def test_share_file_twice_before_a_sync_uploads_once(dserver, fake, dcfg, ddb, embedder, home):
    """Found on a real Drive: asking again before the next sync uploaded a duplicate."""
    from filesonar.indexer import Indexer

    (home / "report.txt").write_bytes(b"annual report")
    Indexer(dcfg, ddb, embedder).run()
    first = call(dserver, "share_file", {"path": str(home / "report.txt")})
    second = call(dserver, "share_file", {"path": str(home / "report.txt")})
    assert first["uploaded"] is True and second["uploaded"] is False
    assert second["link"] == first["link"] and len(fake.uploads) == 1


def test_share_file_refuses_unindexed_paths(dserver, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    with pytest.raises(ToolError, match="not in the index"):
        asyncio.run(dserver.call_tool("share_file", {"path": str(secret)}))


def test_drive_not_signed_in_is_a_clear_tool_error(dcfg, ddb, embedder, home):
    from filesonar.indexer import Indexer

    (home / "a.txt").write_bytes(b"alpha")
    Indexer(dcfg, ddb, embedder).run()

    def not_signed_in():
        raise NotSignedIn()

    server = build_server(dcfg, ddb, embedder, drive=not_signed_in)
    with pytest.raises(ToolError, match="drive login"):
        asyncio.run(server.call_tool("share_file", {"path": str(home / "a.txt")}))


# -- client and auth ---------------------------------------------------------


def test_token_is_refreshed_and_retried(fake):
    client = fake.client()
    assert client.account() == "me@example.com"
    assert fake.access_tokens == 1
    client.token.expires_at = 0  # expired: refreshed before the next call
    client.account()
    assert fake.access_tokens == 2
    fake.expire_access = True  # server-side revocation of the access token
    client.token.access_token = "access-old"
    client.account()
    assert fake.access_tokens == 3


def test_revoked_access_gives_a_clear_error(fake):
    fake.revoked = True
    with pytest.raises(DriveError, match="drive login"):
        fake.client().account()


def test_rate_limits_are_retried(fake):
    fake.fail_next = [429, 503]
    assert fake.client().account() == "me@example.com"
    fake.fail_next = [503] * 10
    with pytest.raises(DriveError, match="not responding"):
        fake.client().account()


def test_download_size_limit(fake):
    f = fake.add("big.txt", b"x" * 5000)
    client = fake.client()
    assert client.download(f.id, 5000) == b"x" * 5000
    with pytest.raises(DriveError, match="size limit"):
        client.download(f.id, 4999)


def test_thumbnail_links_must_be_google(fake):
    assert is_thumbnail_url("https://lh3.googleusercontent.com/abc=s220")
    for bad in (
        "http://lh3.googleusercontent.com/abc",
        "https://evil.example/googleusercontent.com",
        "https://googleusercontent.com.evil.example/x",
        "file:///etc/passwd",
    ):
        assert not is_thumbnail_url(bad)
        with pytest.raises(DriveError):
            fake.client().thumbnail(bad, 1600, 1000)


def test_token_file_is_private(tmp_path):
    path = tmp_path / "data" / "drive-token.json"
    save_token(path, Token("id", "secret", "refresh", account="me@example.com"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_token(path).refresh_token == "refresh"
    assert "access" not in path.read_text()  # short-lived tokens are not stored
    os.chmod(path, 0o644)
    with pytest.raises(DriveError, match="chmod 600"):
        load_token(path)
    with pytest.raises(NotSignedIn):
        load_token(tmp_path / "missing.json")


def test_client_secret_file(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"installed": {"client_id": "a", "client_secret": "b"}}))
    assert read_client_secret(good) == ("a", "b")
    web = tmp_path / "web.json"
    web.write_text(json.dumps({"web": {"client_id": "a", "client_secret": "b"}}))
    with pytest.raises(DriveError, match="Desktop app"):
        read_client_secret(web)


def _browser(response: dict | None = None, tamper_state: bool = False):
    """Pretend to be the user's browser: follow the sign-in link back to the redirect."""
    seen: dict = {}

    def open_url(url: str) -> bool:
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        seen.update(params)
        back = dict(response or {"code": "auth-code"})
        back["state"] = "forged" if tamper_state else params["state"]
        target = params["redirect_uri"] + "?" + urllib.parse.urlencode(back)
        threading.Thread(target=lambda: urllib.request.urlopen(target).read()).start()
        return True

    return open_url, seen


def _token_endpoint(scope: str | None = None, seen: dict | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        if seen is not None:
            seen.update(form)
        return httpx.Response(
            200,
            json={
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
                "scope": scope if scope is not None else " ".join(drive_mod.SCOPES),
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_login_flow(monkeypatch):
    import base64
    import hashlib

    open_url, auth_params = _browser()
    monkeypatch.setattr(drive_mod.webbrowser, "open", open_url)
    form: dict = {}
    token = drive_mod.login(
        "cid", "csecret", show=lambda _m: None, timeout=10, http_client=_token_endpoint(seen=form)
    )
    assert token.refresh_token == "refresh-1"
    assert auth_params["redirect_uri"].startswith("http://127.0.0.1:")
    assert auth_params["code_challenge_method"] == "S256"
    # The verifier sent with the code matches the challenge sent to the browser.
    digest = hashlib.sha256(form["code_verifier"].encode()).digest()
    assert base64.urlsafe_b64encode(digest).decode().rstrip("=") == auth_params["code_challenge"]
    assert form["code"] == "auth-code" and form["redirect_uri"] == auth_params["redirect_uri"]


def test_login_rejects_forged_state(monkeypatch):
    open_url, _ = _browser(tamper_state=True)
    monkeypatch.setattr(drive_mod.webbrowser, "open", open_url)
    with pytest.raises(DriveError, match="did not match"):
        drive_mod.login("cid", "s", show=lambda _m: None, timeout=10, http_client=_token_endpoint())


def test_login_denied_or_partial(monkeypatch):
    open_url, _ = _browser({"error": "access_denied"})
    monkeypatch.setattr(drive_mod.webbrowser, "open", open_url)
    with pytest.raises(DriveError, match="access_denied"):
        drive_mod.login("cid", "s", show=lambda _m: None, timeout=10, http_client=_token_endpoint())
    open_url, _ = _browser()
    monkeypatch.setattr(drive_mod.webbrowser, "open", open_url)
    readonly_only = drive_mod.SCOPES[0]
    with pytest.raises(DriveError, match=r"drive\.file"):
        drive_mod.login(
            "cid", "s", show=lambda _m: None, timeout=10, http_client=_token_endpoint(readonly_only)
        )


# -- storage -----------------------------------------------------------------


def test_v1_index_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema_version', '1');
        CREATE TABLE files (
            id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
            kind TEXT NOT NULL, size INTEGER NOT NULL, mtime REAL NOT NULL,
            sha256 TEXT NOT NULL, captioned INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
            error TEXT, details TEXT, indexed_at REAL NOT NULL);
        INSERT INTO files VALUES (1, '/x/a.txt', 'a.txt', 'text', 1, 1.0, 'abc', 0, 'ok',
                                  NULL, NULL, 1.0);
        """
    )
    conn.commit()
    conn.close()
    db = Database(path)
    row = db.get_file("/x/a.txt")
    assert row["origin"] == "local" and row["md5"] is None
    assert db.get_meta("schema_version") == "2"
    db.close()
    Database(path).close()  # opening again is a no-op


def test_local_md5_is_backfilled(cfg, db, embedder, home):
    from filesonar.indexer import Indexer

    write(home / "a.txt", "alpha")
    Indexer(cfg, db, embedder).run()
    db.conn.execute("UPDATE files SET md5 = NULL")
    db.conn.commit()
    calls = embedder.calls
    report = Indexer(cfg, db, embedder).run()
    assert report.unchanged == 1 and embedder.calls == calls  # hashed, not re-indexed
    assert db.conn.execute("SELECT md5 FROM files").fetchone()[0] is not None


# -- command line ------------------------------------------------------------


@pytest.fixture
def drive_cli(tmp_path, home, fake, monkeypatch):
    from conftest import FakeCaptioner, FakeEmbedder

    from filesonar import cli

    config = tmp_path / "config.toml"
    config.write_text(
        f'roots = ["{home}"]\ndata_dir = "{tmp_path / "data"}"\n'
        f'outbox_dir = "{tmp_path / "outbox"}"\n[drive]\nenabled = true\n'
    )
    monkeypatch.setattr(cli, "OllamaEmbedder", lambda client: FakeEmbedder())
    monkeypatch.setattr(cli, "OllamaCaptioner", lambda client: FakeCaptioner())
    monkeypatch.setattr(DriveClient, "from_data_dir", classmethod(lambda cls, d: fake.client()))

    def run(*args):
        return cli.main(["-c", str(config), *args])

    return run


def test_cli_drive_sync_status_logout(drive_cli, fake, tmp_path, capsys):
    fake.add("a.txt", b"alpha drive file")
    save_token(tmp_path / "data" / "drive-token.json", Token("c", "s", "refresh-1", "me@x"))
    assert drive_cli("drive", "sync") == 0
    assert "1 added" in capsys.readouterr().out
    assert drive_cli("search", "alpha", "--source", "drive") == 0
    assert "Google Drive: My Drive/a.txt" in capsys.readouterr().out
    assert drive_cli("drive", "status") == 0
    assert "indexed:     1 files" in capsys.readouterr().out
    assert drive_cli("drive", "logout") == 0
    out = capsys.readouterr().out
    assert "1 Drive entries removed" in out and fake.revoked
    assert not (tmp_path / "data" / "drive-token.json").exists()


def test_cli_index_includes_drive(drive_cli, fake, home, capsys):
    fake.add("a.txt", b"alpha drive file")
    write(home / "b.txt", "bravo local file")
    assert drive_cli("index") == 0
    out = capsys.readouterr().out
    assert "Google Drive: " in out and "1 added" in out


def test_office_lock_files_in_drive_are_skipped(fake, dindexer, ddb):
    fake.add("~$notated Bibliography.docx", b"owner file")
    fake.add("bibliography.txt", b"annotated bibliography")
    dindexer.sync()
    assert [r["name"] for r in ddb.conn.execute("SELECT name FROM files")] == ["bibliography.txt"]
