"""Getting files to the user: sizes that fit a chat, Drive links for large videos, and
Drive files saved onto the computer."""

import asyncio
import json
import os
import stat

import pytest
from fakedrive import FakeDrive
from mcp.server.mcpserver.exceptions import ToolError

from filesonar.db import Database
from filesonar.drive_index import KEY_PREFIX, DriveIndexer
from filesonar.server import build_server


def call(server, name, args):
    return json.loads(asyncio.run(server.call_tool(name, args)).content[0].text)


def call_error(server, name, args):
    with pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool(name, args))
    return str(exc.value)


@pytest.fixture
def fake():
    return FakeDrive()


@pytest.fixture
def setup(make_cfg, tmp_path, embedder, captioner, fake):
    cfg = make_cfg(
        drive={"enabled": True},
        index={"max_file_size_mb": 1},
        outbox_max_mb=2,
        downloads_dir=str(tmp_path / "Downloads"),
    )
    db = Database(cfg.db_path)
    client = fake.client()
    server = build_server(cfg, db, embedder, drive=lambda: client)
    drive = DriveIndexer(cfg, db, embedder, fake.client(), captioner)
    yield cfg, db, server, drive
    db.close()


def add_local_video(db, path, size):
    """An indexed recording, as the indexer would store it (no ffmpeg needed here)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    st = path.stat()
    db.ensure_vector_table("fake-embed", 128)
    db.replace_file(
        path=str(path), name=path.name, kind="video", size=st.st_size, mtime=st.st_mtime,
        sha256="0" * 64, captioned=True, status="ok", error=None, details="duration: 1:00",
        indexed_at=0.0, chunks=[(None, "caption", "a birthday party")], vectors=[[1.0] * 128],
    )  # fmt: skip


def test_prepare_file_says_when_a_video_is_too_big_for_chat(setup, home):
    cfg, db, server, _ = setup
    add_local_video(db, home / "party.mp4", 3 * 1024 * 1024)
    msg = call_error(server, "prepare_file", {"path": str(home / "party.mp4")})
    assert "more than chat apps send (2 MB)" in msg and "share_file" in msg
    assert not cfg.outbox_dir.exists() or not any(cfg.outbox_dir.iterdir())


def test_small_video_is_prepared_even_above_the_index_size_limit(setup, home):
    _, db, server, _ = setup
    add_local_video(db, home / "clip.mp4", 1_500_000)  # > max_file_size_mb, < outbox_max_mb
    out = call(server, "prepare_file", {"path": str(home / "clip.mp4")})
    assert out["size_bytes"] == 1_500_000


def test_share_file_streams_a_large_video(setup, home, fake):
    _, db, server, _ = setup
    video = home / "wedding.mp4"
    add_local_video(db, video, 5 * 1024 * 1024)  # far above max_file_size_mb
    fake.fail_uploads = 1  # one server error: the retry must send the whole file again
    out = call(server, "share_file", {"path": str(video)})
    assert out["uploaded"] is True
    assert fake.files[fake.uploads[-1]].content == video.read_bytes()


def test_drive_video_too_big_for_chat_points_to_its_link(setup, fake):
    _, _, server, drive = setup
    f = fake.add("trip.mp4", b"x" * 10, size=str(300 * 1024 * 1024))
    drive.sync()
    msg = call_error(server, "prepare_file", {"path": KEY_PREFIX + f.id})
    assert "Google Drive link" in msg and f"/file/d/{f.id}/" in msg


# -- save_to_computer ------------------------------------------------------------------


def test_save_to_computer_downloads_without_overwriting(setup, fake, tmp_path):
    _, _, server, drive = setup
    f = fake.add("notes.txt", b"from drive")
    drive.sync()
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "notes.txt").write_text("my own file")

    out = call(server, "save_to_computer", {"path": KEY_PREFIX + f.id})
    saved = downloads / "notes (1).txt"
    assert out["downloaded"] is True and out["path"] == str(saved)
    assert saved.read_bytes() == b"from drive"
    assert (downloads / "notes.txt").read_text() == "my own file"  # untouched
    assert not [p for p in downloads.iterdir() if p.name.startswith(".filesonar-")]
    from filesonar.server import _umask

    assert stat.S_IMODE(saved.stat().st_mode) == 0o644 & ~_umask()  # like any download

    again = call(server, "save_to_computer", {"path": KEY_PREFIX + f.id})
    assert again["path"].endswith("notes (2).txt")


def test_save_google_doc_as_pdf(setup, fake, tmp_path):
    _, _, server, drive = setup
    f = fake.add(
        "Trip plan",
        mime="application/vnd.google-apps.document",
        export={"text/plain": b"Lisbon", "application/pdf": b"%PDF-1.4 trip"},
    )
    drive.sync()
    out = call(server, "save_to_computer", {"path": KEY_PREFIX + f.id})
    assert out["path"].endswith("Trip plan.pdf")
    assert (tmp_path / "Downloads" / "Trip plan.pdf").read_bytes() == b"%PDF-1.4 trip"


def test_save_returns_the_copy_already_on_this_computer(setup, fake, home, embedder):
    from filesonar.indexer import Indexer

    cfg, db, server, drive = setup
    (home / "lease.txt").write_bytes(b"lease for apartment 4B")
    Indexer(cfg, db, embedder).run()
    f = fake.add("lease.txt", b"lease for apartment 4B")
    drive.sync()
    out = call(server, "save_to_computer", {"path": KEY_PREFIX + f.id})
    assert out == {
        "path": str(home / "lease.txt"),
        "downloaded": False,
        "note": "The same file is already on this computer.",
    }
    assert f.id not in fake.downloads


def test_save_refuses_names_that_try_to_escape(setup, fake, tmp_path):
    _, _, server, drive = setup
    f = fake.add("evil.txt", b"x")
    drive.sync()
    fake.files[f.id].name = "../../.bashrc"  # renamed in Drive after indexing
    out = call(server, "save_to_computer", {"path": KEY_PREFIX + f.id})
    saved = out["path"]
    assert os.path.dirname(saved) == str(tmp_path / "Downloads")
    assert not (tmp_path / ".bashrc").exists()


def test_save_to_computer_is_offered_only_with_drive_and_needs_no_approval(
    setup, cfg, db, embedder
):
    _, _, server, _ = setup
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    notes = tools["save_to_computer"].annotations
    assert notes.destructive_hint is False and notes.open_world_hint is False
    plain = build_server(cfg, db, embedder)
    assert "save_to_computer" not in {t.name for t in asyncio.run(plain.list_tools())}
