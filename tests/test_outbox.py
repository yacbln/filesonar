import os
import time

import pytest

from filesonar.config import ConfigError, config_from_dict
from filesonar.outbox import KEEP_HOURS, OutboxError, clean, safe_name, stage

SHA = "ab" * 32


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("photo.jpg", "photo.jpg"),
        ("../../etc/passwd", ".._.._etc_passwd"),
        ("a\\b.pdf", "a_b.pdf"),
        ("bad\nname.png", "bad_name.png"),
        ("..", "file"),
        ("", "file"),
    ],
)
def test_safe_name(name, expected):
    assert safe_name(name) == expected


def test_safe_name_keeps_extension_when_trimming():
    name = "x" * 300 + ".pdf"
    out = safe_name(name)
    assert out.endswith(".pdf")
    assert len(out.encode()) <= 200


def test_stage_writes_private_copy(tmp_path):
    outbox = tmp_path / "outbox"
    staged = stage(outbox, "report.pdf", SHA, b"%PDF-1.4 data")
    assert staged.path == outbox / SHA[:16] / "report.pdf"
    assert staged.path.read_bytes() == b"%PDF-1.4 data"
    assert not staged.reused
    assert outbox.stat().st_mode & 0o777 == 0o700
    assert stage(outbox, "report.pdf", SHA, b"%PDF-1.4 data").reused
    assert [p.name for p in staged.path.parent.iterdir()] == ["report.pdf"]  # no temp files


def test_stage_never_writes_through_a_symlink(tmp_path):
    outbox = tmp_path / "outbox"
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    entry = outbox / SHA[:16]
    entry.mkdir(parents=True)
    (entry / "report.pdf").symlink_to(victim)
    staged = stage(outbox, "report.pdf", SHA, b"new")
    assert victim.read_text() == "keep me"
    assert not staged.path.is_symlink()
    assert staged.path.read_bytes() == b"new"


def test_stage_refuses_symlinked_entry_folder(tmp_path):
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (outbox / SHA[:16]).symlink_to(elsewhere)
    with pytest.raises(OutboxError):
        stage(outbox, "report.pdf", SHA, b"data")
    assert list(elsewhere.iterdir()) == []


def test_stage_rejects_bad_hash(tmp_path):
    with pytest.raises(OutboxError):
        stage(tmp_path, "x.txt", "../../x", b"data")


def test_clean_removes_only_old_entries_it_created(tmp_path):
    outbox = tmp_path / "outbox"
    old = stage(outbox, "old.txt", "0" * 64, b"old").path.parent
    new = stage(outbox, "new.txt", "1" * 64, b"new").path.parent
    unrelated = outbox / "keep-me"
    unrelated.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "file").write_text("x")
    link = outbox / ("2" * 16)
    link.symlink_to(elsewhere)
    past = time.time() - (KEEP_HOURS + 1) * 3600
    for path in (old, unrelated):
        os.utime(path, (past, past))
    os.utime(link, (past, past), follow_symlinks=False)

    assert clean(outbox) == 1
    assert not old.exists()
    assert new.exists() and unrelated.exists() and link.is_symlink()
    assert (elsewhere / "file").exists()


def test_clean_missing_outbox(tmp_path):
    assert clean(tmp_path / "nope") == 0


def test_outbox_inside_a_root_is_not_indexed(tmp_path):
    cfg = config_from_dict(
        {"roots": [str(tmp_path)], "outbox_dir": str(tmp_path / "outbox")}, base=tmp_path
    )
    assert cfg.excluded(tmp_path / "outbox/abc/photo.jpg", tmp_path)
    assert not cfg.excluded(tmp_path / "photo.jpg", tmp_path)


def test_outbox_may_not_contain_a_root(tmp_path):
    with pytest.raises(ConfigError, match="outbox_dir"):
        config_from_dict(
            {"roots": [str(tmp_path / "docs")], "outbox_dir": str(tmp_path)}, base=tmp_path
        )


def test_default_outbox_follows_openclaw_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "state"))
    cfg = config_from_dict({"roots": [str(tmp_path / "docs")]}, base=tmp_path)
    assert cfg.outbox_dir == (tmp_path / "state/media/filesonar").resolve()
