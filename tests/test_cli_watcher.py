import json
import os
import stat
import threading
import time

import pytest
from conftest import FakeCaptioner, FakeEmbedder, write

from filesonar import cli
from filesonar.indexer import Indexer
from filesonar.watcher import watch


@pytest.fixture
def fake_models(monkeypatch):
    emb = FakeEmbedder()
    monkeypatch.setattr(cli, "OllamaEmbedder", lambda client: emb)
    monkeypatch.setattr(cli, "OllamaCaptioner", lambda client: FakeCaptioner())
    return emb


@pytest.fixture
def config_file(tmp_path, home):
    path = tmp_path / "config.toml"
    path.write_text(f'roots = ["{home}"]\ndata_dir = "{tmp_path / "data"}"\n')
    return path


def test_init_writes_private_config_and_refuses_overwrite(tmp_path, capsys):
    path = tmp_path / "cfg/config.toml"
    assert cli.main(["-c", str(path), "init"]) == 0
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert cli.main(["-c", str(path), "init"]) == 1
    nested = tmp_path / "fresh" / "config.toml"
    assert cli.main(["-c", str(nested), "init"]) == 0
    assert stat.S_IMODE(nested.parent.stat().st_mode) == 0o700
    assert cli.main(["-c", str(path), "init", "--force"]) == 0


def test_index_search_status(config_file, home, fake_models, capsys):
    write(home / "visa/application.txt", "Schengen visa application, appointment on May 3.")
    assert cli.main(["-c", str(config_file), "index"]) == 0
    assert "1 added" in capsys.readouterr().out

    assert cli.main(["-c", str(config_file), "search", "visa", "appointment", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["path"] == str(home / "visa/application.txt")

    assert cli.main(["-c", str(config_file), "search", "visa"]) == 0
    assert "application.txt" in capsys.readouterr().out

    assert cli.main(["-c", str(config_file), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["files"] == 1


def test_errors_are_reported_not_raised(tmp_path, config_file, fake_models, capsys):
    assert cli.main(["-c", str(tmp_path / "nope.toml"), "status"]) == 2
    assert "not found" in capsys.readouterr().err
    assert cli.main(["-c", str(config_file), "search", "x", "--after", "soon"]) == 2
    assert "date" in capsys.readouterr().err


def test_watcher_picks_up_changes(home, cfg, db):
    emb = FakeEmbedder()
    indexer = Indexer(cfg, db, emb)
    stop = threading.Event()
    thread = threading.Thread(
        target=watch, args=(indexer,), kwargs={"quiet_seconds": 0.3, "stop": stop}, daemon=True
    )
    thread.start()
    try:
        time.sleep(0.5)
        f = write(home / "new/receipt.txt", "hardware store receipt")
        _wait(lambda: db.get_file(str(f)) is not None)
        f.rename(home / "new/receipt-2024.txt")
        _wait(
            lambda: (
                db.get_file(str(home / "new/receipt-2024.txt")) is not None
                and db.get_file(str(f)) is None
            )
        )
        (home / "new/receipt-2024.txt").unlink()
        _wait(lambda: db.all_paths() == [])
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive()


def _wait(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def test_doctor_reports_missing_pieces(config_file, monkeypatch, capsys):
    monkeypatch.setattr(cli.OllamaClient, "installed_models", lambda self: set())
    assert cli.main(["-c", str(config_file), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "not reachable" in out and "missing: ollama pull nomic-embed-text" in out

    monkeypatch.setattr(
        cli.OllamaClient, "installed_models", lambda self: {"nomic-embed-text", "qwen2.5vl:7b"}
    )
    assert cli.main(["-c", str(config_file), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "nomic-embed-text ok" in out and "qwen2.5vl:7b ok" in out
    assert not any(line.endswith(" ") for line in out.splitlines())
    assert "GPU memory" not in out


def test_doctor_warns_about_partly_offloaded_model(config_file, monkeypatch, capsys):
    monkeypatch.setattr(
        cli.OllamaClient, "installed_models", lambda self: {"nomic-embed-text", "qwen2.5vl:7b"}
    )
    monkeypatch.setattr(
        cli.OllamaClient,
        "loaded_models",
        lambda self: {"qwen2.5vl:7b": (2_900, 6_200), "nomic-embed-text": (400, 400)},
    )
    cli.main(["-c", str(config_file), "doctor"])
    out = capsys.readouterr().out
    assert "qwen2.5vl:7b is only 46% in GPU memory" in out
    assert "ollama stop qwen2.5vl:7b" in out and "nomic-embed-text is only" not in out


def test_index_progress_is_quiet_by_default(config_file, home, fake_models, capsys):
    for n in range(120):
        write(home / f"notes/{n}.txt", f"note number {n}")
    assert cli.main(["-c", str(config_file), "index"]) == 0
    err = capsys.readouterr().err
    assert err.count("files read") == 2  # at 50 and 100
