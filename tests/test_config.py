from pathlib import Path

import pytest

from filesonar.config import (
    ALWAYS_EXCLUDED,
    ConfigError,
    config_from_dict,
    default_config_path,
    is_loopback_url,
    load_config,
)


def test_minimal_config_uses_safe_defaults(tmp_path):
    cfg = config_from_dict({"roots": [str(tmp_path)]}, base=tmp_path)
    assert cfg.roots == (tmp_path.resolve(),)
    assert cfg.models.host == "http://127.0.0.1:11434"
    assert not cfg.models.allow_remote_host
    assert cfg.excluded(tmp_path.resolve() / ".ssh/id_rsa", tmp_path.resolve())
    assert not cfg.excluded(tmp_path.resolve() / "notes.txt", tmp_path.resolve())
    assert ALWAYS_EXCLUDED


def test_roots_are_expanded_resolved_and_deduplicated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "docs").mkdir()
    cfg = config_from_dict({"roots": ["~/docs", str(tmp_path / "docs"), "docs/."]}, base=tmp_path)
    assert cfg.roots == ((tmp_path / "docs").resolve(),)


def test_relative_root_is_relative_to_config_file(tmp_path):
    cfg = config_from_dict({"roots": ["photos"]}, base=tmp_path)
    assert cfg.roots == ((tmp_path / "photos").resolve(),)


def test_no_roots_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="no folders"):
        config_from_dict({}, base=tmp_path)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"roots": "~/Documents"}, "list"),
        ({"roots": ["x"], "bogus": 1}, "unknown top-level"),
        ({"roots": ["x"], "models": {"nope": 1}}, "unknown option"),
        ({"roots": ["x"], "models": {"caption_images": "yes"}}, "wrong type"),
        ({"roots": ["x"], "index": {"max_pdf_pages": True}}, "boolean"),
        ({"roots": ["x"], "index": {"max_file_size_mb": -1}}, "positive"),
        ({"roots": ["x"], "index": {"chunk_chars": 50}}, "chunk_chars"),
        ({"roots": ["x"], "index": {"chunk_overlap": 900}}, "chunk_overlap"),
        ({"roots": ["x"], "exclude": "*.tmp"}, "exclude"),
        ({"roots": ["x"], "models": "fast"}, "table"),
        ({"roots": ["x"], "models": {"host": "file:///etc/passwd"}}, "http"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, raw, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw, base=tmp_path)


def test_integer_accepted_for_float_option(tmp_path):
    cfg = config_from_dict({"roots": ["x"], "index": {"max_file_size_mb": 10}}, base=tmp_path)
    assert cfg.index.max_file_size_mb == 10.0


@pytest.mark.parametrize(
    "host", ["http://192.168.1.20:11434", "http://example.com", "http://127.0.0.1.evil.com:11434"]
)
def test_remote_model_host_needs_explicit_opt_in(tmp_path, host):
    with pytest.raises(ConfigError, match="another machine"):
        config_from_dict({"roots": ["x"], "models": {"host": host}}, base=tmp_path)
    cfg = config_from_dict(
        {"roots": ["x"], "models": {"host": host, "allow_remote_host": True}}, base=tmp_path
    )
    assert cfg.models.host == host


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:11434", True),
        ("http://[::1]:11434", True),
        ("http://127.5.5.5", True),
        ("http://10.0.0.2", False),
        ("http://localhost.example.com", False),
        ("not a url", False),
    ],
)
def test_loopback_detection(url, expected):
    assert is_loopback_url(url) is expected


def test_load_config_errors_are_friendly(tmp_path):
    with pytest.raises(ConfigError, match="filesonar init"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("roots = [")
    with pytest.raises(ConfigError, match=r"bad\.toml"):
        load_config(bad)


def test_relative_xdg_config_home_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/dir")
    assert default_config_path() == Path(tmp_path) / ".config/filesonar/config.toml"


def test_example_config_is_valid(tmp_path, monkeypatch):
    from filesonar.config import EXAMPLE_CONFIG

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    path.write_text(EXAMPLE_CONFIG)
    cfg = load_config(path)
    assert len(cfg.roots) == 3
    assert cfg.source == path
