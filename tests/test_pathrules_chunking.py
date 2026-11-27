import itertools
from pathlib import Path

import pytest

from filesonar.chunking import chunk_text
from filesonar.config import ALWAYS_EXCLUDED
from filesonar.pathrules import is_hidden, matches_any, within


@pytest.mark.parametrize(
    "path, pattern, expected",
    [
        ("/home/u/.ssh/id_rsa", "**/.ssh/**", True),
        ("/home/u/.ssh", "**/.ssh/**", False),
        ("/home/u/project/.git/config", "**/.git/**", True),
        ("/home/u/notes.git", "**/.git/**", False),
        ("/home/u/app/.env", "**/.env", True),
        ("/home/u/app/.env.local", "**/.env.*", True),
        ("/home/u/app/environment.txt", "**/.env", False),
        ("/home/u/keys/server.PEM", "**/*.pem", True),
        ("/home/u/a/b/c.tmp", "*.tmp", True),
        ("/home/u/a/b/c.tmp", "a/*.tmp", False),
        ("/home/u/a/b/c.tmp", "a/**/*.tmp", True),
        ("/home/u/report-2024.pdf", "report-????.pdf", True),
        ("/home/u/x/report-2024.pdf", "/home/u/*.pdf", False),
    ],
)
def test_glob_rules(path, pattern, expected):
    assert matches_any(Path(path), [pattern]) is expected


@pytest.mark.parametrize(
    "path",
    [
        "/home/u/.ssh/id_ed25519",
        "/home/u/.gnupg/private-keys-v1.d/x.key",
        "/home/u/.password-store/bank.gpg",
        "/home/u/.mozilla/firefox/p/logins.json",
        "/home/u/.config/google-chrome/Default/Login Data",
        "/home/u/code/app/.env",
        "/home/u/Documents/id_rsa.pub",
        "/home/u/Documents/vault.kdbx",
        "/home/u/.aws/credentials",
    ],
)
def test_sensitive_locations_always_excluded(path):
    assert matches_any(Path(path), ALWAYS_EXCLUDED)


def test_hidden_is_relative_to_root():
    root = Path("/home/u/.notes")
    assert not is_hidden(root / "todo.md", root)
    assert is_hidden(root / ".draft" / "todo.md", root)
    assert not is_hidden(Path("/elsewhere/.x"), root)


def test_within():
    roots = [Path("/a/b"), Path("/c")]
    assert within(Path("/a/b/x.txt"), roots) == Path("/a/b")
    assert within(Path("/a/bc/x.txt"), roots) is None
    assert within(Path("/c"), roots) == Path("/c")


# -- chunking ---------------------------------------------------------------


def test_short_text_is_one_chunk():
    assert chunk_text("  hello world  ", 1200, 200) == ["hello world"]
    assert chunk_text("   ", 1200, 200) == []


def test_long_text_is_split_with_overlap_and_full_coverage():
    words = [f"word{i}" for i in range(2000)]
    text = " ".join(words)
    chunks = chunk_text(text, 500, 100)
    assert len(chunks) > 5
    assert all(len(c) <= 500 for c in chunks)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)  # nothing lost
    # consecutive chunks overlap
    for a, b in itertools.pairwise(chunks):
        assert a.split()[-1] in b or b.split()[0] in a


def test_prefers_paragraph_breaks():
    text = ("A" * 300 + ".\n\n") + ("B" * 300)
    chunks = chunk_text(text, 400, 50)
    assert chunks[0].endswith(".")


def test_text_without_spaces_terminates():
    text = "x" * 10_000
    chunks = chunk_text(text, 1000, 200)
    assert 10 <= len(chunks) <= 13
    assert sum(len(c) for c in chunks) >= 10_000
