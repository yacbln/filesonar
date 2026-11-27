from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from filesonar.config import config_from_dict
from filesonar.db import Database
from filesonar.indexer import Indexer
from filesonar.models import ModelError

DIM = 128


@pytest.fixture(autouse=True)
def _no_real_servers(request, monkeypatch):
    """Tests talk to fakes only. A request to a real server (the Ollama on the developer's
    machine, Google) would make results depend on that machine, or leak a test request
    out. Local test servers on other ports (the OAuth redirect) stay reachable; tests
    marked `ollama` use the real model server on purpose."""
    if request.node.get_closest_marker("ollama"):
        return
    real = httpx.HTTPTransport.handle_request

    def guarded(self, req):
        host, port = req.url.host, req.url.port
        if host not in ("127.0.0.1", "localhost", "::1") or port == 11434:
            raise httpx.ConnectError(f"tests must not reach a real server: {req.url}")
        return real(self, req)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded)


needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract is not installed"
)


class FakeEmbedder:
    """Deterministic bag-of-words embedding: texts that share words are close."""

    name = "fake-embed"

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []
        self.fail = False

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v] if any(v) else [1.0 / math.sqrt(DIM)] * DIM

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise ModelError("model server down")
        self.calls += 1
        self.texts.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise ModelError("model server down")
        return self._vec(text)


class FakeCaptioner:
    name = "fake-vision"

    def __init__(
        self, caption: str = "a photo of a red bicycle parked by a lake", available: bool = True
    ) -> None:
        self.text = caption
        self.is_available = available
        self.calls = 0
        self.fail = False

    def available(self) -> bool:
        return self.is_available

    def caption(self, image_png: bytes) -> str:
        self.calls += 1
        assert image_png.startswith(b"\x89PNG")
        if self.fail:
            raise ModelError("vision model crashed")
        return self.text


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "files"
    root.mkdir()
    return root


@pytest.fixture
def make_cfg(tmp_path: Path, home: Path):
    def build(**overrides):
        raw = {
            "roots": [str(home)],
            "data_dir": str(tmp_path / "data"),
            "outbox_dir": str(tmp_path / "outbox"),
        }
        raw.update(overrides)
        return config_from_dict(raw, base=tmp_path)

    return build


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


@pytest.fixture
def db(cfg):
    database = Database(cfg.db_path)
    yield database
    database.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def captioner() -> FakeCaptioner:
    return FakeCaptioner()


@pytest.fixture
def indexer(cfg, db, embedder, captioner) -> Indexer:
    return Indexer(cfg, db, embedder, captioner)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
