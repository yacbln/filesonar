"""End-to-end tests against a real Ollama server and the real MCP stdio transport.

Run with: pytest -m ollama
They need `ollama pull embeddinggemma` (and the vision model for the caption test).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from conftest import write
from samples import docx, photo, text_image, text_pdf

from filesonar.config import config_from_dict
from filesonar.db import Database
from filesonar.indexer import Indexer
from filesonar.models import OllamaCaptioner, OllamaClient, OllamaEmbedder
from filesonar.search import Searcher

pytestmark = pytest.mark.ollama


@pytest.fixture
def real(tmp_path, home):
    cfg = config_from_dict(
        {
            "roots": [str(home)],
            "data_dir": str(tmp_path / "data"),
            "outbox_dir": str(tmp_path / "outbox"),
        },
        base=tmp_path,
    )
    client = OllamaClient(cfg.models)
    if cfg.models.embedding_model not in client.installed_models():
        pytest.skip("Ollama or the embedding model is not available")
    db = Database(cfg.db_path)
    yield cfg, db, client
    db.close()


def build_corpus(home):
    text_pdf(
        home / "insurance/auto-policy.pdf",
        ["Car insurance policy. Vehicle: 2019 Toyota Corolla. Premium 640 EUR per year."],
    )
    docx(
        home / "housing/lease.docx",
        [
            "Residential lease agreement for the apartment.",
            "Monthly rent is 950 and the deposit is two months.",
        ],
    )
    write(home / "recipes/couscous.md", "Couscous with lamb and seven vegetables, serves six.")
    write(home / "work/notes.txt", "Meeting notes: migrate the billing service to Postgres.")
    text_image(
        home / "scans/receipt.png",
        ["SUPERMARKET", "Milk 1.20", "Bread 2.10", "Eggs 3.40", "TOTAL 6.70"],
    )


@pytest.mark.parametrize(
    "query, expected",
    [
        ("automobile coverage", "auto-policy.pdf"),  # no shared words
        ("renting a flat", "lease.docx"),
        ("north african dinner", "couscous.md"),
        ("moving our database at work", "notes.txt"),
        ("grocery shopping bill", "receipt.png"),  # found through OCR text
    ],
)
def test_semantic_search_with_real_embeddings(real, home, query, expected):
    cfg, db, client = real
    build_corpus(home)
    Indexer(cfg, db, OllamaEmbedder(client), None).run()
    searcher = Searcher(db, OllamaEmbedder(client), cfg.search)
    hits = searcher.search(query)
    assert hits and os.path.basename(hits[0].path) == expected
    assert searcher.last_confidence == "high"  # the right file, so no "not found" note


def test_unrelated_query_finds_nothing(real, home):
    cfg, db, client = real
    build_corpus(home)
    Indexer(cfg, db, OllamaEmbedder(client), None).run()
    assert Searcher(db, OllamaEmbedder(client), cfg.search).search("quantum physics") == []


def test_real_image_caption(real, home):
    cfg, db, client = real
    captioner = OllamaCaptioner(client)
    if not captioner.available():
        pytest.skip("vision model not installed")
    photo(home / "IMG_0001.jpg")  # a blue circle on a red background
    write(home / "other.txt", "Notes about the garden fence.")
    Indexer(cfg, db, OllamaEmbedder(client), captioner).run()
    assert db.get_file(str(home / "IMG_0001.jpg"))["captioned"]
    hits = Searcher(db, OllamaEmbedder(client), cfg.search).search("blue circle")
    assert hits and hits[0].name == "IMG_0001.jpg"
    assert hits[0].matched_in == "caption"


def test_mcp_over_stdio(real, home, tmp_path):
    """Start `filesonar serve` as a real subprocess and talk to it as an agent would."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cfg, db, client = real
    build_corpus(home)
    Indexer(cfg, db, OllamaEmbedder(client), None).run()
    config = tmp_path / "config.toml"
    config.write_text(
        f'roots = ["{home}"]\ndata_dir = "{tmp_path / "data"}"\n'
        f'outbox_dir = "{tmp_path / "outbox"}"\n'
    )

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "filesonar.cli", "-c", str(config), "serve"]
        )
        async with stdio_client(params) as (read, write_), ClientSession(read, write_) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            assert tools == {"search_files", "file_info", "prepare_file", "index_status"}
            res = await s.call_tool("search_files", {"query": "automobile coverage"})
            data = json.loads(res.content[0].text)
            assert data["results"][0]["name"] == "auto-policy.pdf"
            info = await s.call_tool("file_info", {"path": data["results"][0]["path"]})
            assert "Toyota" in json.loads(info.content[0].text)["preview"][0]["text"]
            denied = await s.call_tool("file_info", {"path": "/etc/passwd"})
            assert denied.is_error and "not in the index" in denied.content[0].text
            original = data["results"][0]["path"]
            staged = await s.call_tool("prepare_file", {"path": original})
            copy = Path(json.loads(staged.content[0].text)["path"])
            assert copy.is_relative_to(tmp_path / "outbox")
            assert copy.read_bytes() == Path(original).read_bytes()

    asyncio.run(scenario())
