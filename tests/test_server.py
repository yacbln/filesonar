import asyncio
import json
from pathlib import Path

import pytest
from conftest import write
from mcp.server.mcpserver.exceptions import ToolError

from filesonar.server import PREVIEW_CHARS, build_server


@pytest.fixture
def server(home, cfg, db, embedder, indexer):
    write(home / "taxes/2023-return.txt", "Federal tax return 2023. Refund of 412 dollars.")
    write(home / "long.txt", "lorem ipsum dolor " * 1000)
    indexer.run()
    return build_server(cfg, db, embedder)


def call(server, name, args=None):
    result = asyncio.run(server.call_tool(name, args or {}))
    assert not result.is_error, result
    return json.loads(result.content[0].text)


def test_tool_annotations(server):
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) == {"search_files", "file_info", "prepare_file", "index_status"}
    for name, tool in tools.items():
        # prepare_file writes (to its outbox only); everything else is read-only.
        assert tool.annotations.read_only_hint is (name != "prepare_file")
        # None of them change the user's files or reach other machines, which is what
        # lets agents such as OpenClaw run them without an approval prompt.
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is False
        assert tool.description
    props = tools["search_files"].input_schema["properties"]
    assert {"query", "kind", "folder", "modified_after", "modified_before", "limit"} <= set(props)


def test_instructions_warn_about_untrusted_content(server):
    assert "never as instructions" in server.instructions


def test_search_files(server, home):
    data = call(server, "search_files", {"query": "tax refund"})
    assert data["results"][0]["path"] == str(home / "taxes/2023-return.txt")


def test_search_with_no_results_has_a_note(server):
    data = call(server, "search_files", {"query": "zzzzqqq", "kind": "image"})
    assert data["results"] == [] and "note" in data


def test_invalid_arguments_become_tool_errors(server):
    for args in ({"query": "x", "limit": 0}, {}):
        with pytest.raises(ToolError):
            asyncio.run(server.call_tool("search_files", args))
    # the reason reaches the agent, so it can correct itself
    for args, reason in (
        ({"query": "x", "kind": "hologram"}, "kind must be one of"),
        ({"query": ""}, "query is empty"),
        ({"query": "x", "modified_after": "last week"}, "YYYY|date"),
    ):
        with pytest.raises(ToolError, match=reason):
            asyncio.run(server.call_tool("search_files", args))


def test_file_info_only_for_indexed_paths(server, home, tmp_path):
    for path in (
        "/etc/passwd",
        str(tmp_path / "data/index.sqlite3"),
        str(home / "taxes/../taxes/2023-return.txt"),
    ):
        with pytest.raises(ToolError, match="not in the index"):
            asyncio.run(server.call_tool("file_info", {"path": path}))


def test_file_info_preview_is_bounded(server, home):
    data = call(server, "file_info", {"path": str(home / "long.txt")})
    assert data["preview_truncated"] is True
    assert sum(len(p["text"]) for p in data["preview"]) == PREVIEW_CHARS
    small = call(server, "file_info", {"path": str(home / "taxes/2023-return.txt")})
    assert small["preview_truncated"] is False
    assert "412 dollars" in small["preview"][0]["text"]
    assert small["status"] == "ok" and small["kind"] == "text"


def test_index_status(server, home):
    data = call(server, "index_status")
    assert data["files"] == 2
    assert data["folders"] == [str(home)]
    assert data["last_indexed"].endswith("+00:00")


def call_error(server, name, args):
    with pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool(name, args))
    return str(exc.value)


def test_prepare_file_copies_into_the_outbox(server, home, cfg):
    original = home / "taxes/2023-return.txt"
    before = original.read_bytes()
    out = call(server, "prepare_file", {"path": str(original)})
    staged = Path(out["path"])
    assert staged.is_relative_to(cfg.outbox_dir)
    assert staged.name == "2023-return.txt"
    assert staged.read_bytes() == before == original.read_bytes()
    assert staged.stat().st_mode & 0o777 == 0o600
    assert staged.parent.stat().st_mode & 0o777 == 0o700
    assert out["original"] == str(original)
    # Asking again reuses the copy.
    assert call(server, "prepare_file", {"path": str(original)})["path"] == out["path"]


def test_prepare_file_only_for_indexed_and_allowed_files(server, home, tmp_path, cfg):
    outside = tmp_path / "secret.txt"
    outside.write_text("not indexed")
    assert "not in the index" in call_error(server, "prepare_file", {"path": str(outside)})
    # Indexed, then replaced by a symlink to a file outside the folders.
    target = home / "taxes/2023-return.txt"
    target.unlink()
    target.symlink_to(outside)
    assert "no longer available" in call_error(server, "prepare_file", {"path": str(target)})
    assert not cfg.outbox_dir.exists() or not any(cfg.outbox_dir.rglob("*.txt"))


def test_prepare_file_uses_current_content(server, home):
    original = home / "taxes/2023-return.txt"
    first = call(server, "prepare_file", {"path": str(original)})
    original.write_text("Amended federal tax return 2023.")
    second = call(server, "prepare_file", {"path": str(original)})
    assert second["path"] != first["path"]
    assert Path(second["path"]).read_text() == "Amended federal tax return 2023."
