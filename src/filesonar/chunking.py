"""Split long text into overlapping chunks that fit an embedding model comfortably."""

from __future__ import annotations

import re

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _cut_point(text: str, start: int, limit: int) -> int:
    """Find a natural place to end a chunk that starts at ``start``."""
    end = start + limit
    if end >= len(text):
        return len(text)
    window = text[start:end]
    floor = limit // 2  # never make a chunk shorter than half the limit
    for sep in _BREAKS:
        pos = window.rfind(sep)
        if pos >= floor:
            return start + pos + len(sep)
    return end


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _cut_point(text, start, size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Step back by the overlap, then forward to a word boundary.
        nxt = max(end - overlap, start + 1)
        match = re.search(r"\s", text[nxt:end])
        start = nxt + match.end() if match else nxt
    return chunks
