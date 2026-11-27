"""Hybrid search: keyword (BM25) and meaning (vectors) combined with reciprocal rank fusion."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from .config import SearchConfig
from .db import Database
from .models import Embedder, ModelError

UTC = timezone.utc  # datetime.UTC needs Python 3.11

# Reciprocal rank fusion constant. The textbook value is 60; measured on a real
# 16,000-file index (local + Google Drive), 5 lets a strong keyword match on rare
# words win (top-3 recall 64-76% -> 86-95%) without hurting meaning-only queries
# (top-3 77% -> 78%). A smaller constant gives the first ranks of each list more say.
_RRF_K = 5
_WORD = re.compile(r"\w+", re.UNICODE)
KINDS = ("document", "pdf", "image", "text", "audio", "video")
SOURCES = ("local", "drive")
MAX_LIMIT = 50
SNIPPET_CHARS = 280


@dataclass
class Hit:
    path: str
    name: str
    kind: str
    score: float
    modified: str
    size_bytes: int
    page: int | None
    matched_in: str
    snippet: str
    details: str | None
    source: str = "local"
    # For Drive files: the folder in Drive and a link that only the owner can open.
    location: str | None = None
    link: str | None = None
    other_copies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        if self.source == "local":
            for key in ("location", "link"):
                if out[key] is None:
                    del out[key]
        return out


def fts_query(text: str) -> str | None:
    """Build a safe FTS5 query: every word quoted, combined with OR.

    Quoting means user input can never be interpreted as FTS5 syntax (NEAR, column
    filters, ...), so a query like ``name:foo OR`` cannot cause an error.
    """
    words = [w for w in _WORD.findall(text.lower()) if len(w) > 1 or w.isdigit()]
    if not words:
        return None
    unique = list(dict.fromkeys(words))[:32]
    return " OR ".join('"' + w.replace('"', '""') + '"' for w in unique)


def name_phrase_query(text: str) -> str | None:
    """FTS5 query matching the words of ``text`` as a phrase in the file name."""
    words = _WORD.findall(text.lower())[:32]
    if not words:
        return None
    return 'name : "' + " ".join(w.replace('"', '""') for w in words) + '"'


def _parse_day(value: str | None, field: str) -> float | None:
    if not value:
        return None
    try:
        day = date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError(f"{field} must be a date like 2024-05-31") from None
    return datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()


def _snippet(text: str, query: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= SNIPPET_CHARS:
        return flat
    lowered = flat.lower()
    positions = [lowered.find(w) for w in _WORD.findall(query.lower()) if len(w) > 2]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return flat[: SNIPPET_CHARS - 1] + "…"
    start = max(0, min(positions) - SNIPPET_CHARS // 3)
    end = start + SNIPPET_CHARS
    piece = flat[start:end]
    return ("…" if start else "") + piece + ("…" if end < len(flat) else "")


def fuse(lists: list[tuple[list[int], float]], k: int | None = None) -> dict[int, float]:
    """Reciprocal rank fusion: each list adds ``weight / (k + rank)`` to an item's score."""
    k = _RRF_K if k is None else k
    scores: dict[int, float] = {}
    for items, weight in lists:
        for rank, item in enumerate(items):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores


def _preference(row) -> tuple[bool, float]:
    return (row["origin"] == "local", row["mtime"])


def _in_folder(row, prefix: str) -> bool:
    """Local files by path; Drive files by their folder ("My Drive/Taxes")."""
    if row["origin"] == "drive":
        where = (row["location"] or "") + "/"
        wanted = prefix.removeprefix("/")
        return where.startswith(wanted) or where.startswith(f"My Drive/{wanted}")
    return row["path"].startswith(prefix)


class Searcher:
    def __init__(self, db: Database, embedder: Embedder, cfg: SearchConfig | None = None):
        self.db = db
        self.embedder = embedder
        self.cfg = cfg or SearchConfig()
        self.last_warning: str | None = None

    def _vector_hits(self, query: str, depth: int) -> list[int]:
        """Chunk ids close in meaning to ``query``, best first.

        Nearest-neighbour search always returns *something*, so matches that are not
        close enough (in absolute terms, or relative to the best match) are dropped.
        """
        if not self.db.has_vectors():
            return []
        stored = self.db.get_meta("embedding_model")
        if stored != self.embedder.name:
            self.last_warning = (
                f"index was built with {stored}, config uses {self.embedder.name}; "
                "keyword only until `filesonar index --rebuild`"
            )
            return []
        try:
            vector = self.embedder.embed_query(query)
        except ModelError as exc:
            # Keyword search still works without the model server.
            self.last_warning = f"meaning-based search unavailable ({exc}); keyword only"
            return []
        found = self.db.vector_search(vector, depth)
        if not found:
            return []
        cutoff = min(self.cfg.max_distance, found[0][1] + self.cfg.relative_margin)
        return [cid for cid, dist in found if dist <= cutoff]

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        kind: str | None = None,
        folder: str | None = None,
        modified_after: str | None = None,
        modified_before: str | None = None,
        source: str | None = None,
    ) -> list[Hit]:
        query = query.strip()
        if not query:
            raise ValueError("query is empty")
        if len(query) > 1000:
            raise ValueError("query is too long (max 1000 characters)")
        if kind is not None and kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        if source is not None and source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        limit = max(1, min(int(limit), MAX_LIMIT))
        after = _parse_day(modified_after, "modified_after")
        before = _parse_day(modified_before, "modified_before")
        if before is not None:
            before += 86400  # include the whole day
        prefix = None
        if folder:
            prefix = str(Path(folder).expanduser()).rstrip("/") + "/"

        filtered = any(v is not None for v in (kind, prefix, after, before, source))
        depth = 400 if filtered else max(60, limit * 8)

        lists: list[tuple[list[int], float]] = []
        match = fts_query(query)
        if match:
            lists.append(([cid for cid, _ in self.db.fts_search(match, depth)], 1.0))
        # Typing (part of) a file name exactly should put that file first.
        name_match = name_phrase_query(query)
        if name_match:
            lists.append(([cid for cid, _ in self.db.fts_search(name_match, depth)], 2.0))
        self.last_warning = None
        lists.append((self._vector_hits(query, depth), 1.0))
        ranks = fuse(lists)
        if not ranks:
            return []

        # Grouped by content hash: identical copies of a file are one result.
        best: dict[str, tuple[float, int, object]] = {}
        per_path: dict[str, int] = {}
        for row in self.db.chunks_with_files(list(ranks)):
            if kind is not None and row["kind"] != kind:
                continue
            if prefix is not None and not _in_folder(row, prefix):
                continue
            if source is not None and row["origin"] != source:
                continue
            if after is not None and row["mtime"] < after:
                continue
            if before is not None and row["mtime"] >= before:
                continue
            score = ranks[row["chunk_id"]]
            # The same file on this computer and in Drive is one result.
            key = row["md5"] or row["sha256"]
            per_path[row["path"]] = per_path.get(row["path"], 0) + 1
            prev = best.get(key)
            if (
                prev is None
                or score > prev[0]
                or (
                    # Ties go to a copy on this computer, then the most recent one.
                    score == prev[0] and _preference(row) > _preference(prev[2])
                )
            ):
                best[key] = (score, 0, row)
        # How many chunks of the chosen copy matched (copies would double count).
        best = {k: (top, per_path[row["path"]], row) for k, (top, _n, row) in best.items()}

        # A file matched by several chunks ranks slightly higher than a single hit.
        scored = sorted(
            ((top * (1 + 0.05 * min(count - 1, 4)), row) for top, count, row in best.values()),
            key=lambda item: item[0],
            reverse=True,
        )[:limit]
        if not scored:
            return []
        ceiling = scored[0][0]
        hits = []
        for score, row in scored:
            hits.append(
                Hit(
                    path=row["path"],
                    name=row["name"],
                    kind=row["kind"],
                    score=round(score / ceiling, 3),
                    modified=datetime.fromtimestamp(row["mtime"], UTC).date().isoformat(),
                    size_bytes=row["size"],
                    page=row["page"],
                    matched_in=row["source"],
                    snippet=_snippet(row["text"], query),
                    details=row["details"],
                    source=row["origin"],
                    location=row["location"],
                    link=row["link"],
                    other_copies=self.db.copies_of(row["sha256"], row["path"], row["md5"]),
                )
            )
        return hits
