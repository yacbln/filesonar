"""Measure search quality on your own index (read-only).

For a random sample of indexed files, build a query from the file's content and check
where search ranks that file. Run it before and after changing anything in ranking:

    python scripts/eval_search.py                    # distinctive words from each file
    python scripts/eval_search.py --mode random      # a run of ordinary words
    python scripts/eval_search.py --mode paraphrase  # the local model rewrites the
                                                     # content as a query in its own words
    python scripts/eval_search.py --rrf-k 60         # try another fusion constant
    python scripts/eval_search.py -c after.toml --same-files-as before.sqlite3
                                                     # same queries on two indexes

Queries never use the file name. Nothing is written to the index, and nothing leaves
your machine (paraphrases come from the Ollama model in your config).
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import time
from pathlib import Path

import httpx

import filesonar.search as fsearch
from filesonar.config import load_config
from filesonar.db import Database
from filesonar.models import OllamaClient, OllamaEmbedder

_COMMON = set(
    [
        "about",
        "above",
        "after",
        "again",
        "their",
        "there",
        "these",
        "those",
        "which",
        "would",
        "could",
        "should",
        "where",
        "while",
        "with",
        "from",
        "have",
        "this",
        "that",
        "were",
        "been",
        "being",
        "into",
        "your",
        "they",
        "them",
        "what",
        "when",
        "will",
        "image",
        "shows",
        "appears",
    ]
)
_PARAPHRASE = (
    "Here is the content of one of my files:\n\n{text}\n\nWrite ONE short search query "
    "(3 to 7 words) that I might type to find this file later from memory. Use your own "
    "words: avoid copying distinctive words from the text. Reply with the query only."
)


def build_query(text: str, name: str, mode: str, df: dict[str, int], cfg) -> str | None:
    if mode == "paraphrase":
        resp = httpx.post(
            cfg.models.host.rstrip("/") + "/api/chat",
            json={
                "model": cfg.models.vision_model,
                "stream": False,
                "messages": [{"role": "user", "content": _PARAPHRASE.format(text=text[:1200])}],
                "options": {"temperature": 0, "num_predict": 30},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip().strip('"').split("\n")[0][:120] or None
    skip = {w.lower() for w in re.findall(r"\w+", name)} | _COMMON
    words = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", text) if w.lower() not in skip]
    if len(words) < 4:
        return None
    if mode == "rare":  # what people remember: the unusual words
        unique = sorted(dict.fromkeys(words), key=lambda w: df.get(w, 10**9))
        picked = [w for w in unique if df.get(w, 0) > 1][:4]
        return " ".join(picked) or None
    start = random.randrange(0, max(1, len(words) - 5))
    return " ".join(words[start : start + 5])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("rare", "random", "paraphrase"), default="rare")
    ap.add_argument("-n", type=int, default=150, help="files to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rrf-k", type=int, help="override the fusion constant")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("-c", "--config", type=Path, default=None)
    ap.add_argument(
        "--same-files-as",
        type=Path,
        metavar="INDEX",
        help="only sample files also in this other index file, so two indexes (before "
        "and after a change) are measured with exactly the same queries",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    if args.rrf_k:
        fsearch._RRF_K = args.rrf_k
    cfg = load_config(args.config)
    db = Database(cfg.db_path)
    searcher = fsearch.Searcher(db, OllamaEmbedder(OllamaClient(cfg.models)), cfg.search)
    df: dict[str, int] = {}
    if args.mode == "rare":
        db.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab USING fts5vocab(main, chunks_fts, 'row')"
        )
        df = dict(db.conn.execute("SELECT term, doc FROM temp.vocab"))

    rows = db.conn.execute(
        "SELECT id, path, name, origin, COALESCE(md5, sha256) AS key FROM files f "
        "WHERE status = 'ok' AND EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id "
        "AND c.source != 'metadata') ORDER BY path"
    ).fetchall()
    if args.same_files_as:
        import sqlite3

        other = sqlite3.connect(f"file:{args.same_files_as}?mode=ro", uri=True)
        keep = {r[0] for r in other.execute("SELECT path FROM files WHERE status = 'ok'")}
        other.close()
        rows = [r for r in rows if r["path"] in keep]
    ranks: dict[str, list[int | None]] = {}
    times: list[float] = []
    low: list[int | None] = []  # rank of the wanted file when search said "low confidence"
    for row in random.sample(rows, min(args.n, len(rows))):
        chunks = db.conn.execute(
            "SELECT text FROM chunks WHERE file_id = ? AND source != 'metadata'", (row["id"],)
        ).fetchall()
        query = build_query(random.choice(chunks)["text"], row["name"], args.mode, df, cfg)
        if not query:
            continue
        started = time.perf_counter()
        hits = searcher.search(query, limit=5)
        times.append(time.perf_counter() - started)
        keys = [db.get_file(h.path) for h in hits]
        rank = next(
            (i for i, f in enumerate(keys) if f["path"] == row["path"] or f["md5"] == row["key"]
             or f["sha256"] == row["key"]),
            None,
        )  # fmt: skip
        ranks.setdefault(row["origin"], []).append(rank)
        if searcher.last_confidence == "low":
            low.append(rank)
        if args.show_misses and (rank is None or rank > 2):
            print(f"  miss: {query!r} -> wanted {row['name'][:40]!r}, got",
                  [h.name[:30] for h in hits[:3]])  # fmt: skip

    print(f"mode={args.mode} rrf_k={fsearch._RRF_K} seed={args.seed}")
    for origin, found in sorted(ranks.items()):
        n = len(found)
        top1 = sum(r == 0 for r in found) / n
        top3 = sum(r is not None and r < 3 for r in found) / n
        print(f"  {origin:<6} n={n:<4} top1={top1:.0%}  top3={top3:.0%}")
    n_all = sum(len(r) for r in ranks.values())
    if n_all:
        right = sum(r == 0 for r in low)
        print(f"  low confidence: {len(low)}/{n_all} searches ({len(low) / n_all:.0%}), "
              f"right file first in {right} of them")  # fmt: skip
    if times:
        times.sort()
        print(
            f"  latency median={statistics.median(times) * 1000:.0f} ms "
            f"p95={times[int(len(times) * 0.95)] * 1000:.0f} ms"
        )
    db.close()


if __name__ == "__main__":
    main()
