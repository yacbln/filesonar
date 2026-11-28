"""Index the benchmark's files with the real models and measure how well searches find them.

    python benchmark/run.py                       # build files, index, search, report
    python benchmark/run.py --keep DIR            # keep the files and index in DIR
    python benchmark/run.py --report report.md    # also write a Markdown report

Compares search methods on the same index and the same searches:
  filename  - matching words in file names, what most file managers do
  keyword   - filesonar's keyword search alone (BM25)
  meaning   - filesonar's meaning search alone (embeddings)
  filesonar - the hybrid filesonar uses (both, fused, plus file-name matches)

Uses the Ollama host and models from your filesonar config (or the defaults) and, if it
is set up, speech recognition. Your own index and files are never touched.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_corpus import corpus  # noqa: E402

import filesonar  # noqa: E402
from filesonar.config import ConfigError, config_from_dict, load_config  # noqa: E402
from filesonar.db import Database  # noqa: E402
from filesonar.indexer import Indexer  # noqa: E402
from filesonar.media import WhisperTranscriber  # noqa: E402
from filesonar.models import OllamaCaptioner, OllamaClient, OllamaEmbedder  # noqa: E402
from filesonar.search import Searcher  # noqa: E402

METHODS = ["filename", "keyword", "meaning", "filesonar"]
WORD = re.compile(r"[a-z0-9]{2,}")
KINDS = ["describe", "question", "exact", "visual", "photo", "spoken", "sound", "video", "language",
         "typo", "vague"]  # fmt: skip


def filename_search(files: list[str], query: str, limit: int) -> list[str]:
    """Rank files by how many query words appear in their path, like a file manager."""
    words = set(WORD.findall(query.lower()))
    scored = []
    for f in files:
        name = set(WORD.findall(f.lower().replace("_", " ").replace("-", " ")))
        hits = len(words & name)
        if hits:
            scored.append((-hits, len(f), f))
    return [f for _, _, f in sorted(scored)[:limit]]


def run_method(method: str, searcher: Searcher, db: Database, query: str, limit: int, corpus_dir: Path,
               files: list[str]) -> list[str]:  # fmt: skip
    if method == "filename":
        return filename_search(files, query, limit)
    real_fts, real_vec = db.fts_search, searcher._vector_hits
    try:
        if method == "keyword":
            searcher._vector_hits = lambda q, depth: []
        elif method == "meaning":
            db.fts_search = lambda match, depth: []
        hits = searcher.search(query, limit=limit)
    finally:
        db.fts_search, searcher._vector_hits = real_fts, real_vec
    return [Path(h.path).relative_to(corpus_dir).as_posix() for h in hits]


def score(results: list[tuple[dict, list[str]]]) -> dict:
    """top-1, top-3 and MRR over searches that have a right answer; 'absent' searches
    are right only when nothing comes back."""
    ranks, absent_ok, absent_n = [], 0, 0
    for item, got in results:
        if not item["want"]:
            absent_n += 1
            absent_ok += not got
            continue
        ranks.append(next((i for i, g in enumerate(got) if g in item["want"]), None))
    n = len(ranks) or 1
    out = {
        "n": len(ranks),
        "top1": sum(r == 0 for r in ranks) / n,
        "top3": sum(r is not None and r < 3 for r in ranks) / n,
        "mrr": sum(1 / (r + 1) for r in ranks if r is not None) / n,
    }
    if absent_n:
        out["absent_n"], out["absent_ok"] = absent_n, absent_ok / absent_n
    return out


def interval(results: list[tuple[dict, list[str]]], key: str = "top3", rounds: int = 1000) -> tuple:
    """95% bootstrap interval, so small differences are not over-read."""
    rnd = random.Random(1)
    rows = [r for r in results if r[0]["want"]]
    values = sorted(score([rnd.choice(rows) for _ in rows])[key] for _ in range(rounds))
    return values[int(rounds * 0.025)], values[int(rounds * 0.975)]


def machine() -> str:
    gpu = ""
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True)  # fmt: skip
        gpu = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    return f"{platform.system()} {platform.machine()}, {platform.processor() or 'CPU'}" + (
        f", {gpu}" if gpu else ""
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep", type=Path, help="build and keep the files and index here")
    ap.add_argument("--report", type=Path, help="write a Markdown report here")
    ap.add_argument("--json", type=Path, default=HERE / "results.json")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    try:
        mine = load_config()
        models, media, speech_dir = mine.models, mine.media, mine.models_dir
    except ConfigError:
        from filesonar.config import MediaConfig, ModelConfig, default_data_dir

        models, media, speech_dir = ModelConfig(), MediaConfig(), default_data_dir() / "models"

    tmp = tempfile.TemporaryDirectory(prefix="filesonar-bench-")
    root = args.keep or Path(tmp.name)
    corpus_dir = root / "Sam"
    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        raise SystemExit(f"{corpus_dir} is not empty")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    files, searches, problems = corpus(corpus_dir, Path.home() / ".cache" / "filesonar-benchmark")
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)
    print(f"{len(files)} files and {len(searches)} searches ({time.monotonic() - started:.0f}s)")

    cfg = config_from_dict(
        {
            "roots": [str(corpus_dir)],
            "data_dir": str(root / "data"),
            "outbox_dir": str(root / "outbox"),
            "models": {"host": models.host, "embedding_model": models.embedding_model,
                       "vision_model": models.vision_model, "caption_images": models.caption_images},
            "media": {"speech_model": media.speech_model},
        },
        base=root,
    )  # fmt: skip
    client = OllamaClient(cfg.models)
    speech = WhisperTranscriber(cfg.media.speech_model, speech_dir)
    if not speech.ready():
        print("note: speech recognition is not set up, so spoken searches will miss")
    db = Database(cfg.db_path)
    started = time.monotonic()
    report = Indexer(
        cfg, db, OllamaEmbedder(client), OllamaCaptioner(client), transcriber=speech
    ).run()
    index_seconds = time.monotonic() - started
    print(f"indexed: {report.summary()}")

    searcher = Searcher(db, OllamaEmbedder(client), cfg.search)
    results: dict[str, list] = {}
    times: dict[str, list[float]] = defaultdict(list)
    for method in METHODS:
        results[method] = []
        for item in searches:
            t = time.perf_counter()
            got = run_method(method, searcher, db, item["q"], args.limit, corpus_dir, files)
            times[method].append(time.perf_counter() - t)
            results[method].append((item, got))
    db.close()

    def ext_of(item):
        exts = {Path(w).suffix.lower() for w in item["want"]}
        return exts.pop() if len(exts) == 1 else "mixed"

    summary = {
        "filesonar_version": filesonar.__version__,
        "machine": machine(),
        "models": {
            "embedding": models.embedding_model,
            "vision": models.vision_model,
            "speech": cfg.media.speech_model if speech.ready() else None,
        },
        "files": len(files),
        "searches": len(searches),
        "index_seconds": round(index_seconds),
        "index_report": report.summary(),
        "methods": {},
    }
    for method in METHODS:
        res = results[method]
        entry = {"all": score(res), "interval_top3": interval(res),
                 "median_ms": round(statistics.median(times[method]) * 1000, 1), "by_kind": {}, "by_type": {}}  # fmt: skip
        for kind in [*KINDS, "absent"]:
            part = [r for r in res if r[0]["cat"] == kind]
            if part:
                entry["by_kind"][kind] = score(part)
        by_ext = defaultdict(list)
        for r in res:
            if r[0]["want"]:
                by_ext[ext_of(r[0])].append(r)
        entry["by_type"] = {e: score(rs) for e, rs in sorted(by_ext.items()) if len(rs) >= 5}
        summary["methods"][method] = entry
    args.json.write_text(json.dumps(summary, indent=1))

    print_table(summary)
    if args.report:
        args.report.write_text(markdown(summary))
        print(f"\nreport written to {args.report}")
    if args.show_misses:
        print("\nfilesonar searches where the right file was not in the top 3:")
        for item, got in results["filesonar"]:
            if item["want"] and not any(g in item["want"] for g in got[:3]):
                print(f"  [{item['cat']}] {item['q']!r} -> {got[:2]}")
    return 0


def print_table(s: dict) -> None:
    print(f"\n{'':12}" + "".join(f"{m:>18}" for m in METHODS))
    print(f"{'top-1':12}" + "".join(f"{s['methods'][m]['all']['top1']:>18.0%}" for m in METHODS))
    print(f"{'top-3':12}" + "".join(f"{s['methods'][m]['all']['top3']:>18.0%}" for m in METHODS))
    print(f"{'MRR':12}" + "".join(f"{s['methods'][m]['all']['mrr']:>18.2f}" for m in METHODS))
    print(f"{'ms/search':12}" + "".join(f"{s['methods'][m]['median_ms']:>18}" for m in METHODS))
    print("\ntop-3 by kind of search:")
    for kind in [*KINDS, "absent"]:
        row = s["methods"]["filesonar"]["by_kind"].get(kind)
        if not row:
            continue
        key = "absent_ok" if kind == "absent" else "top3"
        cells = "".join(f"{s['methods'][m]['by_kind'][kind].get(key, 0):>18.0%}" for m in METHODS)
        print(f"  {kind:10}" + cells + f"   (n={row.get('absent_n', row['n'])})")


def markdown(s: dict) -> str:
    m = s["methods"]
    lines = [
        f"# filesonar benchmark ({s['files']} files, {s['searches']} searches)",
        "",
        f"filesonar {s['filesonar_version']} on {s['machine']}. Models: {s['models']['embedding']} "
        f"(text), {s['models']['vision']} (pictures), whisper {s['models']['speech']} (speech). "
        f"Indexing took {s['index_seconds'] // 60} min {s['index_seconds'] % 60} s.",
        "",
        "| | " + " | ".join(METHODS) + " |",
        "|---|" + "---|" * len(METHODS),
    ]
    for key, label in (
        ("top1", "Right file first"),
        ("top3", "Right file in top 3"),
        ("mrr", "MRR"),
    ):
        fmt = (lambda v: f"{v:.2f}") if key == "mrr" else (lambda v: f"{v:.0%}")
        lines.append(f"| {label} | " + " | ".join(fmt(m[x]["all"][key]) for x in METHODS) + " |")
    lines.append("| 95% range, top 3 | " + " | ".join(
        f"{m[x]['interval_top3'][0]:.0%}–{m[x]['interval_top3'][1]:.0%}" for x in METHODS) + " |")  # fmt: skip
    lines.append(
        "| Time per search | " + " | ".join(f"{m[x]['median_ms']} ms" for x in METHODS) + " |"
    )
    lines += ["", "## Right file in the top 3, by kind of search", "",
              "| Kind | Searches | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]  # fmt: skip
    for kind in [*KINDS, "absent"]:
        row = m["filesonar"]["by_kind"].get(kind)
        if not row:
            continue
        key = "absent_ok" if kind == "absent" else "top3"
        n = row.get("absent_n", row["n"])
        lines.append(
            f"| {kind} | {n} | "
            + " | ".join(f"{m[x]['by_kind'][kind].get(key, 0):.0%}" for x in METHODS)
            + " |"
        )
    lines += ["", "## filesonar by file type (right file in the top 3)", "", "| Type | Searches | Top 1 | Top 3 |",
              "|---|---|---|---|"]  # fmt: skip
    for ext, row in m["filesonar"]["by_type"].items():
        lines.append(f"| {ext} | {row['n']} | {row['top1']:.0%} | {row['top3']:.0%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
