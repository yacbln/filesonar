"""Choose the benchmark's real photos and sounds (run once by a maintainer).

    python benchmark/pick_media.py

Searches Openverse (https://openverse.org) for public-domain and CC0 media on everyday
subjects, downloads each candidate once to check it and record its checksum, and writes
benchmark/media.json: for every file its subject, title, creator, license, source page
and download address. make_corpus.py downloads exactly these files and verifies them, so
everyone benchmarks the same media. Only CC0 and public-domain-mark media are used;
mature or flagged results are skipped.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image

HERE = Path(__file__).parent
API = "https://api.openverse.org/v1"
UA = {"User-Agent": "filesonar-benchmark/1.0"}

# (subject, search words, how many). Subjects people keep photos of, and photos of paper.
PHOTOS = [
    ("dog", "dog", 8), ("cat", "cat", 8), ("puppy", "puppy", 5), ("horse", "horse field", 4),
    ("pizza", "pizza", 6), ("sushi", "sushi", 5), ("salad", "salad bowl", 5),
    ("burger", "hamburger fries", 5), ("pancakes", "pancakes breakfast", 5),
    ("birthday cake", "birthday cake candles", 6), ("latte art", "latte art coffee", 5),
    ("barbecue", "barbecue grill", 5), ("beach sunset", "beach sunset", 7),
    ("mountain hike", "mountain hiking trail", 7), ("snowy forest", "snow forest winter", 6),
    ("waterfall", "waterfall", 5), ("kayak", "kayak lake", 5), ("camping", "camping tent", 5),
    ("fireworks", "fireworks night", 5), ("rainbow", "rainbow sky", 4),
    ("eiffel tower", "eiffel tower", 5), ("golden gate bridge", "golden gate bridge", 5),
    ("times square", "times square night", 4), ("street market", "street market stall", 5),
    ("subway", "subway train station", 5), ("airport", "airport terminal", 5),
    ("airplane window", "airplane window wing", 5), ("living room", "living room interior", 5),
    ("kitchen", "kitchen interior", 5), ("garden flowers", "garden flowers", 6),
    ("christmas tree", "christmas tree lights", 5), ("vintage car", "vintage car", 5),
    ("bicycle", "bicycle parked", 4), ("wedding", "wedding ceremony", 5),
    ("concert", "concert crowd stage", 5), ("soccer", "soccer match", 5),
    ("graduation", "graduation cap gown", 4), ("lighthouse", "lighthouse coast", 4),
    ("farmers market vegetables", "vegetables market", 5), ("library", "library bookshelves", 4),
    # Photos of paper and signs: what people snap instead of writing down.
    ("whiteboard", "whiteboard", 10), ("handwritten letter", "handwritten letter", 8),
    ("sticky notes", "sticky notes", 5), ("restaurant menu", "restaurant menu", 5),
    ("flyer", "poster flyer", 5), ("nutrition label", "nutrition facts label", 4),
    ("street sign", "street sign", 5), ("map", "paper map", 4),
]  # fmt: skip
SOUNDS = [
    ("rain", "rain", 2), ("thunderstorm", "thunderstorm", 2), ("birds", "birds singing", 2),
    ("ocean waves", "ocean waves", 2), ("applause", "applause", 2), ("piano", "piano music", 2),
    ("guitar", "acoustic guitar", 2), ("dog barking", "dog barking", 2),
    ("crowd cheering", "crowd cheering", 2), ("train", "train passing", 2),
    ("church bells", "church bells", 1), ("campfire", "campfire crackling", 1),
]  # fmt: skip
LARGE = {"whiteboard", "handwritten letter", "sticky notes", "restaurant menu", "flyer",
         "nutrition label", "street sign", "map"}  # text in the picture: keep detail  # fmt: skip
MAX_AUDIO_SECONDS = 90


def search(client: httpx.Client, kind: str, words: str, extra: dict) -> list[dict]:
    params = {"q": words, "license": "cc0,pdm", "page_size": 20, "mature": "false", **extra}
    for _ in range(5):
        r = client.get(f"{API}/{kind}/", params=params)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", "10")))
            continue
        r.raise_for_status()
        return r.json()["results"]
    raise RuntimeError("Openverse keeps refusing; try again later")


def fetch(client: httpx.Client, url: str) -> bytes | None:
    try:
        r = client.get(url, follow_redirects=True, timeout=60)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 and len(r.content) > 2000 else None


def main() -> None:
    chosen: list[dict] = []
    seen: set[str] = set()
    with httpx.Client(headers=UA, timeout=60) as client:
        for subject, words, count in PHOTOS:
            got = 0
            for item in search(client, "images", words, {"category": "photograph"}):
                if got >= count:
                    break
                if item["id"] in seen or item.get("mature") or item.get("unstable__sensitivity"):
                    continue
                if (item.get("width") or 0) < 500:
                    continue
                url = item["url"] if subject in LARGE else item["thumbnail"]
                data = fetch(client, url)
                if data is None:
                    continue
                try:
                    Image.open(io.BytesIO(data)).verify()
                except Exception as exc:  # broken or not an image: pick another
                    print(f"  skipped {item['id']}: {exc}")
                    continue
                seen.add(item["id"])
                got += 1
                chosen.append(entry("image", subject, item, url, data))
                time.sleep(0.5)
            print(f"{subject}: {got}")
        for subject, words, count in SOUNDS:
            got = 0
            for item in search(client, "audio", words, {}):
                if got >= count:
                    break
                seconds = (item.get("duration") or 0) / 1000
                if item["id"] in seen or not 5 <= seconds <= MAX_AUDIO_SECONDS:
                    continue
                if item.get("filetype") not in ("mp3", "ogg", "wav", "flac"):
                    continue
                data = fetch(client, item["url"])
                if data is None:
                    continue
                seen.add(item["id"])
                got += 1
                chosen.append(entry("audio", subject, item, item["url"], data))
                time.sleep(0.5)
            print(f"{subject} (sound): {got}")
    out = HERE / "media.json"
    out.write_text(json.dumps({"source": "Openverse, CC0 and public domain only",
                               "files": chosen}, indent=1, ensure_ascii=False))  # fmt: skip
    print(f"wrote {len(chosen)} entries to {out}")


def entry(kind: str, subject: str, item: dict, url: str, data: bytes) -> dict:
    return {
        "kind": kind,
        "subject": subject,
        "title": (item.get("title") or "").strip()[:120],
        "tags": [t["name"] for t in (item.get("tags") or [])][:12],
        "creator": item.get("creator") or "",
        "license": item["license"],
        "license_url": item.get("license_url") or "",
        "source_page": item.get("foreign_landing_url") or "",
        "provider": item.get("provider") or "",
        "url": url,
        "filetype": item.get("filetype") or ("jpg" if kind == "image" else "mp3"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


if __name__ == "__main__":
    main()
