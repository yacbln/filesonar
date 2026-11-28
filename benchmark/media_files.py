"""Real photos and sounds from benchmark/media.json, plus searches for them.

Files get neutral camera-style names (IMG_5012.jpg, Audio 3.mp3) so a search can only
find them by what they show or sound like, never by their name.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import httpx
from families import Out

HERE = Path(__file__).parent

# How people ask for photos and sounds of each subject (subject -> searches).
PHOTO_ASKS = {
    "dog": ["photos of our dog", "pictures of dogs"], "cat": ["photos of the cat"], "puppy": ["puppy pictures"],
    "horse": ["photo of horses in a field"], "pizza": ["pizza photos"], "sushi": ["the sushi dinner photos"],
    "salad": ["photo of a salad"], "burger": ["picture of burgers and fries"], "pancakes": ["breakfast pancakes photo"],
    "birthday cake": ["birthday cake with candles"], "latte art": ["latte art photos"], "barbecue": ["barbecue grill pictures"],
    "beach sunset": ["sunset at the beach", "beach photos"], "mountain hike": ["photos from the mountain hike"],
    "snowy forest": ["snowy forest pictures"], "waterfall": ["waterfall photos"], "kayak": ["kayaking on the lake"],
    "camping": ["camping trip photos with the tent"], "fireworks": ["fireworks pictures"], "rainbow": ["photo of a rainbow"],
    "eiffel tower": ["pictures of the Eiffel Tower"], "golden gate bridge": ["Golden Gate Bridge photos"],
    "times square": ["Times Square at night"], "street market": ["photos of a street market"],
    "subway": ["subway station photos"], "airport": ["pictures at the airport"], "airplane window": ["view of the wing from the plane window"],
    "living room": ["living room photos"], "kitchen": ["pictures of a kitchen"], "garden flowers": ["flowers in the garden"],
    "christmas tree": ["christmas tree photos"], "vintage car": ["old classic car photos"], "bicycle": ["photo of a bike"],
    "wedding": ["wedding photos"], "concert": ["concert photos"], "soccer": ["soccer game pictures"],
    "graduation": ["graduation photos"], "lighthouse": ["lighthouse pictures"], "farmers market vegetables": ["vegetables at the market"],
    "library": ["photo of library shelves"], "whiteboard": ["photos of whiteboards"], "handwritten letter": ["old handwritten letters"],
    "sticky notes": ["photos of sticky notes"], "restaurant menu": ["picture of a restaurant menu"], "flyer": ["photo of a poster"],
    "nutrition label": ["photo of a nutrition label"], "street sign": ["picture of a street sign"], "map": ["photo of a paper map"],
}  # fmt: skip
SOUND_ASKS = {
    "rain": "recording of rain", "thunderstorm": "thunderstorm sounds", "birds": "birds singing recording",
    "ocean waves": "sound of ocean waves", "applause": "recording of applause", "piano": "piano music recording",
    "guitar": "guitar recording", "dog barking": "dog barking sound", "crowd cheering": "crowd cheering",
    "train": "sound of a train passing", "church bells": "church bells ringing", "campfire": "campfire sounds",
}  # fmt: skip
# Short videos made of a real photo and a real sound: (photo subject, sound subject, search).
CLIPS = [
    ("beach sunset", "ocean waves", "video of the beach with the waves"),
    ("fireworks", "crowd cheering", "video of the fireworks with people cheering"),
    ("concert", "applause", "video from the concert"),
    ("camping", "campfire", "video at the campsite by the fire"),
    ("subway", "train", "video of the train station"),
    ("wedding", "church bells", "video from the wedding with the bells"),
]  # fmt: skip


def download(entry: dict, cache: Path) -> bytes:
    path = cache / entry["sha256"]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
        return path.read_bytes()
    r = httpx.get(entry["url"], timeout=90, follow_redirects=True,
                  headers={"User-Agent": "filesonar-benchmark/1.0"})  # fmt: skip
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"{entry['url']} changed since media.json was made")
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return r.content


def media(o: Out, cache: Path) -> list[str]:
    manifest = HERE / "media.json"
    if not manifest.exists():
        return ["media.json is missing: run benchmark/pick_media.py"]
    entries = json.loads(manifest.read_text())["files"]
    by_subject: dict[str, list[str]] = defaultdict(list)
    sound_files: dict[str, list[str]] = defaultdict(list)
    problems = []
    sounds = 0
    for i, e in enumerate(entries):
        try:
            data = download(e, cache)
        except (httpx.HTTPError, RuntimeError) as exc:
            problems.append(f"{e['title'][:40]}: {exc}")
            continue
        if e["kind"] == "image":
            rel = f"Pictures/Camera/IMG_{5000 + i}.jpg"
            by_subject[e["subject"]].append(rel)
        else:
            sounds += 1
            rel = f"Sounds/Audio {sounds}.{e['filetype']}"
            sound_files[e["subject"]].append(rel)
        o.path(rel).write_bytes(data)
    groups = {"dog": ["dog", "puppy"], "puppy": ["puppy", "dog"], "beach sunset": ["beach sunset"],
              "whiteboard": ["whiteboard"], "sticky notes": ["sticky notes"]}  # fmt: skip
    for subject, asks in PHOTO_ASKS.items():
        want = [r for s in groups.get(subject, [subject]) for r in by_subject.get(s, [])]
        if subject == "whiteboard":
            want += [f for f in o.files if f.startswith("Pictures/Whiteboard/")]
        if subject == "sticky notes":
            want += [f for f in o.files if f.startswith("Pictures/Notes/IMG_33")]
        for q in asks:
            if want:
                o.ask(q, "photo", *want)
    for subject, q in SOUND_ASKS.items():
        if sound_files.get(subject):
            o.ask(q, "sound", *sound_files[subject])
    # Short videos from a photo and a sound.
    for n, (photo, sound, q) in enumerate(CLIPS):
        if not by_subject.get(photo) or not sound_files.get(sound):
            continue
        rel = f"Videos/VID_{6000 + n}.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1", "-framerate", "2",
                            "-i", str(o.root / by_subject[photo][0]), "-i", str(o.root / sound_files[sound][0]),
                            "-t", "12", "-vf", "scale=1280:-2,format=yuv420p", "-c:v", "libx264",
                            "-tune", "stillimage", "-c:a", "aac", "-shortest", str(Path(tmp) / "v.mp4")],
                           check=True)  # fmt: skip
            o.path(rel).write_bytes((Path(tmp) / "v.mp4").read_bytes())
        o.ask(q, "video", rel)
    return problems


NEGATIVE = [
    "boat registration", "divorce papers", "mortgage statement", "photo of a giraffe", "jury duty summons",
    "COVID test result", "pet insurance policy", "snowboard rental receipt", "my motorcycle title",
    "timeshare contract", "photos of a hot air balloon", "chess tournament certificate", "wine cellar inventory",
    "hunting license", "the karaoke video", "crypto exchange statement", "piano lesson invoice",
    "photo of the northern lights", "lottery ticket", "golf club membership card",
]  # fmt: skip


def hard_queries(o: Out, base: list[dict]) -> None:
    """Misspellings and vague asks built on real searches, and searches for things
    that do not exist (the right answer is no result)."""
    typos = {
        "car insurance policy": "car insurence polcy", "receipt for the dishwasher": "reciept for the dishwaser",
        "blood test results": "blod test resutls", "chickpea curry recipe": "chikpea cury recipe",
        "boarding pass": "bording pass", "Maya's report card": "Mayas reprot card",
        "W-2 form": "W2 from", "packing list": "pakcing list", "marriage certificate": "marrige certificat",
        "parking ticket": "parkng tiket", "lasagna recipe": "lasanga recipie", "soccer practice schedule": "socer practise schedule",
    }  # fmt: skip
    for item in base:
        if item["q"] in typos:
            o.ask(typos[item["q"]], "typo", *item["want"])
    car = ["Insurance/auto-policy-2024.pdf", "Finance/car-loan-statement.pdf", "Home/car-maintenance-log.xlsx",
           *[f for f in o.files if f.startswith("Car/")]]  # fmt: skip
    o.ask("car stuff", "vague", *car)
    o.ask("the kids' school things", "vague", *[f for f in o.files if f.startswith("Kids/")])
    o.ask(
        "stuff from our trip to Japan",
        "vague",
        *[f for f in o.files if f.startswith("Travel/tokyo/")],
    )
    o.ask("money from the credit union", "vague", *[f for f in o.files if "cedar" in f.lower()])
    for q in NEGATIVE:
        o.ask(q, "absent")
