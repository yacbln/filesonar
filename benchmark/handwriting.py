"""Photos of handwriting: notebook pages, whiteboards and sticky notes, with known text.

Real photos of handwriting are in media.json too, but their exact words are unknown, so
searches can only target their subject. These are rendered with openly licensed
handwriting fonts (Google Fonts: OFL and Apache 2.0), then made to look photographed:
uneven lines, tilt, shadow, glare, blur and noise. Fonts are downloaded once into the
cache folder and checked against fixed checksums.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = "https://raw.githubusercontent.com/google/fonts/main/"
FONTS = {
    "caveat": ("ofl/caveat/Caveat%5Bwght%5D.ttf",
               "0bdb6b660482d31531b3945849fba5916b3ef8695da7024a9e6b9ee3c4157988"),
    "patrick": ("ofl/patrickhand/PatrickHand-Regular.ttf",
                "0f173b3e6cb6d1af25babf7f0057c5ac4ee11f9992b0469bb817e967ef4ad0fc"),
    "indie": ("ofl/indieflower/IndieFlower-Regular.ttf",
              "ccc94b22b156e9c5dfe50fd051f01b097600b252c24473e624bb43a143140a94"),
    "marker": ("apache/permanentmarker/PermanentMarker-Regular.ttf",
               "28f82c8a7943cb8e9d599f8554da1d4fc75dbcf69b9885ad6c0611d20c6946c5"),
}  # fmt: skip
PEN_FONTS = ("caveat", "patrick", "indie")
INKS = [(20, 30, 90), (15, 15, 20), (30, 60, 140)]  # blue and black ballpoint
MARKERS = [(20, 20, 25), (25, 70, 170), (190, 30, 40), (20, 120, 60)]


def fonts(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (rel, sha) in FONTS.items():
        path = cache / Path(rel).name.replace("%5Bwght%5D", "")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            data = httpx.get(BASE + rel, timeout=60, follow_redirects=True,
                             headers={"User-Agent": "filesonar-benchmark/1.0"}).content  # fmt: skip
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f"font {name} does not match its checksum")
            path.write_bytes(data)
        out[name] = path
    return out


def _wobbly_text(draw, xy, text, font, fill, rnd: random.Random) -> None:
    """Write word by word with a little drift, like a hand does."""
    x, y = xy
    for word in text.split(" "):
        dy = rnd.uniform(-2.5, 2.5)
        draw.text((x, y + dy), word, font=font, fill=fill)
        x += draw.textlength(word + " ", font=font) + rnd.uniform(-2, 4)


def _photograph(img: Image.Image, rnd: random.Random, glare: bool = False,
                background=(70, 62, 55)) -> Image.Image:  # fmt: skip
    """Perspective tilt, shadow, optional glare, blur and noise."""
    w, h = img.size
    canvas = Image.new("RGB", (w + 240, h + 240), background)
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 90)).filter(ImageFilter.GaussianBlur(12))
    canvas.paste(shadow, (135, 140), shadow)
    canvas.paste(img, (120, 120))
    # A gentle perspective: one side slightly further from the camera.
    dx, dy = rnd.uniform(10, 45), rnd.uniform(5, 30)
    cw, ch = canvas.size
    # Corners of the source area, in Pillow's order: upper-left, lower-left, lower-right,
    # upper-right.
    quad = (dx, dy, 0, ch - dy * 0.5, cw, ch, cw - dx * 0.3, 0)
    canvas = canvas.transform((cw, ch), Image.QUAD, quad, Image.BICUBIC)
    canvas = canvas.rotate(rnd.uniform(-4, 4), resample=Image.BICUBIC, fillcolor=background)
    if glare:  # a bright reflection, as on a glossy whiteboard
        layer = Image.new("L", canvas.size, 0)
        d = ImageDraw.Draw(layer)
        cx, cy = rnd.uniform(0.3, 0.7) * cw, rnd.uniform(0.2, 0.5) * ch
        for r in range(int(cw * 0.25), 0, -12):
            d.ellipse(
                (cx - r * 1.6, cy - r, cx + r * 1.6, cy + r), fill=int(70 * (1 - r / (cw * 0.25)))
            )
        canvas = Image.composite(Image.new("RGB", canvas.size, "white"), canvas, layer)
    canvas = canvas.filter(ImageFilter.GaussianBlur(rnd.uniform(0.4, 0.9)))
    px = canvas.load()
    for _ in range(cw * ch // 30):
        x, y = rnd.randrange(cw), rnd.randrange(ch)
        r, g, b = px[x, y]
        n = rnd.randint(-14, 14)
        px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return canvas


def notebook_page(lines: list[str], fontfiles: dict[str, Path], seed: int) -> Image.Image:
    rnd = random.Random(seed)
    w, h, gap = 1100, 1400, 58
    page = Image.new("RGB", (w, h), (252, 251, 245))
    d = ImageDraw.Draw(page)
    for y in range(140, h - 40, gap):
        d.line((0, y, w, y), fill=(170, 200, 230), width=2)
    d.line((110, 0, 110, h), fill=(230, 150, 150), width=2)
    f = ImageFont.truetype(str(fontfiles[rnd.choice(PEN_FONTS)]), rnd.randint(44, 52))
    ink = rnd.choice(INKS)
    for i, line in enumerate(lines[: (h - 180) // gap]):
        _wobbly_text(d, (135 + rnd.uniform(-6, 6), 140 + i * gap - 46), line, f, ink, rnd)
    return _photograph(
        page, rnd, background=rnd.choice([(70, 62, 55), (120, 100, 80), (60, 70, 80)])
    )


def whiteboard(
    lines: list[str], fontfiles: dict[str, Path], seed: int, boxes: int = 2
) -> Image.Image:
    rnd = random.Random(seed)
    w, h = 1600, 1000
    board = Image.new("RGB", (w, h), (243, 245, 244))
    d = ImageDraw.Draw(board)
    d.rectangle((0, 0, w - 1, h - 1), outline=(150, 150, 155), width=22)
    f = ImageFont.truetype(str(fontfiles["marker"]), 50)
    y = 70
    for i, line in enumerate(lines):
        color = rnd.choice(MARKERS) if i else MARKERS[0]
        _wobbly_text(d, (90 + (40 if line.startswith("-") else 0), y), line, f, color, rnd)
        y += 85
    for _ in range(boxes):  # arrows and boxes people draw around ideas
        x0, y0 = rnd.randint(950, 1250), rnd.randint(100, 650)
        d.rounded_rectangle(
            (x0, y0, x0 + 260, y0 + 130), radius=18, outline=rnd.choice(MARKERS), width=6
        )
        d.line((x0 - 120, y0 + 65, x0, y0 + 65), fill=MARKERS[0], width=6)
        d.polygon(((x0, y0 + 65), (x0 - 22, y0 + 52), (x0 - 22, y0 + 78)), fill=MARKERS[0])
    return _photograph(board, rnd, glare=True, background=(200, 196, 188))


def sticky_notes(notes: list[str], fontfiles: dict[str, Path], seed: int) -> Image.Image:
    rnd = random.Random(seed)
    desk = Image.new(
        "RGB", (1400, 1000), rnd.choice([(150, 112, 80), (205, 205, 200), (90, 90, 95)])
    )
    colors = [(255, 236, 120), (255, 170, 190), (170, 235, 160), (150, 210, 255)]
    f = ImageFont.truetype(str(fontfiles[rnd.choice(PEN_FONTS)]), 40)
    for i, text in enumerate(notes[:6]):
        note = Image.new("RGB", (380, 380), colors[i % len(colors)])
        d = ImageDraw.Draw(note)
        words, line, y = text.split(), "", 30
        for word in words:  # wrap by hand
            if d.textlength(line + word, font=f) > 320:
                d.text((28, y), line, font=f, fill=(25, 25, 35))
                line, y = "", y + 52
            line += word + " "
        d.text((28, y), line, font=f, fill=(25, 25, 35))
        note = note.rotate(rnd.uniform(-8, 8), expand=True, fillcolor=desk.getpixel((0, 0)))
        desk.paste(note, (60 + (i % 3) * 440, 60 + (i // 3) * 450))
    return _photograph(desk, random.Random(seed + 1), background=(60, 55, 50))
