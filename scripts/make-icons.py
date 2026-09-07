#!/usr/bin/env python3
"""Generate the extension icons.

The mark is three stacked cards collapsing into one: the top card full-width and solid,
the two beneath progressively narrower and dimmer, i.e. duplicates folding away behind
the post you keep. Drawn at 8x and downsampled so the edges stay clean at 16px, which is
the size that actually has to survive -- a mark that only reads at 128 is a mark nobody
sees, since the toolbar and the extensions menu both render it small.
"""
from PIL import Image, ImageDraw
from pathlib import Path

OUT = Path(__file__).parent.parent / "store" / "icons"
BLUE = (29, 155, 240)          # X's link blue, so it sits naturally in the UI
SIZES = (16, 32, 48, 128)
SS = 8                         # supersample factor


def rounded(d, box, r, fill):
    d.rounded_rectangle(box, radius=r, fill=fill)


def draw(px):
    n = px * SS
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = n / 32.0               # design on a 32-unit grid, scaled

    # Back cards: narrower and more transparent the further back they sit.
    rounded(d, [9 * u, 4 * u, 23 * u, 9 * u], 2 * u, BLUE + (70,))
    rounded(d, [6.5 * u, 9 * u, 25.5 * u, 15 * u], 2.2 * u, BLUE + (130,))
    # Front card: the one post that survives.
    rounded(d, [4 * u, 15 * u, 28 * u, 28 * u], 3 * u, BLUE + (255,))

    # Two text lines on the front card, knocked out so the shape reads as a post.
    d.rounded_rectangle([7.5 * u, 19 * u, 21 * u, 21 * u], radius=1 * u, fill=(255, 255, 255, 235))
    d.rounded_rectangle([7.5 * u, 23 * u, 16.5 * u, 25 * u], radius=1 * u, fill=(255, 255, 255, 180))
    return img.resize((px, px), Image.LANCZOS)


OUT.mkdir(parents=True, exist_ok=True)
for s in SIZES:
    draw(s).save(OUT / f"icon{s}.png")
    print(f"  icon{s}.png")
# 440x280 small promo tile for the store listing, mark centred on a soft field.
tile = Image.new("RGBA", (440, 280), (14, 24, 33, 255))
mark = draw(128).resize((150, 150), Image.LANCZOS)
tile.alpha_composite(mark, (145, 65))
tile.convert("RGB").save(OUT.parent / "store-assets" / "promo-440x280.png")
print("  promo-440x280.png")
