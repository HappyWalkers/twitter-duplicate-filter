#!/usr/bin/env python3
"""
Format raw screen captures into Chrome Web Store screenshots.

The store wants exactly 1280x800. A desktop capture is 2559x1492, which is 1.715:1
against the store's 1.6:1 -- so it has to be CROPPED, not scaled. Scaling a 1.715 image
into a 1.6 frame stretches every face and letterform vertically, which looks subtly wrong
in a way reviewers and users notice without being able to say why.

Cropping from full height and trimming the right edge keeps the layout intact: the left
navigation, the timeline, and the news sidebar all survive, and what goes is empty margin
and X's floating action buttons.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# A caption is worth the vertical space it costs. At 1280x800 the "seen before" control is
# small, and a reviewer scanning a listing has seconds to work out what the extension does.
# The band is drawn INSIDE the frame rather than added to it, so the result stays exactly
# 1280x800 without a second resize.
CAPTIONS = [
    "Posts you have already been shown collapse behind a control",
    "Click it to bring the post back \u2014 nothing is ever deleted",
]
BAND_H = 68
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

TARGET = (1280, 800)
OUT = Path(__file__).parent.parent / "store" / "store-assets"


def caption(im, text):
    """Draw a caption band across the bottom, over the image rather than beside it."""
    font = None
    for f in FONTS:
        try:
            font = ImageFont.truetype(f, 25)
            break
        except OSError:
            continue
    if font is None:
        return im                      # no usable font: ship the clean version instead
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    d.rectangle([0, h - BAND_H, w, h], fill=(0, 0, 0, 255))
    d.line([(0, h - BAND_H), (w, h - BAND_H)], fill=(29, 155, 240, 255), width=3)
    tw = d.textlength(text, font=font)
    d.text(((w - tw) / 2, h - BAND_H + 20), text, font=font, fill=(255, 255, 255, 255))
    return im


def fit(src: Path, dst: Path, text=None):
    im = Image.open(src)
    if im.mode != "RGB":
        im = im.convert("RGB")          # drop alpha: the store wants opaque PNG/JPEG
    w, h = im.size
    want = TARGET[0] / TARGET[1]

    if w / h > want:
        # Wider than the target: keep full height, trim width from the right, where the
        # content is margin rather than substance.
        new_w = int(round(h * want))
        im = im.crop((0, 0, min(new_w, w), h))
    else:
        # Taller than the target: keep full width and trim from the BOTTOM, since the
        # thing being demonstrated sits at the top of the timeline.
        new_h = int(round(w / want))
        im = im.crop((0, 0, w, min(new_h, h)))

    im = im.resize(TARGET, Image.LANCZOS)
    if text:
        im = caption(im, text)
    im.save(dst, "PNG", optimize=True)
    return im.size


def main():
    srcs = sys.argv[1:]
    if not srcs:
        sys.exit("usage: format-screenshots.py <capture.png> [more.png ...]\n"
                 "First image should show the collapsed state, second the expanded one.")
    names = ["screenshot-1-collapsed.png", "screenshot-2-expanded.png",
             "screenshot-3.png", "screenshot-4.png", "screenshot-5.png"]
    for i, s in enumerate(srcs[:5]):
        src = Path(s)
        # Clean and captioned versions of each, so the choice is a matter of taste rather
        # than of re-running anything.
        for suffix, text in (("", None),
                             ("-captioned", CAPTIONS[i] if i < len(CAPTIONS) else None)):
            dst = OUT / names[i].replace(".png", f"{suffix}.png")
            size = fit(src, dst, text)
            print(f"  -> {dst.name}  {size[0]}x{size[1]}  {dst.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
