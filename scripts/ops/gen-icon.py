#!/usr/bin/env python3
"""Generate the `primitive` app icon — a flat geometric mark of overlapping translucent triangles
(the app's own motif: it rebuilds images *from* triangles). Deterministic, offline (PIL only).

Outputs into assets/icons/:
  - icon_{512,256,128,64,32,16}.png   (the plain size set the bundle / docs use)
  - primitive.iconset/ + primitive.icns (Apple bundle icon, via iconutil — invoked by the caller)

Run: python3 scripts/ops/gen-icon.py   (then the wrapper runs `iconutil` to emit the .icns)
"""
from __future__ import annotations

import os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "assets", "icons")
SS = 4  # supersample factor for crisp anti-aliased edges
MASTER = 1024

# Flat, intentional palette (dark navy field; warm + cool translucent primitives).
BG = (29, 29, 43, 255)
TRIS = [
    # (points as fractions of the canvas, RGBA fill)
    ([(0.18, 0.78), (0.52, 0.16), (0.80, 0.74)], (244, 114, 96, 205)),   # coral
    ([(0.30, 0.30), (0.86, 0.40), (0.58, 0.86)], (64, 196, 180, 200)),   # teal
    ([(0.12, 0.46), (0.66, 0.22), (0.50, 0.80)], (245, 197, 66, 190)),   # gold
]


def rounded_mask(size: int, radius_frac: float = 0.225) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    r = int(size * radius_frac)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    return m


def render_master() -> Image.Image:
    s = MASTER * SS
    base = Image.new("RGBA", (s, s), BG)
    # Each translucent triangle on its own layer so alpha composites cleanly (like the app's blend).
    for pts, rgba in TRIS:
        layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon([(x * s, y * s) for x, y in pts], fill=rgba)
        base = Image.alpha_composite(base, layer)
    base = base.resize((MASTER, MASTER), Image.LANCZOS)
    # Round the corners (macOS app-icon convention).
    out = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    out.paste(base, (0, 0), rounded_mask(MASTER))
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    master = render_master()

    for size in (512, 256, 128, 64, 32, 16):
        master.resize((size, size), Image.LANCZOS).save(
            os.path.join(OUT, f"icon_{size}.png")
        )

    # Apple .iconset layout (base + @2x for each logical size).
    iconset = os.path.join(OUT, "primitive.iconset")
    os.makedirs(iconset, exist_ok=True)
    for logical in (16, 32, 128, 256, 512):
        master.resize((logical, logical), Image.LANCZOS).save(
            os.path.join(iconset, f"icon_{logical}x{logical}.png")
        )
        dbl = logical * 2
        master.resize((dbl, dbl), Image.LANCZOS).save(
            os.path.join(iconset, f"icon_{logical}x{logical}@2x.png")
        )
    print(f"icons → {OUT}")


if __name__ == "__main__":
    main()
