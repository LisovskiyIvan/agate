#!/usr/bin/env python3
"""Generate the 512x512 ASCII SDF font atlas embedded by zenderer (ui.zig).

Layout matches UICanvas.getGlyphUV: 16 columns x 8 rows of 32x64 cells,
ASCII 32..126. The glyph edge is encoded at 0.5 (128) and the distance
saturates after `--spread` pixels, which is the standard SDF convention the
UI shader (shaders/ui.glsl) expects.

Usage:
    python3 tools/generate_font_sdf.py \
        --font ~/Library/Fonts/SauceCodeProNerdFontMono-SemiBold.ttf \
        --size 50 --spread 5 --out src/zenderer/assets/font_sdf.png

The atlas was generated with SauceCodePro Nerd Font Mono SemiBold (a patched
Source Code Pro, SIL OFL 1.1); any monospace TTF/OTF works. Re-run with a
different --size/--spread to change weight or edge softness.
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 16, 8
CELL_W, CELL_H = 32, 64
FIRST, LAST = 32, 126


def edt1d(f: np.ndarray) -> np.ndarray:
    """Felzenszwalb 1D squared distance transform of a sampled function."""
    n = len(f)
    v = np.zeros(n, dtype=np.int64)
    z = np.zeros(n + 1, dtype=np.float64)
    k = 0
    z[0], z[1] = -np.inf, np.inf
    for q in range(1, n):
        while True:
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
            if s <= z[k]:
                k -= 1
            else:
                break
        k += 1
        v[k] = q
        z[k], z[k + 1] = s, np.inf
    d = np.empty(n, dtype=np.float64)
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def edt(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest True pixel."""
    inf = np.float64(1e12)
    f = np.where(mask, 0.0, inf)
    for y in range(f.shape[0]):
        f[y, :] = edt1d(f[y, :])
    for x in range(f.shape[1]):
        f[:, x] = edt1d(f[:, x])
    return np.sqrt(np.maximum(f, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help="path to a monospace TTF/OTF")
    ap.add_argument("--size", type=int, default=50, help="em size the atlas is rendered at")
    ap.add_argument("--spread", type=float, default=5.0, help="distance range stored, in px")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Render the mask at a multiple of the final cell so distances are stored
    # with sub-pixel precision instead of one value per pixel step.
    ss = 4
    font = ImageFont.truetype(args.font, args.size * ss)
    chars = [chr(c) for c in range(FIRST, LAST + 1)]

    # Common drawing origin for every glyph (in supersampled px), then center
    # the full ink union in the cell.
    minx = min(ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), ch, font=font)[0] for ch in chars)
    miny = min(ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), ch, font=font)[1] for ch in chars)
    maxx = max(ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), ch, font=font)[2] for ch in chars)
    maxy = max(ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), ch, font=font)[3] for ch in chars)

    off_x = round((CELL_W * ss - (maxx - minx)) / 2) - minx
    off_y = round((CELL_H * ss - (maxy - miny)) / 2) - miny

    atlas = np.zeros((ROWS * CELL_H, COLS * CELL_W), dtype=np.uint8)
    for code in range(FIRST, LAST + 1):
        glyph = Image.new("L", (CELL_W * ss, CELL_H * ss), 0)
        ImageDraw.Draw(glyph).text((off_x, off_y), chr(code), font=font, fill=255)
        mask = np.asarray(glyph) > 127

        to_ink = edt(mask)      # distance from any pixel to the nearest ink
        to_bg = edt(~mask)      # distance from any pixel to the nearest background
        signed = to_bg - to_ink  # positive inside the glyph, in supersampled px
        values = np.clip(128.0 + signed * (127.0 / (args.spread * ss)), 0.0, 255.0)

        # Pick one sample per final cell pixel.
        sampled = values[ss // 2::ss, ss // 2::ss]
        col, row = (code - FIRST) % COLS, (code - FIRST) // COLS
        atlas[row * CELL_H:(row + 1) * CELL_H, col * CELL_W:(col + 1) * CELL_W] = sampled.astype(np.uint8)

    Image.fromarray(atlas).save(args.out)
    print(
        f"wrote {args.out}: {COLS * CELL_W}x{ROWS * CELL_H}, "
        f"size={args.size}px spread={args.spread}px "
        f"ink={(maxx - minx) / ss:.1f}x{(maxy - miny) / ss:.1f} off=({off_x / ss:.1f},{off_y / ss:.1f})"
    )


if __name__ == "__main__":
    main()
