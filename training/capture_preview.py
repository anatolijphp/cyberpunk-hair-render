#!/usr/bin/env python3
"""Checks a training capture (hairfix_capture/<date_time>/) and writes preview PNGs into it.

usage: capture_preview.py <capture dir>

- reference.png:   mean game colour of all frames (dither and FSR jitter average out)
- input.png:       one game colour frame (what FSR / the mod start from)
- resolved.png:    the mod's resolve of that frame
- output_on.png / output_off.png: FSR output with / without the mod (display resolution)
- sheet.png:       all of the above side by side (render-resolution images zoomed 2x)
Prints how much each frame differs from the mean: a still scene stays low and flat.
"""
import os
import sys

import numpy as np
from PIL import Image

FLOAT_FORMATS = {10: np.float16, 2: np.float32}   # DXGI R16G16B16A16_FLOAT, R32G32B32A32_FLOAT


def read_meta(d):
    meta = {"buffers": {}, "frames": []}
    for line in open(os.path.join(d, "meta.txt")):
        p = line.split()
        if not p:
            continue
        if p[0] in ("render", "output"):
            meta[p[0]] = (int(p[1]), int(p[2]))
        elif p[0] in ("box", "output_box"):
            meta[p[0]] = tuple(int(v) for v in p[1:5])
        elif p[0] == "path":
            meta["path"] = p[1]
        elif p[0] == "buffer":
            meta["buffers"][p[1]] = (int(p[3]), int(p[5]))
        elif p[0] == "frame":
            f = {"index": int(p[1]), "fix": p[5], "targets": p[7] == "1", "output": p[9] == "1"}
            if "jitter" in p:
                i = p.index("jitter")
                f["jitter"] = (float(p[i + 1]), float(p[i + 2]))
            if "game" in p:   # upscaler frame number since the capture started (the dither phase)
                f["game"] = int(p[p.index("game") + 1])
            meta["frames"].append(f)
    return meta


def load(d, meta, index, name):
    path = os.path.join(d, "%04d_%s.raw" % (index, name))
    if not os.path.exists(path):
        return None
    fmt, bpp = meta["buffers"][name]
    w, h = meta["output_box"][2:] if name == "output" else meta["box"][2:]
    raw = np.fromfile(path, dtype=np.uint8)
    if bpp == 1:
        return raw.reshape(h, w).astype(np.float32) / 255.0
    if bpp == 2:   # R16_UNORM coverage
        return raw.view(np.uint16).reshape(h, w).astype(np.float32) / 65535.0
    if fmt in FLOAT_FORMATS:
        return raw.view(FLOAT_FORMATS[fmt]).reshape(h, w, -1)[..., :3].astype(np.float32)
    raise SystemExit("unsupported format %d for %s" % (fmt, name))


def still_frames(stack):
    """Mask of the frames that show the same scene: dither and jitter make every frame differ a
    little from the median, a frame after the camera moved or a menu closed differs a lot."""
    med = np.median(stack, axis=0)
    diff = np.abs(stack - med).mean(axis=(1, 2, 3)) / max(float(med.mean()), 1e-6)
    m = np.median(diff)
    mad = np.median(np.abs(diff - m))
    return diff <= m + 6 * mad + 0.02


def to_png(img, exposure):
    """HDR linear colour -> displayable 8 bit (Reinhard + sRGB-ish gamma)."""
    x = np.clip(img * exposure, 0, None)
    x = x / (1 + x)
    x = np.power(x, 1 / 2.2)
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    d = sys.argv[1]
    meta = read_meta(d)
    frames = [f for f in meta["frames"] if os.path.exists(os.path.join(d, "%04d_color.raw" % f["index"]))]
    if not frames:
        raise SystemExit("no colour frames")
    stack = np.stack([load(d, meta, f["index"], "color") for f in frames])
    good = still_frames(stack)
    rejected = [f["index"] for f, g in zip(frames, good) if not g]
    if rejected:
        print("rejected %d frames that differ from the scene (camera / menu changed): %s" % (len(rejected), rejected))
    frames = [f for f, g in zip(frames, good) if g]
    stack = stack[good]
    colors = list(stack)
    ref = stack.mean(axis=0)
    exposure = 0.5 / max(float(np.median(ref)), 1e-4)

    diffs = np.abs(stack - ref).mean(axis=(1, 2, 3)) / max(float(ref.mean()), 1e-6)
    print("%s: %d frames, render %s, box %s" % (meta.get("path"), len(colors), meta["render"], meta["box"]))
    print("difference from the mean (relative): min %.4f  median %.4f  max %.4f" %
          (diffs.min(), np.median(diffs), diffs.max()))
    half = len(diffs) // 2
    if diffs[half:].mean() > 1.5 * diffs[:half].mean() + 0.01 or diffs.max() > 0.25:
        print("WARNING: the scene does not look still (camera / hair moved during the capture)")
    jit = [f["jitter"] for f in frames if "jitter" in f]
    if jit:
        print("jitter: %d distinct offsets" % len(set(jit)))

    first_on = next(f for f in frames if f["targets"])
    images = {"reference": to_png(ref, exposure),
              "input": to_png(load(d, meta, first_on["index"], "color"), exposure),
              "resolved": to_png(load(d, meta, first_on["index"], "resolved"), exposure)}
    for fix in ("on", "off"):
        outs = [load(d, meta, f["index"], "output") for f in frames if f["fix"] == fix and f["output"]]
        outs = [o for o in outs if o is not None]
        if outs:
            images["output_" + fix] = to_png(outs[-1], exposure)
    for name, im in images.items():
        im.save(os.path.join(d, name + ".png"))

    # sheet: render-resolution images zoomed 2x (nearest) next to the display-resolution outputs
    tiles = []
    for name, im in images.items():
        if not name.startswith("output"):
            im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
        tiles.append(im)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), h), (255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    sheet.save(os.path.join(d, "sheet.png"))
    print("previews: " + ", ".join(n + ".png" for n in images) + ", sheet.png")


if __name__ == "__main__":
    main()
