#!/usr/bin/env python3
"""Training scenes for the resolve network from photo mode capture pairs (see README.md).

usage: prepare.py <capture root> <out dir> [sub-samples per axis: 1 = point sample (default), 3 = area]

For every photo mode pair (see pairs.find_sweep_pairs) writes <out>/<low>.npz, at render
resolution, cropped to where both captures overlap:
  raw       (N, h, w, 3) float16   the game's frame before the mod and FSR (dithered hair), per frame
  cov, kept (N, h, w)    float16   the mod's hair coverage / hair-surface mask of the same frames
  resolved  (N, h, w, 3) float16   the mod's resolve of the same frames (what FSR gets today)
  target    (N, h, w, 3) float16   the Native AA reference at each render pixel's centre, moved by that
                                   frame's jitter: what the game would render without the dither. A point
                                   sample, not an area average: FSR rebuilds detail from sharp jittered
                                   frames, pre-filtered ones only make its output softer
The sign convention of the jitter is found from the data (the one under which the mod's resolve
matches the reference best).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from capture_preview import read_meta, load, still_frames   # noqa: E402
from pairs import bilinear, find_sweep_pairs, aligned          # noqa: E402

SUB = 1   # sub-samples per axis inside a render pixel (1: the pixel centre)


def area_target(native, nbox, scale, rx, ry, jx, jy):
    """Mean of the Native AA image over each render pixel (rx, ry: render pixel indices) shifted by
    the jitter (jx, jy) in render pixels."""
    acc = 0
    for i in range(SUB):
        xs = (rx + (i + 0.5) / SUB + jx) * scale - 0.5 - nbox[0]
        for k in range(SUB):
            ys = (ry + (k + 0.5) / SUB + jy) * scale - 0.5 - nbox[1]
            acc = acc + bilinear(native, xs, ys)
    return acc / (SUB * SUB)


def prepare(root, low, native, out_dir):
    dl, dn = os.path.join(root, low), os.path.join(root, native)
    ml, mn = read_meta(dl), read_meta(dn)
    scale = mn["render"][0] / ml["render"][0]

    nf = [f for f in mn["frames"] if os.path.exists(os.path.join(dn, "%04d_color.raw" % f["index"]))]
    stack = np.stack([load(dn, mn, f["index"], "color") for f in nf])
    ref = stack[still_frames(stack)].mean(axis=0)
    nbox = mn["box"]

    frames = [f for f in ml["frames"] if f["targets"] and "jitter" in f and
              os.path.exists(os.path.join(dl, "%04d_color.raw" % f["index"]))]
    raws = np.stack([load(dl, ml, f["index"], "color") for f in frames])
    keep = still_frames(raws)
    frames = [f for f, k in zip(frames, keep) if k]
    raws = raws[keep]

    # render pixels whose whole area (plus the jitter) lies inside the Native AA crop
    lx, ly, lw, lh = ml["box"]
    x0 = max(lx, int(np.ceil((nbox[0] + 2) / scale)) + 1)
    y0 = max(ly, int(np.ceil((nbox[1] + 2) / scale)) + 1)
    x1 = min(lx + lw, int((nbox[0] + nbox[2] - 2) / scale) - 1)
    y1 = min(ly + lh, int((nbox[1] + nbox[3] - 2) / scale) - 1)
    if x1 - x0 < 64 or y1 - y0 < 64:
        print("%s: overlap too small, skipped" % low)
        return
    rx, ry = np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64)
    crop = lambda img: img[y0 - ly:y1 - ly, x0 - lx:x1 - lx]

    resolved = np.stack([crop(load(dl, ml, f["index"], "resolved")) for f in frames])
    # jitter sign: the one under which the mod's resolve matches the reference best
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            err = 0
            for k in range(0, len(frames), max(1, len(frames) // 8)):
                jx, jy = frames[k]["jitter"]
                t = area_target(ref, nbox, scale, rx, ry, sx * jx, sy * jy)
                err += np.abs(t - resolved[k]).mean()
            if best is None or err < best[0]:
                best = (err, sx, sy)
    _, sx, sy = best

    targets = np.stack([area_target(ref, nbox, scale, rx, ry, sx * f["jitter"][0], sy * f["jitter"][1])
                        for f in frames])
    data = {
        "raw": np.stack([crop(r) for r in raws]).astype(np.float16),
        "cov": np.stack([crop(load(dl, ml, f["index"], "coverage")) for f in frames]).astype(np.float16),
        "kept": np.stack([crop(load(dl, ml, f["index"], "kept")) for f in frames]).astype(np.float16),
        "resolved": resolved.astype(np.float16),
        "target": targets.astype(np.float16),
        "jitter_sign": np.array([sx, sy], np.int8),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, low + ".npz"), **data)
    print("%s: %dx%d render px, %d frames, jitter sign (%+d, %+d)" % (low, x1 - x0, y1 - y0, len(frames), sx, sy),
          flush=True)


def main():
    global SUB
    root, out_dir = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        SUB = int(sys.argv[3])
    for low, native, _ in find_sweep_pairs(root):
        if os.path.exists(os.path.join(out_dir, low + ".npz")):   # done by an earlier run
            continue
        ok, err = aligned(root, low, native)
        if not ok:
            print("%s: skipped, not the same view (error %.3f)" % (low, err))
            continue
        prepare(root, low, native, out_dir)


if __name__ == "__main__":
    main()
