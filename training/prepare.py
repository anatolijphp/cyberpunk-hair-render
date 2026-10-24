#!/usr/bin/env python3
"""Training scenes for the resolve network (see README.md).

usage: prepare.py <capture root> <out dir> [sub-samples per axis: 1 = point sample (default), 3 = area]
       prepare.py <capture root> <out dir> averaged

Writes <out>/<capture>.npz at render resolution:
  raw       (N, h, w, 3) float16   the game's frame before the mod and FSR (dithered hair), per frame
  cov, kept (N, h, w)    float16   the mod's hair coverage / hair-surface mask of the same frames
  resolved  (N, h, w, 3) float16   the mod's resolve of the same frames (what FSR gets today)
  target    (N, h, w, 3) float16   what the game would render without the dither, at each render pixel's
                                   centre moved by that frame's jitter
  index     (N,)         int32     the frames' capture indices

Targets, two ways:
- pairs (default): every photo mode pair (see pairs.find_sweep_pairs), the Native AA reference
  sampled at each render pixel's centre plus the jitter, cropped to where both captures overlap. A
  point sample, not an area average: FSR rebuilds detail from sharp jittered frames, pre-filtered
  ones only make its output softer. The sign convention of the jitter is found from the data (the
  one under which the mod's resolve matches the reference best).
- averaged: every single capture with frame numbers (src/capture.cpp): the mean of the capture's own
  frames with the same jitter over all phases of the game's hair dither (frame number modulo 5),
  each phase weighted equally. Same resolution and jitter as the input, no Native AA capture.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from capture_preview import read_meta, load, still_frames   # noqa: E402
from pairs import bilinear, find_sweep_pairs, aligned          # noqa: E402

SUB = 1             # sub-samples per axis inside a render pixel (1: the pixel centre)
DITHER_PERIOD = 5   # the game's hair dither repeats every 5 frames (shaders/hair_coverage.hlsl)


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


def has(d, f, name="color"):
    return os.path.exists(os.path.join(d, "%04d_%s.raw" % (f["index"], name)))


def input_frames(d, m):
    """The frames with the mod's targets and a jitter, without the ones where the scene moved."""
    frames = [f for f in m["frames"] if f["targets"] and "jitter" in f and has(d, f)]
    raws = np.stack([load(d, m, f["index"], "color") for f in frames])
    keep = still_frames(raws)
    return [f for f, k in zip(frames, keep) if k], raws[keep]


def scene_data(d, m, frames, raws, targets, crop):
    return {
        "raw": np.stack([crop(r) for r in raws]).astype(np.float16),
        "cov": np.stack([crop(load(d, m, f["index"], "coverage")) for f in frames]).astype(np.float16),
        "kept": np.stack([crop(load(d, m, f["index"], "kept")) for f in frames]).astype(np.float16),
        "resolved": np.stack([crop(load(d, m, f["index"], "resolved")) for f in frames]).astype(np.float16),
        "target": targets.astype(np.float16),
        "index": np.array([f["index"] for f in frames], np.int32),
    }


def pair_scene(root, low, native):
    """Scene data with Native AA targets (dict), the crop (x0, y0, x1, y1) in render pixels, and a
    description; None when the overlap is too small."""
    dl, dn = os.path.join(root, low), os.path.join(root, native)
    ml, mn = read_meta(dl), read_meta(dn)
    scale = mn["render"][0] / ml["render"][0]

    nf = [f for f in mn["frames"] if has(dn, f)]
    stack = np.stack([load(dn, mn, f["index"], "color") for f in nf])
    ref = stack[still_frames(stack)].mean(axis=0)
    nbox = mn["box"]
    frames, raws = input_frames(dl, ml)

    # render pixels whose whole area (plus the jitter) lies inside the Native AA crop
    lx, ly, lw, lh = ml["box"]
    x0 = max(lx, int(np.ceil((nbox[0] + 2) / scale)) + 1)
    y0 = max(ly, int(np.ceil((nbox[1] + 2) / scale)) + 1)
    x1 = min(lx + lw, int((nbox[0] + nbox[2] - 2) / scale) - 1)
    y1 = min(ly + lh, int((nbox[1] + nbox[3] - 2) / scale) - 1)
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    rx, ry = np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64)
    crop = lambda img: img[y0 - ly:y1 - ly, x0 - lx:x1 - lx]

    resolved = [crop(load(dl, ml, f["index"], "resolved")) for f in frames]
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
    data = scene_data(dl, ml, frames, raws, targets, crop)
    data["jitter_sign"] = np.array([sx, sy], np.int8)
    return data, (x0, y0, x1, y1), "%d frames, jitter sign (%+d, %+d)" % (len(frames), sx, sy)


def averaged_scene(root, name):
    """Scene data with averaged targets (dict), the crop (x0, y0, x1, y1) in render pixels (the whole
    capture box), and a description; None without frame numbers or complete jitters."""
    d = os.path.join(root, name)
    m = read_meta(d)
    if not any("game" in f for f in m["frames"]):
        return None
    frames, raws = input_frames(d, m)
    # every recorded frame of each jitter, by dither phase
    groups = {}
    for f in m["frames"]:
        if "jitter" in f and "game" in f and has(d, f):
            groups.setdefault(f["jitter"], {}).setdefault(f["game"] % DITHER_PERIOD, []).append(f["index"])
    usable = [k for k, f in enumerate(frames) if len(groups.get(f["jitter"], {})) == DITHER_PERIOD]
    if not usable:
        return None
    targets, cache = [], {}
    same, other = [], []   # self-check: frames of one jitter and phase must agree, phases must not
    for k in usable:
        j = frames[k]["jitter"]
        if j not in cache:
            phases = []
            for p in range(DITHER_PERIOD):
                imgs = [load(d, m, i, "color").astype(np.float32) for i in groups[j][p]]
                scale = max(float(imgs[0].mean()), 1e-6)
                same += [np.abs(a - imgs[0]).mean() / scale for a in imgs[1:]]
                phases.append(np.mean(imgs, axis=0))
            other += [np.abs(phases[p] - phases[0]).mean() / max(float(phases[0].mean()), 1e-6)
                      for p in range(1, DITHER_PERIOD)]
            cache[j] = np.mean(phases, axis=0)
        targets.append(cache[j])
    frames = [frames[k] for k in usable]
    raws = raws[usable]
    lx, ly, lw, lh = m["box"]
    data = scene_data(d, m, frames, raws, np.stack(targets), lambda img: img)
    check = "same jitter + phase differ %.4f, phases differ %.4f" % (
        np.median(same) if same else float("nan"), np.median(other) if other else float("nan"))
    return data, (lx, ly, lx + lw, ly + lh), "%d frames, %d jitters averaged (%s)" % (len(frames), len(cache), check)


def main():
    global SUB
    root, out_dir = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "1"
    os.makedirs(out_dir, exist_ok=True)
    if mode == "averaged":
        for name in sorted(os.listdir(root)):
            if not os.path.exists(os.path.join(root, name, "meta.txt")) or \
                    os.path.exists(os.path.join(out_dir, name + ".npz")):
                continue
            r = averaged_scene(root, name)
            if r is None:
                print("%s: no frame numbers or no jitter in every dither phase, skipped" % name)
                continue
            data, (x0, y0, x1, y1), info = r
            np.savez(os.path.join(out_dir, name + ".npz"), **data)
            print("%s: %dx%d render px, %s" % (name, x1 - x0, y1 - y0, info), flush=True)
        return
    SUB = int(mode)
    for low, native, _ in find_sweep_pairs(root):
        if os.path.exists(os.path.join(out_dir, low + ".npz")):   # done by an earlier run
            continue
        ok, err = aligned(root, low, native)
        if not ok:
            print("%s: skipped, not the same view (error %.3f)" % (low, err))
            continue
        r = pair_scene(root, low, native)
        if r is None:
            print("%s: overlap too small, skipped" % low)
            continue
        data, (x0, y0, x1, y1), info = r
        np.savez(os.path.join(out_dir, low + ".npz"), **data)
        print("%s: %dx%d render px, %s" % (low, x1 - x0, y1 - y0, info), flush=True)


if __name__ == "__main__":
    main()
