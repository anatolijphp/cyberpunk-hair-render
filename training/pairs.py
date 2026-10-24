#!/usr/bin/env python3
"""Capture pairs (see src/capture.h, README.md): which captures show the same view in two FSR
qualities, whether they really line up, and bilinear sampling. Used by prepare.py.

A pair is two captures of the same still scene: one at a low FSR quality (render < display) and one
at Native AA (render == display).
"""
import os
import re

import numpy as np

from capture_preview import read_meta, load, still_frames


def bilinear(img, xs, ys):
    """Samples img (H, W[, C]) at float pixel coordinates xs (W'), ys (H') -> (H', W'[, C])."""
    h, w = img.shape[:2]
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(xs - x0, 0, 1)[None, :]
    fy = np.clip(ys - y0, 0, 1)[:, None]
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    a = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    b = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return a * (1 - fy) + b * fy


def find_sweep_pairs(root):
    """Photo mode sweeps (folders <time>_pm<set>_r<rotation>_w<render width>): consecutive sets
    with different render widths are the same scene in two FSR qualities; pairs by rotation."""
    sets = {}
    for name in sorted(os.listdir(root)):
        m = re.match(r"^\d+_\d+_pm(\d+)_r(\d+)_w(\d+)$", name)
        if m:
            sets.setdefault(m.group(1), {"width": int(m.group(3)), "views": {}})["views"][m.group(2)] = name
    order = sorted(sets, key=lambda s: min(sets[s]["views"].values()))
    pairs, i = [], 0
    while i + 1 < len(order):
        a, b = sets[order[i]], sets[order[i + 1]]
        if a["width"] == b["width"]:   # a leftover sweep: the scene starts with the next one
            i += 1
            continue
        low, native = (a, b) if a["width"] < b["width"] else (b, a)
        for rot in sorted(low["views"]):
            if rot in native["views"]:
                pairs.append((low["views"][rot], native["views"][rot], "scene%d" % (len(pairs) // 1000)))
        print("scene: %s (%d) + %s (%d)" % (order[i], a["width"], order[i + 1], b["width"]))
        i += 2
    return pairs


def aligned(root, low, native):
    """The two captures show the same view: low-res mean vs the Native AA mean scaled down."""
    dl, dn = os.path.join(root, low), os.path.join(root, native)
    ml, mn = read_meta(dl), read_meta(dn)

    def mean(d, m):
        fs = [f for f in m["frames"] if os.path.exists(os.path.join(d, "%04d_color.raw" % f["index"]))][:32]
        st = np.stack([load(d, m, f["index"], "color") for f in fs])
        return st[still_frames(st)].mean(axis=0)

    lo, hi = mean(dl, ml), mean(dn, mn)
    s = mn["render"][0] / ml["render"][0]
    lx, ly = ml["box"][:2]
    hx, hy, hw, hh = mn["box"]

    def error(dx, dy):
        ys, xs = np.mgrid[0:lo.shape[0], 0:lo.shape[1]]
        X = ((xs + lx + 0.5 + dx) * s - 0.5 - hx).astype(int)
        Y = ((ys + ly + 0.5 + dy) * s - 0.5 - hy).astype(int)
        m = (X >= 1) & (Y >= 1) & (X < hw - 2) & (Y < hh - 2)
        v = sum(hi[Y[m] + a, X[m] + b] for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9
        return np.abs(v - lo[m]).mean() / lo[m].mean()

    e0 = error(0, 0)
    best_shifted = min(error(dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return e0 < 0.12 and e0 <= best_shifted, e0
