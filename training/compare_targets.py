#!/usr/bin/env python3
"""Compares the two kinds of training target (prepare.py) on one view: the Native AA reference of a
photo mode pair, and the average of the low quality capture's own frames per jitter.

usage: compare_targets.py <capture root> <low quality capture> <Native AA capture> [<sheet.png>]

Prints hair PSNR (tone-mapped, as train.py measures) between the targets, and of the raw frame and the
mod's resolve against each. The sheet shows raw | resolve | averaged | Native AA for one frame.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import prepare   # noqa: E402
import train     # noqa: E402


def main():
    root, low, native = sys.argv[1:4]
    pair = prepare.pair_scene(root, low, native)
    avg = prepare.averaged_scene(root, low)
    if pair is None or avg is None:
        raise SystemExit("need a Native AA pair with overlap and a capture with averaged jitters")
    (p, (px0, py0, px1, py1), pinfo), (a, (ax0, ay0, _, _), ainfo) = pair, avg
    print("Native AA: %s\naveraged:  %s" % (pinfo, ainfo))
    common = sorted(set(p["index"]) & set(a["index"]))
    if not common:
        raise SystemExit("no frame has both targets")
    pi = {int(i): k for k, i in enumerate(p["index"])}
    ai = {int(i): k for k, i in enumerate(a["index"])}
    crop = lambda img: img[py0 - ay0:py1 - ay0, px0 - ax0:px1 - ax0]   # the averaged scene to the pair's crop

    rows = []
    for i in common:
        k, j = pi[int(i)], ai[int(i)]
        mask = train.hair_mask(p["cov"][k], p["kept"][k])
        tm = lambda img: train.tonemap(img, train.EXPOSURE)
        nat, av = tm(p["target"][k]), tm(crop(a["target"][j]))
        raw, res = tm(p["raw"][k]), tm(p["resolved"][k])
        rows.append({"averaged vs Native AA": train.psnr(av, nat, mask),
                     "raw vs Native AA": train.psnr(raw, nat, mask), "raw vs averaged": train.psnr(raw, av, mask),
                     "resolve vs Native AA": train.psnr(res, nat, mask),
                     "resolve vs averaged": train.psnr(res, av, mask)})
    print("hair PSNR over %d frames:" % len(rows))
    for key in rows[0]:
        print("  %-22s %6.2f dB" % (key, np.mean([r[key] for r in rows])))
    if len(sys.argv) > 4:
        k, j = pi[int(common[0])], ai[int(common[0])]
        tm = lambda img: train.tonemap(img, train.EXPOSURE)
        imgs = [tm(p["raw"][k]), tm(p["resolved"][k]), tm(crop(a["target"][j])), tm(p["target"][k])]
        h, w = imgs[0].shape[:2]
        train.sheet(imgs, sys.argv[4], (0, h, 0, w))
        print("sheet: raw | resolve | averaged | Native AA -> %s" % sys.argv[4])


if __name__ == "__main__":
    main()
