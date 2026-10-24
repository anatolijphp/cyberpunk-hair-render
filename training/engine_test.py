#!/usr/bin/env python3
"""Checks the mod's network engine (engine_test.exe: src/neural.cpp + shaders/nn_*.hlsl on the GPU)
against PyTorch on one frame of a prepared scene (prepare.py). Build engine_test.exe as README.md says.

usage: engine_test.py <engine_test.exe> <model.pt> <scene.npz> <work dir>
"""
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import export   # noqa: E402
import train    # noqa: E402

BORDER = 32   # at the frame edge PyTorch pads with zeros, the mod repeats the screen edge


def main():
    exe, model, scene, work = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    s = train.Scene(scene)
    net, info = train.load(model)
    sys.argv = ["export.py", model, os.path.join(work, "hairfix_resolve.bin")]
    export.main()
    net.eval()
    s.exposure = info["exposure"]
    # the game's coverage / kept targets are 8-bit
    q8 = lambda a: np.round(np.clip(a.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    cov8, kept8 = q8(s.cov[0]), q8(s.kept[0])
    s.cov = (cov8[None] / 255.0).astype(np.float16)
    s.kept = (kept8[None] / 255.0).astype(np.float16)
    s.mask = np.stack([train.hair_mask(s.cov[0], s.kept[0])]).astype(np.float16)
    h, w = s.target.shape[:2]
    rgba = lambda img: np.concatenate([img.astype(np.float16), np.ones((h, w, 1), np.float16)], 2)
    with open(os.path.join(work, "input.bin"), "wb") as f:
        f.write(np.array([w, h, 1], np.uint32).tobytes())
        f.write(rgba(s.frames[0]).tobytes())
        f.write(rgba(s.extra[0]).tobytes())
        f.write(cov8.tobytes())
        f.write(kept8.tobytes())
    r = subprocess.run([exe, work], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        raise SystemExit("engine_test failed: %s" % r.stderr)

    gpu = np.fromfile(os.path.join(work, "output.bin"), np.float16).reshape(h, w, 4)[..., :3].astype(np.float32)
    gpu_tm = train.tonemap(gpu, s.exposure)
    with torch.no_grad():
        ref = train.run_full(net, s, 0)
    m = s.mask[0].astype(np.float32) > 0
    inner = np.zeros_like(m)
    inner[BORDER:-BORDER, BORDER:-BORDER] = True
    sel = m & inner
    diff = np.abs(gpu_tm - ref)[sel]
    psnr = 10 * np.log10(1 / max(float((diff ** 2).mean()), 1e-12))
    print("hair pixels %d: mean |diff| %.5f, 99.9%% %.5f, max %.5f, PSNR(mod vs PyTorch) %.1f dB" %
          (sel.sum(), diff.mean(), np.percentile(diff, 99.9), diff.max(), psnr))
    # the network's own change, so that a pass that does nothing cannot look like a match
    inp = train.tonemap(s.frames[0], s.exposure)
    print("network change (PyTorch): mean %.5f   mod: mean %.5f   (mod vs PyTorch error should be far smaller)" %
          (np.abs(ref - inp)[sel].mean(), np.abs(gpu_tm - inp)[sel].mean()))
    untouched = ~m & inner
    if untouched.any():
        print("non-hair pixels changed: max %.5f" % np.abs(gpu_tm - inp)[untouched].max())


if __name__ == "__main__":
    main()
