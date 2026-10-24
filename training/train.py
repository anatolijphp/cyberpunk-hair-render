#!/usr/bin/env python3
"""The hair resolve network: refines the mod's resolve before FSR ([Hair] Resolve = neural).

usage: train.py <out dir> <validation pattern> <iterations> <scenes dir> [<scenes dir> ...]
environment: HAIRFIX_CHANNELS (12), HAIRFIX_BATCH (64), HAIRFIX_PATCH (96)

Input (render resolution, see prepare.py), 11 planes: the mod's tone-mapped resolve, its hair
coverage and hair-surface mask, the local line direction of the resolve (structure tensor:
cos 2a, sin 2a, coherence; the network works along strands rather than on pixels) and the game's
raw frame. Target: the Native AA reference at each render pixel's centre plus the frame's jitter.
The network predicts a correction of the resolve, applied only where there is hair. Training uses random exposures; the game uses EXPOSURE (written with the
weights by export.py). The loss adds a term for errors across strands (direction from the target).
Scenes whose file name contains <validation pattern> (e.g. a photo mode set id) are held out.
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import torch_directml
    DEVICE = torch_directml.device()
except ImportError:
    DEVICE = torch.device("cpu")

PATCH = int(os.environ.get("HAIRFIX_PATCH", 96))
BATCH = int(os.environ.get("HAIRFIX_BATCH", 64))
CHANNELS = int(os.environ.get("HAIRFIX_CHANNELS", 12))
INPUTS = 11           # resolve RGB, coverage, kept, line direction (3), raw RGB
EXPOSURE = 2.0        # tone mapping in the game: x * e / (1 + x * e)
EXPOSURE_RANGE = 2.0  # training: EXPOSURE * 2 ** uniform(-range, range)
MARGIN = 3            # the line features of a pixel need 3 pixels around it
LUMA = (0.2126, 0.7152, 0.0722)


def tonemap(x, exposure):
    x = np.maximum(x.astype(np.float32) * exposure, 0)
    return x / (1 + x)


def hair_mask(cov, kept, widen=1):
    """Where the network may change pixels: any hair coverage or hair surface, widened by `widen`
    pixels (as the mod does: nn_apply.hlsl)."""
    m = np.maximum(1 - cov.astype(np.float32), kept.astype(np.float32))
    m = np.clip(m * 4, 0, 1)
    t = torch.from_numpy(m)[None, None]
    return F.max_pool2d(t, 2 * widen + 1, 1, widen)[0, 0].numpy()


def line_features(rgb_tm):
    """Structure tensor of the luminance, smoothed over 5x5: (cos 2a, sin 2a, coherence) of the
    dominant gradient. rgb_tm: B 3 H W -> B 3 (H - 6) (W - 6) (valid region only)."""
    lum = sum(rgb_tm[:, c:c + 1] * w for c, w in enumerate(LUMA))
    gx = (lum[..., 1:-1, 2:] - lum[..., 1:-1, :-2]) * 0.5
    gy = (lum[..., 2:, 1:-1] - lum[..., :-2, 1:-1]) * 0.5
    j = F.avg_pool2d(torch.cat([gx * gx, gy * gy, gx * gy], 1), 5, 1)
    a, b, s = j[:, 0:1] - j[:, 1:2], 2 * j[:, 2:3], j[:, 0:1] + j[:, 1:2]
    r = torch.sqrt(a * a + b * b + 1e-12)
    inv = 1 / (s + 1e-5)
    return torch.cat([a * inv, b * inv, r * inv], 1)


class Scene:
    """One prepared capture pair (prepare.py; float16, linear HDR), one target per frame (each frame
    has its own FSR jitter); tensors on the device for training."""

    def __init__(self, path):
        d = np.load(path)
        self.name = os.path.basename(path)[:-4]
        self.frames = d["resolved"]   # N h w 3: the mod's resolve (the base the network corrects)
        self.extra = d["raw"]         # N h w 3: the game's raw frame
        self.targets = d["target"]    # N h w 3
        self.baselines = {"raw": d["raw"][:1], "mod resolve": d["resolved"][:1]}
        self.target = self.targets[0]
        self.cov, self.kept = d["cov"], d["kept"]
        self.mask = np.stack([hair_mask(c, k) for c, k in zip(self.cov, self.kept)]).astype(np.float16)
        self.exposure = EXPOSURE

    def target_tm(self, i=0):
        return tonemap(self.targets[i], self.exposure)

    def to_gpu(self):
        f16 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float16)).to(DEVICE)
        self.g_rgb = [f16(o.transpose(2, 0, 1)) for o in self.frames]                         # 3 H W linear
        self.g_extra = [f16(o.transpose(2, 0, 1)) for o in self.extra]
        self.g_aux = [f16(np.stack([c, k, m])) for c, k, m in zip(self.cov, self.kept, self.mask)]
        self.g_tgt = [f16(t.transpose(2, 0, 1)) for t in self.targets]
        self.hair = [F.avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None], 8, 8)[0, 0].numpy()
                     for m in self.mask]


def network_input(rgb_tm, aux, extra_tm):
    """rgb_tm B 3 (P+6) (P+6), aux B 2 P P (coverage, kept), extra_tm (raw frame) like rgb_tm
    -> B 11 P P."""
    m = MARGIN
    return torch.cat([rgb_tm[..., m:-m, m:-m], aux, line_features(rgb_tm), extra_tm[..., m:-m, m:-m]], 1)


def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))


class Net(nn.Module):
    """Small U-Net; predicts an RGB correction."""

    def __init__(self, cin=INPUTS, c=CHANNELS):
        super().__init__()
        self.e1 = nn.Sequential(conv(cin, c), conv(c, c))
        self.e2 = nn.Sequential(conv(c, 2 * c), conv(2 * c, 2 * c))
        self.e3 = nn.Sequential(conv(2 * c, 3 * c), conv(3 * c, 3 * c))
        self.d2 = nn.Sequential(conv(3 * c + 2 * c, 2 * c), conv(2 * c, 2 * c))
        self.d1 = nn.Sequential(conv(2 * c + c, c), conv(c, c))
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.out(d1)


def apply(net, x, mask):
    """Corrected image: input RGB + masked correction."""
    return x[:, :3] + net(x) * mask


def grad_l1(a, b):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return (dx(a) - dx(b)).abs().mean() + (dy(a) - dy(b)).abs().mean()


def across_strand_l1(out, tgt, tgt_lines, m):
    """Error of the derivative across the target's strands (along its dominant gradient), weighted
    by how line-like the target is there."""
    d = sum((out - tgt)[:, c:c + 1] * w for c, w in enumerate(LUMA))
    dx = (d[..., 1:-1, 2:] - d[..., 1:-1, :-2]) * 0.5
    dy = (d[..., 2:, 1:-1] - d[..., :-2, 1:-1]) * 0.5
    t = tgt_lines[..., 1:-1, 1:-1]
    unit = 1 / (torch.sqrt(t[:, 0:1] ** 2 + t[:, 1:2] ** 2) + 1e-6)
    c2, s2, coherence = t[:, 0:1] * unit, t[:, 1:2] * unit, t[:, 2:3]
    across2 = 0.5 * ((dx * dx + dy * dy) + (dx * dx - dy * dy) * c2 + 2 * dx * dy * s2)
    return (torch.sqrt(across2.clamp(min=0) + 1e-8) * coherence * m[..., 1:-1, 1:-1]).mean()


_rev = None


def reversed_index(n):
    global _rev
    if _rev is None or _rev.shape[0] != n:
        _rev = torch.arange(n - 1, -1, -1, device=DEVICE)
    return _rev


def batch(scenes, rng):
    """BATCH random patches (mostly with hair), cut on the device with a MARGIN for the line
    features; random exposure and mirroring. Returns input, target, target lines, mask."""
    n = PATCH + 2 * MARGIN
    rgbs, auxs, tgts, extras = [], [], [], []
    while len(rgbs) < BATCH:
        s = scenes[rng.integers(len(scenes))]
        i = rng.integers(len(s.g_rgb))
        h, w = s.target.shape[:2]
        if h <= n or w <= n:
            continue
        y, x = rng.integers(0, h - n), rng.integers(0, w - n)
        hair = s.hair[i][(y + MARGIN) // 8:(y + MARGIN + PATCH) // 8, (x + MARGIN) // 8:(x + MARGIN + PATCH) // 8]
        if hair.mean() < 0.05 and rng.random() < 0.9:
            continue
        rgb = s.g_rgb[i][:, y:y + n, x:x + n]
        aux = s.g_aux[i][:, y:y + n, x:x + n]
        tgt = s.g_tgt[i][:, y:y + n, x:x + n]
        extra = s.g_extra[i][:, y:y + n, x:x + n]
        if rng.random() < 0.5:   # mirror; Tensor.flip crashes on DirectML, a reversed index_select works
            rev = reversed_index(n)
            rgb, aux, tgt, extra = (torch.index_select(t, 2, rev) for t in (rgb, aux, tgt, extra))
        rgbs.append(rgb), auxs.append(aux), tgts.append(tgt), extras.append(extra)
    e = torch.from_numpy((EXPOSURE * 2.0 ** rng.uniform(-EXPOSURE_RANGE, EXPOSURE_RANGE, (BATCH, 1, 1, 1)))
                         .astype(np.float32)).to(DEVICE)
    tm = lambda t: (t * e).clamp(min=0) / (1 + (t * e).clamp(min=0))
    rgb_tm, tgt_tm = tm(torch.stack(rgbs).float()), tm(torch.stack(tgts).float())
    aux = torch.stack(auxs).float()[..., MARGIN:-MARGIN, MARGIN:-MARGIN]
    extra_tm = tm(torch.stack(extras).float())
    x = network_input(rgb_tm, aux[:, 0:2], extra_tm)
    return x, tgt_tm[..., MARGIN:-MARGIN, MARGIN:-MARGIN], line_features(tgt_tm), aux[:, 2:3]


def psnr(a, b, m):
    mse = (((a - b) ** 2) * m[..., None]).sum() / max(m.sum() * 3, 1)
    return 10 * np.log10(1 / max(mse, 1e-10))


def full_input(s, i):
    """Network input for a whole frame (edges repeated, as the game does at the screen border)."""
    tm = lambda img: torch.from_numpy(tonemap(img, s.exposure).transpose(2, 0, 1)[None]).to(DEVICE)
    aux = torch.from_numpy(np.stack([s.cov[i], s.kept[i]]).astype(np.float32)[None]).to(DEVICE)
    pad = lambda t: F.pad(t, (MARGIN,) * 4, mode="replicate")
    return network_input(pad(tm(s.frames[i])), aux, pad(tm(s.extra[i])))


def run_full(net, s, i):
    x = full_input(s, i)
    m = torch.from_numpy(s.mask[i].astype(np.float32)[None, None]).to(DEVICE)
    h, w = x.shape[2:]
    ph, pw = (-h) % 4, (-w) % 4   # two poolings
    xp, mp = F.pad(x, (0, pw, 0, ph), mode="replicate"), F.pad(m, (0, pw, 0, ph))
    with torch.no_grad():
        y = apply(net, xp, mp)[..., :h, :w]
    return y[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def sheet(images, path, region):
    y0, y1, x0, x1 = region
    tiles = [Image.fromarray((np.clip(im[y0:y1, x0:x1], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)) for im in images]
    w, h = tiles[0].size
    out = Image.new("RGB", (w * len(tiles) + 8 * (len(tiles) - 1), h), (255, 255, 255))
    for k, t in enumerate(tiles):
        out.paste(t, (k * (w + 8), 0))
    out.save(path)


def evaluate(net, scenes, out_dir, label, sheets):
    """Hair PSNR against the reference of frame 0: each baseline and the network (dict, means)."""
    net.eval()
    rows = []
    for s in scenes:
        m = s.mask[0].astype(np.float32)
        y = run_full(net, s, 0)
        tgt = s.target_tm(0)
        base = {k: tonemap(v[0], s.exposure) for k, v in s.baselines.items()}
        r = {k: psnr(v, tgt, m) for k, v in base.items()}
        r["network"] = psnr(y, tgt, m)
        rows.append(r)
        if sheets:
            ys, xs = np.nonzero(m > 0.5)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            h, w = tgt.shape[:2]
            reg = (max(0, cy - 200), min(h, cy + 200), max(0, cx - 250), min(w, cx + 250))
            sheet(list(base.values()) + [y, tgt], os.path.join(out_dir, "%s_%s.png" % (label, s.name)), reg)
    net.train()
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def describe(result):
    return "  ".join("%s %.2f" % (k, v) for k, v in result.items()) + " dB"


def save(net, path):
    torch.save({"state": net.state_dict(), "channels": CHANNELS, "inputs": net.e1[0][0].in_channels,
                "exposure": EXPOSURE}, path)


def load(path):
    """Model saved by save()."""
    d = torch.load(path, map_location="cpu")
    net = Net(cin=d["inputs"], c=d["channels"])
    net.load_state_dict(d["state"])
    return net.to(DEVICE), d


def main():
    out_dir, val_pattern, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(p for d in sys.argv[4:] for p in glob.glob(os.path.join(d, "*.npz")))
    t0 = time.time()
    scenes = [Scene(p) for p in paths]
    train = [s for s in scenes if val_pattern not in s.name]
    val = [s for s in scenes if val_pattern in s.name]
    for s in train:
        s.to_gpu()
    print("device %s; %d training pairs, %d validation pairs (%s); loaded in %.0f s" %
          (DEVICE, len(train), len(val), val_pattern, time.time() - t0), flush=True)

    net = Net().to(DEVICE)
    print("batch %d x %dpx, %d channels, %d inputs, parameters: %d" %
          (BATCH, PATCH, CHANNELS, INPUTS, sum(p.numel() for p in net.parameters())), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    rng = np.random.default_rng(1)
    t0 = time.time()
    for it in range(1, iters + 1):
        x, y, lines, m = batch(train, rng)
        out = apply(net, x, m)
        w = 0.2 + m   # hair counts most, the rest must stay unchanged
        loss = ((out - y).abs() * w).mean() + 0.5 * grad_l1(out * m, y * m) + across_strand_l1(out, y, lines, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == 1:
            print("it %6d  loss %.5f  %.0f s" % (it, loss.item(), time.time() - t0), flush=True)
        if it % 5000 == 0 or it == iters:
            save(net, os.path.join(out_dir, "net.pt"))
            print("  validation hair PSNR vs Native AA: " + describe(evaluate(net, val, out_dir, "val", it == iters)),
                  flush=True)
    print("training pairs hair PSNR: " + describe(evaluate(net, train, out_dir, "train", False)))
    print("sheets: baselines | network | Native AA")


if __name__ == "__main__":
    main()
