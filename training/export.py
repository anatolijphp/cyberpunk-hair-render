#!/usr/bin/env python3
"""Writes a trained model (train.py) as hairfix_resolve.bin for the mod (src/neural.cpp).

usage: export.py <model.pt> <hairfix_resolve.bin>

Format (little endian): "HFNN", u32 version 2, u32 channels, u32 inputs, f32 exposure, u32 count,
count float32 weights: for every convolution in module order, weight [out][in][3][3] then bias[out].
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import train   # noqa: E402


def main():
    net, info = train.load(sys.argv[1])
    weights = []
    for name, p in net.state_dict().items():   # e1, e2, e3, d2, d1, out; weight before bias
        weights.append(p.detach().cpu().numpy().astype(np.float32).ravel())
    data = np.concatenate(weights)
    with open(sys.argv[2], "wb") as f:
        f.write(b"HFNN")
        f.write(struct.pack("<IIIfI", 2, info["channels"], info["inputs"], info["exposure"], data.size))
        f.write(data.tobytes())
    print("%s: %d channels, %d inputs, exposure %.2f, %d weights" %
          (sys.argv[2], info["channels"], info["inputs"], info["exposure"], data.size))


if __name__ == "__main__":
    main()
