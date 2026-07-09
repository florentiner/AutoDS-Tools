"""Identify-Contrails-style segmentation (synthetic, offline).

32x32 grayscale-ish RGB images; ~60% contain a bright diagonal "contrail" line.
The target is a coarse 8x8 binary mask (downsampled) so the submission + Dice
scoring stay tractable while the task still needs a segmentation model.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
SIZE, G = 32, 8


def _sample(rng):
    img = rng.integers(20, 80, (SIZE, SIZE, 3), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    if rng.random() < 0.6:
        b = int(rng.integers(-8, 8)); thick = int(rng.integers(1, 3))
        for x in range(SIZE):
            y = x + b
            for t in range(-thick, thick + 1):
                if 0 <= y + t < SIZE:
                    img[y + t, x] = [230, 230, 230]; mask[y + t, x] = 1
    coarse = mask.reshape(G, SIZE // G, G, SIZE // G).max(axis=(1, 3))   # 8x8 downsample
    return img, coarse.reshape(-1)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True); ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5); n = 1500
    imgs, masks = [], []
    for _ in range(n):
        im, mk = _sample(rng); imgs.append(im); masks.append(mk)
    imgs = np.stack(imgs); masks = np.stack(masks)
    cols = [f"m{i:02d}" for i in range(G * G)]
    cut = int(n * 0.8)
    np.save(WORKSPACE / "train_images.npy", imgs[:cut])
    pd.DataFrame({"id": np.arange(cut), **{cols[i]: masks[:cut, i] for i in range(G * G)}}).to_csv(WORKSPACE / "train.csv", index=False)
    np.save(WORKSPACE / "test_images.npy", imgs[cut:])
    pd.DataFrame({"id": np.arange(cut, n)}).to_csv(WORKSPACE / "test.csv", index=False)
    pd.DataFrame({"id": np.arange(cut, n), **{cols[i]: masks[cut:, i] for i in range(G * G)}}).to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] identify-contrails train={cut} test={n-cut} mask={G}x{G}")


if __name__ == "__main__":
    main()
