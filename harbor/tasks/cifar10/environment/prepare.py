"""Stage a small CIFAR-style image dataset (synthetic, offline) at build time.

Images are saved as numpy arrays (uint8, N x 32 x 32 x 3) aligned by row with the
id order in train.csv / test.csv, so the task is genuinely image-shaped and calls
for a CNN. Kept small (32x32, ~1800 imgs, 5 classes) so a tiny model trains on CPU.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
SIZE, NCLASS = 32, 5
COLORS = [(230, 30, 30), (30, 220, 30), (40, 40, 240), (230, 230, 20), (30, 220, 220)]


def _img(rng, label: int) -> np.ndarray:
    img = rng.integers(0, 55, (SIZE, SIZE, 3), dtype=np.uint8)   # dark noise
    y0 = (label * 5) % (SIZE - 9); x0 = (label * 7) % (SIZE - 9)  # class-specific patch
    img[y0:y0 + 9, x0:x0 + 9] = COLORS[label]
    return img


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True); ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 1800
    labels = rng.integers(0, NCLASS, n)
    imgs = np.stack([_img(rng, int(y)) for y in labels])
    cut = int(n * 0.8)
    np.save(WORKSPACE / "train_images.npy", imgs[:cut])
    pd.DataFrame({"id": np.arange(cut), "label": labels[:cut]}).to_csv(WORKSPACE / "train.csv", index=False)
    np.save(WORKSPACE / "test_images.npy", imgs[cut:])
    pd.DataFrame({"id": np.arange(cut, n)}).to_csv(WORKSPACE / "test.csv", index=False)
    pd.DataFrame({"id": np.arange(cut, n), "label": labels[cut:]}).to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] cifar10 train={cut} test={n - cut} shape={imgs.shape[1:]}")


if __name__ == "__main__":
    main()
