"""FathomNet-style MULTI-LABEL image classification (synthetic, offline).

32x32 RGB images; each of 4 categories may independently be present (a colored
patch at a category-specific location). Targets are 4 binary columns.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
SIZE, K = 32, 4
COLORS = [(240, 40, 40), (40, 230, 40), (50, 50, 245), (240, 240, 30)]
LOC = [(2, 2), (2, 21), (21, 2), (21, 21)]


def _img(rng, present):
    img = rng.integers(0, 50, (SIZE, SIZE, 3), dtype=np.uint8)
    for k in range(K):
        if present[k]:
            y, x = LOC[k]; img[y:y + 9, x:x + 9] = COLORS[k]
    return img


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True); ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3); n = 1800
    present = (rng.random((n, K)) < 0.4).astype(int)
    imgs = np.stack([_img(rng, present[i]) for i in range(n)])
    cols = [f"label_{k}" for k in range(K)]
    cut = int(n * 0.8)
    np.save(WORKSPACE / "train_images.npy", imgs[:cut])
    pd.DataFrame({"id": np.arange(cut), **{cols[k]: present[:cut, k] for k in range(K)}}).to_csv(WORKSPACE / "train.csv", index=False)
    np.save(WORKSPACE / "test_images.npy", imgs[cut:])
    pd.DataFrame({"id": np.arange(cut, n)}).to_csv(WORKSPACE / "test.csv", index=False)
    pd.DataFrame({"id": np.arange(cut, n), **{cols[k]: present[cut:, k] for k in range(K)}}).to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] fathomnet train={cut} test={n-cut} labels={K}")


if __name__ == "__main__":
    main()
