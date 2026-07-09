"""Stage delivery-eta (TabReD) data — synthetic, temporal 80/20 split (build time).

Real data: load via the yandex-research/tabred loader when available. Here a
schema-shaped synthetic set is generated with a `timestamp` column; the split is
by time (first 80% train / last 20% test) to mirror TabReD's temporal setup.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
ID, TARGET, KIND = "id", "target", "reg"


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 6000
    num = {f"num_{i}": rng.normal(0, 1, n) for i in range(12)}
    cat = {f"cat_{i}": rng.integers(0, 6, n) for i in range(4)}
    signal = (
        1.3 * num["num_0"] - 0.8 * num["num_1"] + 0.5 * num["num_2"]
        + 0.4 * (cat["cat_0"] - 2.5) + rng.normal(0, 0.7, n)
    )
    t = np.arange(n)
    signal = signal + (t / n) * 0.8 * num["num_3"]   # mild temporal drift
    if KIND == "binary":
        target = (signal > np.median(signal)).astype(int)
    else:
        target = (50 + 15 * signal + (t / n) * 5).round(3)
    df = pd.DataFrame({ID: np.arange(n), "timestamp": t, **num, **cat, TARGET: target})

    cut = int(n * 0.8)
    train, holdout = df.iloc[:cut], df.iloc[cut:]
    train.to_csv(WORKSPACE / "train.csv", index=False)
    holdout.drop(columns=[TARGET]).to_csv(WORKSPACE / "test.csv", index=False)
    holdout[[ID, TARGET]].to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] delivery-eta kind={KIND} train={len(train)} test={len(holdout)}")


if __name__ == "__main__":
    main()
