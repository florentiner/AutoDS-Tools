"""Stage weather (TabReD) data — synthetic, temporal 80/20 split (build time).

Real data: load via the yandex-research/tabred loader when available. Here a
schema-shaped synthetic set is generated with a `timestamp` column; the split is
by time (first 80% train / last 20% test) to mirror TabReD's temporal setup.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
ID, TARGET, KIND = "id", "target", "reg"
# Seed from the task name so every task gets a DISTINCT dataset + signal.
SEED = int(hashlib.sha256(b"weather").hexdigest(), 16) % (2**32)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n = 6000
    num = {f"num_{i}": rng.normal(0, 1, n) for i in range(12)}
    cat = {f"cat_{i}": rng.integers(0, 6, n) for i in range(4)}
    coef = rng.normal(0, 1, 12)                       # per-task feature importances
    noise = float(rng.uniform(0.5, 1.3))              # per-task noise level
    signal = sum(coef[i] * num[f"num_{i}"] for i in range(12))
    signal = signal + 0.4 * (cat["cat_0"] - 2.5) + rng.normal(0, noise, n)
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
    print(f"[prepare] weather kind={KIND} train={len(train)} test={len(holdout)}")


if __name__ == "__main__":
    main()
