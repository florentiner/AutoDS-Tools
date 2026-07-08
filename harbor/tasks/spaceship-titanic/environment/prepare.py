"""Stage Spaceship-Titanic data for the Harbor task (runs at image build).

Mirrors MLAgentBench's split: the labeled set is cut 80/20; the first 80% becomes
``train.csv``, the last 20% becomes ``test.csv`` (target dropped) plus a held-out
answer key. Data source:

* Real Kaggle data when ``MLAB_USE_KAGGLE=1`` and the ``kaggle`` CLI has
  credentials (KAGGLE_USERNAME / KAGGLE_KEY). Competition: ``spaceship-titanic``.
* Otherwise a schema-accurate SYNTHETIC dataset so the task is runnable offline.

Outputs:
* ``/workspace/train.csv``           — labeled training data (agent-visible)
* ``/workspace/test.csv``            — test features, no target (agent-visible)
* ``/workspace/task_descriptor.txt`` — column descriptions (agent-visible)
* ``/opt/mlab/answer.csv``           — held-out labels (verifier-only)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
TARGET = "Transported"
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _synthetic(n: int = 2600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    home = rng.choice(["Earth", "Europa", "Mars"], n, p=[0.5, 0.3, 0.2])
    cryo = rng.choice([True, False], n, p=[0.36, 0.64])
    dest = rng.choice(["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"], n)
    deck = rng.choice(list("ABCDEFGT"), n)
    side = rng.choice(["P", "S"], n)
    cabin_num = rng.integers(0, 2000, n)
    age = rng.normal(29, 14, n).clip(0, 80).round(1)
    vip = rng.choice([True, False], n, p=[0.03, 0.97])
    spend = {c: (rng.gamma(1.0, 220, n) * (~cryo)).round(1) for c in SPEND_COLS}

    total_spend = np.sum([spend[c] for c in SPEND_COLS], axis=0)
    logit = (
        0.95 * cryo.astype(float)
        - 0.0009 * total_spend
        + (home == "Europa") * 0.7
        - (home == "Mars") * 0.2
        + rng.normal(0, 0.5, n)
    )
    transported = logit > np.median(logit)
    passenger_id = [f"{i // 2:04d}_{i % 2 + 1:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "PassengerId": passenger_id,
            "HomePlanet": home,
            "CryoSleep": cryo,
            "Cabin": [f"{d}/{n_}/{s}" for d, n_, s in zip(deck, cabin_num, side)],
            "Destination": dest,
            "Age": age,
            "VIP": vip,
            **spend,
            "Name": [f"Passenger {i}" for i in range(n)],
            TARGET: transported,
        }
    )


def _load_kaggle() -> pd.DataFrame | None:
    if os.environ.get("MLAB_USE_KAGGLE") != "1":
        return None
    try:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "spaceship-titanic", "-p", str(WORKSPACE)],
            check=True,
        )
        subprocess.run(["unzip", "-o", str(WORKSPACE / "spaceship-titanic.zip"), "-d", str(WORKSPACE)], check=True)
        return pd.read_csv(WORKSPACE / "train.csv")
    except Exception as exc:  # noqa: BLE001 - fall back to synthetic
        print(f"[prepare] Kaggle download failed ({exc}); using synthetic data")
        return None


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)

    full = _load_kaggle()
    source = "kaggle"
    if full is None:
        full = _synthetic()
        source = "synthetic"
    full = full.sample(frac=1.0, random_state=1).reset_index(drop=True)

    cut = int(len(full) * 0.8)
    train = full.iloc[:cut].reset_index(drop=True)
    holdout = full.iloc[cut:].reset_index(drop=True)
    test = holdout.drop(columns=[TARGET])
    answer = holdout[["PassengerId", TARGET]]

    train.to_csv(WORKSPACE / "train.csv", index=False)
    test.to_csv(WORKSPACE / "test.csv", index=False)
    answer.to_csv(ANSWER_DIR / "answer.csv", index=False)

    descriptor = Path(__file__).parent / "task_descriptor.txt"
    if descriptor.exists():
        (WORKSPACE / "task_descriptor.txt").write_text(descriptor.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[prepare] source={source} train={len(train)} test={len(test)} -> {WORKSPACE}")


if __name__ == "__main__":
    main()
