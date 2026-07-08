"""Stage House-Prices data for the Harbor task (runs at image build).

Mirrors MLAgentBench's split (Kaggle ``home-data-for-ml-course``): the labeled
set is cut 80/20 into ``train.csv`` and ``test.csv`` (target dropped) plus a
held-out answer key. Real Kaggle data is used when ``MLAB_USE_KAGGLE=1`` and the
``kaggle`` CLI has credentials; otherwise a schema-accurate SYNTHETIC dataset is
generated (the numeric feature set the MLAgentBench baseline uses).

Outputs:
* ``/workspace/train.csv``            — labeled training data (agent-visible)
* ``/workspace/test.csv``             — test features, no target (agent-visible)
* ``/workspace/data_description.txt``  — feature description (agent-visible)
* ``/opt/mlab/answer.csv``            — held-out SalePrice (verifier-only)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
TARGET = "SalePrice"
FEATURES = [
    "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold",
]


def _synthetic(n: int = 1600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "MSSubClass": rng.choice([20, 30, 50, 60, 70, 120], n),
        "LotArea": rng.normal(10000, 4000, n).clip(1300, 60000),
        "OverallQual": rng.integers(1, 11, n),
        "OverallCond": rng.integers(1, 11, n),
        "YearBuilt": rng.integers(1900, 2011, n),
        "YearRemodAdd": rng.integers(1950, 2011, n),
        "1stFlrSF": rng.normal(1150, 380, n).clip(300, 4000),
        "2ndFlrSF": (rng.random(n) < 0.45) * rng.normal(700, 300, n).clip(0, 2000),
        "LowQualFinSF": (rng.random(n) < 0.02) * rng.normal(100, 50, n).clip(0, 600),
        "GrLivArea": rng.normal(1500, 500, n).clip(400, 5000),
        "FullBath": rng.integers(0, 4, n),
        "HalfBath": rng.integers(0, 3, n),
        "BedroomAbvGr": rng.integers(0, 6, n),
        "KitchenAbvGr": rng.integers(1, 3, n),
        "TotRmsAbvGrd": rng.integers(2, 13, n),
        "Fireplaces": rng.integers(0, 4, n),
        "WoodDeckSF": (rng.random(n) < 0.5) * rng.normal(200, 120, n).clip(0, 900),
        "OpenPorchSF": (rng.random(n) < 0.6) * rng.normal(80, 60, n).clip(0, 500),
        "EnclosedPorch": (rng.random(n) < 0.15) * rng.normal(150, 80, n).clip(0, 600),
        "3SsnPorch": (rng.random(n) < 0.03) * rng.normal(100, 60, n).clip(0, 500),
        "ScreenPorch": (rng.random(n) < 0.08) * rng.normal(150, 80, n).clip(0, 500),
        "PoolArea": (rng.random(n) < 0.005) * rng.normal(500, 100, n).clip(0, 800),
        "MiscVal": (rng.random(n) < 0.05) * rng.normal(1500, 1000, n).clip(0, 15000),
        "MoSold": rng.integers(1, 13, n),
        "YrSold": rng.integers(2006, 2011, n),
    }
    df = pd.DataFrame(data)
    price = (
        20000
        + df["OverallQual"] * 22000
        + df["GrLivArea"] * 55
        + df["1stFlrSF"] * 25
        + df["2ndFlrSF"] * 22
        + (df["YearBuilt"] - 1900) * 350
        + df["FullBath"] * 9000
        + df["Fireplaces"] * 6000
        + df["OverallCond"] * 3000
        + rng.normal(0, 22000, n)
    ).clip(34900, 755000)
    df.insert(0, "Id", np.arange(1, n + 1))
    df[TARGET] = price.round(0).astype(int)
    return df


def _load_kaggle() -> pd.DataFrame | None:
    if os.environ.get("MLAB_USE_KAGGLE") != "1":
        return None
    try:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "home-data-for-ml-course", "-p", str(WORKSPACE)],
            check=True,
        )
        subprocess.run(["unzip", "-o", str(WORKSPACE / "home-data-for-ml-course.zip"), "-d", str(WORKSPACE)], check=True)
        return pd.read_csv(WORKSPACE / "train.csv")
    except Exception as exc:  # noqa: BLE001
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
    full = full.reset_index(drop=True)

    cut = int(len(full) * 0.8)
    train = full.iloc[:cut].reset_index(drop=True)
    holdout = full.iloc[cut:].reset_index(drop=True)
    test = holdout.drop(columns=[TARGET])
    answer = holdout[["Id", TARGET]]

    train.to_csv(WORKSPACE / "train.csv", index=False)
    test.to_csv(WORKSPACE / "test.csv", index=False)
    answer.to_csv(ANSWER_DIR / "answer.csv", index=False)

    descriptor = Path(__file__).parent / "data_description.txt"
    if descriptor.exists():
        (WORKSPACE / "data_description.txt").write_text(descriptor.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[prepare] source={source} train={len(train)} test={len(test)} -> {WORKSPACE}")


if __name__ == "__main__":
    main()
