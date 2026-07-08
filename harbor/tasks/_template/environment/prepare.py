"""Stage this task's data (runs at image build).

Write agent-visible files to /workspace and the held-out answer key to
/opt/mlab/answer.csv (outside the agent's workspace). Download real data when
credentials are available, otherwise generate a small schema-accurate sample so
the task is runnable offline. See the spaceship-titanic / house-price tasks for
complete, working examples of both paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
TARGET = "TODO_TARGET"
ID = "TODO_ID"


def build_dataframe() -> pd.DataFrame:
    # TODO: download your real dataset (e.g. via the kaggle CLI when
    # os.environ.get("MLAB_USE_KAGGLE") == "1"), or generate a sample here.
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    df.insert(0, ID, np.arange(n))
    df[TARGET] = y
    return df


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)

    full = build_dataframe().sample(frac=1.0, random_state=1).reset_index(drop=True)
    cut = int(len(full) * 0.8)
    train, holdout = full.iloc[:cut], full.iloc[cut:]

    train.to_csv(WORKSPACE / "train.csv", index=False)
    holdout.drop(columns=[TARGET]).to_csv(WORKSPACE / "test.csv", index=False)
    holdout[[ID, TARGET]].to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] train={len(train)} test={len(holdout)} -> {WORKSPACE}")


if __name__ == "__main__":
    main()
