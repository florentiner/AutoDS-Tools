"""Stage IMDb-style sentiment data (synthetic, offline) at image build.

Real data: HuggingFace `datasets.load_dataset("imdb")`. Here a small
schema-shaped synthetic set of positive/negative movie reviews is generated so
the task runs offline. Outputs /workspace/{train,test}.csv (text + label) and
/opt/mlab/answer.csv (held-out labels).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))

POS = ["great", "excellent", "wonderful", "amazing", "loved", "brilliant",
       "fantastic", "superb", "enjoyable", "a masterpiece", "captivating", "moving"]
NEG = ["terrible", "awful", "boring", "the worst", "hated", "dull", "forgettable",
       "disappointing", "poor", "a waste of time", "painful", "cliched"]
FILLER = ["the movie", "this film", "the plot", "the acting", "the story",
          "overall", "honestly", "the cast", "the ending", "the direction", "i thought"]


def _review(rng: np.random.Generator, label: int) -> str:
    words = POS if label == 1 else NEG
    parts = [str(rng.choice(FILLER))]
    for _ in range(int(rng.integers(6, 16))):
        parts.append(str(rng.choice(words)) if rng.random() < 0.45 else str(rng.choice(FILLER)))
    return " ".join(parts)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 2000
    labels = rng.integers(0, 2, n)
    texts = [_review(rng, int(y)) for y in labels]
    df = pd.DataFrame({"id": np.arange(n), "text": texts, "label": labels}).sample(frac=1.0, random_state=1).reset_index(drop=True)

    cut = int(n * 0.8)
    train, holdout = df.iloc[:cut], df.iloc[cut:]
    train.to_csv(WORKSPACE / "train.csv", index=False)
    holdout.drop(columns=["label"]).to_csv(WORKSPACE / "test.csv", index=False)
    holdout[["id", "label"]].to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] imdb train={len(train)} test={len(holdout)}")


if __name__ == "__main__":
    main()
