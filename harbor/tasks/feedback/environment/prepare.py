"""Stage Feedback-Prize-style essay scoring data (synthetic, offline)."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
GOOD = ["furthermore", "consequently", "therefore", "moreover", "in conclusion", "however", "for example", "specifically"]
WEAK = ["um", "like", "stuff", "things", "idk", "whatever", "so yeah", "kinda"]
FILLER = ["the essay argues", "the author", "the topic", "in the text", "the writer", "this point", "the evidence"]


def _essay(rng, quality: float) -> str:
    words = GOOD if quality > 3 else WEAK
    parts = []
    for _ in range(int(rng.integers(20, 60))):
        r = rng.random()
        parts.append(str(rng.choice(words)) if r < (0.2 + quality * 0.1) else str(rng.choice(FILLER)))
    return " ".join(parts)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True); ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 1600
    quality = rng.uniform(1, 5, n).round(2)
    texts = [_essay(rng, float(q)) for q in quality]
    df = pd.DataFrame({"id": np.arange(n), "text": texts, "score": quality}).sample(frac=1.0, random_state=1).reset_index(drop=True)
    cut = int(n * 0.8); train, hold = df.iloc[:cut], df.iloc[cut:]
    train.to_csv(WORKSPACE / "train.csv", index=False)
    hold.drop(columns=["score"]).to_csv(WORKSPACE / "test.csv", index=False)
    hold[["id", "score"]].to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] feedback train={len(train)} test={len(hold)}")


if __name__ == "__main__":
    main()
