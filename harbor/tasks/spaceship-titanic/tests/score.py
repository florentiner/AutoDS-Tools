"""Score a Spaceship-Titanic submission (accuracy) and write a Harbor reward.

Always writes a reward file (0.0 on any failure) so Harbor gets a value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET = "Transported"
ID = "PassengerId"


def _to_bin(series: pd.Series) -> pd.Series:
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--reward-json", required=True)
    args = ap.parse_args()

    reward_path = Path(args.reward_json)
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(reward: float, **extra) -> int:
        payload = {"reward": float(reward), **extra}
        reward_path.write_text(json.dumps(payload), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    try:
        answer = pd.read_csv(args.answer)
        submission = pd.read_csv(args.submission)
    except Exception as exc:  # noqa: BLE001
        return emit(0.0, error=f"failed to read inputs: {exc}")

    if ID not in submission.columns or TARGET not in submission.columns:
        return emit(0.0, error=f"submission must have columns [{ID}, {TARGET}]")

    merged = answer.merge(submission, on=ID, suffixes=("_true", "_pred"))
    if merged.empty:
        return emit(0.0, error="no PassengerId overlap between submission and answer")

    y_true = _to_bin(merged[f"{TARGET}_true"])
    y_pred = _to_bin(merged[f"{TARGET}_pred"]).fillna(-1)
    accuracy = float((y_true == y_pred).mean())
    return emit(accuracy, accuracy=accuracy, n=int(len(merged)))


if __name__ == "__main__":
    raise SystemExit(main())
