"""Score the submission and write a Harbor reward. TODO: match your metric.

Always writes a reward (0.0 on any failure) so Harbor gets a value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ID, TARGET = "TODO_ID", "TODO_TARGET"  # TODO: match prepare.py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--reward-json", required=True)
    args = ap.parse_args()
    reward_path = Path(args.reward_json)
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(reward: float, **extra) -> int:
        reward_path.write_text(json.dumps({"reward": float(reward), **extra}), encoding="utf-8")
        return 0

    try:
        answer = pd.read_csv(args.answer)
        submission = pd.read_csv(args.submission)
    except Exception as exc:  # noqa: BLE001
        return emit(0.0, error=f"read failed: {exc}")
    if ID not in submission.columns or TARGET not in submission.columns:
        return emit(0.0, error=f"submission needs columns [{ID}, {TARGET}]")

    merged = answer.merge(submission, on=ID, suffixes=("_true", "_pred"))
    if merged.empty:
        return emit(0.0, error="no id overlap")
    # TODO: replace with your metric (accuracy shown).
    acc = float((merged[f"{TARGET}_true"].astype(str) == merged[f"{TARGET}_pred"].astype(str)).mean())
    return emit(acc, accuracy=acc, n=int(len(merged)))


if __name__ == "__main__":
    raise SystemExit(main())
