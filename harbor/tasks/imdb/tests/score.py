"""Score IMDb submission (accuracy). Always writes a reward (0.0 on failure)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
ID, TARGET = "id", "label"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--reward-json", required=True)
    a = ap.parse_args()
    rp = Path(a.reward_json); rp.parent.mkdir(parents=True, exist_ok=True)

    def emit(r, **x):
        rp.write_text(json.dumps({"reward": float(r), **x})); print(json.dumps({"reward": float(r), **x})); return 0

    try:
        ans = pd.read_csv(a.answer); sub = pd.read_csv(a.submission)
    except Exception as e:  # noqa: BLE001
        return emit(0.0, error=f"read failed: {e}")
    if ID not in sub.columns or TARGET not in sub.columns:
        return emit(0.0, error="submission needs columns [id, label]")
    m = ans.merge(sub, on=ID, suffixes=("_true", "_pred"))
    if m.empty:
        return emit(0.0, error="no id overlap")
    yt = pd.to_numeric(m["label_true"], errors="coerce")
    yp = pd.to_numeric(m["label_pred"], errors="coerce")
    acc = float((yt == yp).mean())
    return emit(acc, accuracy=acc, n=int(len(m)))


if __name__ == "__main__":
    raise SystemExit(main())
