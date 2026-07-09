"""Score Identify-Contrails (Dice over the 8x8 masks). Always writes a reward."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
COLS = [f"m{i:02d}" for i in range(64)]
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True); ap.add_argument("--answer", required=True); ap.add_argument("--reward-json", required=True)
    a = ap.parse_args(); rp = Path(a.reward_json); rp.parent.mkdir(parents=True, exist_ok=True)
    def emit(r, **x):
        rp.write_text(json.dumps({"reward": float(r), **x})); print(json.dumps({"reward": float(r), **x})); return 0
    try:
        ans = pd.read_csv(a.answer); sub = pd.read_csv(a.submission)
    except Exception as e:  # noqa: BLE001
        return emit(0.0, error=f"read failed: {e}")
    if not all(c in sub.columns for c in ["id", *COLS]):
        return emit(0.0, error="submission needs id + m00..m63")
    m = ans.merge(sub, on="id", suffixes=("_true","_pred"))
    if m.empty: return emit(0.0, error="no id overlap")
    yt = m[[f"{c}_true" for c in COLS]].to_numpy(); yp = np.nan_to_num(m[[f"{c}_pred" for c in COLS]].to_numpy())
    yt = (yt > 0.5).astype(int); yp = (yp > 0.5).astype(int)
    inter = (yt & yp).sum(); denom = yt.sum() + yp.sum()
    dice = (2 * inter / denom) if denom > 0 else 1.0
    return emit(float(dice), dice=float(dice), n=int(len(m)))
if __name__ == "__main__":
    raise SystemExit(main())
