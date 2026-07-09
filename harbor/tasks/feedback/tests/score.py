"""Score feedback (R2). Always writes a reward (0.0 on failure)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ID, TARGET = "id", "score"
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
    if ID not in sub.columns or TARGET not in sub.columns:
        return emit(0.0, error="submission needs columns [id, score]")
    m = ans.merge(sub, on=ID, suffixes=("_true","_pred"))
    if m.empty: return emit(0.0, error="no id overlap")
    yt = pd.to_numeric(m["score_true"], errors="coerce").to_numpy(); yp = pd.to_numeric(m["score_pred"], errors="coerce").to_numpy()
    ok = ~(np.isnan(yt)|np.isnan(yp)); yt, yp = yt[ok], yp[ok]
    if len(yt)==0: return emit(0.0, error="no numeric preds")
    r2 = 1.0 - float(np.sum((yt-yp)**2))/(float(np.sum((yt-yt.mean())**2)) or 1.0)
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    return emit(max(0.0,r2), r2=r2, rmse=rmse, n=int(len(yt)))
if __name__ == "__main__":
    raise SystemExit(main())
