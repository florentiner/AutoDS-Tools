"""Score sberbank-housing (TabReD): R2 (RMSE also reported). Always writes a reward (0.0 on failure)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ID, TARGET = "id", "target"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--reward-json", required=True)
    a = ap.parse_args()
    rp = Path(a.reward_json); rp.parent.mkdir(parents=True, exist_ok=True)

    def emit(r, **x):
        payload = {"reward": float(r), **x}
        rp.write_text(json.dumps(payload)); print(json.dumps(payload)); return 0

    try:
        ans = pd.read_csv(a.answer); sub = pd.read_csv(a.submission)
    except Exception as e:  # noqa: BLE001
        return emit(0.0, error=f"read failed: {e}")
    if ID not in sub.columns or TARGET not in sub.columns:
        return emit(0.0, error="submission needs columns [id, target]")
    m = ans.merge(sub, on=ID, suffixes=("_true", "_pred"))
    if m.empty:
        return emit(0.0, error="no id overlap")
    yt = pd.to_numeric(m["target_true"], errors="coerce").to_numpy()
    yp = pd.to_numeric(m["target_pred"], errors="coerce").to_numpy()
    ok = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[ok], yp[ok]
    if len(yt) == 0:
        return emit(0.0, error="no numeric predictions")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return emit(max(0.0, r2), r2=r2, rmse=rmse, n=int(len(yt)))


if __name__ == "__main__":
    raise SystemExit(main())
