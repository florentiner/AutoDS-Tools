"""Score homecredit-default (TabReD): ROC-AUC. Always writes a reward (0.0 on failure)."""
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
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(yt.astype(int), yp))
    except Exception as e:  # noqa: BLE001
        return emit(0.0, error=f"auc failed: {e}")
    return emit(auc, auc=auc, n=int(len(yt)))


if __name__ == "__main__":
    raise SystemExit(main())
