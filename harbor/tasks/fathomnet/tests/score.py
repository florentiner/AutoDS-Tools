"""Score FathomNet (micro-F1 over 4 labels). Always writes a reward."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
LABELS = [f"label_{k}" for k in range(4)]
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
    if not all(c in sub.columns for c in ["id", *LABELS]):
        return emit(0.0, error="submission needs id + label_0..label_3")
    m = ans.merge(sub, on="id", suffixes=("_true","_pred"))
    if m.empty: return emit(0.0, error="no id overlap")
    tp=fp=fn=0
    for c in LABELS:
        yt=pd.to_numeric(m[f"{c}_true"],errors="coerce").fillna(0).astype(int)
        yp=pd.to_numeric(m[f"{c}_pred"],errors="coerce").fillna(0).astype(int)
        tp+=int(((yt==1)&(yp==1)).sum()); fp+=int(((yt==0)&(yp==1)).sum()); fn+=int(((yt==1)&(yp==0)).sum())
    f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
    return emit(f1, micro_f1=f1, n=int(len(m)))
if __name__ == "__main__":
    raise SystemExit(main())
