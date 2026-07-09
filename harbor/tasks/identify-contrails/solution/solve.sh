#!/bin/bash
set -euo pipefail
/opt/venvs/vision/bin/python - <<'PY'
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
Xtr = np.load("/workspace/train_images.npy").reshape(len(tr), -1) / 255.0
Xte = np.load("/workspace/test_images.npy").reshape(len(te), -1) / 255.0
out = {"id": te["id"]}
for i in range(64):
    c = f"m{i:02d}"; y = tr[c]
    out[c] = (m := (LogisticRegression(max_iter=200).fit(Xtr, y) if y.nunique() > 1 else None)) and m.predict(Xte) or np.zeros(len(te), int)
pd.DataFrame(out).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
