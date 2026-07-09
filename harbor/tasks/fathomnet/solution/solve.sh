#!/bin/bash
set -euo pipefail
/opt/venvs/vision/bin/python - <<'PY'
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
Xtr = np.load("/workspace/train_images.npy").reshape(len(tr), -1) / 255.0
Xte = np.load("/workspace/test_images.npy").reshape(len(te), -1) / 255.0
out = {"id": te["id"]}
for k in range(4):
    m = LogisticRegression(max_iter=300).fit(Xtr, tr[f"label_{k}"])
    out[f"label_{k}"] = m.predict(Xte)
pd.DataFrame(out).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
