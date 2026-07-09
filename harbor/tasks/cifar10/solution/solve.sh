#!/bin/bash
# Oracle: a fast reference (small torchvision CNN would be slow; a linear probe on
# pixels is a valid, quick baseline). The agent is asked to use a CNN (timm/torchvision).
set -euo pipefail
/opt/venvs/vision/bin/python - <<'PY'
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
Xtr = np.load("/workspace/train_images.npy").reshape(len(pd.read_csv("/workspace/train.csv")), -1) / 255.0
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
Xte = np.load("/workspace/test_images.npy").reshape(len(te), -1) / 255.0
m = LogisticRegression(max_iter=300).fit(Xtr, tr["label"])
pd.DataFrame({"id": te["id"], "label": m.predict(Xte)}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
