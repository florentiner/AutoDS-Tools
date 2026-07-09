#!/bin/bash
# Oracle: a fast reference. A tabular model on node features + neighbor-label
# smoothing is a valid baseline; the agent is asked to use a PyG GNN.
set -euo pipefail
/opt/venvs/graph/bin/python - <<'PY'
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
X = np.load("/workspace/node_features.npy"); E = np.load("/workspace/edges.npy")
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
m = LogisticRegression(max_iter=300).fit(X[tr["id"].to_numpy()], tr["label"])
pred = m.predict(X[te["id"].to_numpy()])
pd.DataFrame({"id": te["id"], "label": pred}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
