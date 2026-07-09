#!/bin/bash
set -euo pipefail
/opt/venvs/tabular/bin/python - <<'PY'
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
feats = [c for c in tr.columns if c not in {"id","updrs"} and c in te.columns]
m = HistGradientBoostingRegressor(random_state=0).fit(tr[feats], tr["updrs"])
pd.DataFrame({"id": te["id"], "updrs": m.predict(te[feats])}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
