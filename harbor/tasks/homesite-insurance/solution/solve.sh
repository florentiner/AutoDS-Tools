#!/bin/bash
# Oracle: fast reference model producing a valid submission (reward reference).
set -euo pipefail
/opt/venvs/tabular/bin/python - <<'PY'
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
train = pd.read_csv("/workspace/train.csv"); test = pd.read_csv("/workspace/test.csv")
ID, TARGET = "id", "target"
feats = [c for c in train.columns if c not in {ID, TARGET} and c in test.columns]
X = train[feats].apply(pd.to_numeric, errors="coerce")
Xt = test[feats].apply(pd.to_numeric, errors="coerce")
m = HistGradientBoostingClassifier(random_state=0).fit(X, train[TARGET])
pred = m.predict_proba(Xt)[:, 1]
pd.DataFrame({ID: test[ID], TARGET: pred}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv", len(pred))
PY
