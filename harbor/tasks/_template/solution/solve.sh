#!/bin/bash
# Oracle: a fast reference model producing a valid /workspace/submission.csv.
# Used by `harbor run -a oracle` to establish the reward reference (no LLM).
set -euo pipefail

/opt/venvs/tabular/bin/python - <<'PY'
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

train = pd.read_csv("/workspace/train.csv")
test = pd.read_csv("/workspace/test.csv")
ID, TARGET = "TODO_ID", "TODO_TARGET"  # TODO: match prepare.py

feats = [c for c in train.columns if c not in {ID, TARGET} and c in test.columns]
model = HistGradientBoostingClassifier(random_state=0).fit(train[feats], train[TARGET])
preds = model.predict(test[feats])
pd.DataFrame({ID: test[ID], TARGET: preds}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
