#!/bin/bash
# Oracle solution: a fast reference regressor producing a valid submission.
set -euo pipefail

/opt/venvs/tabular/bin/python - <<'PY'
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

train = pd.read_csv("/workspace/train.csv")
test = pd.read_csv("/workspace/test.csv")
target = "SalePrice"

drop = {"Id", target}
feats = [c for c in train.columns if c not in drop and c in test.columns]
X = train[feats].apply(pd.to_numeric, errors="coerce")
X_test = test[feats].apply(pd.to_numeric, errors="coerce")
y = np.log1p(train[target].astype(float))

model = HistGradientBoostingRegressor(random_state=0).fit(X, y)
preds = np.expm1(model.predict(X_test))

pd.DataFrame({"Id": test["Id"], "SalePrice": preds}).to_csv(
    "/workspace/submission.csv", index=False
)
print(f"[oracle] wrote /workspace/submission.csv ({len(preds)} rows)")
PY
