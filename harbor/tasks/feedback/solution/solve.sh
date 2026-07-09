#!/bin/bash
set -euo pipefail
/opt/venvs/nlp/bin/python - <<'PY'
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
v = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X = v.fit_transform(tr["text"].astype(str)); Xt = v.transform(te["text"].astype(str))
m = Ridge(alpha=1.0).fit(X, tr["score"])
pd.DataFrame({"id": te["id"], "score": m.predict(Xt)}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
