#!/bin/bash
set -euo pipefail
/opt/venvs/nlp/bin/python - <<'PY'
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tr = pd.read_csv("/workspace/train.csv"); te = pd.read_csv("/workspace/test.csv")
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X = v.fit_transform(tr["text"].astype(str)); Xt = v.transform(te["text"].astype(str))
m = LogisticRegression(max_iter=1000).fit(X, tr["label"])
pd.DataFrame({"id": te["id"], "label": m.predict(Xt)}).to_csv("/workspace/submission.csv", index=False)
print("[oracle] wrote submission.csv")
PY
