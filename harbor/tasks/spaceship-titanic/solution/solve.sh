#!/bin/bash
# Oracle solution: a fast, self-contained reference model that produces a valid
# submission (the reward reference the agent is compared against). Uses the
# tabular child venv (scikit-learn); it is intentionally simpler than what the
# AutoDS agent is asked to do with LightAutoML.
set -euo pipefail

/opt/venvs/tabular/bin/python - <<'PY'
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv("/workspace/train.csv")
test = pd.read_csv("/workspace/test.csv")


def engineer(df):
    df = df.copy()
    cabin = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["Side"] = cabin[2]
    return df


train, test = engineer(train), engineer(test)
cat = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
num = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
feats = cat + num

X = train[feats].copy()
X_test = test[feats].copy()
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat] = enc.fit_transform(X[cat].astype(str))
X_test[cat] = enc.transform(X_test[cat].astype(str))
X = X.apply(pd.to_numeric, errors="coerce").fillna(-1)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(-1)

y = train["Transported"].astype(bool)
model = HistGradientBoostingClassifier(random_state=0).fit(X, y)
preds = model.predict(X_test).astype(bool)

pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": preds}).to_csv(
    "/workspace/submission.csv", index=False
)
print(f"[oracle] wrote /workspace/submission.csv ({len(preds)} rows)")
PY
