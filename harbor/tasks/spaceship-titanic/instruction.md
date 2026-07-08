# Spaceship Titanic — tabular binary classification

Predict whether each passenger was `Transported` to an alternate dimension.

## Data (already staged in your workspace, `/workspace`)
- `train.csv` — labeled training data. Target column: **`Transported`** (boolean).
- `test.csv` — test passengers **without** the target.
- `task_descriptor.txt` — full description of every column.

## Specialized library to use — REQUIRED
Use **LightAutoML** as the primary modeling approach. It is pre-installed in this
environment. LightAutoML is a tabular AutoML framework that automatically builds
and blends **gradient-boosting models (CatBoost / LightGBM / XGBoost), linear
models and neural nets** with automatic preprocessing — so do **not** hand-roll
separate boosting models; let LightAutoML do the model selection and blending.

```python
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

task = Task("binary", metric="accuracy")
automl = TabularAutoML(task=task, timeout=600)
oof = automl.fit_predict(train_df, roles={"target": "Transported"})
preds = automl.predict(test_df).data[:, 0]  # probability of Transported=True
```

Feature engineering (do this before fitting):
- Split `Cabin` on "/" into `Deck` / `Cabin_num` / `Side`.
- Group features from `PassengerId` (`gggg_pp` → group `gggg`, size, position).
- Optionally use the pre-installed **featuretools** and **feature-engine** for
  richer features (`pip install category_encoders` if you want target/GLMM
  encoders).

You MAY `pip install` and compare alternatives if helpful: **AutoGluon**, **FLAML**.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `PassengerId` — copied from `test.csv`
- `Transported` — boolean predictions (`True` / `False`)

Scoring: **accuracy** of `Transported` against the held-out labels (higher is
better). Validate your approach on a held-out split of `train.csv` before
predicting on `test.csv`.
