# House Prices — tabular regression

Predict the final **`SalePrice`** (in dollars) for each home.

## Data (already staged in your workspace, `/workspace`)
- `train.csv` — labeled training data. Target column: **`SalePrice`**.
- `test.csv` — test homes **without** the target. Has an `Id` column.
- `data_description.txt` — description of the features.

## Specialized library to use — REQUIRED
Use **LightAutoML** as the primary modeling approach (pre-installed). It is a
tabular AutoML framework that automatically builds and blends gradient-boosting
models (CatBoost / LightGBM / XGBoost), linear models and neural nets with
automatic preprocessing — do **not** hand-roll separate boosting models.

```python
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

task = Task("reg", metric="mae")           # regression; also report RMSE/R2
automl = TabularAutoML(task=task, timeout=600)
automl.fit_predict(train_df, roles={"target": "SalePrice"})
preds = automl.predict(test_df).data[:, 0]
```

Consider log-transforming `SalePrice` for training (prices are right-skewed) and
inverse-transforming predictions. The pre-installed **featuretools** /
**feature-engine** libraries can add interaction and ratio features. You MAY
`pip install` **AutoGluon** or **FLAML** to compare.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `Id` — copied from `test.csv`
- `SalePrice` — predicted price (float)

Scoring: the submission is evaluated by **R²** on the held-out homes (higher is
better; MAE is also reported). Validate on a held-out split of `train.csv` first.
