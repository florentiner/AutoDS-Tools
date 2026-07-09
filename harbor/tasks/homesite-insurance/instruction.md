# Homesite Insurance — tabular (classification) with a temporal split

You are solving the TabReD **homesite-insurance** task: predict whether a customer buys a home-insurance quote. This is real-world
tabular data with a **temporal** train/test split (rows are ordered by time), so
guard against distribution drift.

## Data (already staged in `/workspace`)
- `train.csv` — labeled training data. Target column: **`target`**. Has a
  `timestamp` column (time order) plus numeric (`num_*`) and categorical
  (`cat_*`) features and an `id` column.
- `test.csv` — later-in-time rows to predict, **without** `target`.

## Specialized library to use — REQUIRED
Use **LightAutoML** as the primary approach (pre-installed) — its `TabularAutoML`
preset for a classification task. It handles mixed numeric/categorical data and
**datetime roles** and automatically blends gradient-boosting + linear + NN
models, so do not hand-roll separate boosting models. Give the `target` column
the target role and the `timestamp` column a datetime role so the model can use
temporal structure. Consult the LightAutoML docs for the exact API.

Because the split is temporal, **validate on the latest slice of `train.csv`**
(not a random split). TabReD's strongest models are GBDT and simple MLP-like
tabular-DL — you MAY `pip install` and compare **rtdl** (MLP-PLR),
**pytorch-tabular**, or **tabm**
(`pip install "tabm @ git+https://github.com/yandex-research/tabm"`), and use
**featuretools** for temporal/relational features.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `id` — copied from `test.csv`
- `target` — probability of the positive class

Scoring: **ROC-AUC** on the held-out (future) rows (higher is better).
