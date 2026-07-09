# AMP Parkinson's — disease-progression regression

Predict the UPDRS clinical score **`updrs`** from protein/peptide abundances
measured across patient visits over time.

## Data (already staged in `/workspace`)
- `train.csv` — `id`, `visit_month` (time), protein columns `P000..P039`, target **`updrs`**.
- `test.csv` — same features, **without** `updrs`.

## Specialized library to use — REQUIRED
Use **LightAutoML** (`TabularAutoML`, regression) as the primary model — it is
pre-installed and blends gradient-boosting/linear/NN models automatically. Give
`updrs` the target role and `visit_month` a datetime/numeric role. Engineer
per-visit temporal and relational features with **featuretools** (pre-installed);
you MAY also `pip install tsfresh` for automated time-series feature extraction on
the protein trajectories. Do not hand-roll a single boosting model. Consult the
library docs for the exact API.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with columns:
- `id` — from `test.csv`
- `updrs` — predicted score (float)

Scoring: **R²** on the held-out visits (higher is better; RMSE also reported).
