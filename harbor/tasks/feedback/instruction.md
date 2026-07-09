# Feedback Prize — essay quality scoring (text regression)

Predict a continuous English-language-proficiency **`score`** (roughly 1–5) for
each essay.

## Data (already staged in `/workspace`)
- `train.csv` — columns `id`, `text`, target **`score`** (float).
- `test.csv` — essays with `id` + `text`, **without** `score`.

## Specialized library to use — REQUIRED
This is a text task — use the **HuggingFace transformers ecosystem** (pre-installed).
The recommended CPU-friendly approach is **sentence-transformers**: embed each
essay with a small pretrained sentence encoder and train a regressor (e.g. ridge)
on the embeddings. You may also fine-tune a small **transformers** model. Combine
with **textstat** readability features (sentence length, syllable counts, reading
ease), which are strong signals for essay quality. Do **not** use plain TF-IDF as
the final model. Keep it fast on CPU (small models).

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `id` — copied from `test.csv`
- `score` — predicted score (float)

Scoring: **R²** on the held-out essays (higher is better; RMSE also reported).
