# IMDb — text sentiment classification

Classify each movie review as positive (`1`) or negative (`0`).

## Data (already staged in `/workspace`)
- `train.csv` — labeled reviews. Columns: `id`, `text`, and target **`label`** (0/1).
- `test.csv` — reviews to classify, with `id` + `text`, **without** `label`.

## Specialized library to use — REQUIRED
This is a text task — use the **HuggingFace transformers ecosystem**, which is
pre-installed. Do **not** rely on bag-of-words/TF-IDF as the final model. The
recommended, CPU-friendly approach is **sentence-transformers**: embed each
review with a small pretrained model (e.g. a MiniLM sentence encoder) and train a
lightweight classifier on those embeddings. Alternatively fine-tune a small
**transformers** model (e.g. DistilBERT), or `pip install setfit` and use
**SetFit** for few-shot contrastive fine-tuning. `transformers`,
`sentence-transformers` and `datasets` are pre-installed. You may compute
readability features with **textstat**. Keep it fast on CPU (small models, short sequences); a plain
TF-IDF baseline is fine only as a quick sanity check, not the submission.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `id` — copied from `test.csv`
- `label` — predicted sentiment (`0` or `1`)

Scoring: **accuracy** against the held-out labels (higher is better). Validate on
a held-out split of `train.csv` first.
