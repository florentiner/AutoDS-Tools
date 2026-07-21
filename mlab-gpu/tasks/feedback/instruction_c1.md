# Feedback Prize (ELL) — score essays on 6 analytic measures

Dataset Description
The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions.

Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set.

File and Field Information
train.csv - The training set, comprising the full_text of each essay, identified by a unique text_id. The essays are also given a score for each of the seven analytic measures above: cohesion, etc. These analytic measures comprise the target for the competition.
test.csv - For the test data we give only the full_text of an essay together with its text_id.
sample_submission.csv - A submission file in the correct format. See the evaluation_details.txt for details.


## Data (already staged in your workspace, `/workspace`)
- `train.csv` — `text_id`, `full_text`, and the 6 target columns.
- `test.csv` — `text_id`, `full_text` (no targets).
- `task_descriptor.txt` — original competition description.

## Specialized library to use — REQUIRED
This is a text-regression task — use **transformers** / **sentence-transformers** (pre-installed) to embed or fine-tune a language model over `full_text`, then regress the 6 targets. Do not hand-roll bag-of-words only.

## Training discipline — REQUIRED
Monitor a held-out split every epoch, early-stop on plateau, print `VALIDATION_SCORE=<MCRMSE>`. Keep the best checkpoint.

## Submission — REQUIRED
Write **`/workspace/submission.csv`** with columns:
- `text_id` — from `test.csv`
- `cohesion`, `syntax`, `vocabulary`, `phraseology`, `grammar`, `conventions` — predicted scores (1.0–5.0 scale)

Scoring: **MCRMSE** (mean columnwise RMSE) across the 6 targets (lower is better).
