# <Task title> — <modality> <problem type>

<One-paragraph description of the problem.>

## Data (already staged in your workspace, `/workspace`)
- `train.csv` — labeled training data. Target column: **`<TARGET>`**.
- `test.csv` — rows to predict, without the target.
- `<any other files>` — <description>.

## Specialized library to use — REQUIRED
Use **<library>** as the primary approach (pre-installed in this environment).
<One or two sentences: what the library is and why it fits this data/task, so the
agent doesn't hand-roll a weaker approach.> You MAY `pip install` and compare
these alternatives: **<alt1>**, **<alt2>**.

```python
# Minimal usage sketch of the specialized library
```

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY these columns:
- `<ID>` — copied from `test.csv`
- `<TARGET>` — your predictions

Scoring: **<metric>** against the held-out labels (higher is better). Validate on
a held-out split of `train.csv` first.
