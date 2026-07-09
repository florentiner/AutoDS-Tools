# FathomNet (small) — multi-label image classification

Each 32×32 image may contain any subset of **4 categories**. Predict, for each
image, which categories are present (`label_0`..`label_3`, each 0/1).

## Data (already staged in `/workspace`)
- `train_images.npy` — uint8 `(N, 32, 32, 3)`, row-aligned with `train.csv`.
- `train.csv` — `id` + four binary targets `label_0`..`label_3`.
- `test_images.npy` — row-aligned with `test.csv`.
- `test.csv` — `id` only.

## Specialized library to use — REQUIRED
This is a multi-label image task. You **MUST** instantiate the backbone from
**timm** (`timm.create_model(...)`) or **torchvision.models**, with 4 sigmoid
outputs (one per category) — do **not** hand-roll the network from raw `torch.nn`
layers or flatten pixels into a tabular model. Use **albumentations** /
`torchvision.transforms` for augmentation. Keep it small and CPU-friendly.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with columns:
- `id` — from `test.csv`
- `label_0`, `label_1`, `label_2`, `label_3` — each 0 or 1

Scoring: **micro-F1** across the four labels (higher is better).
