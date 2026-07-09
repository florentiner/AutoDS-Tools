# CIFAR-10 (small) — image classification

Classify each 32×32 RGB image into one of **5 classes** (`label` 0–4).

## Data (already staged in `/workspace`)
- `train_images.npy` — uint8 array, shape `(N, 32, 32, 3)`, row-aligned with `train.csv`.
- `train.csv` — `id`, target **`label`** (0–4).
- `test_images.npy` — uint8 array, row-aligned with `test.csv`.
- `test.csv` — `id` only.

## Specialized library to use — REQUIRED
This is an image task — build a small **convolutional neural network** with
**torchvision** and/or **timm** (both pre-installed). Do **not** flatten the
pixels into a plain tabular/logistic model. Use **albumentations** for light
augmentation if helpful. Keep it small and CPU-friendly: a tiny CNN (or a small
`timm` model like a minimal ResNet) trained for only a few epochs on these 32×32
images is enough. Load the arrays with numpy, convert to tensors, and train with
torch on CPU.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `id` — from `test.csv`
- `label` — predicted class (0–4)

Scoring: **accuracy** against the held-out labels (higher is better). Validate on
a held-out split of the training images first.
