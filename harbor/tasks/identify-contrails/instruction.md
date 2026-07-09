# Identify Contrails (small) — image segmentation

For each 32×32 image, predict a coarse **8×8 binary mask** marking contrail
pixels (1 = contrail, 0 = background).

## Data (already staged in `/workspace`)
- `train_images.npy` — uint8 `(N, 32, 32, 3)`, row-aligned with `train.csv`.
- `train.csv` — `id` + 64 mask columns `m00`..`m63` (the flattened 8×8 target mask).
- `test_images.npy` — row-aligned with `test.csv`.
- `test.csv` — `id` only.

## Specialized library to use — REQUIRED
This is a segmentation task — use **segmentation-models-pytorch** (pre-installed)
to build a small U-Net (or similar) on top of a lightweight encoder, trained with
**torch** on CPU. Predict the mask, then downsample/threshold to the 8×8 grid for
submission. Use **albumentations** for augmentation if helpful. Do not treat it as
64 independent tabular targets. Keep the encoder small and train only a few epochs.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with columns:
- `id` — from `test.csv`
- `m00`..`m63` — the flattened 8×8 predicted mask (each 0 or 1)

Scoring: **Dice coefficient** between predicted and true masks (higher is better).
