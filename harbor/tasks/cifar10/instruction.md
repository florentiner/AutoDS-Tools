# CIFAR-10 (small) — image classification

Classify each 32×32 RGB image into one of **5 classes** (`label` 0–4).

## Data (already staged in `/workspace`)
- `train_images.npy` — uint8 array, shape `(N, 32, 32, 3)`, row-aligned with `train.csv`.
- `train.csv` — `id`, target **`label`** (0–4).
- `test_images.npy` — uint8 array, row-aligned with `test.csv`.
- `test.csv` — `id` only.

## Specialized library to use — REQUIRED
This is an image task. You **MUST** instantiate the model from **timm**
(`timm.create_model(...)`) or **torchvision.models** (e.g. a small ResNet) — do
**not** hand-roll the network from raw `torch.nn` layers, and do **not** flatten
the pixels into a tabular/logistic model. A small backbone (set `num_classes=5`,
`in_chans=3`) trained for a few epochs on these 32×32 images is enough; adapt the
input size if the model requires it. Use **albumentations** or
`torchvision.transforms` for light augmentation. Keep it small and CPU-friendly;
load the arrays with numpy and train with torch on CPU.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with EXACTLY two columns:
- `id` — from `test.csv`
- `label` — predicted class (0–4)

Scoring: **accuracy** against the held-out labels (higher is better). Validate on
a held-out split of the training images first.
