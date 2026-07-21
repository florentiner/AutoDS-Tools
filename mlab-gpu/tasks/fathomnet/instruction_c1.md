# FathomNet 2023 — multi-label underwater species classification

Dataset Description
The training and test images for the competition were all collected in the Monterey Bay Area between the surface and 1300 meters depth. The images contain bounding box annotations of 290 categories of bottom dwelling animals. The training and evaluation data are split across an 800 meter depth threshold: all training data is collected from 0-800 meters, evaluation data comes from the whole 0-1300 meter range. Since an organisms' habitat range is partially a function of depth, the species distributions in the two regions are overlapping but not identical. Test images are drawn from the same region but may come from above or below the depth horizon.

The competition goal is to label the animals present in a given image (i.e. multi-label classification) and determine whether the image is out-of-sample.

Data format
The training dataset are provide in two different formats: multi-label classification and object detection. The different formats live in the corresponding named directories. The datasets in these directories are identical aside from how they are organized.

Multi-label classification
Each line of the csv files indicates an image by its id and a list of corresponding categories present in the frame.

id, categories
4a7f2199-772d-486d-b8e2-b651246316b5, [1.0]
3bddedf6-4ff8-4e81-876a-564d2b03b364, "[1.0, 9.0, 11.0, 88.0]"
3f735021-f5de-4168-b139-74bf2859d12a, "[1.0, 37.0, 51.0, 119.0]"
130e185f-09c5-490c-8d08-641c4cbf6e54, "[1.0, 51.0, 119.0]"
The ids correspond those in the object detection files. The categories are the set of all unique annotations found in the associated image.

Object detection
The datasets are formatted to adhere to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to a category_id ranging from 1 to 290 and a supercategory from 1 to 20. The fine-grained annotations are taxonomic thought not always at the same level of the taxonomic tree.

Supercategories
Each category also belongs to one of 20 semantic supercategories as indicated in category_key.csv:

['Worm', 'Feather star', 'Sea cucumber', 'Squat lobster', 'Fish', 'Soft coral', 'Urchin', 'Sea star', 'Sea fan', 'Sea pen', 'Barnacle', 'Eel', 'Glass sponge', 'Shrimp', 'Black coral', 'Anemone', 'Sea spider', 'Gastropod', 'Crab', 'Stony coral']
These supercategories might be useful for certain training procedures. The supercategories are represented in both the training and validation set. But please note that submissions must be made identifying the 290 fine grained categories.


Files
multilabel_classification/train.csv - csv list of training images and categories
object_detection/train.json - the training images, annotations, and categories in COCO formatted json
object_detection/eval.json - the evaluation images in COCO formatted json
sample_submission.csv - a sample submission file in the correct format
category_key.csv - key mapping numerical index to category and supercategory name
metric.py - a Python implementation of the metric described in the evaluation_details.txt. This is the same metric used to calculate the leaderboard score.

## Data (already staged in your workspace, `/workspace`)
- `images/` — full-resolution competition JPGs, named `<id>.jpg`.
- `train.csv` — `id`, `categories` (list of category ids present in the image).
- `test.csv` — `id` only.
- `task_descriptor.txt` — original competition description.

## Specialized library to use — REQUIRED
Instantiate a CNN/ViT backbone from **timm** (`timm.create_model`) or **torchvision.models** with a 290-way sigmoid head; train multi-label over the real images. Do not hand-roll a trivial classifier.

## Training discipline — REQUIRED
Monitor micro-F1 on a held-out split each epoch, early-stop on plateau, print `VALIDATION_SCORE=<f1>`, keep the best checkpoint.

## Submission — REQUIRED
Write **`/workspace/submission.csv`** with columns:
- `id` — from `test.csv`
- `categories` — predicted list of category ids, e.g. `[1, 5, 12]`

Scoring: **micro-F1** over the 290 categories (higher is better).
