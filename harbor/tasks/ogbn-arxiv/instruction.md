# OGBN-Arxiv (small) — graph node classification

Classify each node in a citation-style graph into one of **6 classes** (`label`).

## Data (already staged in `/workspace`)
- `node_features.npy` — float array, shape `(N, 32)`; row `i` is node `i`'s features.
- `edges.npy` — int array, shape `(2, E)`: the `edge_index` (each column is an edge).
- `train.csv` — `id` (node index) + target **`label`** (0–5), for the training nodes.
- `test.csv` — `id` (node index) only.

## Specialized library to use — REQUIRED
This is a graph task — use **torch-geometric** (PyG, pre-installed) to build a
graph neural network (e.g. a 2-layer `GCNConv`/`SAGEConv` model) that does message
passing over `edges.npy`. Do **not** ignore the graph and train a plain tabular
model on `node_features` alone — the edges carry the signal. Load the arrays,
build a `torch_geometric.data.Data`, and train a small GNN on CPU for a few
epochs. **ogb** is available for reference.

## Submission — REQUIRED
Write predictions to **`/workspace/submission.csv`** with columns:
- `id` — node index from `test.csv`
- `label` — predicted class (0–5)

Scoring: **accuracy** on the held-out nodes (higher is better).
