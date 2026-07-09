"""OGBN-Arxiv-style node classification (synthetic homophilous graph, offline).

A single graph: N nodes with class-correlated features and within-class-preferring
edges. Saved as node_features.npy (N x F), edges.npy (2 x E edge_index), plus
train/test node id splits — so solving it needs a graph neural network.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True); ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    N, C, F = 2400, 6, 32
    y = rng.integers(0, C, N)
    centers = rng.normal(0, 1.5, (C, F))
    X = (centers[y] + rng.normal(0, 1.0, (N, F))).astype(np.float32)
    # homophilous edges: connect nodes, preferring same-class
    src, dst = [], []
    for _ in range(N * 6):
        i = int(rng.integers(0, N))
        if rng.random() < 0.8:
            same = np.where(y == y[i])[0]; j = int(rng.choice(same))
        else:
            j = int(rng.integers(0, N))
        if i != j:
            src += [i, j]; dst += [j, i]
    edges = np.array([src, dst], dtype=np.int64)
    np.save(WORKSPACE / "node_features.npy", X)
    np.save(WORKSPACE / "edges.npy", edges)
    idx = rng.permutation(N); tr_idx, te_idx = idx[: int(N * 0.8)], idx[int(N * 0.8):]
    pd.DataFrame({"id": tr_idx, "label": y[tr_idx]}).to_csv(WORKSPACE / "train.csv", index=False)
    pd.DataFrame({"id": te_idx}).to_csv(WORKSPACE / "test.csv", index=False)
    pd.DataFrame({"id": te_idx, "label": y[te_idx]}).to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] ogbn-arxiv nodes={N} edges={edges.shape[1]} classes={C}")


if __name__ == "__main__":
    main()
