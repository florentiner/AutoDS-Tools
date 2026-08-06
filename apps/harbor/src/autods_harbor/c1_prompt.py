"""Agent-side C1 augmentation for AutoDS (stdlib-only).

The task ships the ORIGINAL benchmark instruction (MLAgentBench / TabReD framing).
When the AutoDS agent runs, it appends this modality-specific "specialized library
+ training discipline" layer — the C1 experimental condition — so only AutoDS sees
it; any other Harbor agent (oracle, mini-swe-agent, …) runs on the plain original
instruction. Disable with ``AUTODS_C1_DISABLED=1`` (AutoDS-without-C1 ablation).

Kept import-free (like ``constants``) so it loads on both host and container.
"""

from __future__ import annotations

FAMILIES = ("tabular", "vision", "graph", "nlp")

# Modality-specific "which library to use" mandate. Generic (no per-task hints):
# the task specifics (target column, task type, metric) come from the original
# instruction the agent is already given.
_LIBRARY: dict[str, str] = {
    "tabular": (
        "## Specialized library to use — REQUIRED, STRICT\n"
        "The final model you submit MUST be **LightAutoML** (pre-installed). Start from this\n"
        "skeleton and adapt only the marked parts:\n"
        "```python\n"
        "from lightautoml.automl.presets.tabular_presets import TabularAutoML\n"
        "from lightautoml.tasks import Task\n"
        "\n"
        "task = Task('reg')            # 'reg' | 'binary' | 'multiclass' — per the instruction\n"
        "roles = {'target': TARGET_COLUMN}\n"
        "automl = TabularAutoML(task=task, timeout=TIME_BUDGET_SECONDS)\n"
        "oof = automl.fit_predict(train_df, roles=roles)      # train_df includes the target\n"
        "pred = automl.predict(test_df).data.reshape(-1)\n"
        "print('MODEL_USED=', type(automl).__name__)          # must print TabularAutoML\n"
        "```\n"
        "You MUST print the `MODEL_USED=` line, and it MUST say `TabularAutoML`.\n"
        "You MUST NOT submit predictions from `RandomForestRegressor`, `GradientBoosting*`,\n"
        "`Ridge`, `LinearRegression` or any other scikit-learn estimator as the final model —\n"
        "they are allowed only as a quick sanity reference you print alongside.\n"
        "If an import or an API call fails, read the error and fix it (check the installed\n"
        "package, e.g. `help(TabularAutoML)`); do NOT abandon LightAutoML.\n"
        "For multi-target problems, fit one TabularAutoML per target.\n"
        "You MAY additionally use the pre-installed **featuretools** / **feature-engine**.\n"
    ),
    "vision": (
        "## Specialized library to use — REQUIRED\n"
        "Build the model from a pre-installed vision library, do **not** hand-roll a CNN\n"
        "from scratch:\n"
        "- **classification** (single- or multi-label): instantiate the backbone from\n"
        "  **timm** (`timm.create_model(..., pretrained=False)`) or **torchvision.models**,\n"
        "  with an appropriate classification head and sigmoid/softmax as the task needs;\n"
        "- **segmentation**: use **segmentation-models-pytorch** (`smp.Unet(...)` or\n"
        "  similar) on a lightweight encoder.\n"
        "Consult the library docs (staged under `/opt/docs/` if present) for exact APIs.\n"
    ),
    "graph": (
        "## Specialized library to use — REQUIRED\n"
        "Use **PyTorch Geometric (torch_geometric)** (pre-installed) to build a real GNN\n"
        "(e.g. GCN / GraphSAGE / GAT) over the provided node features and edge index — do\n"
        "**not** ignore the graph structure and train a plain MLP on node features alone.\n"
        "If the target is defined per node PAIR (i, j) rather than per node, note that a\n"
        "score built only from two node embeddings usually cannot express it and collapses\n"
        "to the majority answer: make sure information can flow through the intermediate\n"
        "nodes the quantity actually depends on, and check that your predictions are not\n"
        "(nearly) constant across pairs.\n"
        "Consult the PyG docs (staged under `/opt/docs/` if present) for exact APIs.\n"
    ),
    "nlp": (
        "## Specialized library to use — REQUIRED\n"
        "Use a pre-installed transformer library as the primary approach: **HuggingFace\n"
        "`transformers`** (fine-tune / embed a pretrained encoder) or **sentence-transformers**\n"
        "for text embeddings + a lightweight head — do **not** rely on bag-of-words alone.\n"
        "Consult the library docs (staged under `/opt/docs/` if present) for exact APIs.\n"
    ),
}

# Modality-agnostic discipline blocks (identical to the current C1 prompts).
_DISCIPLINE = (
    "## Hardware — REQUIRED, STRICT\n"
    "Resolve the compute device with EXACTLY this snippet, before any training:\n"
    "```python\n"
    "import torch\n"
    "DEVICE = ('cuda' if torch.cuda.is_available()\n"
    "          else 'mps' if torch.backends.mps.is_available() else 'cpu')\n"
    "print('DEVICE=', DEVICE)\n"
    "```\n"
    "You MUST NOT hardcode `'cuda'`, and you MUST NOT write a CUDA-only check such as\n"
    "`'cuda' if torch.cuda.is_available() else 'cpu'` — on a machine whose accelerator is\n"
    "Apple MPS that silently drops you to CPU and training becomes ~10x slower.\n"
    "Move the model AND every batch to `DEVICE`; print the `DEVICE=` line so the log\n"
    "proves which device was used. On `mps` keep tensors float32 (no float64).\n"
    "\n"
    "## Training discipline — REQUIRED\n"
    "Do NOT stop after an arbitrary small budget (5 or 15 epochs is almost always badly\n"
    "undertrained). Choose a budget large enough to converge — for a small CNN/U-Net or a\n"
    "fine-tuned encoder on this data ~100 epochs is a sane starting point — and use early\n"
    "stopping with patience on a held-out split. If the validation metric is still improving\n"
    "when you stop, raise the budget and train again.\n"
    "Validate with the TASK'S OWN metric (the one named in the instruction), not just the\n"
    "training loss, and print `VALIDATION_SCORE=<value>`.\n"
    "\n"
    "## Decision threshold — REQUIRED when predictions are thresholded\n"
    "For binary / multi-label / segmentation outputs, do NOT leave the threshold at the\n"
    "default 0.5. Sweep candidate thresholds on the validation split and keep the one that\n"
    "maximises the task metric. With rare positives, 0.5 typically predicts nothing at all.\n"
    "If positives are rare, weight them in the loss accordingly.\n"
    "\n"
    "## Beat a trivial baseline — REQUIRED\n"
    "Before submitting, score at least one trivial predictor on your validation split with\n"
    "the TASK'S OWN metric — per-target median / most-frequent class, or the training mean —\n"
    "and print both numbers. If your trained model does not beat the trivial predictor, do\n"
    "not submit the model: fix it, or submit the simpler predictor that actually scores\n"
    "better. Merely MATCHING the trivial score is also a failure — it means the model\n"
    "collapsed onto the majority answer; change the approach rather than submitting it.\n"
    "Think about what the metric rewards, not just about fitting the data. For example\n"
    "SMAPE applies its maximum penalty whenever the true value is 0 and your prediction is\n"
    "non-zero — however small — so for a target that is often exactly 0, predicting exactly\n"
    "0 on those rows can matter more than fitting the non-zero rows precisely.\n"
    "\n"
    "## Sanity-check the submission — REQUIRED\n"
    "Before finishing, inspect what you are about to submit. A submission that is empty,\n"
    "constant, all-negative or otherwise degenerate scores ~0 even when training looked\n"
    "fine. If you see that, fix the cause (threshold, class weighting, longer training)\n"
    "instead of submitting it.\n"
    "\n"
    "## Time budget management — REQUIRED\n"
    "Assume you have hours, not minutes: this run is not tightly time-boxed. At the start\n"
    "allocate your wall-clock budget across stages (exploration / training / improvement /\n"
    "finalisation) and print the allocation, then check elapsed time between stages.\n"
    "Training the model to convergence is NOT 'optional work' — never cut the number of\n"
    "epochs to finish early. Drop optional exploration instead. As a rule of thumb, if the\n"
    "whole training phase finished within a couple of minutes you almost certainly\n"
    "undertrained: raise the epoch budget and train again until the validation metric\n"
    "stops improving.\n"
    "\n"
    "## Keep the best model — REQUIRED\n"
    "Save every model variant you train together with its validation score. Your FINAL\n"
    "submission predictions MUST come from the variant with the BEST validation score —\n"
    "never simply the last one trained.\n"
)


def normalize_family(family: str | None) -> str:
    """Map a free-form family/modality string onto one of FAMILIES (default tabular)."""
    f = (family or "").strip().lower()
    if f in _LIBRARY:
        return f
    if f in ("image", "images", "cv"):
        return "vision"
    if f in ("text", "language"):
        return "nlp"
    if f in ("node", "graphs"):
        return "graph"
    return "tabular"


def augment(instruction: str, family: str | None) -> str:
    """Append the C1 (specialized-library + discipline) layer for the given family."""
    fam = normalize_family(family)
    return f"{instruction.rstrip()}\n\n{_LIBRARY[fam]}\n{_DISCIPLINE}"


__all__ = ["FAMILIES", "augment", "normalize_family"]
