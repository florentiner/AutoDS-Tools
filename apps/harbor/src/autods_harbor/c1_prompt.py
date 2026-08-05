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
        "## Specialized library to use — REQUIRED\n"
        "Use **LightAutoML** as the primary modeling approach (pre-installed). It is a\n"
        "tabular AutoML framework that automatically builds and blends gradient-boosting\n"
        "models (CatBoost / LightGBM / XGBoost), linear models and neural nets with\n"
        "automatic preprocessing — do **not** hand-roll separate boosting models; let\n"
        "LightAutoML's `TabularAutoML` preset do the model selection and blending. Give\n"
        "the target column the target role, fit on the training frame, predict on the\n"
        "test frame. The API reference is staged at **`/opt/docs/lightautoml.md`** — you\n"
        "**MUST read it (`cat /opt/docs/lightautoml.md`) BEFORE writing your solution**\n"
        "and follow its signatures. You MAY also use the pre-installed **featuretools** /\n"
        "**feature-engine** for feature engineering.\n"
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
    "## Sanity-check the submission — REQUIRED\n"
    "Before finishing, inspect what you are about to submit. A submission that is empty,\n"
    "constant, all-negative or otherwise degenerate scores ~0 even when training looked\n"
    "fine. If you see that, fix the cause (threshold, class weighting, longer training)\n"
    "instead of submitting it.\n"
    "\n"
    "## Time budget management — REQUIRED\n"
    "At the start, allocate your wall-clock budget across stages (exploration /\n"
    "training / improvement / finalisation) and print the allocation. Check elapsed\n"
    "time between stages. If behind schedule, drop optional work — a valid submission\n"
    "written comfortably before the budget ends takes priority over everything else.\n"
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
