# AutoDS × Harbor — MLAB & TabReD

Run **AutoDS** (the multi-agent AutoDS pipeline) as a first-class
[Harbor](https://www.harborframework.com) agent, evaluated on two data-science
benchmarks ported to the Harbor task format:

- **MLAB** — the 13 core [MLAgentBench](https://github.com/snap-stanford/MLAgentBench)
  tasks (mixed modalities: tabular, text, image, graph, systems).
- **TabReD** — the 8 real-world *temporal* tabular tasks from
  [Yandex Research](https://github.com/yandex-research/tabred) (ICLR 2025).

Each trial produces a native **ATIF `trajectory.json`**, so the Harbor Hub renders
per-step reasoning, tool calls, token usage, cost and reward — exactly like any
built-in agent.

## How it works

AutoDS runs **inside** the task container as a Harbor *installed agent*
(`autods_harbor.agent:AutoDSAgent`, `SUPPORTS_ATIF = True`):

1. `install()` verifies the `autods-harbor` entrypoint is baked into the image.
2. `run()` execs `autods-harbor` in the container. It runs the existing
   `build_pipeline(workspace).astream({"task": instruction})` (Analyst →
   Researcher → Planner → Coder → Presenter), writes the submission to
   `/workspace`, and records a light-weight raw trace + token usage to the agent
   log dir. `AUTODS_MODEL/API_KEY/BASE_URL` are forwarded from the host via
   `--ae`.
3. `populate_context_post_run()` converts the raw trace into an ATIF
   `trajectory.json` and backfills `n_input_tokens` / `n_output_tokens` /
   `cost_usd` onto the trial.

AutoDS's code-execution tools run in a **dedicated per-family child venv**
(`/opt/venvs/<family>`, selected via `AUTODS_CHILD_VENV`) that holds the
specialized libraries, isolated from the AutoDS framework interpreter to avoid
dependency conflicts.

```
apps/harbor/                     # the agent adapter (host + container code)
  src/autods_harbor/
    agent.py        # AutoDSAgent(BaseInstalledAgent)   [host]
    atif.py         # raw trace -> ATIF trajectory.json  [host]
    entrypoint.py   # `autods-harbor` CLI                [container]
    usage.py        # LangChain token/cost callback      [container]
harbor/
  environments/     # base + per-family Docker images (build.sh)
  tasks/<task>/     # task.toml, instruction.md, environment/, solution/, tests/
```

## Environment matrix

One image per **family**, all `FROM autods-mlab-base` (AutoDS framework + the
`autods-harbor` entrypoint, no GRAD/libq). The specialized libraries live in a
child venv and are named in each task's `instruction.md`; alternatives are
pip-installable at task time.

| Family image | Tasks | Baked specialized libraries (CPU-only) |
|---|---|---|
| `autods-mlab-tabular` | spaceship-titanic, house-price, amp-parkinsons | lightautoml, featuretools, feature-engine, scikit-learn, pandas, numpy<2 (+torch CPU) |
| `autods-mlab-tabred` | 8 TabReD tasks | lightautoml, pytorch-tabular, rtdl, pytorch-frame, tabm, featuretools, category_encoders |
| `autods-mlab-nlp` | imdb, feedback, babylm, llama-inference | torch(cpu), transformers, datasets, tokenizers, accelerate, sentence-transformers, setfit, textstat, ctranslate2, optimum[onnxruntime] |
| `autods-mlab-vision` | cifar10, fathomnet, identify-contrails | torch(cpu), torchvision, timm, segmentation-models-pytorch, albumentations, kornia, opencv-python-headless |
| `autods-mlab-graph` | ogbn-arxiv | torch(cpu), torch-geometric, ogb, networkx |
| `autods-mlab-clrs` | CLRS | jax[cpu], dm-haiku, optax, jraph, flax, clrs |
| `autods-mlab-numeric` | vectorization | numpy, numba, numexpr, cython |

### Build images

```bash
# base + tabular (default)
harbor/environments/build.sh
# specific families
harbor/environments/build.sh tabular nlp
# everything
harbor/environments/build.sh --all
```

## Run a task

Images must be built locally first (Harbor task `environment/Dockerfile`s use
`FROM autods-mlab-<family>`).

> **Jobs dir must be on a Docker-shared path.** Harbor bind-mounts the trial's
> `agent/` and `verifier/` dirs into the container as `/logs`. With a VM-based
> Docker (colima, Docker Desktop) the jobs dir must be under a path the VM
> mounts (e.g. under `$HOME`), otherwise the container's trace/reward writes are
> lost. Pass `-o "$HOME/harbor-jobs"` (not a `/private/tmp` path).

**Validate task mechanics without an LLM** (Harbor's oracle runs `solution/solve.sh`):

```bash
harbor run -p harbor/tasks/spaceship-titanic -a oracle
```

**Run AutoDS** — provide an OpenAI-compatible LLM endpoint via `--ae`:

```bash
harbor run \
  -p harbor/tasks/spaceship-titanic \
  -a autods_harbor.agent:AutoDSAgent \
  -m gpt-5 \
  --ae AUTODS_MODEL=gpt-5 \
  --ae AUTODS_API_KEY=sk-your-llm-key \
  --ae AUTODS_BASE_URL=https://api.openai.com/v1
```

For the agent import path to resolve, install this package into the harbor
environment:

```bash
uv tool install harbor --with ./apps/harbor
```

### API keys

- **LLM** (`AUTODS_API_KEY` / `AUTODS_BASE_URL` / `AUTODS_MODEL`): any
  OpenAI-compatible endpoint AutoDS should call. Forwarded into the container
  with `--ae`.
- **Harbor Hub** (`HARBOR_API_KEY`, an `sk-harbor-…` key): used by `harbor
  upload` / `harbor publish` / `harbor hub` to push results and datasets to the
  hub. It is **not** an LLM key. Keep it in a gitignored `harbor/.env`, never in
  version control.

## Specialized library per task

For each task the `instruction.md` names the domain-optimal library to use and
what for. LightAutoML already wraps CatBoost/LightGBM/XGBoost, so those are not
named separately.

### MLAB (MLAgentBench)

| Task | Modality | Primary library | Alternatives |
|---|---|---|---|
| spaceship-titanic | tabular clf | LightAutoML | AutoGluon, FLAML, featuretools |
| house-price | tabular reg | LightAutoML | AutoGluon, FLAML, feature-engine |
| amp-parkinsons | proteomic/clinical reg | LightAutoML + tsfresh | sktime, Darts, statsforecast |
| imdb | text clf | transformers / LightAutoML-NLP | sentence-transformers, SetFit, spaCy |
| feedback | text multi-reg | transformers + sentence-transformers + textstat | LightAutoML-NLP, SetFit |
| cifar10 | image clf | timm + torchvision + albumentations | fastai, kornia |
| fathomnet | image multi-label | timm/torchvision + albumentations | kornia |
| identify-contrails | segmentation | segmentation-models-pytorch + albumentations | MONAI, mmsegmentation |
| ogbn-arxiv | graph node clf | torch-geometric + ogb | DGL, networkx |
| CLRS | algorithmic reasoning | clrs + jax + dm-haiku + jraph | flax |
| babylm | LM pretraining | transformers + tokenizers + datasets | nanoGPT patterns |
| llama-inference | CPU inference speedup | ctranslate2 / optimum[onnxruntime] / llama-cpp-python | intel-extension-for-pytorch, openvino |
| vectorization | numpy speedup | numpy + numba | numexpr, cython |

### TabReD (temporal tabular; GBDT + MLP-like DL win)

| Task | Problem | Primary library | Alternatives |
|---|---|---|---|
| homesite-insurance | binary clf (AUC) | LightAutoML | pytorch-tabular, rtdl, TabM |
| ecom-offers | binary clf (AUC) | LightAutoML + featuretools | AutoGluon, TabM |
| homecredit-default | binary clf (AUC) | LightAutoML + featuretools | pytorch-frame, TabM |
| sberbank-housing | reg (RMSE) | LightAutoML | pytorch-tabular, rtdl |
| cooking-time | reg (RMSE) | LightAutoML + featuretools | TabM, rtdl |
| delivery-eta | reg (RMSE) | LightAutoML + featuretools | pytorch-tabular, TabM |
| maps-routing | reg (RMSE) | LightAutoML + TabM/rtdl | AutoGluon |
| weather | reg (RMSE) | LightAutoML + rtdl/TabM | sktime, tsfresh |

## Data & scoring

- Each task's `environment/prepare.py` stages data into `/workspace` and writes a
  held-out answer key to `/opt/mlab/answer.csv` (outside the agent's workspace).
  Real competition data is used when `MLAB_USE_KAGGLE=1` and the `kaggle` CLI has
  credentials; otherwise a schema-accurate **synthetic** dataset is generated so
  tasks run offline.
- `tests/test.sh` scores `/workspace/submission.csv` against the answer key and
  writes the reward to `/logs/verifier/reward.json` (accuracy / R² / AUC / RMSE
  per task).

## Notes

- **CPU-only**: images pin CPU wheels (e.g. torch from the CPU index).
  Vision/graph/LM tasks run but may score modestly under CPU time limits; their
  prompts mandate small models / subsampling.
- Hardening: run the agent as a non-root user (`[agent] user` in `task.toml`)
  with a root-owned `answer.csv` to fully prevent answer-key leakage.
