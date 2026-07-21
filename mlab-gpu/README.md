# mlab-gpu — full-data MLAB tasks, docker-less (venv) runner

Run four MLAB tasks on **full / original-fidelity data** with the AutoDS agent
(baseline + C1) **without docker** — for a GPU box that is itself a container
(docker-in-docker unavailable). Plain venvs reproduce the harbor container: an
agent-brain venv (`packages/autods` + `apps/harbor`, from this repo) and a
per-family child venv (torch + DS libs, via `AUTODS_CHILD_VENV`).

## Where things live

| where | holds |
|---|---|
| **this repo** (`mlab-gpu/`) | `run_venv.sh`, `requirements/`, and the agent (`packages/autods`, `apps/harbor`). AutoDS's C1 specialized-library layer is applied at runtime by `c1_prompt.py` — there are **no per-task C1 files**. |
| **HF dataset** `danil-e/harbor-datasets-mlab` | one folder per task under `datasets/mlab-real/<task>/`: `environment/data/`, `instruction.md` (base, from the original MLAgentBench repo), `score.py`. |

The dataset ships only the **base** instruction (agent-agnostic). AutoDS appends
its C1 layer itself; `base` mode disables it with `AUTODS_C1_DISABLED=1`.

## Tasks (competition-faithful, ≤5 GB each)

| task | family | metric |
|---|---|---|
| `amp-parkinsons` | tabular | SMAPE |
| `feedback` | nlp | MCRMSE (6 targets) |
| `fathomnet` | vision | micro-F1 (290 categories) |
| `identify-contrails` | vision | Dice |

## Run (on the GPU box)

```bash
git clone https://github.com/florentiner/AutoDS-Tools -b mlab-gpu-pub
export AUTODS_MODEL=gemma-4-31b-it AUTODS_API_KEY=sk-or-... \
       AUTODS_BASE_URL=https://openrouter.ai/api/v1

cd AutoDS-Tools/mlab-gpu
./run_venv.sh setup            # build venvs (GPU torch by default; TORCH_CPU=1 forces CPU)
./run_venv.sh feedback c1      # one task/mode  (base = AutoDS without C1)
./run_venv.sh all              # all 4 tasks, baseline + C1
```

The runner pulls each task's data straight from HF via `--repo` (like `harbor run
--repo …`) — **no manual clone needed**. Override the dataset with
`./run_venv.sh --repo https://huggingface.co/datasets/<owner>/<repo> <task> <mode>`
(default: `danil-e/harbor-datasets-mlab`). Only the requested task's folder is
downloaded (so a `feedback` run pulls ~9 MB, not the whole repo).

Per run → `runs/<task>-<mode>/`: `agent.log`, `trace.json`, `workspace/submission.csv`, `score.txt`.
Both modes read the same base `instruction.md`; `c1` lets AutoDS augment it, `base` sets `AUTODS_C1_DISABLED=1`.
