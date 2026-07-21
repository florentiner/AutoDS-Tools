# mlab-gpu — full-data MLAB tasks, docker-less (venv) runner

Run four MLAB tasks on **full / original-fidelity data** with the AutoDS agent
(baseline + C1) **without docker** — for a GPU box that is itself a container
(docker-in-docker unavailable). Plain venvs reproduce what the harbor container
does: an agent-brain venv (`packages/autods` + `apps/harbor`, from this repo) and
a per-family child venv (torch + DS libs, via `AUTODS_CHILD_VENV`).

## Split of responsibilities

| where | holds |
|---|---|
| **this repo** (`mlab-gpu/`) | `run_venv.sh`, `requirements/`, per-task `tasks/<task>/instruction_c1.md` (autods C1 prompt), and the agent (`packages/autods`, `apps/harbor`) |
| **HF dataset** `danil-e/mlab-gpu-fork` | per task: `<task>/data/`, `<task>/score.py`, `<task>/instruction_base.md` (agent-agnostic baseline) |

## Tasks (competition-faithful, ≤5 GB each)

| task | family | metric |
|---|---|---|
| `amp-parkinsons` | tabular | SMAPE |
| `feedback` | nlp | MCRMSE (6 targets) |
| `fathomnet` | vision | micro-F1 (290 categories) |
| `identify-contrails` | vision | Dice |

## Run (on the GPU box)

```bash
git clone https://github.com/florentiner/AutoDS-Tools && cd AutoDS-Tools/mlab-gpu

# data (Git-LFS)
git lfs install
git clone https://huggingface.co/datasets/danil-e/mlab-gpu-fork ~/mlab-gpu-data
export DATA_ROOT=~/mlab-gpu-data

# your LLM
export AUTODS_MODEL=gemma-4-31b-it AUTODS_API_KEY=sk-or-... \
       AUTODS_BASE_URL=https://openrouter.ai/api/v1

./run_venv.sh setup            # build venvs (GPU torch by default; TORCH_CPU=1 forces CPU)
./run_venv.sh feedback c1      # one task/mode
./run_venv.sh all              # all 4 tasks, baseline + C1
```

Per run → `runs/<task>-<mode>/`: `agent.log`, `trace.json`, `workspace/submission.csv`, `score.txt`.
`base` uses the HF `instruction_base.md`; `c1` uses this repo's `instruction_c1.md`.
