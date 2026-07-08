# Adding datasets & running `autods-harbor`

This branch ships the **AutoDS Harbor agent** (`autods_harbor.agent:AutoDSAgent`)
without any bundled datasets. You bring your own tasks (or point the agent at a
Harbor **registry** dataset) and get hub traces just like Harbor's pre-installed
benchmarks.

There are two ways to run AutoDS as a Harbor agent.

---

## A. Author your own Harbor task (recommended)

Copy the template and fill in the `TODO`s:

```bash
cp -r harbor/tasks/_template harbor/tasks/my-task
```

A task is a directory:

```
harbor/tasks/my-task/
├── task.toml                     # metadata; [metadata].family picks the base image
├── instruction.md               # the prompt — NAME the specialized library to use
├── environment/
│   ├── Dockerfile                # FROM autods-mlab-<family>; stage data
│   └── prepare.py                # write /workspace/{train,test}.csv + /opt/mlab/answer.csv
├── solution/solve.sh            # oracle (reward reference; runs without an LLM)
└── tests/
    ├── test.sh                   # scores /workspace/submission.csv -> /logs/verifier/reward.json
    └── score.py                  # your metric
```

Key rules:
- `instruction.md` must tell AutoDS the **submission path/format** and the
  **specialized library** to use (and why) — that is what makes AutoDS reach for
  the right tool instead of a weak baseline.
- `prepare.py` stages agent-visible data in `/workspace` and the held-out answer
  key in `/opt/mlab/answer.csv` (verifier-only).
- `tests/test.sh` must write a reward to `/logs/verifier/reward.json`
  (`{"reward": <float>, ...}`) or `/logs/verifier/reward.txt`.

Build the base + family image once, then run:

```bash
# 1) build the base + the family image your task.toml uses (e.g. tabular)
harbor/environments/build.sh tabular

# 2) validate the task with NO LLM (Harbor's oracle runs solution/solve.sh)
harbor run -p harbor/tasks/my-task -a oracle -o "$HOME/harbor-jobs"

# 3) run AutoDS (provide an OpenAI-compatible LLM via --ae)
harbor run -p harbor/tasks/my-task \
  -a autods_harbor.agent:AutoDSAgent \
  -m gemma-4-31b-it \
  --ae AUTODS_MODEL=gemma-4-31b-it \
  --ae AUTODS_API_KEY=sk-your-llm-key \
  --ae AUTODS_BASE_URL=https://openrouter.ai/api/v1 \
  -o "$HOME/harbor-jobs"
```

The task's `environment/Dockerfile` is `FROM autods-mlab-<family>`, so AutoDS is
already baked in and the agent's `install()` is a no-op.

---

## B. Run against a Harbor registry dataset (any base image)

To run AutoDS on a dataset from the hub (e.g. `terminal-bench`, or one you
`harbor publish`ed) whose task images are **not** `FROM autods-mlab-*`, the agent
installs AutoDS into the container at setup time. Point it at the AutoDS source
with a pip/git spec:

```bash
harbor run -d "<org>/<dataset>@<version>" \
  -a autods_harbor.agent:AutoDSAgent \
  -m gemma-4-31b-it \
  --ae AUTODS_MODEL=gemma-4-31b-it \
  --ae AUTODS_API_KEY=sk-your-llm-key \
  --ae AUTODS_BASE_URL=https://openrouter.ai/api/v1 \
  --ae "AUTODS_INSTALL_SPEC=$(printf '%s\n%s' \
      'git+https://github.com/<you>/AutoDS-Tools.git#subdirectory=packages/autods' \
      'git+https://github.com/<you>/AutoDS-Tools.git#subdirectory=apps/harbor')" \
  -o "$HOME/harbor-jobs"
```

- `AUTODS_INSTALL_SPEC` is one or more pip requirements (newline- or
  space-separated). The agent builds a uv-managed Python 3.12 venv in the
  container, installs them, and puts `autods-harbor` on `PATH`.
- Task code then runs in that venv; the AutoDS Coder can `pip install` any extra
  libraries the task needs at run time.
- Alternatively pass it as `--ak install_spec='...'`.

---

## C. Run locally in venvs (no Docker)

If you don't want Docker/Harbor, run tasks directly in Python venvs. This
provisions a framework venv (autods + autods-harbor) and a per-family child venv
(the specialized libs), prepares/downloads the data, runs the agent, and scores —
one task at a time. Outputs land in `./runs/<task>/` (no hub trace).

```bash
# add a dataset, then run it:
cp -r harbor/tasks/_template harbor/tasks/my-task     # fill in the TODOs
AUTODS_MODEL=gemma-4-31b-it \
AUTODS_API_KEY=sk-your-llm-key \
AUTODS_BASE_URL=https://openrouter.ai/api/v1 \
bash harbor/run_all_local.sh my-task                  # or no arg = every task

cat runs/my-task/reward.json          # the score
cat runs/my-task/autods_trace.json    # the AutoDS trace
```

Needs `python3.12` on PATH. To score against real competition data set
`MLAB_USE_KAGGLE=1` + `KAGGLE_USERNAME`/`KAGGLE_KEY`.

---

## D. Publish a dataset to the hub, then run it from the registry

Bundle tasks into a dataset and publish it (default **private**) so it appears
under https://hub.harborframework.com/datasets:

```bash
harbor dataset init <org>/<name> -o harbor/datasets --description "..."
harbor add harbor/tasks/task-a harbor/tasks/task-b --to harbor/datasets
# publish the tasks AND the dataset together (the manifest pins task digests,
# so the tasks must be published for the dataset to resolve):
HARBOR_API_KEY=sk-harbor-... harbor publish \
  harbor/tasks/task-a harbor/tasks/task-b harbor/datasets --private
# -> https://hub.harborframework.com/datasets/<org>/<name>
```

Then "use after adding" — run it from the registry and **auto-upload the trace**
with `--upload` (one command, no separate upload step):

```bash
export HARBOR_API_KEY=sk-harbor-...        # enables --upload
harbor run -d <org>/<name> -a autods_harbor.agent:AutoDSAgent -m gemma-4-31b-it \
  --ae AUTODS_MODEL=gemma-4-31b-it \
  --ae AUTODS_API_KEY=sk-your-llm-key \
  --ae AUTODS_BASE_URL=https://openrouter.ai/api/v1 \
  -o "$HOME/harbor-jobs" \
  --upload                                  # prints https://hub.harborframework.com/jobs/<id>
  # add --public to make the uploaded job shareable
```

> Portability note: the published tasks' `environment/Dockerfile` is
> `FROM autods-mlab-<family>` (a locally-built image). Build it first on whatever
> machine runs the dataset (`harbor/environments/build.sh tabular`), or change the
> base to a registry-hosted image so anyone can run the dataset without building.

---

## See the trace on the hub

Every trial writes an ATIF `trajectory.json`, so the run renders on the hub
(steps, tool calls, tokens, cost, reward) like a pre-installed dataset. Two ways:

```bash
# automatic — add --upload to `harbor run` (needs HARBOR_API_KEY):
harbor run ... --upload                # prints the hub URL when the job finishes

# after the fact — upload a finished job dir:
HARBOR_API_KEY=sk-harbor-... harbor upload "$HOME/harbor-jobs/<job-timestamp>"
# -> https://hub.harborframework.com/jobs/<id>   (private; add --public to share)
```

> `sk-harbor-...` is your **Hub** key (`HARBOR_API_KEY`) for upload/publish — it is
> NOT the LLM key. Keep it in a gitignored `harbor/.env`, never in version control.

> **Jobs dir must be Docker-shared.** With colima / Docker Desktop, keep `-o`
> under `$HOME` (not `/private/tmp`), or the container's `/logs` writes are lost.
