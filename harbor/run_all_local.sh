#!/usr/bin/env bash
# Run AutoDS tasks LOCALLY in Python venvs — NO Docker — one task at a time.
#
# For each task it (once) provisions:
#   * a framework venv  (.venvs/framework)   — autods + autods-harbor (the pipeline)
#   * a family child venv (.venvs/<family>)  — that family's specialized libraries
# then: prepares/downloads the data, runs the autods-harbor agent, and scores it.
# Per-task outputs land in ./runs/<task>/ (submission.csv, autods_trace.json, reward.json).
#
# NOTE: this local path does NOT create a Harbor hub job/trace. For hub traces,
# use the Docker path (harbor/run_all.sh) or `harbor run` + `harbor upload`.
#
# Required env (an OpenAI-compatible LLM AutoDS calls — NOT the sk-harbor key):
#   AUTODS_MODEL      e.g. gemma-4-31b-it
#   AUTODS_API_KEY    e.g. sk-or-v1-...
#   AUTODS_BASE_URL   e.g. https://openrouter.ai/api/v1
# Optional:
#   PYTHON (default python3.12), RUNS_DIR (./runs), VENVS_DIR (./.venvs)
#   RESEARCH_DISABLED=1 (default), MLAB_USE_KAGGLE=1 + KAGGLE_USERNAME/KAGGLE_KEY
#
# Usage:
#   bash harbor/run_all_local.sh                     # every task (skips _template)
#   bash harbor/run_all_local.sh spaceship-titanic   # specific task(s)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${AUTODS_MODEL:?set AUTODS_MODEL (the LLM model name)}"
: "${AUTODS_API_KEY:?set AUTODS_API_KEY (an OpenAI-compatible LLM key)}"
: "${AUTODS_BASE_URL:?set AUTODS_BASE_URL (the LLM endpoint)}"

PYTHON="${PYTHON:-python3.12}"
RUNS_DIR="${RUNS_DIR:-$ROOT/runs}"
VENVS_DIR="${VENVS_DIR:-$ROOT/.venvs}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "need $PYTHON on PATH (set PYTHON=...)"; exit 1; }

# --- framework venv: autods + autods-harbor -------------------------------
FW="$VENVS_DIR/framework"
if [ ! -x "$FW/bin/autods-harbor" ]; then
  echo ">> creating framework venv ($FW)"
  "$PYTHON" -m venv "$FW"
  "$FW/bin/pip" install -q -U pip
  "$FW/bin/pip" install -q "$ROOT/packages/autods" "$ROOT/apps/harbor"
fi

task_family() { sed -nE 's/^\s*family\s*=\s*"([^"]+)".*/\1/p' "$1/task.toml" | head -1; }

# --- per-family child venv: specialized libraries -------------------------
ensure_family_venv() {  # $1=family; prints venv path on stdout
  local fam="$1" venv="$VENVS_DIR/$1" req="$ROOT/harbor/environments/$1/requirements.txt"
  if [ ! -x "$venv/bin/python" ]; then
    echo ">> creating child venv for family '$fam' (installs specialized libs; first time is slow)" >&2
    "$PYTHON" -m venv "$venv"
    "$venv/bin/pip" install -q -U pip wheel "setuptools<81" >&2
    ( "$venv/bin/pip" install -q "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu \
        || "$venv/bin/pip" install -q "torch==2.2.2" ) >&2 || true
    [ -f "$req" ] && "$venv/bin/pip" install -q -r "$req" >&2
  fi
  echo "$venv"
}

# --- task list ------------------------------------------------------------
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  while IFS= read -r d; do TASKS+=("$(basename "$d")"); done \
    < <(find harbor/tasks -mindepth 1 -maxdepth 1 -type d ! -name '_*' | sort)
fi

for t in "${TASKS[@]}"; do
  tdir="harbor/tasks/$t"
  [ -f "$tdir/task.toml" ] || { echo "!! skip '$t' (no harbor/tasks/$t/task.toml)"; continue; }
  fam="$(task_family "$tdir")"; fam="${fam:-tabular}"
  cvenv="$(ensure_family_venv "$fam")"

  run="$RUNS_DIR/$t"; ws="$run/workspace"; ans="$run/answer"
  mkdir -p "$ws" "$ans"

  echo "== [$t] family=$fam — preparing data"
  MLAB_WORKSPACE="$ws" MLAB_ANSWER_DIR="$ans" "$cvenv/bin/python" "$tdir/environment/prepare.py"

  echo "== [$t] running AutoDS (venv: framework; task code venv: $fam)"
  AUTODS_CHILD_VENV="$cvenv" AUTODS_VENV_SYSTEM_SITE_PACKAGES=0 LIBQ_DISABLED=1 \
  RESEARCH_DISABLED="${RESEARCH_DISABLED:-1}" \
  AUTODS_MODEL="$AUTODS_MODEL" AUTODS_API_KEY="$AUTODS_API_KEY" AUTODS_BASE_URL="$AUTODS_BASE_URL" \
    "$FW/bin/autods-harbor" \
      --instruction-file "$tdir/instruction.md" \
      --workspace "$ws" \
      --trace-out "$run/autods_trace.json" || echo "!! [$t] agent run failed"

  if [ -f "$tdir/tests/score.py" ] && [ -f "$ws/submission.csv" ]; then
    "$cvenv/bin/python" "$tdir/tests/score.py" \
      --submission "$ws/submission.csv" --answer "$ans/answer.csv" \
      --reward-json "$run/reward.json" || true
    echo "== [$t] reward: $(cat "$run/reward.json" 2>/dev/null)"
  else
    echo "!! [$t] no submission.csv produced (see $run/autods_trace.json)"
  fi
done

echo ">> done. Per-task outputs under $RUNS_DIR/"
