#!/usr/bin/env bash
# Docker-less runner for the full-data MLAB tasks. Lives in the AutoDS fork
# (installs the agent from this repo's own packages/autods + apps/harbor), and
# reads the DATA + baseline instruction + scorer from the HF dataset
# `danil-e/mlab-gpu-fork` (clone it, pass its path as DATA_ROOT).
#
# Split:
#   HF dataset  -> <task>/data, <task>/score.py, <task>/instruction_base.md
#   this repo   -> mlab-gpu/tasks/<task>/instruction_c1.md  (autods C1 prompt)
#
# Runs inside a GPU container where docker-in-docker is unavailable: it uses
# plain venvs (agent-brain venv + per-family child venv), GPU torch by default.
#
# Usage:
#   export AUTODS_MODEL=... AUTODS_API_KEY=... AUTODS_BASE_URL=...
#   export DATA_ROOT=/path/to/cloned/mlab-gpu-fork
#   ./run_venv.sh setup
#   ./run_venv.sh <task> <base|c1>
#   ./run_venv.sh all
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"                  # AutoDS repo root (mlab-gpu/ lives under it)
: "${DATA_ROOT:?set DATA_ROOT to the cloned HF dataset danil-e/mlab-gpu-fork}"
: "${AUTODS_MODEL:?set AUTODS_MODEL}"; : "${AUTODS_API_KEY:?set AUTODS_API_KEY}"
: "${AUTODS_BASE_URL:=https://openrouter.ai/api/v1}"
PY="${PYTHON:-python3}"; AGENT="$HERE/.venv-agent"

family_of() { case "$1" in
  amp-parkinsons) echo tabular;; feedback) echo nlp;;
  fathomnet|identify-contrails) echo vision;; *) echo "unknown task $1" >&2; exit 1;; esac; }
torch_pkgs() { case "$1" in vision) echo "torch==2.2.2 torchvision==0.17.2";; *) echo "torch==2.2.2";; esac; }

build_agent_venv() {
  [ -x "$AGENT/bin/autods-harbor" ] && return 0
  echo ">> agent-brain venv (from $REPO/packages/autods + apps/harbor)"
  "$PY" -m venv "$AGENT"
  "$AGENT/bin/pip" install -q --upgrade pip wheel "setuptools<81"
  "$AGENT/bin/pip" install -q "$REPO/packages/autods" "$REPO/apps/harbor"
  "$AGENT/bin/pip" install -q "$REPO/packages/pygrad" fastembed || true
  "$AGENT/bin/autods-harbor" --help >/dev/null 2>&1 || { echo "autods-harbor entrypoint missing" >&2; exit 1; }
}
build_family_venv() {  # $1 = family
  local fam="$1" venv="$HERE/.venv-$fam"; [ -x "$venv/bin/python" ] && return 0
  echo ">> $fam child venv (torch: $([ "${TORCH_CPU:-0}" = 1 ] && echo CPU || echo GPU))"
  "$PY" -m venv "$venv"
  "$venv/bin/pip" install -q --upgrade pip wheel "setuptools<81"
  local pk; pk="$(torch_pkgs "$fam")"
  if [ "${TORCH_CPU:-0}" = 1 ]; then "$venv/bin/pip" install $pk --index-url https://download.pytorch.org/whl/cpu
  else "$venv/bin/pip" install $pk; fi
  "$venv/bin/pip" install -q -r "$HERE/requirements/$fam.txt"
}

run_one() {  # $1=task $2=base|c1
  local task="$1" mode="$2" fam; fam="$(family_of "$task")"
  build_agent_venv; build_family_venv "$fam"
  local dsrc="$DATA_ROOT/$task"
  [ -d "$dsrc/data" ] || { echo "no data at $dsrc/data (clone mlab-gpu-fork, set DATA_ROOT)"; exit 1; }
  # instruction: base from HF dataset; c1 from this repo
  local inst
  if [ "$mode" = base ]; then inst="$dsrc/instruction_base.md"
  else inst="$HERE/tasks/$task/instruction_c1.md"; fi
  [ -f "$inst" ] || { echo "missing instruction: $inst"; exit 1; }
  local run="$HERE/runs/$task-$mode"; rm -rf "$run"; mkdir -p "$run/workspace"
  cp -R "$dsrc/data/." "$run/workspace/"
  echo ">> RUN $task/$mode (family=$fam)  $(date +%H:%M:%S)"
  local extra=(); [ "$mode" = base ] && extra=(RESEARCH_DISABLED=1 DEBUGGER_DISABLED=1) \
                                     || extra=(RESEARCH_DISABLED=1 AUTODS_SUBMISSION_GUARD=1)
  env AUTODS_CHILD_VENV="$HERE/.venv-$fam" \
      AUTODS_MODEL="$AUTODS_MODEL" AUTODS_API_KEY="$AUTODS_API_KEY" AUTODS_BASE_URL="$AUTODS_BASE_URL" \
      "${extra[@]}" \
      "$AGENT/bin/autods-harbor" --instruction-file "$inst" \
        --workspace "$run/workspace" --trace-out "$run/trace.json" 2>&1 | tee "$run/agent.log"
  echo ">> SCORE $task/$mode"
  "$HERE/.venv-$fam/bin/python" "$dsrc/score.py" \
      "$run/workspace/submission.csv" "$dsrc/data/answer.csv" | tee "$run/score.txt"
}

case "${1:-}" in
  setup) build_agent_venv; for f in tabular nlp vision; do build_family_venv "$f"; done;;
  all) for t in amp-parkinsons feedback fathomnet identify-contrails; do run_one "$t" base; run_one "$t" c1; done;;
  "") echo "usage: $0 setup | <task> <base|c1> | all"; exit 1;;
  *) run_one "$1" "${2:?mode base|c1}";;
esac
