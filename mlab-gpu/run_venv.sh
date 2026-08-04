#!/usr/bin/env bash
# Docker-less runner for the full-data MLAB tasks. Lives in the AutoDS fork
# (installs the agent from this repo's packages/autods + apps/harbor) and pulls
# data + the (base) instruction + scorer straight from the HF dataset via
# --repo, exactly like `harbor run --repo …` — no manual clone needed.
#
# One instruction per task: the dataset ships the ORIGINAL (base) instruction.
# AutoDS appends its C1 specialized-library layer at RUNTIME (c1_prompt.py);
# baseline disables it with AUTODS_C1_DISABLED=1. No per-task C1 files.
#
# Runs where docker-in-docker is unavailable: plain venvs (agent-brain venv +
# per-family child venv), GPU torch by default (TORCH_CPU=1 forces CPU).
#
# Usage:
#   export AUTODS_MODEL=... AUTODS_API_KEY=... AUTODS_BASE_URL=...
#   ./run_venv.sh [--repo <hf-url>] setup
#   ./run_venv.sh [--repo <hf-url>] <task> <base|c1>     # base = AutoDS without C1
#   ./run_venv.sh [--repo <hf-url>] all
#
# --repo default: https://huggingface.co/datasets/danil-e/harbor-datasets-mlab
# Full-data tasks: amp-parkinsons feedback fathomnet identify-contrails
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO_ROOT="$(cd "$HERE/.." && pwd)"
REPO="${REPO:-https://huggingface.co/datasets/danil-e/harbor-datasets-mlab}"
[ "${1:-}" = "--repo" ] && { REPO="$2"; shift 2; }
: "${AUTODS_MODEL:?set AUTODS_MODEL}"; : "${AUTODS_API_KEY:?set AUTODS_API_KEY}"
: "${AUTODS_BASE_URL:=https://openrouter.ai/api/v1}"
PY="${PYTHON:-python3}"; AGENT="$HERE/.venv-agent"; CACHE="$HERE/.data"
REPO_ID="${REPO#https://huggingface.co/datasets/}"    # -> danil-e/harbor-datasets-mlab

family_of() { case "$1" in
  amp-parkinsons) echo tabular;; feedback) echo nlp;;
  fathomnet|identify-contrails) echo vision;; *) echo "unknown task $1" >&2; exit 1;; esac; }
torch_pkgs() { case "$1" in vision) echo "torch==2.2.2 torchvision==0.17.2";; *) echo "torch==2.2.2";; esac; }

build_agent_venv() {
  [ -x "$AGENT/bin/autods-harbor" ] && return 0
  echo ">> agent-brain venv (from $REPO_ROOT/packages/autods + apps/harbor)"
  "$PY" -m venv "$AGENT"
  "$AGENT/bin/pip" install -q --upgrade pip wheel "setuptools<81"
  "$AGENT/bin/pip" install -q "$REPO_ROOT/packages/autods" "$REPO_ROOT/apps/harbor" huggingface_hub
  "$AGENT/bin/pip" install -q "$REPO_ROOT/packages/pygrad" fastembed || true
  "$AGENT/bin/autods-harbor" --help >/dev/null 2>&1 || { echo "autods-harbor entrypoint missing" >&2; exit 1; }
}
build_family_venv() {
  local fam="$1"
  local venv="$HERE/.venv-$fam"
  [ -x "$venv/bin/python" ] && return 0
  echo ">> $fam child venv (torch: $([ "${TORCH_CPU:-0}" = 1 ] && echo CPU || echo GPU))"
  "$PY" -m venv "$venv"; "$venv/bin/pip" install -q --upgrade pip wheel "setuptools<81"
  local pk; pk="$(torch_pkgs "$fam")"
  if [ "${TORCH_CPU:-0}" = 1 ]; then "$venv/bin/pip" install $pk --index-url https://download.pytorch.org/whl/cpu
  else "$venv/bin/pip" install $pk; fi
  "$venv/bin/pip" install -q -r "$HERE/requirements/$fam.txt"
}

pull_task() {  # $1=task -> downloads only that task's folder from HF, echoes its local path
  local task="$1"
  "$AGENT/bin/python" - "$REPO_ID" "$task" "$CACHE" <<'PY' >/dev/null
import sys
from huggingface_hub import snapshot_download
repo_id, task, cache = sys.argv[1:4]
snapshot_download(repo_id, repo_type="dataset",
                  allow_patterns=[f"datasets/mlab-real/{task}/**"], local_dir=cache)
PY
  echo "$CACHE/datasets/mlab-real/$task"
}

run_one() {  # $1=task $2=base|c1
  local task="$1" mode="$2" fam; fam="$(family_of "$task")"
  build_agent_venv; build_family_venv "$fam"
  echo ">> PULL $task from $REPO_ID"
  local td; td="$(pull_task "$task")"
  [ -d "$td/environment/data" ] || { echo "no data at $td/environment/data (check --repo / task name)"; exit 1; }
  local inst="$td/instruction.md"                      # base instruction for BOTH modes
  local run="$HERE/runs/$task-$mode"; rm -rf "$run"; mkdir -p "$run/workspace"
  cp -R "$td/environment/data/." "$run/workspace/"
  echo ">> RUN $task/$mode (family=$fam)  $(date +%H:%M:%S)"
  # base = AutoDS without C1 (AUTODS_C1_DISABLED=1); c1 = AutoDS appends its C1 layer at runtime
  local extra=(RESEARCH_DISABLED=1)
  [ "$mode" = base ] && extra+=(DEBUGGER_DISABLED=1 AUTODS_C1_DISABLED=1) || extra+=(AUTODS_SUBMISSION_GUARD=1)
  env AUTODS_CHILD_VENV="$HERE/.venv-$fam" \
      AUTODS_MODEL="$AUTODS_MODEL" AUTODS_API_KEY="$AUTODS_API_KEY" AUTODS_BASE_URL="$AUTODS_BASE_URL" \
      "${extra[@]}" \
      "$AGENT/bin/autods-harbor" --instruction-file "$inst" \
        --workspace "$run/workspace" --trace-out "$run/trace.json" 2>&1 | tee "$run/agent.log"
  echo ">> SCORE $task/$mode"
  "$HERE/.venv-$fam/bin/python" "$td/tests/score.py" \
      --submission "$run/workspace/submission.csv" --answer "$td/environment/data/answer.csv" \
      --reward-json "$run/reward.json" | tee "$run/score.txt"
}

case "${1:-}" in
  setup) build_agent_venv; for f in tabular nlp vision; do build_family_venv "$f"; done;;
  all) for t in amp-parkinsons feedback fathomnet identify-contrails; do run_one "$t" base; run_one "$t" c1; done;;
  "") echo "usage: $0 [--repo <hf-url>] setup | <task> <base|c1> | all"; exit 1;;
  *) run_one "$1" "${2:?mode base|c1}";;
esac
