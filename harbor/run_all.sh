#!/usr/bin/env bash
# Run every AutoDS×MLAB/TabReD task through Harbor with the autods-harbor agent,
# then optionally upload results to the hub. Data for each task is prepared /
# downloaded automatically by that task's environment/prepare.py at build time
# (real Kaggle/TabReD data when MLAB_USE_KAGGLE=1 + creds, else synthetic).
#
# Requirements: Docker daemon running, `uv` installed.
#
# Required env (the OpenAI-compatible LLM AutoDS calls):
#   AUTODS_MODEL      e.g. gemma-4-31b-it
#   AUTODS_API_KEY    e.g. sk-or-v1-...        (an LLM key, NOT the sk-harbor key)
#   AUTODS_BASE_URL   e.g. https://openrouter.ai/api/v1
#
# Optional env:
#   TASKS="spaceship-titanic house-price"   subset (default: all task dirs)
#   JOBS_DIR                                 default: $HOME/harbor-jobs
#                                            (MUST be on a Docker-shared path;
#                                            with colima/Docker Desktop use $HOME)
#   RESEARCH_DISABLED=1                      skip the Researcher stage (faster)
#   UPLOAD=1 + HARBOR_API_KEY=sk-harbor-...  upload each job to the hub for a URL
#   MLAB_USE_KAGGLE=1 + KAGGLE_USERNAME/KAGGLE_KEY   use real competition data
#
# Example:
#   AUTODS_MODEL=gemma-4-31b-it \
#   AUTODS_API_KEY=sk-or-v1-... \
#   AUTODS_BASE_URL=https://openrouter.ai/api/v1 \
#   UPLOAD=1 HARBOR_API_KEY=sk-harbor-... \
#   bash harbor/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${AUTODS_MODEL:?set AUTODS_MODEL (the LLM model name)}"
: "${AUTODS_API_KEY:?set AUTODS_API_KEY (an OpenAI-compatible LLM key)}"
: "${AUTODS_BASE_URL:?set AUTODS_BASE_URL (the LLM endpoint)}"

JOBS_DIR="${JOBS_DIR:-$HOME/harbor-jobs}"
AGENT="autods_harbor.agent:AutoDSAgent"

# Harbor CLI + the autods-harbor agent (injected into the harbor tool env).
if ! command -v harbor >/dev/null 2>&1; then
  echo ">> installing harbor + autods-harbor agent"
  uv tool install harbor --with "$ROOT/apps/harbor"
fi

# Build the base image + every family image referenced by the present tasks.
families="$(for d in harbor/tasks/*/; do
  [ -f "$d/task.toml" ] && sed -nE 's/^[[:space:]]*family[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$d/task.toml"
done | sort -u | tr '\n' ' ')"
echo ">> building images: base ${families:-tabular}"
bash harbor/environments/build.sh ${families:-tabular}

mkdir -p "$JOBS_DIR"
TASKS="${TASKS:-$(ls -1 harbor/tasks | grep -v '^_')}"

for t in $TASKS; do
  tdir="harbor/tasks/$t"
  [ -f "$tdir/task.toml" ] || { echo "!! skip $t (no task.toml)"; continue; }
  echo ">> running $t"
  harbor run \
    -p "$tdir" \
    -a "$AGENT" \
    -m "$AUTODS_MODEL" \
    --ae AUTODS_MODEL="$AUTODS_MODEL" \
    --ae AUTODS_API_KEY="$AUTODS_API_KEY" \
    --ae AUTODS_BASE_URL="$AUTODS_BASE_URL" \
    --ae RESEARCH_DISABLED="${RESEARCH_DISABLED:-1}" \
    ${MLAB_USE_KAGGLE:+--ek MLAB_USE_KAGGLE="$MLAB_USE_KAGGLE"} \
    -o "$JOBS_DIR" -n 1 -y || echo "!! $t run failed"
done

if [ "${UPLOAD:-0}" = "1" ]; then
  : "${HARBOR_API_KEY:?set HARBOR_API_KEY (sk-harbor-...) to upload}"
  echo ">> uploading jobs to the hub"
  for job in "$JOBS_DIR"/*/; do
    [ -f "$job/result.json" ] && harbor upload "$job" || true
  done
fi

echo ">> done. Jobs in: $JOBS_DIR"
