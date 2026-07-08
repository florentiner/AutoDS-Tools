#!/bin/bash
# Harbor verifier: score /workspace/submission.csv against the held-out answer
# key and write the reward. Runs in the task container (FROM autods-mlab-tabular),
# so the tabular child venv's Python (with pandas) is available.
set -uo pipefail

mkdir -p /logs/verifier
PY=/opt/venvs/tabular/bin/python
[ -x "$PY" ] || PY=python

"$PY" /tests/score.py \
  --submission /workspace/submission.csv \
  --answer /opt/mlab/answer.csv \
  --reward-json /logs/verifier/reward.json

# Reward reflects submission quality; always exit 0 so Harbor reads the reward.
exit 0
