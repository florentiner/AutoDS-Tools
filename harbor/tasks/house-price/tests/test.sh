#!/bin/bash
# Harbor verifier: score /workspace/submission.csv (R2) and write the reward.
set -uo pipefail

mkdir -p /logs/verifier
PY=/opt/venvs/tabular/bin/python
[ -x "$PY" ] || PY=python

"$PY" /tests/score.py \
  --submission /workspace/submission.csv \
  --answer /opt/mlab/answer.csv \
  --reward-json /logs/verifier/reward.json

exit 0
