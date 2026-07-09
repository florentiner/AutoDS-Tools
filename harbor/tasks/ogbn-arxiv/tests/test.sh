#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
PY=/opt/venvs/graph/bin/python; [ -x "$PY" ] || PY=python
"$PY" /tests/score.py --submission /workspace/submission.csv --answer /opt/mlab/answer.csv --reward-json /logs/verifier/reward.json
exit 0
