# <task> environment. FROM the family image chosen in task.toml [metadata].family.
# Build the family image first:  harbor/environments/build.sh tabular
FROM autods-mlab-tabular:latest

# Stage data into /workspace (agent-visible) and the answer key into /opt/mlab
# (verifier-only). Runs with the family child venv's Python (has pandas/numpy).
COPY prepare.py /opt/mlab/prepare.py
RUN /opt/venvs/tabular/bin/python /opt/mlab/prepare.py

WORKDIR /workspace
