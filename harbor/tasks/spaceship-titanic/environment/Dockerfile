# Spaceship-Titanic task environment.
# FROM the tabular family image (LightAutoML + feature-engineering libs).
# Build the family images first: harbor/environments/build.sh tabular
FROM autods-mlab-tabular:latest

# Stage the dataset into /workspace and the held-out answer key into /opt/mlab
# (outside the agent's workspace). Uses the tabular child venv's Python (pandas).
COPY prepare.py /opt/mlab/prepare.py
COPY task_descriptor.txt /opt/mlab/task_descriptor.txt
RUN /opt/venvs/tabular/bin/python /opt/mlab/prepare.py

WORKDIR /workspace
