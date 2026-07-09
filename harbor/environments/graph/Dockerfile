# autods-mlab-graph: torch-geometric (PyG) + ogb (CPU). Serves ogbn-arxiv.
# Build from the repository root (after autods-mlab-base):
#   docker build -f harbor/environments/graph/Dockerfile -t autods-mlab-graph:latest .
FROM autods-mlab-base:latest

ENV AUTODS_CHILD_VENV=/opt/venvs/graph

COPY harbor/environments/graph/requirements.txt /tmp/graph-requirements.txt
RUN python -m venv /opt/venvs/graph \
    && /opt/venvs/graph/bin/pip install --upgrade pip wheel "setuptools<81" \
    && ( /opt/venvs/graph/bin/pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu \
         || /opt/venvs/graph/bin/pip install "torch==2.2.2" ) \
    && /opt/venvs/graph/bin/pip install -r /tmp/graph-requirements.txt

RUN /opt/venvs/graph/bin/python -c "import torch, torch_geometric, ogb, sklearn; print('graph env OK; torch', torch.__version__, 'pyg', torch_geometric.__version__)"

WORKDIR /workspace
