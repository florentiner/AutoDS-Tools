# autods-mlab-tabred: LightAutoML + tabular deep-learning (RTDL / TabM family).
#
# Serves the 8 TabReD temporal tabular tasks. Specialized libraries live in a
# dedicated child venv (/opt/venvs/tabred), isolated from the AutoDS framework.
#
# Build from the repository root (after autods-mlab-base):
#   docker build -f harbor/environments/tabred/Dockerfile -t autods-mlab-tabred:latest .
FROM autods-mlab-base:latest

ENV AUTODS_CHILD_VENV=/opt/venvs/tabred

COPY harbor/environments/tabred/requirements.txt /tmp/tabred-requirements.txt
# CPU-only torch (see tabular/Dockerfile) + setuptools<81 (keeps pkg_resources,
# which LightAutoML still imports).
RUN python -m venv /opt/venvs/tabred \
    && /opt/venvs/tabred/bin/pip install --upgrade pip wheel "setuptools<81" \
    && ( /opt/venvs/tabred/bin/pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu \
         || /opt/venvs/tabred/bin/pip install "torch==2.2.2" ) \
    && /opt/venvs/tabred/bin/pip install -r /tmp/tabred-requirements.txt

RUN /opt/venvs/tabred/bin/python -c "import lightautoml, featuretools, sklearn, pandas, numpy, torch; print('tabred env OK; numpy', numpy.__version__)"

WORKDIR /workspace
