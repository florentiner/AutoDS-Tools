# autods-mlab-tabular: LightAutoML + temporal feature engineering.
#
# Serves the tabular MLAB tasks (spaceship-titanic, house-price, amp-parkinsons).
# The specialized libraries live in a DEDICATED child venv (/opt/venvs/tabular),
# kept isolated from the AutoDS framework interpreter to avoid dependency-pin
# conflicts. AutoDS runs task code in that venv via AUTODS_CHILD_VENV, and the
# Coder can `pip install` alternatives (sktime, statsforecast, Darts, AutoGluon)
# into it at task time.
#
# Build from the repository root (after autods-mlab-base):
#   docker build -f harbor/environments/tabular/Dockerfile -t autods-mlab-tabular:latest .
FROM autods-mlab-base:latest

ENV AUTODS_CHILD_VENV=/opt/venvs/tabular

COPY harbor/environments/tabular/requirements.txt /tmp/tabular-requirements.txt
# CPU-only torch first so LightAutoML does not pull the multi-GB CUDA build.
# The CPU wheel index only serves linux/amd64; on linux/arm64 the plain PyPI
# torch==2.2.2 wheel is already CPU-only, so fall back to it.
RUN python -m venv /opt/venvs/tabular \
    # setuptools<81 keeps pkg_resources, which LightAutoML still imports.
    && /opt/venvs/tabular/bin/pip install --upgrade pip wheel "setuptools<81" \
    && ( /opt/venvs/tabular/bin/pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu \
         || /opt/venvs/tabular/bin/pip install "torch==2.2.2" ) \
    && /opt/venvs/tabular/bin/pip install -r /tmp/tabular-requirements.txt

# Fail early if the primary specialized libraries do not import.
RUN /opt/venvs/tabular/bin/python -c "import lightautoml, featuretools, feature_engine, sklearn, pandas, numpy; print('tabular env OK; numpy', numpy.__version__)"

WORKDIR /workspace
