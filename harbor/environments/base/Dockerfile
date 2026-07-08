# autods-mlab-base: the AutoDS agent framework, shared by every family image.
#
# Contains the AutoDS pipeline (langchain/langgraph) + the `autods-harbor`
# entrypoint, but NOT the heavy pygrad/cognee doc-retrieval stack (libq is
# disabled). Family images (tabular/nlp/vision/graph/clrs/numeric) extend this
# and add their specialized libraries into the same system Python, which the
# per-task `.venv` inherits via AUTODS_VENV_SYSTEM_SITE_PACKAGES=1.
#
# Build from the repository root:
#   docker build -f harbor/environments/base/Dockerfile -t autods-mlab-base:latest .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Harbor images run AutoDS without the optional GRAD/libq stack.
    LIBQ_DISABLED=1 \
    # The per-project venv inherits the image's baked specialized libraries.
    AUTODS_VENV_SYSTEM_SITE_PACKAGES=1

# ripgrep powers AutoDS's file-search middleware; build tooling for sdist wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ripgrep \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the AutoDS framework + Harbor entrypoint into the system interpreter.
# pip (not uv) is used so [tool.uv.sources] workspace mappings are ignored and
# the optional `libq`/pygrad extra is never pulled.
COPY packages/autods /opt/autods-src/packages/autods
COPY apps/harbor /opt/autods-src/apps/harbor
RUN pip install \
        /opt/autods-src/packages/autods \
        /opt/autods-src/apps/harbor

# Fail the build early if the entrypoint is not wired up.
RUN autods-harbor --help >/dev/null

WORKDIR /workspace
