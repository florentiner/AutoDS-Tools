# autods-mlab-nlp: HuggingFace transformers + sentence-transformers (CPU).
# Serves the text MLAB tasks (imdb, feedback, ...). Specialized libs live in a
# dedicated child venv (/opt/venvs/nlp), isolated from the AutoDS framework.
#
# Build from the repository root (after autods-mlab-base):
#   docker build -f harbor/environments/nlp/Dockerfile -t autods-mlab-nlp:latest .
FROM autods-mlab-base:latest

ENV AUTODS_CHILD_VENV=/opt/venvs/nlp

COPY harbor/environments/nlp/requirements.txt /tmp/nlp-requirements.txt
# CPU-only torch first (see tabular/Dockerfile) + setuptools<81.
RUN python -m venv /opt/venvs/nlp \
    && /opt/venvs/nlp/bin/pip install --upgrade pip wheel "setuptools<81" \
    && ( /opt/venvs/nlp/bin/pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu \
         || /opt/venvs/nlp/bin/pip install "torch==2.2.2" ) \
    && /opt/venvs/nlp/bin/pip install -r /tmp/nlp-requirements.txt

RUN /opt/venvs/nlp/bin/python -c "import transformers, sentence_transformers, sklearn, pandas, numpy, torch; print('nlp env OK; torch', torch.__version__)"

WORKDIR /workspace
