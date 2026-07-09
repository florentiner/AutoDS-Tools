# IMDb task environment (nlp family: transformers + sentence-transformers).
# Build the family image first: harbor/environments/build.sh nlp
FROM autods-mlab-nlp:latest
COPY prepare.py /opt/mlab/prepare.py
RUN /opt/venvs/nlp/bin/python /opt/mlab/prepare.py
WORKDIR /workspace
