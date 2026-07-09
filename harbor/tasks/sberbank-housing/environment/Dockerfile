# TabReD sberbank-housing task environment (tabular family: LightAutoML).
# Build the family image first: harbor/environments/build.sh tabular
FROM autods-mlab-tabular:latest
COPY prepare.py /opt/mlab/prepare.py
RUN /opt/venvs/tabular/bin/python /opt/mlab/prepare.py
WORKDIR /workspace
