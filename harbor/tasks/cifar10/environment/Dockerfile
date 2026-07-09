# CIFAR-10 task environment (vision family: timm + torchvision).
FROM autods-mlab-vision:latest
COPY prepare.py /opt/mlab/prepare.py
RUN /opt/venvs/vision/bin/python /opt/mlab/prepare.py
WORKDIR /workspace
