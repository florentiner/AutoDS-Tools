# autods-mlab-vision: timm + torchvision + albumentations + segmentation-models (CPU).
# Serves the image MLAB tasks (cifar10, fathomnet, identify-contrails). Keep models
# small — CPU only.
#
# Build from the repository root (after autods-mlab-base):
#   docker build -f harbor/environments/vision/Dockerfile -t autods-mlab-vision:latest .
FROM autods-mlab-base:latest

ENV AUTODS_CHILD_VENV=/opt/venvs/vision

COPY harbor/environments/vision/requirements.txt /tmp/vision-requirements.txt
# CPU torch + matching torchvision (0.17.2 pairs with torch 2.2.2); setuptools<81.
RUN python -m venv /opt/venvs/vision \
    && /opt/venvs/vision/bin/pip install --upgrade pip wheel "setuptools<81" \
    && ( /opt/venvs/vision/bin/pip install "torch==2.2.2" "torchvision==0.17.2" --index-url https://download.pytorch.org/whl/cpu \
         || /opt/venvs/vision/bin/pip install "torch==2.2.2" "torchvision==0.17.2" ) \
    && /opt/venvs/vision/bin/pip install -r /tmp/vision-requirements.txt

RUN /opt/venvs/vision/bin/python -c "import torch, torchvision, timm, albumentations, cv2, sklearn; print('vision env OK; torch', torch.__version__, 'tv', torchvision.__version__)"

WORKDIR /workspace
