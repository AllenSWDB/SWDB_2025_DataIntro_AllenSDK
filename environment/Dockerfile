# hash:sha256:9bc4634465cdddd77dddb1f99d9179ff132cf0a0607e21c368a4f776aa0b4e24
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    matplotlib==3.9.0 \
    numpy==1.23.5 \
    scikit-learn==1.6.1
