# hash:sha256:b82b77e413cfa8ed58ae2eae593d450a38430f1f366e853f26469da9b4178694
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    matplotlib==3.9.0 \
    numpy==1.23.5 \
    scikit-learn
