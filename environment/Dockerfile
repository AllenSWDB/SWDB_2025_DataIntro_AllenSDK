# hash:sha256:44f646a689097b78db9d66c60de5c50c466ebd980a924b0331254bcfb437bfd7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    matplotlib==3.9.0 \
    seaborn==0.13.2
