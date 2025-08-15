# hash:sha256:451642610c51c19731153b3cc59556c0c0da297bd3c6dd68d1b3582cdd775daa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    matplotlib==3.9.0 \
    numpy==1.23.5
