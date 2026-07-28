FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    TRANSFORMERS_CACHE=/data/hf/hub \
    SOULX_PRETRAINED_ROOT=/data/pretrained_models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libsndfile1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/docker/start.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/start.sh"]
