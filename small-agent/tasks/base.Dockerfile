FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm rustc cargo gcc g++ make git curl jq \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest
WORKDIR /workspace
