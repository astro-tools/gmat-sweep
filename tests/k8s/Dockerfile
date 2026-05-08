ARG GMAT_IMAGE_TAG=R2026a
FROM ghcr.io/astro-tools/gmat:${GMAT_IMAGE_TAG}

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY tests /workspace/tests

RUN python3.12 -m pip install --no-cache-dir --upgrade pip \
    && python3.12 -m pip install --no-cache-dir -e ".[k8s]"

ENV PYTHONUNBUFFERED=1
