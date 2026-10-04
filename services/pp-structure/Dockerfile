# syntax=docker/dockerfile:1
ARG VARIANT=gpu
FROM ghcr.io/astral-sh/uv:0.12.20 AS uv

FROM python:3.13-slim AS base
ARG VARIANT
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,id=pp-structure-uv,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra "$VARIANT"
COPY pp_structure ./pp_structure
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --no-dev --extra "$VARIANT"
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLE_PDX_CACHE_HOME=/models
RUN /app/.venv/bin/python -c "from paddleocr import PPStructureV3; from pp_structure.app import DISABLED_STAGES, model_options; PPStructureV3(device='cpu', enable_mkldnn=False, **model_options('cpu' if '$VARIANT' == 'cpu' else 'gpu:0'), **DISABLED_STAGES)" \
    && chmod -R a+rX /models
EXPOSE 8012
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8012/health', timeout=4)" || exit 1
CMD ["/app/.venv/bin/pp-structure", "--host", "0.0.0.0", "--port", "8012"]
