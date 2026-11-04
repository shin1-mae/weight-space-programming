# Reproducible CPU-only environment for the weight-space programming experiments.
#   docker compose run --rm wsp all          # all experiments, offline scripted agent
#   docker compose run --rm tests            # test suite
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN pip install "uv>=0.5"

WORKDIR /app

# CPU wheels of PyTorch keep the image small (the PyPI default wheel bundles CUDA libraries).
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

COPY pyproject.toml README.md ./
COPY src ./src
# Editable install so that results/ and artifacts/ resolve to /app (mounted as volumes).
RUN uv pip install -e ".[llm]" "pytest>=8.3"

COPY tests ./tests
COPY docs ./docs

# Run as an unprivileged user; the sandbox subprocesses inherit it.
RUN useradd --create-home --uid 1000 wsp && mkdir -p results artifacts && chown -R wsp /app
USER wsp

ENTRYPOINT ["wsp"]
CMD ["all"]
