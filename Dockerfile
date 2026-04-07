# Multi-stage build: UV builder → slim runtime
# Follows leartech patterns: non-root user, healthcheck, minimal image

# Stage 1: Build with UV
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cache-friendly)
COPY pyproject.toml ./
RUN uv sync --frozen --no-cache --no-dev 2>/dev/null || uv sync --no-cache --no-dev

# Copy application code and model
COPY app/ app/
COPY models/ models/

# Stage 2: Slim runtime
FROM python:3.12-slim

# Non-root user (UID 1000) — same pattern as auth-service/soc-collector
RUN groupadd -r classifier && useradd -r -g classifier -u 1000 classifier

WORKDIR /app

# Copy virtualenv and app from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/models /app/models

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8080

USER classifier
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
