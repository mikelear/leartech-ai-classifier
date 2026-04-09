# Python service with UV — follows mqube pattern
# Non-root user, healthcheck, uv run entrypoint

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root user (UID 1000) — same pattern as auth-service/soc-collector
RUN groupadd -r classifier && useradd -r -g classifier -u 1000 classifier

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY app/ app/
COPY models/ models/

# Install dependencies
ENV UV_FROZEN=true
RUN uv sync --frozen --no-cache --no-dev 2>/dev/null || uv sync --no-cache --no-dev

ENV PORT=8080
ENV UV_CACHE_DIR=/tmp/uv-cache

# Create home dir for non-root user
RUN mkdir -p /home/classifier && chown classifier:classifier /home/classifier

USER classifier
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
