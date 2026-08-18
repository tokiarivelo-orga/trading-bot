# syntax=docker/dockerfile:1
#
# Production image for the FastAPI backend. Multi-stage: `builder` resolves
# and installs dependencies with uv into a venv; `final` is a slim runtime
# that copies only the built venv + application source — no uv, no C
# toolchain, no dev/test files. Runs as a non-root user.
#
# `.env` and `configs/` are NOT baked in here — they stay compose-level
# concerns (env_file + volume mounts), see docker-compose.prod.yml and
# the dev docker-compose.yml.
#
# Build:  docker build -f backend/Dockerfile -t tradingbot/backend:local backend
# Run:    see docker-compose.prod.yml (pulls a published image instead)

# ─── Builder ──────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install dependencies first, in their own layer, so editing application
# code doesn't invalidate the (slow, torch-containing) dependency install.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Now bring in the actual application source and re-sync (no-op for already
# installed deps; picks up any project-metadata-driven changes).
COPY . .
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# ─── Final ────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS final

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Only the built venv + source the app actually needs at runtime — no uv
# binary, no build cache, no tests/, no scripts/, no local .venv/data/.
COPY --from=builder --chown=appuser:appuser /app/.venv ./.venv
COPY --from=builder --chown=appuser:appuser /app/src ./src
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/uv.lock* ./
COPY --from=builder --chown=appuser:appuser /app/alembic.ini ./alembic.ini
COPY --from=builder --chown=appuser:appuser /app/migrations ./migrations

# Mount points for compose-level volumes (configs read-only, data read-write)
# — created + owned up front so the non-root user can use them even when the
# image is run standalone without those mounts.
RUN mkdir -p /app/configs /app/data && chown appuser:appuser /app/configs /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# GET /health is the unauthenticated liveness probe (src/main.py) —
# independent of gateway/DB connectivity by design.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.main:socket_app", "--host", "0.0.0.0", "--port", "8000"]
