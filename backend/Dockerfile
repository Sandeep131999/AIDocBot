# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
#   Installs all Python dependencies into an isolated venv.
#   Uses a full Python image so compilation tools are available.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Prevent interactive prompts, set sane locale
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System build deps (needed for psycopg, cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt installs the project itself through its -e . entry.
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY requirements.txt ./

# Create a venv and install into it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
#   Slim image, copies only the venv from builder + application code.
#   Runs as non-root user for security.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Tell uvicorn / FastAPI where to find the app
    APP_MODULE="api.main:app"

# Runtime system deps only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Create writable directories (volumes will overlay these in compose)
RUN mkdir -p /app/uploads /app/logs /app/model_cache && \
    chown -R appuser:appgroup /app/uploads /app/logs /app/model_cache

# Switch to non-root
USER appuser

EXPOSE 8000

# Health check — hits the /api/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Entrypoint: production uses multiple workers; dev uses reload
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--log-level", "info"]
