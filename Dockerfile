FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ──────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 argus && \
    useradd --uid 1001 --gid argus --shell /bin/bash --create-home argus

WORKDIR /app

# ── Install dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY --chown=argus:argus . .

# ── Switch to non-root ─────────────────────────────────────────────────────────
USER argus

EXPOSE 8001

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ── Start with configured worker count ────────────────────────────────────────
CMD ["sh", "-c", "uvicorn argus_core.main:app --host 0.0.0.0 --port 8001 --workers ${ARGUS_WORKERS:-2}"]
