# syntax=docker/dockerfile:1

# ---------- base ----------
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (optional minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- prod ----------
FROM base AS prod
COPY src ./src
COPY langgraph.json ./langgraph.json
CMD ["python", "-m", "src.graph.runtime"]

# ---------- dev ----------
FROM base AS dev
RUN pip install --no-cache-dir uvicorn[standard] watchdog alembic
COPY scripts/prestart.sh ./scripts/prestart.sh
RUN chmod +x ./scripts/prestart.sh
# Code will be mounted in dev via compose
CMD ["/bin/sh", "-c", "./scripts/prestart.sh && uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload"]

# ---------- test ----------
FROM base AS test
RUN pip install --no-cache-dir alembic
COPY src ./src
COPY tests ./tests
COPY scripts/prestart.sh ./scripts/prestart.sh
RUN chmod +x ./scripts/prestart.sh
ENV PYTEST_ADDOPTS="-q"
CMD ["/bin/sh", "-c", "./scripts/prestart.sh && pytest"] 