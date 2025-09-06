#!/usr/bin/env sh
set -eu

# Install Guardrails Hub validators
echo "[prestart] Installing Guardrails Hub validators..."
guardrails hub install hub://tryolabs/restricttotopic --quiet || echo "[prestart] restricttotopic already installed"
guardrails hub install hub://guardrails/detect_pii --quiet || echo "[prestart] detect_pii already installed"

# Only run if enabled and alembic.ini exists
RUN_DB_MIGRATIONS="${RUN_DB_MIGRATIONS:-true}"
if [ "$RUN_DB_MIGRATIONS" = "true" ] && [ -f "alembic.ini" ]; then
  echo "[prestart] Running Alembic migrations..."
  # Use AWS_DB_URL if provided; alembic.ini can reference it via %(AWS_DB_URL)s
  if command -v alembic >/dev/null 2>&1; then
    alembic upgrade head || {
      echo "[prestart] WARNING: Alembic migration failed; proceeding anyway" >&2
    }
  else
    echo "[prestart] Alembic not installed; skipping migrations"
  fi
else
  echo "[prestart] Skipping migrations (RUN_DB_MIGRATIONS=$RUN_DB_MIGRATIONS, alembic.ini present=$( [ -f alembic.ini ] && echo yes || echo no ))"
fi