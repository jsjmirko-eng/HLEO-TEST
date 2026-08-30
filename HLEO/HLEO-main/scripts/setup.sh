#!/usr/bin/env bash
# HLEO — configure the environment and initialise the database.
# Idempotent: creating .env and running create_all() are both safe to repeat.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

# 1. Ensure a Python interpreter exists (install.sh first, then fallback).
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PY="${VENV_DIR}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "[setup] ERROR: no Python interpreter found. Run scripts/install.sh first." >&2
    exit 1
fi

# 2. Create .env from the template only if it does not already exist.
if [[ ! -f ".env" ]]; then
    echo "[setup] Creating .env from .env.example ..."
    cp .env.example .env
fi

# 2b. Local default: ensure DATABASE_URL is present so the backend does not
#     fall back to PostgreSQL (localhost:5432) on a fresh machine. Never
#     overwrites an existing value.
if ! grep -qE '^DATABASE_URL=' .env; then
    echo "[setup] Adding local DATABASE_URL (SQLite) to .env ..."
    printf '\nDATABASE_URL=sqlite:///./hleo.db\n' >> .env
fi

# 3. Runtime artifacts directory (pidfile/logs).
mkdir -p "${ROOT_DIR}/.hleo"

# 4. Initialise the database schema (idempotent).
#    DATABASE_URL defaults to SQLite via core/database.py when unset;
#    honour .env by loading it through python-dotenv.
echo "[setup] Initialising database schema ..."
"${PY}" - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")

# Local default: SQLite so setup works with no external Postgres service.
os.environ.setdefault("DATABASE_URL", "sqlite:///./hleo.db")

from core.database import Base, engine
import core.models  # noqa: F401  (register all mapped models)

Base.metadata.create_all(bind=engine)
print("[setup] Database ready:", engine.url)
PY

echo "[setup] OK"
