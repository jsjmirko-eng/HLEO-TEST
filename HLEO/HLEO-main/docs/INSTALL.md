# HLEO — Installation

Two supported paths: **local** (Python venv) and **Docker**.

## Prerequisites

- Python 3.12 or 3.13 (3.13 is what CI validates)
- pip
- Docker + Docker Compose (only for the Docker path)
- PostgreSQL 15 (only if you choose PostgreSQL instead of SQLite)

## Path A — Local (recommended for a new PC)

```bash
# 1. Install dependencies into .venv (idempotent)
bash scripts/install.sh

# 2. Create .env from the template and initialise the database (SQLite by default)
bash scripts/setup.sh

# 3. Start the API in the background
bash scripts/start.sh

# 4. Verify
bash scripts/health-check.sh
```

The default `scripts/setup.sh` uses **SQLite** (`DATABASE_URL=sqlite:///./hleo.db`)
so no PostgreSQL service is required on a new PC.

## Path B — Docker

```bash
# Optional: copy the env template and fill in secrets
cp .env.example .env

docker compose up --build -d
```

`docker-compose.yml` starts:

- `db`  — postgres:15-alpine (health-checked)
- `api` — the HLEO API on http://localhost:8000

## Full boot sequence

```bash
bash scripts/bootloader.sh
```

This runs Environment → Dependencies → Database → Backend → Scientific/RWE
services → Frontend → Health checks, and prints `HLEO READY` only when every
component passed.

## Notes

- `scripts/install.sh`, `scripts/setup.sh`, `scripts/start.sh` and
  `scripts/stop.sh` are all idempotent — safe to run repeatedly.
- The obsolete backup archives previously committed to the repository
  (`HLEO-ultima-versione.zip`, `zipFile.zip`) were removed and are excluded by
  `.gitignore`. They are **not** part of the installation.
