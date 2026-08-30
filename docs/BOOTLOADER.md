# HLEO — Bootloader

`scripts/bootloader.sh` starts the system in the correct order and verifies
every component before declaring HLEO ready.

## Sequence

```
Environment        → Python version check, .env creation if missing
Dependencies       → scripts/install.sh (virtualenv + pinned requirements)
Database           → scripts/setup.sh (create .env, create_all schema)
Backend            → scripts/start.sh (uvicorn in background, pidfile)
Scientific/RWE     → HTTP reachability of /search and /rwe/search
Frontend           → HTTP reachability of / with HLEO marker
Health checks      → scripts/health-check.sh (PASS / FAIL / NOT CONFIGURED)
HLEO READY
```

> Scientific and RWE are **not** separate daemons in this architecture. They
> are in-process services of the FastAPI backend (`api/main.py` and
> `core/relational_search.py`, `core/rwe/*`). The bootloader verifies them
> through the HTTP endpoints they expose, which is the real readiness signal.

## Idempotency

Every stage is idempotent:

- `install.sh` reuses an existing `.venv` and skips already-installed packages.
- `setup.sh` only creates `.env` if absent; `create_all()` is a no-op when the
  schema already exists.
- `start.sh` refuses to start a second instance while the pidfile process is
  alive.
- `stop.sh` is safe when nothing is running.

## Usage

```bash
bash scripts/bootloader.sh
```

Exit code `0` means every mandatory component passed (`HLEO READY`). A `FAIL`
in the final health check produces a non-zero exit code.
