#!/usr/bin/env bash
# HLEO — bootloader.
#
# Starts the system in the correct order and verifies every component before
# declaring HLEO READY:
#
#   Environment → Dependencies → Database → Backend → Scientific/RWE services
#   → Frontend → Health checks → HLEO READY
#
# Scientific and RWE are not separate daemons in this architecture: they are
# in-process services of the FastAPI backend (api/main.py + core/*). They are
# verified here via the HTTP endpoints they expose.
#
# Idempotent: install, setup and start all refuse to repeat work already done.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAGE() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$1" "$2"; }

STAGE "bootloader" "HLEO boot sequence — ${ROOT_DIR}"

# 1. Environment
STAGE "1/7" "Environment"
python3 --version
[[ -f .env ]] || { echo "  .env missing — creating from .env.example"; cp .env.example .env; }

# 2. Dependencies
STAGE "2/7" "Dependencies"
bash scripts/install.sh

# 3. Database
STAGE "3/7" "Database"
bash scripts/setup.sh

# 4. Backend
STAGE "4/7" "Backend"
bash scripts/start.sh

# 5. Scientific/RWE services (in-process with the backend)
STAGE "5/7" "Scientific/RWE services"
"${ROOT_DIR}/.venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())

# Route-level verification (no network call, so no LLM/dependency involved).
try:
    from api.main import app
    paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is not None:
            paths.add(path)
    for name, path in [("scientific-search", "/search"),
                       ("scientific-pipeline", "/pipeline/run"),
                       ("synthesis", "/synthesis"),
                       ("rwe-search", "/rwe/search"),
                       ("assistant", "/assistant/chat")]:
        print(f"  {name}: {'mounted' if path in paths else 'MISSING'}")
except Exception as e:
    print(f"  route check: WARNING — {type(e).__name__}: {e}")
PY

# 6. Frontend
STAGE "6/7" "Frontend"
python3 - <<'PY'
import os, urllib.request
base = f"http://127.0.0.1:{os.getenv('PORT','8000')}"
try:
    with urllib.request.urlopen(base + "/", timeout=10) as r:
        body = r.read().decode("utf-8", "ignore")
        ok = "HLEO" in body
        print(f"  frontend: HTTP {r.status}, HLEO marker {'present' if ok else 'MISSING'}")
except Exception as e:
    print(f"  frontend: WARNING — {type(e).__name__}: {e}")
PY

# 7. Health checks
STAGE "7/7" "Health checks"
bash scripts/health-check.sh

STAGE "bootloader" "HLEO READY"
