#!/usr/bin/env bash
# HLEO — health check wrapper: resolve the right interpreter and run
# scripts/health_check.py.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PY="${VENV_DIR}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "[health-check] ERROR: no Python interpreter found." >&2
    exit 1
fi

exec "${PY}" "${ROOT_DIR}/scripts/health_check.py"
