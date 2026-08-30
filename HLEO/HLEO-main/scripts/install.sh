#!/usr/bin/env bash
# HLEO — install dependencies into a local virtualenv.
# Idempotent: safe to run repeatedly; skips work already done.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

# 1. Environment: pick a usable Python (3.12+; CI targets 3.13).
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[install] ERROR: ${PYTHON_BIN} not found on PATH." >&2
    exit 1
fi

# 2. Create the virtualenv once.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[install] Creating virtualenv at ${VENV_DIR} ..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# 3. Upgrade the installer tooling once per venv (best-effort, not fatal).
"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true

# 4. Install pinned dependencies (idempotent: pip is a no-op when already met).
echo "[install] Installing requirements.txt ..."
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo "[install] OK — virtualenv ready at ${VENV_DIR}"
