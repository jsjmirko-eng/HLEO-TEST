#!/usr/bin/env bash
# HLEO — start the API in the background and write a pidfile.
# Idempotent: refuses to start a second instance while one is already running.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PIDFILE="${ROOT_DIR}/.hleo/hleo.pid"
LOGFILE="${ROOT_DIR}/.hleo/hleo.log"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

cd "${ROOT_DIR}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PY="${VENV_DIR}/bin/python"
else
    echo "[start] ERROR: virtualenv missing. Run scripts/install.sh first." >&2
    exit 1
fi

mkdir -p "${ROOT_DIR}/.hleo"

if [[ -f "${PIDFILE}" ]]; then
    OLD_PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[start] Already running (pid ${OLD_PID}). Use scripts/stop.sh to stop it."
        exit 0
    fi
    rm -f "${PIDFILE}"
fi

echo "[start] Starting HLEO on ${HOST}:${PORT} (log: ${LOGFILE}) ..."
nohup "${PY}" -m uvicorn api.main:app --host "${HOST}" --port "${PORT}" \
    > "${LOGFILE}" 2>&1 &
echo $! > "${PIDFILE}"
echo "[start] Started with pid $(cat "${PIDFILE}")"
