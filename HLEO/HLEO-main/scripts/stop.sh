#!/usr/bin/env bash
# HLEO — stop the background API process recorded in the pidfile.
# Idempotent: safe to run when nothing is running.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${ROOT_DIR}/.hleo/hleo.pid"

if [[ ! -f "${PIDFILE}" ]]; then
    echo "[stop] No pidfile found — nothing to stop."
    exit 0
fi

PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
    echo "[stop] Process not running; removing stale pidfile."
    rm -f "${PIDFILE}"
    exit 0
fi

# A zombie process still answers kill -0 but is already dead; only the parent
# can reap it. Treat zombies as stopped and remove the stale pidfile.
if [[ "$(ps -p "${PID}" -o stat= 2>/dev/null | tr -d ' ')" == Z* ]]; then
    echo "[stop] Process ${PID} is a zombie (already dead); removing stale pidfile."
    rm -f "${PIDFILE}"
    exit 0
fi

echo "[stop] Stopping HLEO (pid ${PID}) ..."
kill "${PID}"
# Wait briefly, then escalate to SIGKILL if needed.
for _ in 1 2 3 4 5; do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 1
done
if kill -0 "${PID}" 2>/dev/null; then
    echo "[stop] Process did not exit; sending SIGKILL."
    kill -9 "${PID}" 2>/dev/null || true
fi
rm -f "${PIDFILE}"
echo "[stop] Stopped."
