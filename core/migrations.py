"""
HLEO — Schema upgrade migrations (idempotent)
=============================================

Runs lightweight ALTER TABLE migrations for columns added after the initial
schema creation.  Every migration checks whether the column already exists
before issuing DDL — safe to run on both fresh installs and existing DBs.

Called once at startup, after Base.metadata.create_all(), from api/main.py.

Adding a new migration
----------------------
1. Write a function `_add_<column>_to_<table>()` that:
   a. Checks information_schema.columns (or pg_attribute for PG) for existence.
   b. Issues ALTER TABLE … ADD COLUMN … DEFAULT … only when absent.
   c. Logs at DEBUG when already present, INFO when added.
2. Call it from run_schema_upgrades() in the ordered list.
3. Add a test in tests/test_migrations.py.

Design rules
------------
- Never DROP or RENAME — only ADD with safe defaults.
- Each migration catches its own exceptions and logs them; a failed migration
  must not prevent the server from starting.
- Works with PostgreSQL (production) and SQLite (tests / local dev).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _column_exists(conn, table: str, column: str) -> bool:
    """
    Return True if *column* already exists in *table*.

    Strategy (in order):
    1. information_schema.columns — works on PostgreSQL and SQLite ≥ 3.37.
    2. PRAGMA table_info — SQLite fallback for older versions.
    Both paths use text() wrappers required by SQLAlchemy 2.x.
    """
    from sqlalchemy import text

    try:
        result = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": column},
        ).fetchone()
        return result is not None
    except Exception:
        # information_schema unavailable (older SQLite) — use PRAGMA
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return any(row[1] == column for row in rows)


# ── Individual migrations (ordered, idempotent) ───────────────────────────────

def _add_collector_max_workers(conn) -> None:
    """
    hleo_global_limits.collector_max_workers INTEGER DEFAULT 6

    Added in FASE 4.1 (parallel Scientific collector).
    """
    table = "hleo_global_limits"
    column = "collector_max_workers"

    if _column_exists(conn, table, column):
        logger.debug("Migration: %s.%s already exists — skipped.", table, column)
        return

    from sqlalchemy import text
    conn.execute(
        text(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER DEFAULT 6")
    )
    logger.info("Migration: added %s.%s (DEFAULT 6).", table, column)


# ── Public API ────────────────────────────────────────────────────────────────

def run_schema_upgrades(engine=None) -> None:
    """
    Run all pending schema upgrades in order.

    Parameters
    ----------
    engine : sqlalchemy Engine, optional
        Defaults to the application engine from core.database.
        Pass an explicit engine in tests to target an in-memory DB.
    """
    if engine is None:
        from core.database import engine as _engine
        engine = _engine

    try:
        with engine.begin() as conn:
            _add_collector_max_workers(conn)
    except Exception as exc:
        # A migration failure must not prevent the server from starting.
        logger.error("Schema upgrade failed (server continues): %s", exc)
