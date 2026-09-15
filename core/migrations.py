"""
HLEO — Schema upgrade migrations (idempotent)
=============================================

Runs lightweight ALTER TABLE migrations for columns added after the initial
schema creation.  Every migration checks whether the column already exists
before issuing DDL — safe to run on both fresh installs and existing DBs.

Called once at startup, after Base.metadata.create_all(), from api/main.py.

Adding a new migration
----------------------
1. Write a call to _add_column() inside the appropriate grouping function.
2. Call that function from run_schema_upgrades() in the ordered list.
3. Add a test in tests/test_migrations.py.

Design rules
------------
- Never DROP or RENAME — only ADD with safe defaults.
- A failed migration logs an error but does NOT raise — server starts anyway.
- Works with PostgreSQL (production) and SQLite (tests / local dev).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TABLE = "hleo_global_limits"


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


def _add_column(conn, table: str, column: str, definition: str) -> None:
    """
    Add *column* to *table* with the given SQL *definition* (type + DEFAULT).
    No-op when the column already exists.
    """
    from sqlalchemy import text

    if _column_exists(conn, table, column):
        logger.debug("Migration: %s.%s already exists — skipped.", table, column)
        return
    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
    logger.info("Migration: added %s.%s (%s).", table, column, definition)


# ── Individual migrations (ordered, idempotent) ───────────────────────────────

def _migrate_fase_4_1(conn) -> None:
    """FASE 4.1 — parallel Scientific collector."""
    _add_column(conn, _TABLE, "collector_max_workers", "INTEGER DEFAULT 6")


def _migrate_fase_4_2b(conn) -> None:
    """FASE 4.2B — per-source semaphores + configurable HTTP settings."""
    _add_column(conn, _TABLE, "pubmed_max_concurrent",     "INTEGER DEFAULT 2")
    _add_column(conn, _TABLE, "epmc_max_concurrent",       "INTEGER DEFAULT 4")
    _add_column(conn, _TABLE, "ct_max_concurrent",         "INTEGER DEFAULT 3")
    _add_column(conn, _TABLE, "pubmed_inter_call_sleep_s", "REAL DEFAULT 0.4")
    _add_column(conn, _TABLE, "collector_timeout_s",       "REAL DEFAULT 20.0")
    _add_column(conn, _TABLE, "collector_max_retries",     "INTEGER DEFAULT 2")


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
            _migrate_fase_4_1(conn)
            _migrate_fase_4_2b(conn)
    except Exception as exc:
        # A migration failure must not prevent the server from starting.
        logger.error("Schema upgrade failed (server continues): %s", exc)
