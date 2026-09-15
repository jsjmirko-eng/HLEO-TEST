"""
HLEO — Test suite per core/migrations.py

Scenari
-------
1.  run_schema_upgrades() su schema già aggiornato (colonna presente) — idempotente
2.  run_schema_upgrades() su schema privo della colonna — aggiunge la colonna
3.  Il valore DEFAULT 6 è applicato alle righe esistenti
4.  Doppia esecuzione — idempotente (nessun errore)
5.  DB nuovo (create_all + upgrade) — tutto funziona
6.  Fallimento DB — run_schema_upgrades() non propaga l'eccezione
"""
from __future__ import annotations

import unittest
from sqlalchemy import create_engine, text, inspect

from core.database import Base


def _fresh_engine():
    """In-memory SQLite engine per i test — isolato, nessun accesso al DB reale."""
    return create_engine("sqlite:///:memory:", echo=False)


class TestMigrationColumnAlreadyPresent(unittest.TestCase):
    """Scenario 1 — schema già aggiornato: run_schema_upgrades() è idempotente."""

    def test_no_error_when_column_exists(self):
        engine = _fresh_engine()
        Base.metadata.create_all(bind=engine)  # crea hleo_global_limits con la colonna

        # Prima passata
        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)  # non deve sollevare

        # Seconda passata — idempotente
        run_schema_upgrades(engine)  # non deve sollevare

        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("hleo_global_limits")]
        self.assertIn("collector_max_workers", cols)


class TestMigrationColumnMissing(unittest.TestCase):
    """Scenario 2 — schema privo della colonna: viene aggiunta."""

    def test_column_added_when_missing(self):
        engine = _fresh_engine()

        # Crea la tabella manualmente SENZA collector_max_workers
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE hleo_global_limits ("
                "  id INTEGER PRIMARY KEY,"
                "  max_total_attempts INTEGER DEFAULT 5,"
                "  pipeline_max_workers INTEGER DEFAULT 8,"
                "  updated_at DATETIME"
                ")"
            ))

        # Verifica che la colonna non esista
        insp = inspect(engine)
        cols_before = [c["name"] for c in insp.get_columns("hleo_global_limits")]
        self.assertNotIn("collector_max_workers", cols_before)

        # Esegui la migration
        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)

        # Verifica che ora esista
        insp2 = inspect(engine)
        cols_after = [c["name"] for c in insp2.get_columns("hleo_global_limits")]
        self.assertIn("collector_max_workers", cols_after)


class TestMigrationDefaultValue(unittest.TestCase):
    """Scenario 3 — righe esistenti: DEFAULT 6 viene applicato."""

    def test_existing_row_gets_default(self):
        engine = _fresh_engine()

        # Tabella senza la colonna + riga esistente
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE hleo_global_limits ("
                "  id INTEGER PRIMARY KEY,"
                "  max_total_attempts INTEGER DEFAULT 5"
                ")"
            ))
            conn.execute(text(
                "INSERT INTO hleo_global_limits (id, max_total_attempts) VALUES (1, 5)"
            ))

        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT collector_max_workers FROM hleo_global_limits WHERE id = 1")
            ).fetchone()

        # SQLite assegna il DEFAULT alle colonne aggiunte con ADD COLUMN
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 6)


class TestMigrationIdempotent(unittest.TestCase):
    """Scenario 4 — doppia esecuzione: nessun errore."""

    def test_double_run_is_safe(self):
        engine = _fresh_engine()

        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE hleo_global_limits ("
                "  id INTEGER PRIMARY KEY"
                ")"
            ))

        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)  # aggiunge la colonna
        run_schema_upgrades(engine)  # già presente — deve essere silenzioso

        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("hleo_global_limits")]
        count = cols.count("collector_max_workers")
        self.assertEqual(count, 1, "La colonna non deve essere duplicata")


class TestMigrationFreshDatabase(unittest.TestCase):
    """Scenario 5 — DB nuovo: create_all + upgrade funziona correttamente."""

    def test_fresh_db_create_all_then_upgrade(self):
        engine = _fresh_engine()
        Base.metadata.create_all(bind=engine)  # schema completo

        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)

        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("hleo_global_limits")]
        self.assertIn("collector_max_workers", cols)
        self.assertIn("pipeline_max_workers", cols)
        self.assertIn("max_total_attempts", cols)


class TestMigrationDBFailure(unittest.TestCase):
    """Scenario 6 — fallimento DB: run_schema_upgrades non propaga l'eccezione."""

    def test_db_failure_does_not_raise(self):
        """Un engine rotto non deve impedire l'avvio del server."""
        from sqlalchemy import create_engine as ce
        broken_engine = ce("sqlite:///", echo=False)  # percorso invalido ma non solleva

        # Usiamo un engine valido ma con connect() patchato per simulare un errore
        from unittest.mock import patch, MagicMock

        engine = _fresh_engine()

        def raise_on_begin(*a, **kw):
            raise Exception("Simulated DB connection failure")

        with patch.object(engine, "begin", side_effect=raise_on_begin):
            from core.migrations import run_schema_upgrades
            # Non deve propagare l'eccezione — il server deve avviarsi comunque
            try:
                run_schema_upgrades(engine)
            except Exception as exc:
                self.fail(f"run_schema_upgrades() ha propagato un'eccezione: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
