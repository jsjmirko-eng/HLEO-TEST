import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy.exc import OperationalError

import api.main as appmod
import core.api_limits as api_limits
from core.rwe.pipeline import RWEPipeline


class _Relation:
    scientific_query = "hair loss induced by dutasteride"

    def to_dict(self):
        return {"scientific_query": self.scientific_query}


class _Plan:
    original_query = "finasteride"
    translated_query = "finasteride"
    canonical_query = "finasteride"
    detected_language = "en"
    translation_applied = False
    expanded_queries = []
    entities = []
    intent = None
    vocabulary = {}

    def to_dict(self):
        return {"expanded_queries": []}


def test_database_setup_failure_does_not_block_startup(monkeypatch):
    db_error = OperationalError("database unavailable", {}, RuntimeError("db down"))

    def fail_create_all(*args, **kwargs):
        raise db_error

    monkeypatch.setattr(appmod.Base.metadata, "create_all", fail_create_all)
    with patch("core.migrations.run_schema_upgrades", side_effect=db_error):
        appmod._initialize_database_best_effort()


def test_app_import_survives_unavailable_database(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'missing' / 'hleo.db'}"
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    result = subprocess.run(
        [sys.executable, "-c", "import api.main"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_non_database_startup_error_is_not_hidden(monkeypatch):
    monkeypatch.setattr(appmod.Base.metadata, "create_all", lambda *args, **kwargs: None)
    with patch("core.migrations.run_schema_upgrades", side_effect=RuntimeError("migration bug")):
        try:
            appmod._initialize_database_best_effort()
        except RuntimeError as exc:
            assert str(exc) == "migration bug"
        else:
            raise AssertionError("non-database startup errors must propagate")



def test_scientific_search_route_does_not_use_database(monkeypatch):
    class FakeRelationalSearch:
        _client = object()

        def search(self, query):
            assert query == "finasteride"
            return {
                "pubmed": [SimpleNamespace(pmid="1", title="Study", url="", doi=None)],
                "europepmc": [],
                "clinicaltrials": [],
                "reddit": [],
                "relation": _Relation(),
                "stats": {"collector_errors": []},
            }

    def fail_db(*args, **kwargs):
        raise AssertionError("scientific search accessed the database")

    monkeypatch.setattr("core.relational_search.RelationalSearch", FakeRelationalSearch)
    monkeypatch.setattr("core.database.SessionLocal", fail_db)

    result = appmod.search("finasteride", mode="scientific", _rate_limit=None)

    assert result["totals"]["pubmed"] == 1
    assert result["pubmed"][0]["pmid"] == "1"


def test_database_rate_limiter_falls_back_to_memory(monkeypatch):
    limiter = api_limits.DatabaseRateLimiter()

    def fail_begin(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(api_limits.engine, "begin", fail_begin)

    allowed, remaining, _ = limiter.allow("offline:/search", limit=1, window_seconds=60)

    assert allowed is True
    assert remaining == 0
    assert limiter._fallback.allow("offline:/search", 1, 60)[0] is False


def test_rwe_search_uses_technical_sources_without_database(monkeypatch):
    def fail_db(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("core.database.SessionLocal", fail_db)
    pipe = RWEPipeline()
    pipe._engine.plan = lambda query: _Plan()
    selected = []

    def collect_source(name, collector, plan, cap):
        selected.append(name)
        return [], "no_results"

    pipe._collect_source = collect_source

    result = pipe.search("finasteride", sources=None)

    assert set(selected) == {
        "reddit",
        "openfda_faers",
        "calvizie",
        "hairlosstalk",
        "hairlossexperiences",
        "maladiesrares",
    }
    assert result.items == []
