"""Shared pytest fixtures for HLEO backend tests.

Creates tables on an in-memory SQLite DB so the API can be exercised via
TestClient without a live Postgres instance. The .env in the repo holds a real
OPENAI_API_KEY (loaded by core.database.load_dotenv at import), so we blank it
again here to keep tests hermetic.
"""
import os
import sys

# Force SQLite in-memory BEFORE importing the app (core.database reads env at import).
# Use StaticPool so the single in-memory DB is shared across threads (TestClient
# runs the ASGI app in a separate thread from the test).
os.environ["DATABASE_URL"] = "sqlite://"

# Tests are hermetic/offline: the Catena C vocabulary layer defaults to ON in
# production, but in tests real providers must never be hit. Tests that need
# terminology patch build_resolver_from_env with a FakeResolver explicitly.
os.environ["HLEO_VOCAB_ENABLED"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Neutralize load_dotenv before any core module re-imports it during a request.
# Several core modules (core.extractor, core.article_extractor, core.database) call
# load_dotenv() at import time, which would re-leak the real .env key into the
# process env mid-request and defeat the per-test env pops.
import dotenv
_real_load_dotenv = dotenv.load_dotenv
def _noop_load_dotenv(*a, **k):
    return False
dotenv.load_dotenv = _noop_load_dotenv
# Also patch already-imported references in modules that did `from dotenv import load_dotenv`.
for _modname in ("core.database", "core.extractor", "core.article_extractor"):
    try:
        _m = sys.modules.get(_modname)
        if _m is not None and hasattr(_m, "load_dotenv"):
            _m.load_dotenv = _noop_load_dotenv
    except Exception:
        pass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

import api.main as appmod
from core.database import Base, engine, SessionLocal
from sqlalchemy import create_engine as _ce

# Replace the engine with a StaticPool in-memory SQLite so the single DB
# instance is shared across threads (TestClient runs ASGI in a separate thread).
_test_engine = _ce(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
from sqlalchemy.orm import sessionmaker as _sm
_test_SessionLocal = _sm(autocommit=False, autoflush=False, bind=_test_engine)
# Rebind the app's DB dependency to the test engine.
appmod.engine = _test_engine
appmod.SessionLocal = _test_SessionLocal
# Also rebind core.database so any module that imported SessionLocal from it
# at module-load time points at the test engine.
import core.database as _coredb
_coredb.engine = _test_engine
_coredb.SessionLocal = _test_SessionLocal


def _get_test_db():
    db = _test_SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override the FastAPI dependency used by the app.
appmod.app.dependency_overrides[appmod.get_db] = _get_test_db


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture(autouse=True)
def _no_openai_key():
    """Ensure no real LLM key leaks into any test (load_dotenv is neutralized).

    PERPLEXITY_API_KEY is popped too: with the multi-provider layer
    (core.llm_provider) a Perplexity key in the environment would otherwise
    make "auto" resolve to Perplexity and break hermeticity.
    """
    for var in ("OPENAI_API_KEY", "PERPLEXITY_API_KEY", "perplexity_api_key",
                "HLEO_LLM_PROVIDER"):
        os.environ.pop(var, None)
    yield
    for var in ("OPENAI_API_KEY", "PERPLEXITY_API_KEY", "perplexity_api_key",
                "HLEO_LLM_PROVIDER"):
        os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def _clean_db():
    """Truncate all tables after each test so tests are order-independent."""
    yield
    with _test_engine.connect() as conn:
        from sqlalchemy import text
        # SQLite: delete from all tables (faster than drop+recreate, keeps schema)
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        for tbl in reversed(Base.metadata.sorted_tables):
            conn.execute(tbl.delete())
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()


@pytest.fixture()
def client():
    return TestClient(appmod.app)


@pytest.fixture()
def db_session():
    s = _test_SessionLocal()
    try:
        yield s
    finally:
        s.close()
