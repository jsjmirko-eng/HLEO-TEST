#!/usr/bin/env python3
"""HLEO component health check.

Emits one line per component with an explicit status:

    PASS            — component is present and working
    FAIL            — component is expected but not available/working
    NOT CONFIGURED  — optional component with no credentials/configuration

Exit code is 0 when no component is FAIL, otherwise 1.

This script is intentionally self-contained and performs NO network calls to
external services: it checks local process reachability, the database
connection and the presence/absence of credentials, exactly matching the
bootloader's readiness contract.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

BASE_URL = os.getenv("HLEO_HEALTH_URL", f"http://127.0.0.1:{os.getenv('PORT', '8000')}")

PASS = "PASS"
FAIL = "FAIL"
NOT_CONFIGURED = "NOT CONFIGURED"


@dataclass
class Component:
    name: str
    status: str
    detail: str = ""


def _line(name: str, status: str, detail: str = "") -> str:
    out = f"{status:<16} {name}"
    if detail:
        out += f" — {detail}"
    return out


def _is_reachable() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _check_db() -> tuple[str, str]:
    """PASS when a SQLAlchemy connection can be established and a trivial
    query executes; FAIL when DATABASE_URL points at something unreachable."""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
    except Exception:
        pass

    try:
        from core.database import engine
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        return PASS, str(engine.url).split("@")[-1] if "@" in str(engine.url) else str(engine.url)
    except Exception as exc:
        return FAIL, f"{type(exc).__name__}: {exc}"


def _check_llm() -> tuple[str, str]:
    try:
        from core.llm_provider import build_provider, configured_provider_name
    except Exception:
        return FAIL, "core.llm_provider could not be imported"

    name = configured_provider_name()
    provider = build_provider()
    if provider is not None:
        return PASS, f"provider={provider.name} (configured mode: {name})"
    return NOT_CONFIGURED, f"no LLM key configured (mode: {name}); scientific search/synthesis/assistant will return 503"


def _check_scientific_rwe() -> tuple[str, str]:
    """Scientific and RWE are in-process services of the FastAPI backend.
    Verify their modules import and their routes are actually mounted."""
    try:
        import core.orchestrator  # noqa: F401
        import core.relational_search  # noqa: F401
        import core.rwe.pipeline  # noqa: F401
        import core.rwe.query_engine  # noqa: F401
    except Exception as exc:
        return FAIL, f"service modules could not be imported: {type(exc).__name__}: {exc}"

    try:
        from api.main import app
        paths = set()
        for route in app.routes:
            path = getattr(route, "path", None)
            if path is None:
                # APIRoute/Route have .path; included Router objects may not.
                continue
            paths.add(path)
        missing = [p for p in ("/search", "/rwe/search", "/pipeline/run", "/assistant/chat", "/synthesis") if p not in paths]
    except Exception as exc:
        return FAIL, f"api.main could not be loaded: {type(exc).__name__}: {exc}"

    if missing:
        return FAIL, f"routes not mounted: {', '.join(missing)}"
    return PASS, "scientific + RWE routes mounted (in-process with backend)"


def _check_optional_env(var: str, label: str) -> tuple[str, str]:
    if os.getenv(var):
        return PASS, f"{var} is set"
    return NOT_CONFIGURED, f"{var} not set"


def main() -> int:
    # Load .env once for the whole script.
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
    except Exception:
        pass

    results: list[tuple[str, str, str]] = []

    # 1. Environment
    results.append(("Python", PASS, sys.version.split()[0]))

    # 2. Dependencies
    missing_deps = []
    for mod in ("fastapi", "uvicorn", "sqlalchemy", "pydantic", "dotenv", "openai"):
        try:
            __import__(mod)
        except Exception:
            missing_deps.append(mod)
    results.append(("Dependencies", PASS if not missing_deps else FAIL,
                    "all core imports ok" if not missing_deps else f"missing: {', '.join(missing_deps)}"))

    # 3. Database
    db_status, db_detail = _check_db()
    results.append(("Database", db_status, db_detail))

    # 4. Backend (HTTP reachability)
    if _is_reachable():
        results.append(("Backend", PASS, f"{BASE_URL}/health responded 200"))
    else:
        results.append(("Backend", FAIL, f"{BASE_URL}/health unreachable"))

    # 5. LLM provider (Scientific/RWE/Assistant all depend on it)
    llm_status, llm_detail = _check_llm()
    results.append(("LLM provider", llm_status, llm_detail))

    # 5. Scientific/RWE services (in-process with the backend)
    scirwe_status, scirwe_detail = _check_scientific_rwe()
    results.append(("Scientific/RWE", scirwe_status, scirwe_detail))

    # 6. Optional external providers (skipped gracefully when unset)
    results.append(("Reddit", *_check_optional_env("REDDIT_CLIENT_ID", "Reddit OAuth2")))
    results.append(("UMLS", *_check_optional_env("UMLS_API_KEY", "UMLS")))
    results.append(("LOINC", *_check_optional_env("HLEO_LOINC_USERNAME", "LOINC")))
    results.append(("Admin auth", *_check_optional_env("HLEO_ADMIN_USERNAME", "Admin")))

    failed = False
    for name, status, detail in results:
        print(_line(name, status, detail))
        if status == FAIL:
            failed = True

    print()
    print("HLEO READY" if not failed else "HLEO NOT READY")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
