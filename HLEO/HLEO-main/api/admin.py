"""
HLEO Admin — Source Management API.
Prefix: /admin

All routes are gated by core.admin_auth.require_admin (HTTP Basic Auth with a
bcrypt-hashed password from env). When the admin env vars are not set, every
route returns 404 (the section is fully disabled).

This is NOT a user/session system — it is a single-admin gate for source
governance. Runtime collectors read env secrets directly; this module only
records WHICH env vars a source needs and whether they are set (it never
reads or returns secret values).

Runtime seeding: the real collectors are registered as SourceRegistry rows
on first access, mapping each runtime collector to its category, evidence
level, and required env vars.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.admin_auth import (
    create_admin_token,
    is_admin_enabled,
    require_admin,
)
from core.database import get_db
from core.models import SourceRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Known runtime collectors (mirrors the real code) ─────────────────────────
# category: scientific | rwe_experience | pharmacovigilance
# evidence_level: registry|observational|anecdotal
_RUNTIME_SOURCES = [
    {
        "source_id": "pubmed", "name": "PubMed", "runtime_collector": "pubmed",
        "category": "scientific", "evidence_level": "registry",
        "integration_type": "api", "source_type": "rest_api",
        "data_category": "clinical_outcomes",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "global", "language_codes": ["en"],
        "description": "NCBI PubMed E-utilities — scientific literature.",
    },
    {
        "source_id": "europepmc", "name": "Europe PMC", "runtime_collector": "europepmc",
        "category": "scientific", "evidence_level": "registry",
        "integration_type": "api", "source_type": "rest_api",
        "data_category": "clinical_outcomes",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "europe", "language_codes": ["en"],
        "description": "EBI Europe PMC — scientific literature.",
    },
    {
        "source_id": "clinicaltrials", "name": "ClinicalTrials.gov",
        "runtime_collector": "clinicaltrials", "category": "scientific",
        "evidence_level": "registry", "integration_type": "api",
        "source_type": "rest_api", "data_category": "clinical_outcomes",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "global", "language_codes": ["en"],
        "description": "ClinicalTrials.gov v2 API — registered trials.",
    },
    {
        "source_id": "reddit", "name": "Reddit (PRAW)", "runtime_collector": "reddit",
        "category": "rwe_experience", "evidence_level": "anecdotal",
        "integration_type": "api", "source_type": "rest_api",
        "data_category": "discussion_posts",
        "requires_credentials": True,
        "credentials_env_vars": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
        "geographic_scope": "global", "language_codes": ["en"],
        "description": "Reddit via PRAW OAuth2 — community discussions.",
    },
    {
        "source_id": "openfda_faers", "name": "openFDA FAERS",
        "runtime_collector": "openfda_faers", "category": "pharmacovigilance",
        "evidence_level": "observational", "integration_type": "api",
        "source_type": "rest_api", "data_category": "clinical_outcomes",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "global", "language_codes": ["en"],
        "description": "openFDA FAERS — spontaneous adverse-event reports.",
    },
    {
        "source_id": "calvizie", "name": "Calvizie.net",
        "runtime_collector": "calvizie", "category": "rwe_experience",
        "evidence_level": "anecdotal", "integration_type": "rss",
        "source_type": "rss", "data_category": "discussion_posts",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "italy", "language_codes": ["it"],
        "description": "Calvizie.net forum (XenForo RSS) — patient experiences.",
    },
    {
        "source_id": "hairlosstalk", "name": "HairLossTalk",
        "runtime_collector": "hairlosstalk", "category": "rwe_experience",
        "evidence_level": "anecdotal", "integration_type": "rss",
        "source_type": "rss", "data_category": "discussion_posts",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "global", "language_codes": ["en"],
        "description": "HairLossTalk forum (XenForo RSS) — patient experiences.",
    },
    {
        "source_id": "hairlossexperiences", "name": "HairLossExperiences",
        "runtime_collector": "hairlossexperiences", "category": "rwe_experience",
        "evidence_level": "anecdotal", "integration_type": "rss",
        "source_type": "rss", "data_category": "discussion_posts",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "global", "language_codes": ["en"],
        "description": "HairLossExperiences forum (XenForo RSS).",
    },
    {
        "source_id": "maladiesrares", "name": "MaladiesRaresInfo",
        "runtime_collector": "maladiesrares", "category": "rwe_experience",
        "evidence_level": "anecdotal", "integration_type": "rss",
        "source_type": "atom", "data_category": "discussion_posts",
        "requires_credentials": False, "credentials_env_vars": [],
        "geographic_scope": "france", "language_codes": ["fr"],
        "description": "MaladiesRaresInfo (Atom feed) — patient experiences (FR).",
    },
]


def _credentials_status(env_vars: List[str]) -> dict:
    """Report whether each required env var is set. Never returns values."""
    return {
        "configured": bool(env_vars) and all(os.getenv(v) for v in env_vars),
        "missing": [v for v in env_vars if not os.getenv(v)],
        "required_vars": list(env_vars),
    }


def _source_to_dict(s: SourceRegistry) -> dict:
    env_vars = s.credentials_env_vars or []
    creds = _credentials_status(env_vars) if s.requires_credentials else None
    return {
        "id": s.id,
        "source_id": s.source_id,
        "name": s.name,
        "description": s.description,
        "category": s.category,
        "runtime_collector": s.runtime_collector,
        "integration_type": s.integration_type,
        "source_type": s.source_type,
        "data_category": s.data_category,
        "evidence_level": s.evidence_level,
        "requires_credentials": s.requires_credentials,
        "credentials": creds,
        "status": s.status,
        "authorization_status": s.authorization_status,
        "geographic_scope": s.geographic_scope,
        "language_codes": s.language_codes or [],
        "estimated_records": s.estimated_records,
        "last_sync_at": s.last_sync_at.isoformat() if s.last_sync_at else None,
        "registered_at": s.registered_at.isoformat() if s.registered_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _seed_runtime_sources(db: Session) -> int:
    """Register the known runtime collectors as SourceRegistry rows if absent.

    Idempotent: only inserts rows whose source_id does not yet exist.
    Returns the number of newly inserted rows.
    """
    inserted = 0
    for spec in _RUNTIME_SOURCES:
        existing = db.execute(
            select(SourceRegistry).where(SourceRegistry.source_id == spec["source_id"])
        ).scalar_one_or_none()
        if existing:
            continue
        row = SourceRegistry(
            source_id=spec["source_id"],
            name=spec["name"],
            description=spec.get("description"),
            category=spec.get("category", "scientific"),
            runtime_collector=spec.get("runtime_collector"),
            integration_type=spec.get("integration_type"),
            source_type=spec.get("source_type"),
            data_category=spec.get("data_category"),
            evidence_level=spec.get("evidence_level"),
            requires_credentials=spec.get("requires_credentials", False),
            credentials_env_vars=spec.get("credentials_env_vars", []),
            geographic_scope=spec.get("geographic_scope"),
            language_codes=spec.get("language_codes", []),
            status="active",
            authorization_status="authorized" if not spec.get("requires_credentials") else "unauthorized",
        )
        db.add(row)
        inserted += 1
    if inserted:
        db.commit()
    return inserted


# ── Pydantic models ──────────────────────────────────────────────────────────

class AdminLoginRequest(BaseModel):
    username: str
    password: str

    model_config = {"extra": "ignore"}


class SourceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    authorization_status: Optional[str] = None
    requires_credentials: Optional[bool] = None
    credentials_env_vars: Optional[List[str]] = None
    connection_spec: Optional[dict] = None
    evidence_level: Optional[str] = None
    geographic_scope: Optional[str] = None
    language_codes: Optional[List[str]] = None
    estimated_records: Optional[int] = None

    model_config = {"extra": "ignore"}


class SourceCreate(BaseModel):
    source_id: str
    name: str
    description: Optional[str] = ""
    category: str = "scientific"
    runtime_collector: Optional[str] = None
    integration_type: Optional[str] = "api"
    source_type: Optional[str] = "rest_api"
    data_category: Optional[str] = ""
    evidence_level: Optional[str] = "anecdotal"
    requires_credentials: bool = False
    credentials_env_vars: List[str] = []
    connection_spec: Optional[dict] = None
    geographic_scope: Optional[str] = "global"
    language_codes: List[str] = ["en"]

    model_config = {"extra": "ignore"}


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/ping")
def admin_ping():
    """Public probe: returns whether the admin section is enabled.

    Does NOT require authentication and returns NO sensitive data — only a
    boolean. The frontend uses this to decide whether to show the Admin tab.
    Authentication still happens on /admin/* via require_admin.
    """
    return {"admin_enabled": is_admin_enabled()}


@router.post("/login")
def admin_login(body: AdminLoginRequest):
    """Authenticate with username + password → stateless Bearer token.

    Validates credentials against the same bcrypt hash used by Basic Auth.
    On success returns a signed token the frontend stores in sessionStorage
    and sends as `Authorization: Bearer <token>` on subsequent /admin/* calls.
    The plaintext password is NEVER stored; the token is HMAC-signed with the
    bcrypt hash (server-side only) and expires automatically.
    """
    if not is_admin_enabled():
        # Admin not configured — hide the section (404, not 401).
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    from core.admin_auth import _verify_credentials
    if not _verify_credentials(body.username, body.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials.",
        )
    token, exp = create_admin_token(body.username)
    return {
        "token": token,
        "expires_at": exp,
        "username": body.username,
    }


@router.post("/logout")
def admin_logout(_: str = Depends(require_admin)):
    """Logout endpoint (stateless tokens — client just discards the token).

    Kept for API symmetry; the frontend clears sessionStorage on its own.
    """
    return {"logged_out": True}


@router.get("/status")
def admin_status(request: Request, _: str = Depends(require_admin)):
    """Admin availability + auth check (returns 200 only if authenticated).

    When called with ?return=admin (after the browser's native Basic-Auth
    prompt succeeds), returns a tiny HTML page that redirects back to the
    app and re-opens the admin tab. This lets the frontend trigger the
    browser's native auth dialog via a top-level navigation.
    """
    ret = request.query_params.get("return")
    if ret == "admin":
        return HTMLResponse(
            content="""<!doctype html><html><head><meta http-equiv="refresh" content="0; url=/?page=admin"></head>
            <body style="font-family:system-ui;background:#0d1117;color:#8b949e;text-align:center;padding:3rem">
              <p>Authentication successful. Redirecting…</p>
              <p><a href="/?page=admin" style="color:#3b82f6">Click here if not redirected.</a></p>
            </body></html>""",
            status_code=200,
        )
    return {"admin_enabled": True, "authenticated": True}


@router.get("/sources")
def list_sources(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    _: str = Depends(require_admin),
):
    """List all registered sources. Seeds the known runtime collectors first."""
    _seed_runtime_sources(db)
    q = select(SourceRegistry).order_by(SourceRegistry.source_id)
    if category:
        q = q.where(SourceRegistry.category == category)
    if status_filter:
        q = q.where(SourceRegistry.status == status_filter)
    rows = db.execute(q).scalars().all()
    return {"total": len(rows), "sources": [_source_to_dict(s) for s in rows]}


@router.post("/sources", status_code=201)
def create_source(
    body: SourceCreate,
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
):
    existing = db.execute(
        select(SourceRegistry).where(SourceRegistry.source_id == body.source_id)
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(409, f"Source '{body.source_id}' already exists.")
    row = SourceRegistry(
        source_id=body.source_id,
        name=body.name,
        description=body.description,
        category=body.category,
        runtime_collector=body.runtime_collector,
        integration_type=body.integration_type,
        source_type=body.source_type,
        data_category=body.data_category,
        evidence_level=body.evidence_level,
        requires_credentials=body.requires_credentials,
        credentials_env_vars=body.credentials_env_vars,
        connection_spec=body.connection_spec,
        geographic_scope=body.geographic_scope,
        language_codes=body.language_codes,
        status="active",
        authorization_status="unauthorized" if body.requires_credentials else "authorized",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _source_to_dict(row)


@router.patch("/sources/{source_id}")
def update_source(
    source_id: str,
    body: SourceUpdate,
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
):
    row = db.execute(
        select(SourceRegistry).where(SourceRegistry.source_id == source_id)
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(404, f"Source '{source_id}' not found.")
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(row, k, v)
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return _source_to_dict(row)


@router.delete("/sources/{source_id}", status_code=204)
def delete_source(
    source_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
):
    row = db.execute(
        select(SourceRegistry).where(SourceRegistry.source_id == source_id)
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(404, f"Source '{source_id}' not found.")
    db.delete(row)
    db.commit()
    return None


@router.post("/sources/{source_id}/test")
def test_source_connection(
    source_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
):
    """Test connectivity + credentials for a source (no secrets returned).

    Instantiates the real collector and runs a tiny probe query. For sources
    requiring credentials, reports configured=True/False without exposing
    values.
    """
    row = db.execute(
        select(SourceRegistry).where(SourceRegistry.source_id == source_id)
    ).scalar_one_or_none()
    if not row:
        raise HTTPException(404, f"Source '{source_id}' not found.")

    creds = _credentials_status(row.credentials_env_vars or [])
    result = {
        "source_id": source_id,
        "name": row.name,
        "credentials": creds if row.requires_credentials else None,
        "reachable": False,
        "items_returned": 0,
        "error": None,
    }

    collector_key = row.runtime_collector or source_id
    probe_query = "finasteride"
    try:
        if collector_key == "pubmed":
            from collectors.pubmed import PubMedCollector
            items = PubMedCollector().search(probe_query, limit=1)
            result["reachable"] = True
            result["items_returned"] = len(items)
        elif collector_key == "europepmc":
            from collectors.europepmc import EuropePMCCollector
            items = EuropePMCCollector().search(probe_query, limit=1)
            result["reachable"] = True
            result["items_returned"] = len(items)
        elif collector_key == "clinicaltrials":
            from collectors.clinicaltrials import ClinicalTrialsCollector
            items = ClinicalTrialsCollector().search(probe_query, limit=1)
            result["reachable"] = True
            result["items_returned"] = len(items)
        elif collector_key == "openfda_faers":
            from core.rwe.openfda_collector import OpenFDACollector
            res = OpenFDACollector().search(probe_query, limit=1)
            result["reachable"] = True
            result["items_returned"] = len(res) if isinstance(res, list) else (res.get("items", []) if isinstance(res, dict) else 0)
        elif collector_key == "generic_rest":
            # Generic REST collector configured via connection_spec
            from collectors.generic_rest import GenericRESTCollector
            gen = GenericRESTCollector(row.connection_spec or {}, source_id=row.source_id, category=row.category)
            if row.category == "rwe_experience":
                items, st, reason = gen.search_with_status(probe_query, limit=1)
                result["reachable"] = (st == "ok")
                result["items_returned"] = len(items) if isinstance(items, list) else 0
                if st != "ok":
                    result["error"] = reason
            else:
                try:
                    items = gen.search(probe_query, limit=1)
                    result["reachable"] = True
                    result["items_returned"] = len(items)
                except Exception as exc:
                    result["error"] = str(exc)
        elif collector_key == "calvizie":

            from core.rwe.calvizie_collector import CalvizieCollector
            items = CalvizieCollector().search_with_status(probe_query, limit=1)
            result["reachable"] = bool(items and items[0])
            result["items_returned"] = len(items[0]) if (items and isinstance(items[0], list)) else 0
        elif collector_key == "hairlosstalk":
            from core.rwe.hairlosstalk_collector import HairLossTalkCollector
            items = HairLossTalkCollector().search_with_status(probe_query, limit=1)
            result["reachable"] = bool(items and items[0])
            result["items_returned"] = len(items[0]) if (items and isinstance(items[0], list)) else 0
        elif collector_key == "hairlossexperiences":
            from core.rwe.hairlossexperiences_collector import HairLossExperiencesCollector
            items = HairLossExperiencesCollector().search_with_status(probe_query, limit=1)
            result["reachable"] = bool(items and items[0])
            result["items_returned"] = len(items[0]) if (items and isinstance(items[0], list)) else 0
        elif collector_key == "maladiesrares":
            from core.rwe.maladiesrares_collector import MaladiesRaresCollector
            items = MaladiesRaresCollector().search_with_status(probe_query, limit=1)
            result["reachable"] = bool(items and items[0])
            result["items_returned"] = len(items[0]) if (items and isinstance(items[0], list)) else 0
        elif collector_key == "reddit":
            if not creds["configured"]:
                result["error"] = "Reddit credentials not configured."
            else:
                from collectors.reddit import RedditCollector
                _, st, _ = RedditCollector().search_with_status(probe_query, limit=1)
                result["reachable"] = (st == "ok")
                result["error"] = None if st == "ok" else f"Reddit status: {st}"
        else:
            result["error"] = f"No test probe for collector '{collector_key}'."
    except Exception as exc:
        logger.exception(f"Source test failed for {source_id}")
        result["error"] = str(exc)

    return result
