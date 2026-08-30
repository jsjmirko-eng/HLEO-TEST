"""
HLEO Real World Evidence & Partnership Management API
Prefix: /rwe

Partner lifecycle: proposed → contacted → in_discussion → legal_review
                   → approved → active  (or → declined / → disabled)

Routes
──────
GET  /rwe/partners                   list partners
POST /rwe/partners                   register a partner
PATCH /rwe/partners/{id}             update partner (status, notes, …)
DELETE /rwe/partners/{id}            remove partner (soft delete via status)

GET  /rwe/partners/{id}/contacts     list contacts
POST /rwe/partners/{id}/contacts     add contact
PATCH /rwe/contacts/{id}             update contact
DELETE /rwe/contacts/{id}            remove contact

GET  /rwe/partners/{id}/agreements   list agreements
POST /rwe/partners/{id}/agreements   add agreement
PATCH /rwe/agreements/{id}           update agreement (status, sign, etc.)

GET  /rwe/sources                    list sources (filter by partner / status)
POST /rwe/sources                    register a source (partner must be approved/active)
PATCH /rwe/sources/{id}              update source

POST /rwe/sources/{id}/authorize     create authorization record
DELETE /rwe/sources/{id}/authorize   revoke authorization

GET  /rwe/sources/{id}/licensing     list licenses for a source
POST /rwe/sources/{id}/licensing     attach license

GET  /rwe/agreements                 list all agreements (optional partner_id filter)
GET  /rwe/licenses                   list all licenses  (optional source_id filter)

GET  /rwe/evidence-comparisons       list comparisons
POST /rwe/evidence-comparisons       create comparison scaffold
GET  /rwe/evidence-comparisons/{id}  get comparison detail
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.database import get_db
from core.models import (
    DataLicensing,
    EvidenceComparison,
    PartnerAgreement,
    PartnerAuthorization,
    PartnerContact,
    PartnerRegistry,
    SourceRegistry,
)

router = APIRouter(prefix="/rwe", tags=["rwe"])

# ─── Valid lifecycle transitions ──────────────────────────────────────────────
_PARTNER_TRANSITIONS: Dict[str, List[str]] = {
    "proposed":       ["contacted", "declined", "disabled"],
    "contacted":      ["in_discussion", "declined", "disabled"],
    "in_discussion":  ["legal_review", "declined", "disabled"],
    "legal_review":   ["approved", "declined", "disabled"],
    "approved":       ["active", "disabled"],
    "active":         ["disabled"],
    "declined":       ["contacted"],   # re-engage allowed
    "disabled":       [],
}

_AGREEMENT_TRANSITIONS: Dict[str, List[str]] = {
    "draft":        ["under_review", "terminated"],
    "under_review": ["negotiating", "draft", "terminated"],
    "negotiating":  ["under_review", "signed", "terminated"],
    "signed":       ["active", "terminated"],
    "active":       ["expired", "terminated"],
    "expired":      ["draft"],   # renew
    "terminated":   [],
}


# ═══════════════════ Pydantic schemas ════════════════════════════════════════

class PartnerCreate(BaseModel):
    name: str
    short_name: Optional[str] = None
    description: Optional[str] = None
    partner_type: str              # forum|patient_org|healthcare_community|research_network
    integration_type: Optional[str] = None
    website_url: Optional[str] = None
    internal_notes: Optional[str] = None
    tags: Optional[List[str]] = None


class PartnerUpdate(BaseModel):
    status: Optional[str] = None
    short_name: Optional[str] = None
    description: Optional[str] = None
    integration_type: Optional[str] = None
    website_url: Optional[str] = None
    data_sharing_agreement: Optional[bool] = None
    first_contact_date: Optional[datetime] = None
    last_contact_date: Optional[datetime] = None
    next_followup_date: Optional[datetime] = None
    outreach_notes: Optional[str] = None
    legal_review_started: Optional[datetime] = None
    legal_review_completed: Optional[datetime] = None
    internal_notes: Optional[str] = None
    tags: Optional[List[str]] = None


class ContactCreate(BaseModel):
    full_name: str
    role: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    is_primary: bool = False
    notes: Optional[str] = None


class ContactUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    is_primary: Optional[bool] = None
    notes: Optional[str] = None
    last_contacted_at: Optional[datetime] = None


class AgreementCreate(BaseModel):
    agreement_type: str           # mou|data_sharing_agreement|nda|research_collaboration_agreement|api_license
    title: str
    summary: Optional[str] = None
    key_terms: Optional[List[Dict[str, str]]] = None
    file_reference: Optional[str] = None
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    notes: Optional[str] = None


class AgreementUpdate(BaseModel):
    status: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    key_terms: Optional[List[Dict[str, str]]] = None
    file_reference: Optional[str] = None
    signed_at: Optional[datetime] = None
    signed_by_partner: Optional[str] = None
    signed_by_hleo: Optional[str] = None
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    sent_for_review_at: Optional[datetime] = None
    notes: Optional[str] = None
    version: Optional[str] = None


class SourceCreate(BaseModel):
    partner_id: str
    name: str
    description: Optional[str] = None
    integration_type: str         # api|licensed_dataset|rss|manual_import|research_collaboration
    source_type: Optional[str] = None
    data_category: Optional[str] = None
    connection_spec: Optional[Dict[str, Any]] = None
    evidence_level: Optional[str] = None
    population_tags: Optional[List[str]] = None
    geographic_scope: Optional[str] = None
    language_codes: Optional[List[str]] = None
    estimated_records: Optional[int] = None
    record_update_frequency: Optional[str] = None


class SourceUpdate(BaseModel):
    status: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    integration_type: Optional[str] = None
    connection_spec: Optional[Dict[str, Any]] = None
    evidence_level: Optional[str] = None
    population_tags: Optional[List[str]] = None
    estimated_records: Optional[int] = None
    record_update_frequency: Optional[str] = None


class AuthorizationCreate(BaseModel):
    auth_type: str                # api_key|oauth2|basic_auth|ip_whitelist|manual|mtls
    auth_config_schema: Optional[Dict[str, Any]] = None  # field schema — no actual secrets
    scope: Optional[List[str]] = None
    rate_limit_rpm: Optional[int] = None
    expires_at: Optional[datetime] = None
    granted_by: Optional[str] = "system"


class LicensingCreate(BaseModel):
    agreement_id: Optional[str] = None
    license_type: str             # cc_by|cc_by_nc|proprietary|research_only|open_data|custom
    permitted_uses: Optional[List[str]] = None
    restrictions: Optional[List[str]] = None
    attribution_required: bool = True
    attribution_text: Optional[str] = None
    geographic_restrictions: Optional[List[str]] = None
    retention_period_days: Optional[int] = None
    deletion_on_request: bool = True
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    notes: Optional[str] = None


class ComparisonCreate(BaseModel):
    query_topic: str
    dimensions: Optional[List[str]] = None
    scientific_profile_ids: Optional[List[str]] = None
    community_source_ids: Optional[List[str]] = None


# ═══════════════════ Partners ════════════════════════════════════════════════

@router.get("/partners")
async def list_partners(
    status: Optional[str] = None,
    partner_type: Optional[str] = None,
    limit: int = Query(200, le=500),
    db: Session = Depends(get_db),
):
    q = select(PartnerRegistry).order_by(PartnerRegistry.created_at.desc())
    if status:
        q = q.where(PartnerRegistry.status == status)
    if partner_type:
        q = q.where(PartnerRegistry.partner_type == partner_type)
    rows = db.execute(q.limit(limit)).scalars().all()
    return [_partner_out(p) for p in rows]


@router.post("/partners", status_code=201)
async def create_partner(body: PartnerCreate, db: Session = Depends(get_db)):
    partner = PartnerRegistry(
        partner_id=str(uuid.uuid4()),
        **body.model_dump(),
        status="proposed",
    )
    db.add(partner)
    db.commit()
    db.refresh(partner)
    return _partner_out(partner)


@router.patch("/partners/{partner_id}")
async def update_partner(
    partner_id: str, body: PartnerUpdate, db: Session = Depends(get_db)
):
    p = _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, partner_id, "Partner")
    updates = body.model_dump(exclude_none=True)
    # Validate lifecycle transition
    if "status" in updates and updates["status"] != p.status:
        allowed = _PARTNER_TRANSITIONS.get(p.status, [])
        if updates["status"] not in allowed:
            raise HTTPException(
                400,
                f"Cannot transition from '{p.status}' to '{updates['status']}'. "
                f"Allowed next states: {allowed}",
            )
        # Auto-set timestamps on key transitions
        if updates["status"] == "contacted" and not p.first_contact_date:
            p.first_contact_date = datetime.now(timezone.utc)
        if updates["status"] == "legal_review" and not p.legal_review_started:
            p.legal_review_started = datetime.now(timezone.utc)
        if updates["status"] in ("approved", "active") and p.legal_review_started and not p.legal_review_completed:
            p.legal_review_completed = datetime.now(timezone.utc)
    for k, v in updates.items():
        setattr(p, k, v)
    p.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(p)
    return _partner_out(p)


@router.delete("/partners/{partner_id}", status_code=204)
async def delete_partner(partner_id: str, db: Session = Depends(get_db)):
    p = _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, partner_id, "Partner")
    db.delete(p)
    db.commit()


# ═══════════════════ Contacts ════════════════════════════════════════════════

@router.get("/partners/{partner_id}/contacts")
async def list_contacts(partner_id: str, db: Session = Depends(get_db)):
    _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, partner_id, "Partner")
    rows = db.execute(
        select(PartnerContact)
        .where(PartnerContact.partner_id == partner_id)
        .order_by(PartnerContact.is_primary.desc(), PartnerContact.full_name)
    ).scalars().all()
    return [_contact_out(c) for c in rows]


@router.post("/partners/{partner_id}/contacts", status_code=201)
async def add_contact(
    partner_id: str, body: ContactCreate, db: Session = Depends(get_db)
):
    _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, partner_id, "Partner")
    # Demote existing primary if this one is primary
    if body.is_primary:
        existing = db.execute(
            select(PartnerContact)
            .where(PartnerContact.partner_id == partner_id, PartnerContact.is_primary == True)
        ).scalars().all()
        for c in existing:
            c.is_primary = False
    contact = PartnerContact(
        contact_id=str(uuid.uuid4()),
        partner_id=partner_id,
        **body.model_dump(),
    )
    db.add(contact)
    db.commit()
    db.refresh(contact)
    return _contact_out(contact)


@router.patch("/contacts/{contact_id}")
async def update_contact(
    contact_id: str, body: ContactUpdate, db: Session = Depends(get_db)
):
    c = _get_or_404(db, PartnerContact, PartnerContact.contact_id, contact_id, "Contact")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c)
    return _contact_out(c)


@router.delete("/contacts/{contact_id}", status_code=204)
async def delete_contact(contact_id: str, db: Session = Depends(get_db)):
    c = _get_or_404(db, PartnerContact, PartnerContact.contact_id, contact_id, "Contact")
    db.delete(c)
    db.commit()


# ═══════════════════ Agreements ══════════════════════════════════════════════

@router.get("/partners/{partner_id}/agreements")
async def list_partner_agreements(partner_id: str, db: Session = Depends(get_db)):
    _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, partner_id, "Partner")
    rows = db.execute(
        select(PartnerAgreement)
        .where(PartnerAgreement.partner_id == partner_id)
        .order_by(PartnerAgreement.created_at.desc())
    ).scalars().all()
    return [_agreement_out(a) for a in rows]


@router.post("/partners/{partner_id}/agreements", status_code=201)
async def add_agreement(
    partner_id: str, body: AgreementCreate, db: Session = Depends(get_db)
):
    _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, partner_id, "Partner")
    agmt = PartnerAgreement(
        agreement_id=str(uuid.uuid4()),
        partner_id=partner_id,
        **body.model_dump(),
        status="draft",
    )
    db.add(agmt)
    db.commit()
    db.refresh(agmt)
    return _agreement_out(agmt)


@router.patch("/agreements/{agreement_id}")
async def update_agreement(
    agreement_id: str, body: AgreementUpdate, db: Session = Depends(get_db)
):
    a = _get_or_404(db, PartnerAgreement, PartnerAgreement.agreement_id, agreement_id, "Agreement")
    updates = body.model_dump(exclude_none=True)
    if "status" in updates and updates["status"] != a.status:
        allowed = _AGREEMENT_TRANSITIONS.get(a.status, [])
        if updates["status"] not in allowed:
            raise HTTPException(
                400,
                f"Cannot transition agreement from '{a.status}' to '{updates['status']}'. "
                f"Allowed: {allowed}",
            )
        if updates["status"] == "signed" and not a.signed_at:
            a.signed_at = datetime.now(timezone.utc)
    for k, v in updates.items():
        setattr(a, k, v)
    a.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(a)
    return _agreement_out(a)


@router.get("/agreements")
async def list_all_agreements(
    partner_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(200, le=500),
    db: Session = Depends(get_db),
):
    q = select(PartnerAgreement).order_by(PartnerAgreement.created_at.desc())
    if partner_id:
        q = q.where(PartnerAgreement.partner_id == partner_id)
    if status:
        q = q.where(PartnerAgreement.status == status)
    return [_agreement_out(a) for a in db.execute(q.limit(limit)).scalars().all()]


# ═══════════════════ Sources ═════════════════════════════════════════════════

@router.get("/sources")
async def list_sources(
    partner_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(200, le=500),
    db: Session = Depends(get_db),
):
    q = select(SourceRegistry).order_by(SourceRegistry.registered_at.desc())
    if partner_id:
        q = q.where(SourceRegistry.partner_id == partner_id)
    if status:
        q = q.where(SourceRegistry.status == status)
    return [_source_out(s) for s in db.execute(q.limit(limit)).scalars().all()]


@router.post("/sources", status_code=201)
async def create_source(body: SourceCreate, db: Session = Depends(get_db)):
    p = _get_or_404(db, PartnerRegistry, PartnerRegistry.partner_id, body.partner_id, "Partner")
    if p.status not in ("approved", "active"):
        raise HTTPException(
            400,
            f"Partner must be in 'approved' or 'active' state to register sources (current: '{p.status}').",
        )
    source = SourceRegistry(
        source_id=str(uuid.uuid4()),
        **body.model_dump(),
        status="pending",
        authorization_status="unauthorized",
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return _source_out(source)


@router.patch("/sources/{source_id}")
async def update_source(
    source_id: str, body: SourceUpdate, db: Session = Depends(get_db)
):
    s = _get_or_404(db, SourceRegistry, SourceRegistry.source_id, source_id, "Source")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(s, k, v)
    s.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(s)
    return _source_out(s)


# ═══════════════════ Authorizations ══════════════════════════════════════════

@router.post("/sources/{source_id}/authorize", status_code=201)
async def authorize_source(
    source_id: str, body: AuthorizationCreate, db: Session = Depends(get_db)
):
    s = _get_or_404(db, SourceRegistry, SourceRegistry.source_id, source_id, "Source")
    if s.authorization_status == "authorized":
        raise HTTPException(400, "Source already authorized. Revoke existing authorization first.")
    auth = PartnerAuthorization(
        auth_id=str(uuid.uuid4()),
        source_id=source_id,
        partner_id=s.partner_id,
        **body.model_dump(),
        status="active",
    )
    db.add(auth)
    s.authorization_status = "authorized"
    s.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(auth)
    return {"auth_id": auth.auth_id, "status": "authorized", "source_id": source_id}


@router.delete("/sources/{source_id}/authorize", status_code=200)
async def revoke_authorization(
    source_id: str,
    reason: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    s = _get_or_404(db, SourceRegistry, SourceRegistry.source_id, source_id, "Source")
    auths = db.execute(
        select(PartnerAuthorization)
        .where(PartnerAuthorization.source_id == source_id, PartnerAuthorization.status == "active")
        .order_by(PartnerAuthorization.granted_at.desc())
    ).scalars().all()
    for a in auths:
        a.status = "revoked"
        a.revoked_at = datetime.now(timezone.utc)
        a.revoked_reason = reason
    s.authorization_status = "revoked"
    s.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "revoked", "source_id": source_id}


# ═══════════════════ Licensing ═══════════════════════════════════════════════

@router.get("/sources/{source_id}/licensing")
async def list_source_licensing(source_id: str, db: Session = Depends(get_db)):
    _get_or_404(db, SourceRegistry, SourceRegistry.source_id, source_id, "Source")
    rows = db.execute(
        select(DataLicensing).where(DataLicensing.source_id == source_id)
        .order_by(DataLicensing.created_at.desc())
    ).scalars().all()
    return [_license_out(l) for l in rows]


@router.post("/sources/{source_id}/licensing", status_code=201)
async def add_licensing(
    source_id: str, body: LicensingCreate, db: Session = Depends(get_db)
):
    s = _get_or_404(db, SourceRegistry, SourceRegistry.source_id, source_id, "Source")
    lic = DataLicensing(
        license_id=str(uuid.uuid4()),
        source_id=source_id,
        partner_id=s.partner_id,
        **body.model_dump(),
        status="active",
    )
    db.add(lic)
    db.commit()
    db.refresh(lic)
    return _license_out(lic)


@router.get("/licenses")
async def list_all_licenses(
    source_id: Optional[str] = None,
    partner_id: Optional[str] = None,
    limit: int = Query(200, le=500),
    db: Session = Depends(get_db),
):
    q = select(DataLicensing).order_by(DataLicensing.created_at.desc())
    if source_id:
        q = q.where(DataLicensing.source_id == source_id)
    if partner_id:
        q = q.where(DataLicensing.partner_id == partner_id)
    return [_license_out(l) for l in db.execute(q.limit(limit)).scalars().all()]


# ═══════════════════ Evidence Comparisons ════════════════════════════════════

@router.get("/evidence-comparisons")
async def list_comparisons(
    limit: int = Query(50, le=200), db: Session = Depends(get_db)
):
    rows = db.execute(
        select(EvidenceComparison).order_by(EvidenceComparison.created_at.desc()).limit(limit)
    ).scalars().all()
    return [_comparison_out(c) for c in rows]


@router.post("/evidence-comparisons", status_code=201)
async def create_comparison(body: ComparisonCreate, db: Session = Depends(get_db)):
    authorized = db.execute(
        select(SourceRegistry).where(SourceRegistry.authorization_status == "authorized")
    ).scalars().all()
    comparison = EvidenceComparison(
        comparison_id=str(uuid.uuid4()),
        query_topic=body.query_topic,
        dimensions=body.dimensions or [
            "efficacy", "safety", "adherence", "tolerability", "quality_of_life"
        ],
        scientific_profile_ids=body.scientific_profile_ids or [],
        community_source_ids=(
            body.community_source_ids
            if body.community_source_ids is not None
            else [s.source_id for s in authorized]
        ),
        comparison_framework="rwe_v1",
        scientific_evidence={},
        community_evidence={},
        status="draft",
    )
    db.add(comparison)
    db.commit()
    db.refresh(comparison)
    return _comparison_out(comparison)


@router.get("/evidence-comparisons/{comparison_id}")
async def get_comparison(comparison_id: str, db: Session = Depends(get_db)):
    c = _get_or_404(
        db, EvidenceComparison,
        EvidenceComparison.comparison_id, comparison_id, "Comparison"
    )
    return _comparison_out(c)


# ═══════════════════ Serialisers ═════════════════════════════════════════════

def _ts(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def _partner_out(p: PartnerRegistry) -> dict:
    return {
        "partner_id": p.partner_id, "name": p.name, "short_name": p.short_name,
        "description": p.description, "partner_type": p.partner_type,
        "integration_type": p.integration_type, "website_url": p.website_url,
        "status": p.status, "data_sharing_agreement": p.data_sharing_agreement,
        "first_contact_date": _ts(p.first_contact_date),
        "last_contact_date": _ts(p.last_contact_date),
        "next_followup_date": _ts(p.next_followup_date),
        "outreach_notes": p.outreach_notes,
        "legal_review_started": _ts(p.legal_review_started),
        "legal_review_completed": _ts(p.legal_review_completed),
        "internal_notes": p.internal_notes, "tags": p.tags,
        "created_at": _ts(p.created_at), "updated_at": _ts(p.updated_at),
    }


def _contact_out(c: PartnerContact) -> dict:
    return {
        "contact_id": c.contact_id, "partner_id": c.partner_id,
        "full_name": c.full_name, "role": c.role, "email": c.email,
        "phone": c.phone, "department": c.department, "is_primary": c.is_primary,
        "notes": c.notes, "last_contacted_at": _ts(c.last_contacted_at),
        "created_at": _ts(c.created_at),
    }


def _agreement_out(a: PartnerAgreement) -> dict:
    return {
        "agreement_id": a.agreement_id, "partner_id": a.partner_id,
        "agreement_type": a.agreement_type, "title": a.title, "status": a.status,
        "version": a.version, "summary": a.summary, "key_terms": a.key_terms,
        "file_reference": a.file_reference,
        "signed_by_partner": a.signed_by_partner, "signed_by_hleo": a.signed_by_hleo,
        "drafted_at": _ts(a.drafted_at), "sent_for_review_at": _ts(a.sent_for_review_at),
        "signed_at": _ts(a.signed_at), "effective_from": _ts(a.effective_from),
        "effective_until": _ts(a.effective_until), "terminated_at": _ts(a.terminated_at),
        "notes": a.notes, "created_at": _ts(a.created_at), "updated_at": _ts(a.updated_at),
    }


def _source_out(s: SourceRegistry) -> dict:
    return {
        "source_id": s.source_id, "partner_id": s.partner_id, "name": s.name,
        "description": s.description, "integration_type": s.integration_type,
        "source_type": s.source_type, "data_category": s.data_category,
        "category": s.category, "runtime_collector": s.runtime_collector,
        "requires_credentials": s.requires_credentials,
        "credentials_env_vars": s.credentials_env_vars or [],
        "connection_spec": s.connection_spec, "evidence_level": s.evidence_level,
        "population_tags": s.population_tags, "geographic_scope": s.geographic_scope,
        "language_codes": s.language_codes, "status": s.status,
        "authorization_status": s.authorization_status,
        "estimated_records": s.estimated_records,
        "record_update_frequency": s.record_update_frequency,
        "last_sync_at": _ts(s.last_sync_at), "registered_at": _ts(s.registered_at),
        "updated_at": _ts(s.updated_at),
    }


def _license_out(l: DataLicensing) -> dict:
    return {
        "license_id": l.license_id, "source_id": l.source_id,
        "partner_id": l.partner_id, "agreement_id": l.agreement_id,
        "license_type": l.license_type, "permitted_uses": l.permitted_uses,
        "restrictions": l.restrictions, "attribution_required": l.attribution_required,
        "attribution_text": l.attribution_text,
        "geographic_restrictions": l.geographic_restrictions,
        "retention_period_days": l.retention_period_days,
        "deletion_on_request": l.deletion_on_request, "status": l.status,
        "valid_from": _ts(l.valid_from), "valid_until": _ts(l.valid_until),
        "notes": l.notes, "created_at": _ts(l.created_at), "updated_at": _ts(l.updated_at),
    }


def _comparison_out(c: EvidenceComparison) -> dict:
    return {
        "comparison_id": c.comparison_id, "query_topic": c.query_topic,
        "comparison_framework": c.comparison_framework, "dimensions": c.dimensions,
        "scientific_profile_ids": c.scientific_profile_ids,
        "community_source_ids": c.community_source_ids,
        "scientific_evidence": c.scientific_evidence,
        "community_evidence": c.community_evidence,
        "agreement_score": c.agreement_score, "divergence_points": c.divergence_points,
        "synthesis": c.synthesis, "status": c.status,
        "created_at": _ts(c.created_at), "updated_at": _ts(c.updated_at),
    }


# ═══════════════════ Helpers ═════════════════════════════════════════════════

def _get_or_404(db, model, col, value, label):
    obj = db.execute(select(model).where(col == value)).scalar_one_or_none()
    if not obj:
        raise HTTPException(404, f"{label} '{value}' not found.")
    return obj
