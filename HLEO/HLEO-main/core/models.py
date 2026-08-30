from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text,
)

from core.database import Base


# ═══════════════════════ PARTNER MANAGEMENT SYSTEM ═══════════════════════════
#
# Lifecycle states
# ── Partner:     proposed→contacted→in_discussion→legal_review→approved→active
#                 or →declined / →disabled (from any state)
# ── Source:      pending→approved→active→paused→suspended
# ── Agreement:   draft→under_review→negotiating→signed→active→expired→terminated
# ── Licensing:   active→expired→revoked
# ── Auth:        active→expired→revoked
#
# Integration types: api | licensed_dataset | rss | manual_import |
#                    research_collaboration
# Partner types:     forum | patient_org | healthcare_community |
#                    research_network
# ─────────────────────────────────────────────────────────────────────────────

class PartnerRegistry(Base):
    """Community partner organisation — complete CRM-style record."""
    __tablename__ = "hleo_partner_registry"

    id             = Column(Integer, primary_key=True, index=True)
    partner_id     = Column(String, unique=True, index=True)    # uuid
    name           = Column(String, nullable=False)
    short_name     = Column(String)
    description    = Column(Text)
    partner_type   = Column(String)    # forum|patient_org|healthcare_community|research_network
    website_url    = Column(String)
    status         = Column(String, default="proposed")  # full 8-stage lifecycle
    integration_type = Column(String)  # api|licensed_dataset|rss|manual_import|research_collaboration
    data_sharing_agreement = Column(Boolean, default=False)
    # Outreach & CRM tracking
    first_contact_date  = Column(DateTime(timezone=True))
    last_contact_date   = Column(DateTime(timezone=True))
    next_followup_date  = Column(DateTime(timezone=True))
    outreach_notes      = Column(Text)
    # Legal review tracking
    legal_review_started   = Column(DateTime(timezone=True))
    legal_review_completed = Column(DateTime(timezone=True))
    # Internal
    internal_notes = Column(Text)
    tags           = Column(JSON)    # ["dermatology","alopecia","EU"]
    created_at     = Column(DateTime(timezone=True),
                            default=lambda: datetime.now(timezone.utc))
    updated_at     = Column(DateTime(timezone=True))


class PartnerContact(Base):
    """Individual contacts at a partner organisation."""
    __tablename__ = "hleo_partner_contacts"

    id             = Column(Integer, primary_key=True, index=True)
    contact_id     = Column(String, unique=True, index=True)
    partner_id     = Column(String, index=True)
    full_name      = Column(String, nullable=False)
    role           = Column(String)    # "CTO", "Data Officer", "Community Manager"
    email          = Column(String)
    phone          = Column(String)
    department     = Column(String)
    is_primary     = Column(Boolean, default=False)
    notes          = Column(Text)
    last_contacted_at = Column(DateTime(timezone=True))
    created_at     = Column(DateTime(timezone=True),
                            default=lambda: datetime.now(timezone.utc))


class PartnerAgreement(Base):
    """Legal agreements and contracts with partner organisations."""
    __tablename__ = "hleo_partner_agreements"

    id             = Column(Integer, primary_key=True, index=True)
    agreement_id   = Column(String, unique=True, index=True)
    partner_id     = Column(String, index=True)
    agreement_type = Column(String)  # mou|data_sharing_agreement|nda|research_collaboration_agreement|api_license
    title          = Column(String)
    status         = Column(String, default="draft")  # draft→under_review→negotiating→signed→active→expired→terminated
    version        = Column(String, default="1.0")
    # Timeline
    drafted_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc))
    sent_for_review_at  = Column(DateTime(timezone=True))
    signed_at           = Column(DateTime(timezone=True))
    effective_from      = Column(DateTime(timezone=True))
    effective_until     = Column(DateTime(timezone=True))
    terminated_at       = Column(DateTime(timezone=True))
    # Content
    summary          = Column(Text)
    key_terms        = Column(JSON)    # [{"term": "...", "value": "..."}]
    file_reference   = Column(String)  # path/to/agreement.pdf
    signed_by_partner = Column(String)
    signed_by_hleo    = Column(String)
    notes            = Column(Text)
    created_at       = Column(DateTime(timezone=True),
                              default=lambda: datetime.now(timezone.utc))
    updated_at       = Column(DateTime(timezone=True))


class DataLicensing(Base):
    """Data licensing and rights management per source."""
    __tablename__ = "hleo_data_licensing"

    id             = Column(Integer, primary_key=True, index=True)
    license_id     = Column(String, unique=True, index=True)
    source_id      = Column(String, index=True)    # → hleo_source_registry
    partner_id     = Column(String, index=True)
    agreement_id   = Column(String, index=True)    # → hleo_partner_agreements
    # License terms
    license_type           = Column(String)   # cc_by|cc_by_nc|proprietary|research_only|open_data|custom
    permitted_uses         = Column(JSON)     # ["research","internal_analytics"]
    restrictions           = Column(JSON)     # ["no_redistribution","aggregated_only"]
    attribution_required   = Column(Boolean, default=True)
    attribution_text       = Column(Text)
    geographic_restrictions = Column(JSON)   # ["EU","US"] or null = global
    # Retention
    retention_period_days  = Column(Integer)  # null = indefinite
    deletion_on_request    = Column(Boolean, default=True)
    # Validity
    status      = Column(String, default="active")  # active|expired|revoked
    valid_from  = Column(DateTime(timezone=True))
    valid_until = Column(DateTime(timezone=True))
    notes       = Column(Text)
    created_at  = Column(DateTime(timezone=True),
                         default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime(timezone=True))


class SourceRegistry(Base):
    """Data source contributed by a partner — technical integration spec."""
    __tablename__ = "hleo_source_registry"

    id               = Column(Integer, primary_key=True, index=True)
    source_id        = Column(String, unique=True, index=True)
    partner_id       = Column(String, index=True)
    name             = Column(String, nullable=False)
    description      = Column(Text)
    integration_type = Column(String)   # api|licensed_dataset|rss|manual_import|research_collaboration
    source_type      = Column(String)   # rest_api|graphql|sftp|email|manual …
    data_category    = Column(String)   # patient_experiences|clinical_outcomes|survey_data|discussion_posts
    # Extensible 4-bucket classification used by the runtime pipeline + UI:
    #   scientific | rwe_experience | pharmacovigilance | other
    category         = Column(String, default="scientific")
    # Collector key that the runtime pipeline dispatches to (e.g. "openfda_faers",
    # "calvizie", "pubmed"). Lets the admin map a registry row to the real
    # collector without storing secrets or URLs.
    runtime_collector   = Column(String)
    requires_credentials = Column(Boolean, default=False)
    credentials_env_vars = Column(JSON)   # list of env-var names needed, e.g. ["REDDIT_CLIENT_ID","REDDIT_CLIENT_SECRET"]
    connection_spec  = Column(JSON)     # field schema — no secrets stored here
    evidence_level   = Column(String)   # anecdotal|observational|survey|registry
    population_tags  = Column(JSON)
    geographic_scope = Column(String)
    language_codes   = Column(JSON)
    status                = Column(String, default="pending")        # pending|approved|active|paused|suspended
    authorization_status  = Column(String, default="unauthorized")   # unauthorized|authorized|expired|revoked
    estimated_records     = Column(Integer)
    record_update_frequency = Column(String)   # daily|weekly|monthly|on_demand
    last_sync_at     = Column(DateTime(timezone=True))
    registered_at    = Column(DateTime(timezone=True),
                              default=lambda: datetime.now(timezone.utc))
    updated_at       = Column(DateTime(timezone=True))


class PartnerAuthorization(Base):
    """Technical authorization record for a specific source — schema only, no secrets."""
    __tablename__ = "hleo_partner_authorizations"

    id                = Column(Integer, primary_key=True, index=True)
    auth_id           = Column(String, unique=True, index=True)
    source_id         = Column(String, index=True)
    partner_id        = Column(String, index=True)
    auth_type         = Column(String)  # api_key|oauth2|basic_auth|ip_whitelist|manual|mtls
    auth_config_schema = Column(JSON)  # field names and types only — actual values live in secrets manager
    scope             = Column(JSON)   # ["read:posts","read:metadata"]
    rate_limit_rpm    = Column(Integer)
    granted_at        = Column(DateTime(timezone=True),
                               default=lambda: datetime.now(timezone.utc))
    expires_at        = Column(DateTime(timezone=True))
    granted_by        = Column(String, default="system")
    revoked_at        = Column(DateTime(timezone=True))
    revoked_reason    = Column(Text)
    status            = Column(String, default="active")  # active|expired|revoked


class EvidenceComparison(Base):
    """Comparison run linking scientific profiles to community source evidence."""
    __tablename__ = "hleo_evidence_comparisons"

    id                   = Column(Integer, primary_key=True, index=True)
    comparison_id        = Column(String, unique=True, index=True)
    query_topic          = Column(String, nullable=False)
    scientific_profile_ids = Column(JSON)   # episode_ids from hleo_clinical_profiles
    community_source_ids   = Column(JSON)   # source_ids from hleo_source_registry
    comparison_framework   = Column(String, default="rwe_v1")
    dimensions             = Column(JSON)   # ["efficacy","safety","adherence","tolerability","quality_of_life"]
    scientific_evidence    = Column(JSON)   # structured findings (populated when run)
    community_evidence     = Column(JSON)   # structured findings (populated when sources active)
    agreement_score        = Column(Float)  # 0–1 concordance (populated after analysis)
    divergence_points      = Column(JSON)
    synthesis              = Column(Text)   # LLM-generated synthesis (future)
    status                 = Column(String, default="draft")  # draft|complete|archived
    created_at  = Column(DateTime(timezone=True),
                         default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime(timezone=True))


class RawSource(Base):
    __tablename__ = "hleo_raw_sources"

    id            = Column(Integer, primary_key=True, index=True)
    episode_id    = Column(String, unique=True, index=True)
    user_id       = Column(String, index=True)
    source_type   = Column(String)          # pubmed | europepmc | clinicaltrials | reddit
    platform      = Column(String)
    external_url  = Column(String, unique=True)
    post_timestamp = Column(DateTime(timezone=True))
    raw_text      = Column(Text)
    ingested_at   = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class ClinicalProfile(Base):
    __tablename__ = "hleo_clinical_profiles"

    id                   = Column(Integer, primary_key=True, index=True)
    episode_id           = Column(String, unique=True, index=True)
    user_id              = Column(String, index=True)       # source platform
    final_category       = Column(String)
    confidence_score     = Column(Float)
    adjudication_required = Column(Boolean, default=False)
    extracted_payload    = Column(JSON)                    # ClinicalProfile dict
    validation_payload   = Column(JSON)                    # metadata: title, abstract_chars, source url
    processed_at         = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class PatientExperience(Base):
    """
    Structured patient-reported experience extracted from Reddit posts.
    One row per Reddit post that passes extraction.
    """
    __tablename__ = "hleo_patient_experiences"

    id               = Column(Integer, primary_key=True, index=True)
    episode_id       = Column(String, unique=True, index=True)
    source_platform  = Column(String, default="reddit")
    source_url       = Column(String)
    author           = Column(String)
    raw_text         = Column(Text)
    extracted_profile = Column(JSON)           # PatientExperienceProfile dict
    query_context    = Column(String)          # search query that surfaced this post
    ingested_at      = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class RWEProfile(Base):
    """FASE 13: Structured RWE profile extracted from RWE items.

    One row per RWE item that passes extraction. Bridges RWE search results
    to Profiles and Testimonianze. Separate from PatientExperience (Reddit
    OAuth) — RWE items come from the RWE pipeline (forums + openFDA).
    """
    __tablename__ = "hleo_rwe_profiles"

    id               = Column(Integer, primary_key=True, index=True)
    episode_id       = Column(String, unique=True, index=True)
    source           = Column(String, index=True)       # reddit|openfda_faers|calvizie|...
    source_type      = Column(String)                   # community_forum|pharmacovigilance
    evidence_tier    = Column(String, default="anecdotal")
    source_url       = Column(String)
    external_id      = Column(String, index=True)
    title            = Column(Text)
    raw_text         = Column(Text)
    extracted_profile = Column(JSON)           # structured RWE profile dict
    treatment        = Column(String)
    condition        = Column(String)
    experience_type  = Column(String, default="discussion")
    query_context    = Column(String)
    language         = Column(String, default="en")
    is_testimonial   = Column(Boolean, default=False)   # curated for Testimonianze
    ingested_at      = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class SourceAttribution(Base):
    """
    Links a clinical profile (or patient experience) to its provenance.
    One row per source for each profile (could be multiple citations).
    """
    __tablename__ = "hleo_source_attributions"

    id                  = Column(Integer, primary_key=True, index=True)
    profile_episode_id  = Column(String, index=True)  # FK to hleo_clinical_profiles.episode_id
    source_type         = Column(String)               # pubmed | europepmc | clinicaltrials | reddit
    source_title        = Column(String)
    source_url          = Column(String)
    external_id         = Column(String)               # PMID / NCT / DOI
    journal             = Column(String)
    pub_year            = Column(String)
    abstract_excerpt    = Column(Text)
    added_at            = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class ChatSession(Base):
    __tablename__ = "hleo_chat_sessions"

    id             = Column(Integer, primary_key=True, index=True)
    session_id     = Column(String, unique=True, index=True)
    title          = Column(String)
    status         = Column(String, default="active")       # "active" | "closed"
    search_query   = Column(Text, nullable=True)            # human-readable query
    search_context = Column(JSON, nullable=True)            # serialised SearchContext dict
    created_at     = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at     = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    closed_at      = Column(DateTime(timezone=True), nullable=True)


class ChatMessage(Base):
    __tablename__ = "hleo_chat_messages"

    id           = Column(Integer, primary_key=True, index=True)
    session_id   = Column(String, index=True)
    role         = Column(String)              # user | assistant
    content      = Column(Text)
    context_used = Column(JSON)               # list of episode_ids used as RAG context
    created_at   = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class AuditLog(Base):
    __tablename__ = "hleo_audit_log"

    id         = Column(Integer, primary_key=True, index=True)
    episode_id = Column(String, index=True)
    action     = Column(String)
    status     = Column(String)
    details    = Column(Text)
    timestamp  = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
