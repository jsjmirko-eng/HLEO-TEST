"""
HLEO v1.0 — FastAPI application
Endpoints:
  GET  /                     → dashboard UI
  GET  /health               → key/version status
  GET  /stats                → DB row counts
  GET  /search?q=&mode=      → scientific (relational, L2) or global (keyword) collect
  POST /pipeline/run?q=&mode=→ collect + LLM-extract articles → DB (scientific: relevant-only)
  POST /synthesis            → Level 3 scientific synthesis (reuses L2 relevant articles, no re-search)
  GET  /rwe/search?q=        → RWE search (Reddit + openFDA FAERS + Calvizie.net) — patient experiences & community
  GET  /profiles?limit=       → saved clinical profiles
  POST /experiences/ingest?q= → collect Reddit + LLM-extract patient experiences → DB
  GET  /experiences?limit=   → saved patient experiences
  POST /assistant/chat       → AI Clinical Assistant (RAG over DB; accepts scientific + RWE context)
  GET  /assistant/sessions/{session_id} → chat history
"""
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List

from fastapi import FastAPI, Depends, Query, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select, func, desc, or_
from sqlalchemy.orm import Session

from core.database import get_db, engine, Base
from core.models import (
    ClinicalProfile, RawSource, AuditLog,
    PatientExperience, SourceAttribution, ChatSession, ChatMessage,
    RWEProfile,
)
from api.partners import router as rwe_router
from api.admin import router as admin_router
from core.orchestrator import QueryOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level orchestrator instance (stateless, safe to share across requests)
_orchestrator = QueryOrchestrator()

# Bounded in-memory translation cache (avoids re-calling LLM for identical text+lang pairs).
# Eviction is FIFO via OrderedDict: oldest entry removed when the limit is hit.
# Limit is intentionally generous (UI translations are small) but prevents unbounded growth.
import collections as _collections
_TRANSLATE_CACHE_MAXSIZE = int(os.getenv("TRANSLATE_CACHE_MAX_KEYS", "256"))
_translate_cache: _collections.OrderedDict = _collections.OrderedDict()

app = FastAPI(title="HLEO API", version="1.0.0")

# Allow browser clients and reverse proxies (and any origin) to send cross-origin
# POST/PATCH/DELETE requests with JSON bodies. The preflight OPTIONS can otherwise
# block fetch() calls from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

Base.metadata.create_all(bind=engine)
from core.migrations import run_schema_upgrades
run_schema_upgrades()
app.include_router(rwe_router)
app.include_router(admin_router)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)


def _article_from_pubmed(item: Any) -> dict:
    return {
        "source":      "pubmed",
        "episode_id":  f"pubmed-{item.pmid}",
        "title":        item.title,
        "abstract":     item.abstract or "",
        "url":          f"https://pubmed.ncbi.nlm.nih.gov/{item.pmid}/",
        "external_id":  item.pmid,
        "journal":      (item.metadata or {}).get("journal", ""),
        "pub_year":     str((item.metadata or {}).get("pubdate", ""))[:4],
        "meta":         item.metadata or {},
    }


def _article_from_europepmc(item: Any) -> dict:
    ep_id = (item.metadata or {}).get("id") or (item.doi or "").replace("/", "-")
    return {
        "source":      "europepmc",
        "episode_id":  f"europepmc-{ep_id}",
        "title":        item.title,
        "abstract":     item.abstract or "",
        "url":          f"https://doi.org/{item.doi}" if item.doi else "",
        "external_id":  item.doi or ep_id,
        "journal":      (item.metadata or {}).get("journal", ""),
        "pub_year":     str(item.year or ""),
        "meta":         item.metadata or {},
    }


def _article_from_clinicaltrials(item: Any) -> dict:
    nct = (item.metadata or {}).get("nct_id", "unknown")
    return {
        "source":      "clinicaltrials",
        "episode_id":  f"clinicaltrial-{nct}",
        "title":        item.title,
        "abstract":     item.abstract or "",
        "url":          f"https://clinicaltrials.gov/study/{nct}" if nct != "unknown" else "",
        "external_id":  nct,
        "journal":      "",
        "pub_year":     str(item.year or ""),
        "meta":         item.metadata or {},
    }


def _article_from_generic(item: Any, source_key: Optional[str] = None) -> dict:
    """Fallback generic mapper from SearchResult-like object to API article dict."""
    title = getattr(item, 'title', '') or (item.get('title') if isinstance(item, dict) else '')
    abstract = getattr(item, 'abstract', '') or (item.get('abstract') if isinstance(item, dict) else '')
    url = getattr(item, 'url', '') or (item.get('url') if isinstance(item, dict) else '')
    doi = getattr(item, 'doi', None) or (item.get('doi') if isinstance(item, dict) else None)
    pmid = getattr(item, 'pmid', None) or (item.get('pmid') if isinstance(item, dict) else None)
    metadata = getattr(item, 'metadata', None) or (item if isinstance(item, dict) else {})
    # episode id fallback
    if pmid:
        episode_id = f"pubmed-{pmid}"
    elif doi:
        episode_id = f"doi-{str(doi).replace('/', '-') }"
    else:
        # try a stable key from metadata or url
        if isinstance(metadata, dict) and metadata.get('id'):
            episode_id = f"item-{metadata.get('id')}"
        elif url:
            episode_id = f"url-{hash(url) & 0xffffffff}"
        else:
            episode_id = f"{source_key or 'generic'}-{abs(hash(title)) % (10**8)}"
    return {
        "source": source_key or (metadata.get('source') if isinstance(metadata, dict) else 'generic'),
        "episode_id": episode_id,
        "title": title,
        "abstract": abstract or "",
        "url": url,
        "external_id": pmid or doi or "",
        "journal": (metadata or {}).get('journal', ''),
        "pub_year": str((getattr(item, 'year', '') or (metadata or {}).get('year', ''))),
        "meta": metadata or {},
    }


def _articles_from_raw(raw: dict) -> list[dict]:
    """Build the article list from a HLEOPipeline.collect() result (global mode).

    Supports dynamic keys returned by HLEOPipeline.collect(). Uses known builders
    for pubmed/europepmc/clinicaltrials and a generic mapper for others.
    """
    articles: list[dict] = []
    builders = {
        'pubmed': _article_from_pubmed,
        'europepmc': _article_from_europepmc,
        'clinicaltrials': _article_from_clinicaltrials,
    }
    for key, items in raw.items():
        if key == 'reddit' or not items:
            continue
        builder = builders.get(key)
        if builder:
            for item in items:
                articles.append(builder(item))
        else:
            for item in items:
                articles.append(_article_from_generic(item, source_key=key))
    return articles


def _articles_from_relational(rel_out: dict, only_relevant: bool = True) -> list[dict]:
    """Build the article list from a RelationalSearch result.

    In scientific mode, profiles are extracted ONLY from articles the relational
    judge labeled `relevant` (the core rule). relevance_label/score/reason are
    carried into validation_payload so each profile stays anchored to the relation.

    This function is defensive: RelationalSearch.search() returns a dict that
    may include non-iterable keys such as 'relation' (ClinicalRelation) and
    'stats'. Iterate only over the known per-source lists to avoid treating
    relational metadata as iterables.
    """
    articles: list[dict] = []
    builders = {
        'pubmed': _article_from_pubmed,
        'europepmc': _article_from_europepmc,
        'clinicaltrials': _article_from_clinicaltrials,
    }

    if not rel_out:
        return articles

    # Preferred: explicit per-source extraction avoids accidentally iterating
    # over the 'relation' or 'stats' objects that appear in rel_out.
    if isinstance(rel_out, dict):
        for source_key in ('pubmed', 'europepmc', 'clinicaltrials'):
            items = rel_out.get(source_key)
            if not items:
                continue
            builder = builders.get(source_key)
            # Normalize single-item responses into a list
            if not isinstance(items, (list, tuple)):
                items = [items]
            for item in items:
                if item is None:
                    continue
                # metadata may be a dict or an attribute on the item
                md = getattr(item, 'metadata', None) or (item if isinstance(item, dict) else {})
                if isinstance(md, dict):
                    label = md.get('relevance_label')
                else:
                    label = None
                if only_relevant and label != 'relevant':
                    continue
                if builder:
                    art = builder(item)
                else:
                    art = _article_from_generic(item, source_key=source_key)
                # Attach provenance/relevance info when available
                try:
                    art['relevance_label'] = md.get('relevance_label') if isinstance(md, dict) else None
                    art['relevance_score'] = md.get('relevance_score') if isinstance(md, dict) else None
                    art['relevance_reason'] = md.get('relevance_reason') if isinstance(md, dict) else None
                except Exception:
                    art['relevance_label'] = art.get('relevance_label')
                    art['relevance_score'] = art.get('relevance_score')
                    art['relevance_reason'] = art.get('relevance_reason')
                articles.append(art)
        return articles

    # Fallback: if rel_out is a list/iterable (legacy shape), map generically
    if isinstance(rel_out, (list, tuple)):
        for item in rel_out:
            try:
                art = _article_from_generic(item, source_key='pubmed')
            except Exception:
                # Best-effort fallback for unknown legacy items
                art = {
                    'source': 'unknown',
                    'episode_id': '',
                    'title': str(item)[:120],
                    'abstract': ''
                }
            articles.append(art)
        return articles

    logger.warning("_articles_from_relational: unexpected rel_out type %s", type(rel_out))
    return articles


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health_check():
    import os
    key = os.getenv("OPENAI_API_KEY", "")
    from core.llm_provider import build_provider, configured_provider_name
    provider = build_provider()
    return {
        "status": "ok",
        "version": "1.0.0",
        "openai_key_set": bool(key),
        "openai_key_prefix": key[:8] + "…" if key else None,
        "llm_provider": provider.name if provider else None,
        "llm_provider_configured": configured_provider_name(),
        "llm_fallback": provider.fallback.name if (provider and provider.fallback) else None,
    }


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    return {
        "clinical_profiles": db.execute(
            select(func.count()).select_from(ClinicalProfile)
        ).scalar(),
        "patient_experiences": db.execute(
            select(func.count()).select_from(PatientExperience)
        ).scalar(),
        "rwe_profiles": db.execute(
            select(func.count()).select_from(RWEProfile)
        ).scalar(),
        "raw_sources": db.execute(
            select(func.count()).select_from(RawSource)
        ).scalar(),
        "source_attributions": db.execute(
            select(func.count()).select_from(SourceAttribution)
        ).scalar(),
        "chat_sessions": db.execute(
            select(func.count()).select_from(ChatSession)
        ).scalar(),
    }


# ── Search (fast, no LLM) ─────────────────────────────────────────────────────

@app.get("/search")
def search(q: str = Query(..., description="Search query"),
           mode: str = Query("scientific", description="Search mode: 'scientific' (relational, Level 2) | 'global' (broad keyword).")):
    """Collect results from all sources.

    Two modes:
      - scientific (default): relational pipeline (Level 2) — AI-extracted
        clinical relation → per-source structured queries → hard filter →
        LLM relational judge re-ranking. Requires OPENAI_API_KEY (503 if absent,
        no silent keyword fallback, so scientific profiles stay relation-pure).
      - global: broad exploratory keyword pipeline (orchestrator + collect),
        no relational filter, all sources including Reddit.

    ``mode=rwe`` is NOT a scientific mode — it must use the dedicated RWE
    pipeline at ``GET /rwe/search``. To prevent silently returning scientific
    results under an RWE label, we reject it here with an explicit 400.
    """
    if mode == "rwe":
        raise HTTPException(
            status_code=400,
            detail="mode='rwe' is not supported on /search. Use GET /rwe/search?q= instead.",
        )

    from core.pipeline import HLEOPipeline

    if mode == "scientific":
        from core.relational_search import RelationalSearch
        rs = RelationalSearch()
        if rs._client is None:
            raise HTTPException(
                status_code=503,
                detail="Scientific mode requires OPENAI_API_KEY (relational search unavailable). Use Global mode.",
            )
        try:
            rel_out = rs.search(q)
        except Exception as exc:
            logger.warning(f"/search scientific: relational search failed ({exc}).")
            raise HTTPException(
                status_code=503,
                detail=f"Scientific search failed: {exc}. Try Global mode.",
            )
        if rel_out is None:
            raise HTTPException(
                status_code=503,
                detail="Scientific mode could not extract a clinical relation. Try rephrasing or use Global mode.",
            )
        # Convert raw items into dictionaries reliably. _to_dict may return a string
        # for certain lightweight objects (e.g. SimpleNamespace used in tests). In
        # that case, fall back to the original item's __dict__ when available.
        raw_pubmed = rel_out.get("pubmed") or []
        if not isinstance(raw_pubmed, (list, tuple)):
            raw_pubmed = [raw_pubmed] if raw_pubmed else []
        pubmed = []
        for item in raw_pubmed:
            d = _to_dict(item)
            if isinstance(d, str) and hasattr(item, "__dict__"):
                try:
                    d = dict(vars(item))
                except Exception:
                    d = {"title": str(item)}
            pubmed.append(d)

        raw_europepmc = rel_out.get("europepmc") or []
        if not isinstance(raw_europepmc, (list, tuple)):
            raw_europepmc = [raw_europepmc] if raw_europepmc else []
        europepmc = []
        for item in raw_europepmc:
            d = _to_dict(item)
            if isinstance(d, str) and hasattr(item, "__dict__"):
                try:
                    d = dict(vars(item))
                except Exception:
                    d = {}
            europepmc.append(d)

        raw_clinicaltrials = rel_out.get("clinicaltrials") or []
        if not isinstance(raw_clinicaltrials, (list, tuple)):
            raw_clinicaltrials = [raw_clinicaltrials] if raw_clinicaltrials else []
        clinicaltrials = []
        for item in raw_clinicaltrials:
            d = _to_dict(item)
            if isinstance(d, str) and hasattr(item, "__dict__"):
                try:
                    d = dict(vars(item))
                except Exception:
                    d = {}
            clinicaltrials.append(d)

        raw_reddit = rel_out.get("reddit") or []
        if not isinstance(raw_reddit, (list, tuple)):
            raw_reddit = [raw_reddit] if raw_reddit else []
        reddit_raw = []
        for p in raw_reddit:
            reddit_raw.append(_to_dict(p))

        # Normalize fields so clients/tests receive strings instead of nulls.
        # Specifically: when PMID is present but url is null, build the canonical
        # PubMed URL. Also ensure doi is an empty string when missing (do not invent).
        for a in pubmed:
            if not isinstance(a, dict):
                continue
            # Ensure url is a string; derive from pmid when available
            if a.get("url") is None or a.get("url") == "":
                pmid = a.get("pmid") or a.get("external_id") or ""
                a["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            if a.get("doi") is None:
                a["doi"] = ""

        for a in europepmc:
            if not isinstance(a, dict):
                continue
            # If doi present but url missing, build DOI link; otherwise ensure empty string
            if a.get("url") is None or a.get("url") == "":
                doi = a.get("doi") or ""
                a["url"] = f"https://doi.org/{doi}" if doi else ""
            if a.get("doi") is None:
                a["doi"] = ""

        for a in clinicaltrials:
            if not isinstance(a, dict):
                continue
            # Use nct_id from meta or nct_id field to build clinicaltrials.gov URL
            if a.get("url") is None or a.get("url") == "":
                nct = (a.get("meta") or {}).get("nct_id") or a.get("nct_id") or ""
                a["url"] = f"https://clinicaltrials.gov/study/{nct}" if nct else ""
            if a.get("doi") is None:
                a["doi"] = ""

        rel = rel_out.get("relation")
        # Normalise relation -> dict form for the orchestration payload
        rel_dict = None
        if rel is None:
            rel_dict = {}
        elif hasattr(rel, "to_dict"):
            try:
                rel_dict = rel.to_dict()
            except Exception:
                rel_dict = _to_dict(rel)
        elif isinstance(rel, dict):
            rel_dict = rel
        else:
            rel_dict = _to_dict(rel)

        # Determine search_query robustly
        search_query = q
        if isinstance(rel, dict):
            search_query = rel.get("scientific_query") or q
        else:
            try:
                search_query = getattr(rel, "scientific_query", q) or q
            except Exception:
                search_query = q

        orch_dict = {
            "original_query":      q,
            "search_query":        search_query,
            "detected_language":   "",
            "translation_applied": True,
            "confidence":          1.0,
            "relation":            rel_dict,
            "retrieval_stats":     rel_out.get("stats"),
        }
        return {
            "query":          q,
            "orchestration":  orch_dict,
            "llm_extraction": True,
            "totals": {
                "pubmed":         len(pubmed),
                "europepmc":      len(europepmc),
                "clinicaltrials": len(clinicaltrials),
                "reddit":         len(reddit_raw),
            },
            "pubmed": pubmed, "europepmc": europepmc,
            "clinicaltrials": clinicaltrials, "reddit": reddit_raw,
        }

    # ── Global mode: plain keyword pipeline (orchestrator + collect) ─────────
    orch = _orchestrator.process(q)
    pipeline = HLEOPipeline()
    raw = pipeline.collect(orch.search_query)

    # Convert raw results (which can contain dynamic keys) into plain dicts
    converted: dict = {}
    for key, items in (raw or {}).items():
        converted[key] = [_to_dict(a) for a in (items or [])]

    # Ensure legacy keys are present for backward compatibility
    for k in ("pubmed", "europepmc", "clinicaltrials", "reddit"):
        if k not in converted:
            converted[k] = []

    totals = {
        "pubmed": len(converted.get("pubmed", [])),
        "europepmc": len(converted.get("europepmc", [])),
        "clinicaltrials": len(converted.get("clinicaltrials", [])),
        "reddit": len(converted.get("reddit", [])),
    }

    resp = {
        "query": q,
        "orchestration": orch.to_dict(),
        "llm_extraction": pipeline.extractor.client is not None,
        "totals": totals,
    }

    # Attach all per-source arrays to the response (including dynamic ones)
    for key, items in converted.items():
        resp[key] = items

    return resp


# ── Article pipeline ──────────────────────────────────────────────────────────

@app.post("/pipeline/run")
def run_pipeline(q: str = Query(...), db: Session = Depends(get_db),
                 mode: str = Query("scientific", description="Extraction mode: 'scientific' (relational, relevant-only) | 'global' (broad, all articles)."),
                 max_results: Optional[int] = Query(None, description="Optional cap on number of articles to process (testing)")):
    """
    Full article pipeline (NO persistent storage of extracted profiles by default):
    1. Collect from PubMed, EuropePMC, ClinicalTrials (Reddit only in global)
    2. LLM-extract a ClinicalProfile from each abstract
    3. RETURN extracted profiles in the response (do NOT save them to the DB)

    Modes:
      - scientific (default): uses RelationalSearch (Level 2) and extracts profiles
        ONLY from articles judged `relevant` by the relational judge. Requires
        OPENAI_API_KEY.
      - global: broad keyword pipeline (orchestrator + collect), extracts from all
        retrieved articles.

    NOTE: This endpoint no longer persists ClinicalProfile/SourceAttribution.
    """
    from core.article_extractor import ArticleExtractor

    extractor = ArticleExtractor()
    if extractor.client is None:
        return {"error": "LLM configuration is not set or incomplete — cannot run extraction."}

    import datetime as _dt  # local alias to avoid shadowing module-level names

    # ── Retrieval dispatch (mode-aware) ──────────────────────────────────────
    rel_dict: Optional[dict] = None   # relation preserved from scientific search
    if mode == "scientific":
        from core.relational_search import RelationalSearch
        rs = RelationalSearch()
        if rs._client is None:
            return {"error": "Scientific mode requires a valid LLM configuration."}
        try:
            rel_out = rs.search(q)
        except Exception as exc:
            logger.warning(f"/pipeline/run scientific: relational search failed ({exc}).")
            return {"error": f"Scientific search failed: {exc}. Try Global mode."}
        if rel_out is None:
            return {"error": "Scientific mode could not extract a clinical relation. Try rephrasing or use Global mode."}
        rel = rel_out["relation"]
        rel_dict = rel.to_dict()
        orch_search_query = rel.scientific_query or q
        # Scientific rule: profiles only from articles judged 'relevant'.
        articles = _articles_from_relational(rel_out, only_relevant=True)
        if not articles:
            return {
                "query":           q,
                "orchestration":   {"original_query": q, "search_query": orch_search_query,
                                    "relation": rel_dict, "translation_applied": True},
                "processed":       0, "saved": 0, "already_existed": 0, "errors": 0,
                "episode_ids":     [], "results": [], "error_details": [],
                "warning":         "No articles judged relevant; no profiles extracted.",
            }
    else:
        from core.pipeline import HLEOPipeline
        orch = _orchestrator.process(q)
        orch_search_query = orch.search_query
        pipeline = HLEOPipeline()
        raw = pipeline.collect(orch_search_query)
        articles = _articles_from_raw(raw)

    # Optional testing cap: restrict number of articles processed to speed up E2E in CI/dev
    if max_results and isinstance(articles, list) and len(articles) > max_results:
        articles = articles[:max_results]

    # ── Phase 1: Pre-checks (no DB lookups for previous results) ─────────────
    pre_results: dict[int, dict] = {}        # index → resolved entry (skip cases)
    needs_llm:   list[tuple[int, dict]] = [] # (index, art) requiring extraction

    for i, art in enumerate(articles):
        episode_id = art["episode_id"]

        # Skip items with no abstract (can't extract)
        if not art.get("abstract"):
            pre_results[i] = {
                "_is_error":  True,
                "episode_id": episode_id,
                "error":      "No abstract available — skipped.",
            }
            continue

        needs_llm.append((i, art))

    # ── Phase 2: Parallel LLM extraction (unchanged) ────────────────────────
    from core.llm_limits import get_limits as _get_limits
    _MAX_WORKERS = _get_limits().pipeline_max_workers
    llm_results: dict[int, tuple] = {}

    if needs_llm:
        def _extract_one(idx_art: tuple) -> tuple:
            idx, art = idx_art
            try:
                payload = extractor.extract(
                    title=art["title"],
                    abstract=art["abstract"],
                    source=art["source"],
                )
                return idx, "ok", payload, None
            except Exception as exc:
                return idx, "error", None, str(exc)

        logger.info(
            "Extraction | %d articles → parallel (max_workers=%d)",
            len(needs_llm), _MAX_WORKERS,
        )
        t_llm_start = _dt.datetime.now(_dt.timezone.utc)
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_extract_one, item): item[0] for item in needs_llm}
            for fut in as_completed(futures):
                idx, status, payload, err = fut.result()
                llm_results[idx] = (status, payload, err)
                if status == "error":
                    logger.error(
                        "Extraction failed [%s]: %s",
                        articles[idx]["episode_id"], err,
                    )
                else:
                    logger.info("Extracted %s", articles[idx]["episode_id"])
        t_llm_elapsed = (_dt.datetime.now(_dt.timezone.utc) - t_llm_start).total_seconds()
        logger.info(
            "Extraction | done in %.1fs (%d ok, %d failed)",
            t_llm_elapsed,
            sum(1 for v in llm_results.values() if v[0] == "ok"),
            sum(1 for v in llm_results.values() if v[0] == "error"),
        )

    # ── Phase 3: Build results for response (NO DB writes) ────────────────
    saved = []
    errors = []

    for i, art in enumerate(articles):
        episode_id = art["episode_id"]

        if i in pre_results:
            entry = pre_results[i]
            if entry.get("_is_error"):
                errors.append({"episode_id": entry["episode_id"], "error": entry["error"]})
            else:
                saved.append(entry)
            continue

        if i not in llm_results:
            errors.append({"episode_id": episode_id, "error": "Extraction result missing."})
            continue

        status, payload, err = llm_results[i]
        if status == "error":
            errors.append({"episode_id": episode_id, "error": err})
            continue

        # Build in-memory result entry (no DB persistence)
        entry = {
            "episode_id": episode_id,
            "status": "extracted",
            "source": art.get("source"),
            "title": art.get("title"),
            "profile": payload,
            "validation_payload": {
                "source": art.get("source"),
                "title": art.get("title"),
                "url": art.get("url"),
                "abstract_chars": len(art.get("abstract") or ""),
                "journal": art.get("journal"),
                "pub_year": art.get("pub_year"),
                "meta": art.get("meta", {}),
                "relevance_label": art.get("relevance_label"),
                "relevance_score": art.get("relevance_score"),
                "relevance_reason": art.get("relevance_reason"),
            },
        }
        saved.append(entry)

    # Flat list of ALL episode_ids processed — used by frontend to scope the profiles view
    all_episode_ids = [s["episode_id"] for s in saved]

    # Build orchestration dict for the response (mode-aware).
    if mode == "scientific":
        orchestration_out = {
            "original_query":      q,
            "search_query":        orch_search_query,
            "detected_language":   "",
            "translation_applied": True,
            "relation":            rel_dict,
        }
    else:
        orchestration_out = orch.to_dict()

    import uuid
    import datetime as _dt
    from core.temp_store import temp_store

    # Persist results only in the ephemeral temp-store (no DB writes)
    search_id = str(uuid.uuid4())
    temp_payload = {
        "type": "clinical_profiles",
        "query": q,
        "orchestration": orchestration_out,
        "results": saved,
        "episode_ids": all_episode_ids,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    temp_store.set(search_id, temp_payload)

    return {
        "query":           q,
        "orchestration":   orchestration_out,
        "processed":       len(articles),
        "saved":           len(saved),
        "already_existed": 0,
        "errors":          len(errors),
        "episode_ids":     all_episode_ids,
        "results":         saved,
        "error_details":   errors,
        "search_id":       search_id,
    }


def _get_attribution(db: Session, episode_id: str) -> Optional[dict]:
    attr = db.execute(
        select(SourceAttribution).where(SourceAttribution.profile_episode_id == episode_id)
    ).scalar_one_or_none()
    if not attr:
        return None
    return {
        "source_type":  attr.source_type,
        "source_title": attr.source_title,
        "source_url":   attr.source_url,
        "external_id":  attr.external_id,
        "journal":      attr.journal,
        "pub_year":     attr.pub_year,
        "abstract_excerpt": attr.abstract_excerpt,
    }


# ── Clinical profiles ─────────────────────────────────────────────────────────

@app.get("/profiles")
def list_profiles(
    limit:       int           = Query(20, ge=1, le=100),
    episode_ids: Optional[str] = Query(None),   # comma-separated; filters to current search
    search_id: Optional[str] = Query(None, description="Optional ephemeral search_id returned by /pipeline/run"),
    db:          Session       = Depends(get_db),
):
    """Return saved clinical profiles with source attribution.

    When `episode_ids` is supplied (comma-separated), only those profiles are
    returned — this powers the per-search isolation in the Profiles view.
    If `search_id` is provided and references an ephemeral result, return that
    transient result set instead of reading the persistent DB.
    """
    id_filter = [eid.strip() for eid in episode_ids.split(",") if eid.strip()] \
                if episode_ids else None

    # Ephemeral search path: prefer transient store when search_id provided
    if search_id:
        from core.temp_store import temp_store
        data = temp_store.get(search_id)
        if data and data.get("type") == "clinical_profiles":
            saved = data.get("results", [])
            mapped = []
            for s in saved[:limit]:
                vp = s.get("validation_payload", {}) or {}
                prof = s.get("profile", {}) or {}
                mapped.append({
                    "id": None,
                    "episode_id": s.get("episode_id"),
                    "user_id": s.get("source") or vp.get("source"),
                    "final_category": None,
                    "confidence_score": None,
                    "adjudication_required": False,
                    "processed_at": None,
                    "title": vp.get("title", ""),
                    "source": vp.get("source", s.get("source", "")),
                    "url": vp.get("url", ""),
                    "journal": vp.get("journal", ""),
                    "pub_year": vp.get("pub_year", ""),
                    "profile": prof,
                    "attribution": None,
                })
            return {"total": len(saved), "profiles": mapped}

    q = select(ClinicalProfile).order_by(desc(ClinicalProfile.processed_at))
    if id_filter is not None:
        q = q.where(ClinicalProfile.episode_id.in_(id_filter))
    rows = db.execute(q.limit(limit)).scalars().all()

    result = []
    for r in rows:
        vp = r.validation_payload or {}
        attr = _get_attribution(db, r.episode_id)
        result.append({
            "id":                   r.id,
            "episode_id":           r.episode_id,
            "user_id":              r.user_id,
            "final_category":       r.final_category,
            "confidence_score":     r.confidence_score,
            "adjudication_required": r.adjudication_required,
            "processed_at":         r.processed_at.isoformat() if r.processed_at else None,
            "title":    vp.get("title", ""),
            "source":   vp.get("source", r.user_id),
            "url":      vp.get("url", ""),
            "journal":  vp.get("journal", ""),
            "pub_year": vp.get("pub_year", ""),
            "profile":  r.extracted_payload,
            "attribution": attr,
        })

    # Total count respects the same filter so the badge is accurate
    total_q = select(func.count()).select_from(ClinicalProfile)
    if id_filter is not None:
        total_q = total_q.where(ClinicalProfile.episode_id.in_(id_filter))

    return {
        "total":    db.execute(total_q).scalar(),
        "profiles": result,
    }


# ── Patient experiences ───────────────────────────────────────────────────────

@app.post("/experiences/ingest")
def ingest_experiences(q: str = Query(...), db: Session = Depends(get_db)):
    """
    Collect Reddit posts via PRAW OAuth, LLM-extract patient experiences, save to DB.

    Always returns a structured response including:
      reddit_status  — ok | no_credentials | auth_error | rate_limited | no_results | network_error
      reddit_reason  — human-readable explanation of the status
    """
    from collectors.reddit import RedditCollector, STATUS_OK, STATUS_NO_CREDENTIALS
    from core.patient_extractor import PatientExperienceExtractor

    extractor = PatientExperienceExtractor()
    if extractor.client is None:
        return {
            "query": q, "collected": 0, "saved": 0, "errors": 0, "results": [],
            "reddit_status": "no_llm_config",
            "reddit_reason": "LLM configuration is not set or incomplete — extraction is disabled.",
        }

    # ── Orchestrate: detect language, translate to scientific English if needed ──
    orch = _orchestrator.process(q)

    # ── Collect from Reddit via PRAW ────────────────────────────────────────
    collector = RedditCollector()
    raw_reddit, reddit_status, reddit_reason = collector.search_with_status(
        orch.search_query, limit=15
    )
    logger.info(f"Reddit [{reddit_status}] for '{orch.search_query}': {reddit_reason}")

    if reddit_status != STATUS_OK:
        return {
            "query":          q,
            "orchestration":  orch.to_dict(),
            "collected":      0,
            "saved":          0,
            "already_existed": 0,
            "errors":         0,
            "results":        [],
            "error_details":  [],
            "reddit_status":  reddit_status,
            "reddit_reason":  reddit_reason,
        }

    # ── Extract and save ────────────────────────────────────────────────────
    saved  = []
    errors = []

    seen_urls = set()

    for post in raw_reddit:
        episode_id = f"reddit-exp-{abs(hash(post.url))}"

        if episode_id in seen_urls:
            saved.append({"episode_id": episode_id, "status": "skipped", "title": post.title})
            continue
        seen_urls.add(episode_id)

        body = (post.text or "").strip()
        if len(body) < 50:
            errors.append({"episode_id": episode_id, "error": "Post body too short — skipped."})
            continue

        try:
            profile = extractor.extract(
                title=post.title,
                text=body,
                author=post.author or "",
                url=post.url or "",
            )

            # Build in-memory patient experience result (no DB persistence)
            saved.append({
                "episode_id": episode_id,
                "status": "extracted",
                "title": post.title,
                "url": post.url,
                "profile": profile,
            })
            logger.info(f"Extracted patient experience {episode_id}")

        except Exception as exc:
            logger.exception(f"Failed to extract experience {episode_id}: {exc}")
            errors.append({"episode_id": episode_id, "error": str(exc)})

    n_saved = len([s for s in saved if s.get('status') in ('saved','extracted')])
    return {
        "query":           q,
        "orchestration":   orch.to_dict(),
        "collected":       len(raw_reddit),
        "saved":           n_saved,
        "already_existed": len([s for s in saved if s["status"] == "already_exists"]),
        "errors":          len(errors),
        "results":         saved,
        "error_details":   errors,
        "reddit_status":   STATUS_OK,
        "reddit_reason":   f"Retrieved {len(raw_reddit)} post(s); {n_saved} new experience(s) saved.",
    }


@app.get("/experiences")
def list_experiences(
    limit: int = Query(20, ge=1, le=100),
    db:    Session = Depends(get_db),
):
    rows = db.execute(
        select(PatientExperience)
        .order_by(desc(PatientExperience.ingested_at))
        .limit(limit)
    ).scalars().all()

    return {
        "total": db.execute(select(func.count()).select_from(PatientExperience)).scalar(),
        "experiences": [
            {
                "id":           r.id,
                "episode_id":   r.episode_id,
                "source_url":   r.source_url,
                "query_context": r.query_context,
                "ingested_at":  r.ingested_at.isoformat() if r.ingested_at else None,
                "profile":      r.extracted_profile,
            }
            for r in rows
        ],
    }


# ── RWE search (Real World Evidence — patient experiences & community) ────────
#
# Independent from the scientific /search pipeline. Collects from Reddit
# (PRAW OAuth2) and openFDA FAERS (official API, no key required) only —
# sources that require authorization are registered in /rwe/sources but
# never scraped. RWE items are stamped evidence_tier=anecdotal /
# spontaneous_report and never presented as clinical evidence.

@app.get("/rwe/search")
def rwe_search(
    q: str = Query(..., description="RWE search query"),
    limit: int = Query(30, ge=1, le=400),
    page: int = Query(1, ge=1),
    search_id: Optional[str] = Query(None),
    sources: Optional[str] = Query(
        None, description="Comma-separated subset: reddit,openfda_faers,calvizie,hairlosstalk,hairlossexperiences,maladiesrares"
    ),
):
    """Run RWE once, then serve cached final results in pages of 30."""
    from core.rwe.pipeline import RWEPipeline
    from core.temp_store import temp_store

    page_size = 30
    cached = temp_store.get(search_id) if search_id else None
    if cached is None:
        src_list = [s.strip() for s in sources.split(",")] if sources else None
        result = RWEPipeline().search(q, limit=400, sources=src_list)
        body = result.model_dump()
        search_id = str(uuid.uuid4())
        temp_store.set(search_id, body)
        body = dict(body)  # page on a copy; cache keeps the full result
    else:
        body = dict(cached)

    all_items = body.get("items", [])[:400]
    total = len(all_items)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    start = (page - 1) * page_size
    body["items"] = all_items[start:start + page_size]
    body["pagination"] = {
        "search_id": search_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
        "max_results": 400,
    }
    return body


# ── FASE 13-14: RWE Profiles + Testimonianze ────────────────────────────────

class RWEExtractRequest(BaseModel):
    """FASE 13: extract a profile from a single RWE item."""
    title: str = ""
    text: str = ""
    source: str = ""
    source_type: str = ""
    evidence_tier: str = "anecdotal"
    source_url: str = ""
    external_id: str = ""
    treatment: str = ""
    condition: str = ""
    experience_type: str = "discussion"
    language: str = "en"
    query_context: str = ""


@app.post("/rwe/extract")
def rwe_extract_profile(body: RWEExtractRequest, db: Session = Depends(get_db)):
    """FASE 13: LLM-extract a structured profile from a single RWE item.

    NOTE: Under the new architecture, extracted RWE profiles are NOT persisted
    by default. This endpoint returns the extracted profile in the response
    so the frontend can use it as part of the current search context.
    """
    import os, hashlib

    from core.llm_provider import llm_available
    if not llm_available():
        return {"error": "LLM configuration is not set or incomplete."}

    from core.rwe.profile_extractor import RWEProfileExtractor
    extractor = RWEProfileExtractor()
    try:
        profile = extractor.extract(
            title=body.title,
            text=body.text,
            source=body.source,
            source_type=body.source_type,
            treatment=body.treatment,
            condition=body.condition,
        )
    except RuntimeError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception(f"/rwe/extract failed — {exc}")
        return {"error": f"RWE extraction failed: {exc}"}

    dedup_key = body.external_id or hashlib.md5(
        f"{body.source}:{body.external_id or body.title}:{(body.text or '')[:120]}".encode()
    ).hexdigest()[:16]
    episode_id = f"rwe-{dedup_key}"

    # Return extracted profile without persisting to DB
    return {
        "episode_id": episode_id,
        "extracted_profile": profile,
        "status": "extracted",
        "source": body.source,
    }


# ── FASE 13b: Batch extraction from real RWE search items ────────────────────

class RWEBatchExtractItem(BaseModel):
    """One RWE item from the current RWE search, sent by the frontend.

    These are the REAL items returned by /rwe/search — not a re-collection
    of Reddit. The backend extracts a structured profile from each and
    returns the extracted profiles in the response. Persistence of RWEProfile
    rows is an explicit operation and is NOT performed automatically by this
    endpoint.
    All fields are Optional so that null values from /rwe/search are accepted.
    """
    source: Optional[str] = ""
    source_type: Optional[str] = ""
    evidence_tier: Optional[str] = "anecdotal"
    collection_method: Optional[str] = ""
    source_url: Optional[str] = ""
    external_id: Optional[str] = ""
    title: Optional[str] = ""
    text: Optional[str] = ""
    date: Optional[str] = ""
    language: Optional[str] = "en"
    topic: Optional[str] = ""
    treatment: Optional[str] = ""
    condition: Optional[str] = ""
    experience_type: Optional[str] = "discussion"

    model_config = {"extra": "ignore"}


class RWEBatchExtractRequest(BaseModel):
    """Batch of RWE items to extract experiences from."""
    query: Optional[str] = ""
    items: List[RWEBatchExtractItem] = []

    model_config = {"extra": "ignore"}


@app.post("/rwe/extract-batch")
def rwe_extract_batch(body: RWEBatchExtractRequest, db: Session = Depends(get_db)):
    """Extract structured experiences from the REAL RWE items in the current
    RWE search.

    Does NOT re-collect Reddit. Accepts the items returned by /rwe/search and
    runs the RWE profile extractor on each, returning a status report
    (found/extracted/updated/skipped/errors). This endpoint does NOT persist
    RWEProfile rows automatically; persistence must be explicit when required.
    When no items are provided, returns a clear message and does NOT simulate
    any extraction.
    """
    import os, hashlib

    found = len(body.items)
    if found == 0:
        return {
            "query": body.query,
            "rwe_records_found": 0,
            "experiences_extracted": 0,
            "profiles_created": 0,
            "profiles_updated": 0,
            "skipped": 0,
            "errors": 0,
            "error_details": [],
            "message": "Non sono disponibili record RWE da estrarre.",
        }

    from core.llm_provider import llm_available
    if not llm_available():
        return {
            "query": body.query,
            "rwe_records_found": found,
            "experiences_extracted": 0,
            "profiles_created": 0,
            "profiles_updated": 0,
            "skipped": 0,
            "errors": found,
            "error_details": [{"error": "LLM configuration is not set or incomplete."}],
            "message": "Configurazione LLM non impostata o incompleta — estrazione LLM non disponibile.",
        }

    from core.rwe.profile_extractor import RWEProfileExtractor
    extractor = RWEProfileExtractor()

    created = 0
    updated = 0
    skipped = 0
    errors = 0
    error_details = []
    results = []

    seen_keys = set()

    for it in body.items:
        # Dedup by (source, external_id) — if external_id missing, fall back to
        # a content hash so the same item isn't extracted twice.
        dedup_key = it.external_id or hashlib.md5(
            f"{it.source}:{it.title}:{(it.text or '')[:120]}".encode()
        ).hexdigest()[:16]

        if dedup_key in seen_keys:
            skipped += 1
            results.append({"episode_id": f"rwe-{dedup_key}", "source": it.source, "status": "skipped"})
            continue
        seen_keys.add(dedup_key)

        # Skip near-empty items — nothing to extract.
        if not (it.title or "").strip() and not (it.text or "").strip():
            skipped += 1
            error_details.append({"source": it.source, "error": "empty content — skipped."})
            continue

        try:
            profile = extractor.extract(
                title=it.title,
                text=it.text,
                source=it.source,
                source_type=it.source_type,
                treatment=it.treatment,
                condition=it.condition,
            )
        except RuntimeError as exc:
            errors += 1
            error_details.append({"source": it.source, "error": str(exc)})
            continue
        except Exception as exc:
            errors += 1
            error_details.append({"source": it.source, "error": str(exc)})
            continue

        episode_id = f"rwe-{dedup_key}"
        results.append({
            "episode_id": episode_id,
            "source": it.source,
            "status": "extracted",
            "profile": profile,
        })
        created += 1

    import uuid
    import datetime as _dt
    from core.temp_store import temp_store

    search_id = str(uuid.uuid4())
    temp_payload = {
        "type": "rwe_profiles",
        "query": body.query,
        "results": results,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    temp_store.set(search_id, temp_payload)

    return {
        "query": body.query,
        "rwe_records_found": found,
        "experiences_extracted": created,
        "profiles_created": created,
        "profiles_updated": updated,
        "skipped": skipped,
        "errors": errors,
        "error_details": error_details,
        "results": results,
        "message": (
            f"Estratte {created} esperienze da {found} record RWE "
            f"({skipped} saltati, {errors} errori)."
            if found else "Non sono disponibili record RWE da estrarre."
        ),
        "search_id": search_id,
    }


@app.get("/rwe/profiles")
def list_rwe_profiles(
    db: Session = Depends(get_db),
    source: Optional[str] = None,
    treatment: Optional[str] = None,
    limit: int = 50,
    search_id: Optional[str] = Query(None, description="Optional ephemeral search_id returned by /rwe/extract-batch"),
):
    """FASE 13: list stored RWE profiles.

    When `search_id` is supplied and references an ephemeral result, return
    the transient result set stored in the temp-store instead of reading the
    persistent RWEProfile table.
    """
    if search_id:
        from core.temp_store import temp_store
        data = temp_store.get(search_id)
        if data and data.get("type") == "rwe_profiles":
            saved = data.get("results", [])[:limit]
            mapped = []
            for s in saved:
                mp = {
                    "episode_id": s.get("episode_id"),
                    "source": s.get("source"),
                    "source_type": s.get("source_type"),
                    "evidence_tier": s.get("evidence_tier"),
                    "title": s.get("title"),
                    "treatment": s.get("treatment"),
                    "condition": s.get("condition"),
                    "experience_type": s.get("experience_type"),
                    "is_testimonial": False,
                    "extracted_profile": s.get("profile", {}),
                    "source_url": s.get("source_url"),
                    "external_id": s.get("external_id"),
                    "query_context": s.get("query_context"),
                    "language": s.get("language"),
                    "ingested_at": s.get("created_at"),
                }
                mapped.append(mp)
            return {"count": len(saved), "profiles": mapped}

    q = db.query(RWEProfile)
    if source:
        q = q.filter(RWEProfile.source == source)
    if treatment:
        q = q.filter(RWEProfile.treatment.ilike(f"%{treatment}%"))
    rows = q.order_by(RWEProfile.ingested_at.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "profiles": [{
            "episode_id": r.episode_id,
            "source": r.source,
            "source_type": r.source_type,
            "evidence_tier": r.evidence_tier,
            "title": r.title,
            "treatment": r.treatment,
            "condition": r.condition,
            "experience_type": r.experience_type,
            "is_testimonial": r.is_testimonial,
            "extracted_profile": r.extracted_profile,
            "source_url": r.source_url,
            "external_id": r.external_id,
            "query_context": r.query_context,
            "language": r.language,
            "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
        } for r in rows],
    }


@app.get("/rwe/testimonianze")
def list_rwe_testimonianze(
    db: Session = Depends(get_db),
    treatment: Optional[str] = None,
    limit: int = 20,
):
    """FASE 14: curated testimonials (is_testimonial=True) derived from RWE profiles.

    A testimonial is a community_forum RWE profile curated for display.
    FAERS records are excluded — spontaneous reports are NOT testimonials.
    """
    q = db.query(RWEProfile).filter(
        RWEProfile.is_testimonial.is_(True),
        RWEProfile.source_type == "community_forum",
    )
    if treatment:
        q = q.filter(RWEProfile.treatment.ilike(f"%{treatment}%"))
    rows = q.order_by(RWEProfile.ingested_at.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "testimonials": [{
            "episode_id": r.episode_id,
            "source": r.source,
            "title": r.title,
            "treatment": r.treatment,
            "condition": r.condition,
            "extracted_profile": r.extracted_profile,
            "source_url": r.source_url,
            "language": r.language,
        } for r in rows],
    }


@app.post("/rwe/testimonianze/{episode_id}/curate")
def curate_testimonial(episode_id: str, db: Session = Depends(get_db)):
    """FASE 14: mark an RWE profile as a curated testimonial."""
    row = db.query(RWEProfile).filter(RWEProfile.episode_id == episode_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="RWE profile not found.")
    if row.source_type != "community_forum":
        raise HTTPException(
            status_code=400,
            detail="Only community_forum profiles can be testimonials "
                    "(FAERS records are not testimonials)."
        )
    row.is_testimonial = True
    db.commit()
    return {"episode_id": episode_id, "is_testimonial": True}


# ── AI Clinical Assistant ─────────────────────────────────────────────────────

class SearchArticleCtx(BaseModel):
    """One article from the active search, sent by the frontend."""
    source: str                     # "pubmed" | "europepmc" | "clinicaltrials"
    title: str
    abstract: Optional[str] = ""
    url: Optional[str] = ""
    # Bibliographic identifiers — used by the AI for accurate Key Studies citations
    pmid: Optional[str] = ""
    doi: Optional[str] = ""
    nct_id: Optional[str] = ""
    year: Optional[str] = ""
    journal: Optional[str] = ""

class RWEItemCtx(BaseModel):
    """One RWE item forwarded by the frontend with a chat message.

    RWE is kept strictly separate from scientific evidence: ``source_type``
    and ``evidence_tier`` let the Assistant distinguish testimonials /
    pharmacovigilance reports from clinical studies.
    """
    source: str
    source_type: str
    evidence_tier: str = "anecdotal"
    collection_method: str = "official_api"
    source_url: Optional[str] = ""
    external_id: Optional[str] = ""
    title: str = ""
    text: str = ""
    date: Optional[str] = ""
    language: str = "en"
    topic: str = ""
    treatment: Optional[str] = ""
    condition: Optional[str] = ""
    experience_type: str = "discussion"
    relevance: str = "unknown"
    relevance_reason: Optional[str] = ""
    privacy_status: str = "redacted"
    # Search-engine provenance (Phase: RWE Search Engine)
    matched_query: Optional[str] = ""
    matched_query_type: Optional[str] = ""
    source_language: str = "en"
    relevance_score: float = 0.0
    match_reason: Optional[str] = ""
    metadata: dict = {}

class SearchContext(BaseModel):
    """
    Active search context forwarded by the frontend with every chat message.
    Populated by the orchestrator output + raw collector results.

    ``articles`` is the scientific evidence; ``rwe_evidence`` is the optional
    RWE evidence. They are kept separate so the Assistant never conflates a
    testimonial with a clinical study.
    """
    original_query: str
    search_query: str               # English query actually sent to collectors
    detected_language: str          # ISO-639-1 code from orchestrator
    articles: List[SearchArticleCtx] = []
    rwe_evidence: List[RWEItemCtx] = []   # Feature: RWE convergence

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    language: Optional[str] = "en"         # ISO 639-1 code, e.g. "en" / "it"
    search_context: Optional[SearchContext] = None   # Feature 002: active search


# ── Level 3: Scientific Synthesis (reuses Level 2 results, no re-search) ──────

class SynthesisArticle(BaseModel):
    """One article already judged relevant by the Level 2 relational judge."""
    source: str
    title: str
    abstract: Optional[str] = ""
    url: Optional[str] = ""
    pmid: Optional[str] = ""
    doi: Optional[str] = ""
    nct_id: Optional[str] = ""
    year: Optional[str] = ""
    journal: Optional[str] = ""
    relevance_label: Optional[str] = ""
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = ""


class SynthesisRelation(BaseModel):
    """The ClinicalRelation extracted by Level 2 (passed through from /search)."""
    original_query: str = ""
    agent: dict = {}
    event: dict = {}
    manifestation: dict = {}
    temporal: str = ""
    relation_type: str = "unknown"
    scientific_query: str = ""
    relation_phrases: list = []
    fallback_needed: bool = False


class SynthesisRequest(BaseModel):
    query: str
    search_query: Optional[str] = ""
    detected_language: Optional[str] = "en"
    language: Optional[str] = "en"
    relation: Optional[SynthesisRelation] = None
    articles: List[SynthesisArticle] = []


class CardSynthesisRequest(BaseModel):
    """On-demand per-card synthesis (FASE 8).

    Synthesises a SINGLE article or RWE record on demand — no auto-synthesis,
    no global batch. The user clicks 'Ricava sintesi' on a specific card.
    """
    query: str
    title: str
    abstract: str = ""
    text: str = ""
    source: str = ""
    url: str = ""
    pmid: str = ""
    doi: str = ""
    nct_id: str = ""
    external_id: str = ""
    source_type: str = ""       # scientific_article | community_forum | pharmacovigilance
    evidence_tier: str = ""     # RCT | anecdotal | spontaneous_report | …
    treatment: str = ""
    condition: str = ""
    language: str = "en"        # output language


@app.post("/assistant/chat")
def assistant_chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    AI Clinical Assistant — RAG over stored profiles and patient experiences.
    Creates a session if none provided; returns assistant response + session_id.
    """
    import os, json as _json

    from core.llm_provider import build_provider
    client = build_provider()
    if client is None:
        return {"error": "LLM configuration is not set or incomplete."}

    # ── Session management ──────────────────────────────────────────
    import json as _json
    session_id = body.session_id or str(uuid.uuid4())
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()

    if not session:
        # Auto-create only when no session_id was supplied (new chat from welcome screen)
        title = body.message[:80]
        session = ChatSession(session_id=session_id, title=title, status="active")
        db.add(session)
        db.commit()
    elif session.status == "closed":
        raise HTTPException(
            status_code=409,
            detail="This session is closed. Reopen it before sending messages.",
        )

    # ── Persist ONLY minimal search metadata (history of last 10 searches) on session
    # This stores metadata for chat continuity but NOT the full profiles or extracted data.
    now_utc = datetime.now(timezone.utc)
    if body.search_context is not None:
        try:
            sc = body.search_context
            meta = {
                "original_query": getattr(sc, "original_query", ""),
                "search_query": getattr(sc, "search_query", ""),
                "detected_language": getattr(sc, "detected_language", ""),
                "articles_count": len(getattr(sc, "articles", []) or []),
                "rwe_count": len(getattr(sc, "rwe_evidence", []) or []),
                "timestamp": now_utc.isoformat(),
            }
        except Exception:
            meta = {"original_query": "", "search_query": "", "detected_language": "", "articles_count": 0, "rwe_count": 0, "timestamp": now_utc.isoformat()}

        try:
            existing = session.search_context or {}
            history = existing.get("history", []) if isinstance(existing, dict) else []
        except Exception:
            history = []
        # Prepend and trim to last 10
        history = [meta] + history
        history = history[:10]
        session.search_query = meta.get("original_query", "")
        session.search_context = {"history": history, "last_updated": now_utc.isoformat()}
        session.updated_at = now_utc
        db.commit()
    else:
        session.updated_at = now_utc
        db.commit()

    # ── Resolve effective search context: use the live context provided by the client only.
    # Do NOT reconstruct full search contexts from stored session metadata (which is only history).
    effective_search_ctx = body.search_context if body.search_context is not None else None

    # ── Load conversation history ───────────────────────────────────
    history_rows = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    ).scalars().all()

    messages_history = [
        {"role": r.role, "content": r.content} for r in history_rows
    ]

    # ── RAG: retrieve relevant context from DB ──────────────────────
    # If the effective search context includes explicit episode-id lists
    # for clinical and/or RWE profiles, fetch those DB rows and include them
    # in the LLM context. This guarantees that ALL profiles produced by the
    # current search are included automatically.
    user_msg_lower = body.message.lower()
    context_snippets: list = []
    context_episode_ids: list = []

    # Build a raw dict from the provided search_context or from the stored session
    sc_raw = None
    if body.search_context is not None:
        try:
            sc_raw = body.search_context.model_dump()
        except Exception:
            try:
                sc_raw = dict(body.search_context)
            except Exception:
                sc_raw = None
    elif session.search_context:
        sc_raw = session.search_context

    clinical_ids = sc_raw.get("clinical_profile_episode_ids", []) if sc_raw else []
    rwe_ids = sc_raw.get("rwe_profile_episode_ids", []) if sc_raw else []

    # Support ephemeral clinical & RWE search ids in the provided search_context
    clinical_search_id = sc_raw.get("clinical_profile_search_id") if sc_raw else None
    rwe_search_id = sc_raw.get("rwe_profile_search_id") if sc_raw else None
    if clinical_search_id:
        from core.temp_store import temp_store
        t = temp_store.get(clinical_search_id)
        if t and t.get("type") == "clinical_profiles":
            cp_rows = []
            for s in (t.get("results", []) or [])[:30]:
                vp = s.get("validation_payload", {}) or {}
                payload = s.get("profile", {}) or {}
                class _Tmp:
                    pass
                tmp = _Tmp()
                tmp.episode_id = s.get("episode_id")
                tmp.validation_payload = vp
                tmp.extracted_payload = payload
                tmp.user_id = s.get("source") or vp.get("source", "")
                tmp.processed_at = None
                cp_rows.append(tmp)
            # Avoid DB lookup path below
            clinical_ids = []

    if rwe_search_id:
        from core.temp_store import temp_store
        t = temp_store.get(rwe_search_id)
        if t and t.get("type") == "rwe_profiles":
            rp_rows = []
            for s in (t.get("results", []) or [])[:30]:
                class _TmpRWE:
                    pass
                tmp = _TmpRWE()
                tmp.episode_id = s.get("episode_id")
                tmp.extracted_profile = s.get("profile", {})
                tmp.source = s.get("source")
                tmp.title = s.get("title")
                rp_rows.append(tmp)
            rwe_ids = []


    included_eps = set()

    # Fetch ClinicalProfile rows referenced by the search context, if any
    if clinical_ids:
        cp_rows = db.execute(
            select(ClinicalProfile).where(ClinicalProfile.episode_id.in_(clinical_ids))
        ).scalars().all()
        for cp in cp_rows:
            vp = cp.validation_payload or {}
            payload = cp.extracted_payload or {}
            diag = ", ".join(payload.get("diagnosis", [])[:3])
            treats = ", ".join(payload.get("treatments", [])[:4])
            dosages = ", ".join(payload.get("dosages", [])[:4])
            outcomes = ", ".join(payload.get("outcomes", [])[:3])
            ae = ", ".join(payload.get("adverse_effects", [])[:3])
            ev_level = payload.get("evidence_level", "")
            study_pop = payload.get("study_population", "")
            snippet = (
                f"[Clinical Profile — {cp.user_id.upper()}, {vp.get('pub_year','')}] "
                f"'{vp.get('title','')[:100]}'"
                f"{f' ({ev_level})' if ev_level else ''}"
                f"{f' [{study_pop}]' if study_pop else ''}\n"
                f"  Diagnosis: {diag or 'N/A'}\n"
                f"  Treatments: {treats or 'N/A'}"
                f"{f' | Dosages: {dosages}' if dosages else ''}\n"
                f"  Outcomes: {outcomes or 'N/A'}\n"
                f"  Adverse effects: {ae or 'N/A'}"
            )
            context_snippets.append(snippet)
            context_episode_ids.append(cp.episode_id)
            included_eps.add(cp.episode_id)

    # Fetch RWEProfile rows referenced by the search context, if any
    if rwe_ids:
        rp_rows = db.execute(
            select(RWEProfile).where(RWEProfile.episode_id.in_(rwe_ids))
        ).scalars().all()
        for rp in rp_rows:
            p = rp.extracted_profile or {}


            condition = p.get("condition", "unknown condition")
            summary = p.get("experience_summary", "")
            treats = ", ".join(p.get("treatments_tried", [])[:3]) or p.get("treatment", "")
            outcomes = ", ".join(p.get("reported_outcomes", [])[:3])
            snippet = (
                f"[RWEProfile — {rp.source}] {rp.title or rp.episode_id}\n"
                f"  Condition: {condition}\n"
                f"  Summary: {summary[:200] if summary else 'N/A'}\n"
                f"  Treatments tried: {treats or 'N/A'}\n"
                f"  Reported outcomes: {outcomes or 'N/A'}"
            )
            context_snippets.append(snippet)
            context_episode_ids.append(rp.episode_id)
            included_eps.add(rp.episode_id)

    # Existing fallback DB retrieval (RAG) remains but will avoid duplicating snippets
    _has_search_articles_early = bool(
        effective_search_ctx and effective_search_ctx.articles
    )

    if not _has_search_articles_early:
        # Search clinical profiles (existing behaviour) but skip ones already included
        cp_rows = db.execute(
            select(ClinicalProfile)
            .order_by(desc(ClinicalProfile.processed_at))
            .limit(30)
        ).scalars().all()

        for cp in cp_rows:
            if cp.episode_id in included_eps:  # avoid duplicate snippet
                continue
            vp = cp.validation_payload or {}
            title = vp.get("title", "").lower()
            payload = cp.extracted_payload or {}
            payload_text = _json.dumps(payload).lower()
            if any(word in title or word in payload_text
                   for word in user_msg_lower.split() if len(word) > 3):
                diag       = ", ".join(payload.get("diagnosis", [])[:3])
                treats     = ", ".join(payload.get("treatments", [])[:4])
                dosages    = ", ".join(payload.get("dosages", [])[:4])
                outcomes   = ", ".join(payload.get("outcomes", [])[:3])
                ae         = ", ".join(payload.get("adverse_effects", [])[:3])
                ev_level   = payload.get("evidence_level", "")
                study_pop  = payload.get("study_population", "")
                snippet  = (
                    f"[Clinical Profile — {cp.user_id.upper()}, {vp.get('pub_year','')}] "
                    f"'{vp.get('title','')[:100]}'"
                    f"{f' ({ev_level})' if ev_level else ''}"
                    f"{f' [{study_pop}]' if study_pop else ''}\n"
                    f"  Diagnosis: {diag or 'N/A'}\n"
                    f"  Treatments: {treats or 'N/A'}"
                    f"{f' | Dosages: {dosages}' if dosages else ''}\n"
                    f"  Outcomes: {outcomes or 'N/A'}\n"
                    f"  Adverse effects: {ae or 'N/A'}"
                )
                context_snippets.append(snippet)
                context_episode_ids.append(cp.episode_id)
                if len(context_snippets) >= 5:
                    break

        # Search patient experiences as before, skipping included ones
        pe_rows = db.execute(
            select(PatientExperience)
            .order_by(desc(PatientExperience.ingested_at))
            .limit(30)
        ).scalars().all()

        for pe in pe_rows:
            if pe.episode_id in included_eps:
                continue
            p = pe.extracted_profile or {}
            payload_text = _json.dumps(p).lower()
            if any(word in payload_text for word in user_msg_lower.split() if len(word) > 3):
                condition = p.get("condition", "unknown condition")
                summary   = p.get("experience_summary", "")
                treats    = ", ".join(p.get("treatments_tried", [])[:3])
                outcomes  = ", ".join(p.get("reported_outcomes", [])[:3])
                snippet   = (
                    f"[Patient Experience — Reddit]\n"
                    f"  Condition: {condition}\n"
                    f"  Summary: {summary[:200] if summary else 'N/A'}\n"
                    f"  Treatments tried: {treats or 'N/A'}\n"
                    f"  Reported outcomes: {outcomes or 'N/A'}"
                )
                context_snippets.append(snippet)
                context_episode_ids.append(pe.episode_id)
                if len(context_snippets) >= 8:
                    break

    # ── Build system prompt ─────────────────────────────────────────

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: You must respond entirely in {_resp_lang}. "
        f"All your answers, labels, and explanations must be written in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    # ── Priority 1: current search context (Feature 002 + 003 isolation) ──
    search_block = ""
    has_search_articles = False
    sc = effective_search_ctx   # backend-resolved; never leaks from other sessions

    if sc and sc.articles:
        has_search_articles = True
        lang_note_query = (
            f" (original query in {sc.detected_language.upper()}: '{sc.original_query}',"
            f" translated to: '{sc.search_query}')"
            if sc.original_query.lower() != sc.search_query.lower()
            else f" (query: '{sc.original_query}')"
        )
        art_lines = []
        for i, art in enumerate(sc.articles, 1):
            src_label = {"pubmed": "PubMed", "europepmc": "Europe PMC",
                         "clinicaltrials": "ClinicalTrials.gov"}.get(art.source, art.source)
            abstract_excerpt = (art.abstract or "").strip()
            url_part = f"\n     URL: {art.url}" if art.url else ""
            # Build identifier string so the AI can cite accurately without hallucinating IDs
            id_parts = []
            if art.pmid:   id_parts.append(f"PMID {art.pmid}")
            if art.doi:    id_parts.append(f"DOI {art.doi}")
            if art.nct_id: id_parts.append(f"NCT {art.nct_id}")
            id_str = f" | {' | '.join(id_parts)}" if id_parts else ""
            meta_parts = []
            if art.journal: meta_parts.append(art.journal)
            if art.year:    meta_parts.append(art.year)
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            art_lines.append(
                f"  [{i}] [{src_label}]{id_str}{meta_str} {art.title}\n"
                f"      Abstract: {abstract_excerpt or 'N/A'}"
                f"{url_part}"
            )
        source_labels = sorted(set(
            {"pubmed": "PubMed", "europepmc": "Europe PMC",
             "clinicaltrials": "ClinicalTrials.gov"}.get(a.source, a.source)
            for a in sc.articles
        ))
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH RESULTS{lang_note_query} ══\n"
            f"{len(sc.articles)} article(s) retrieved from: {', '.join(source_labels)}\n\n"
            + "\n\n".join(art_lines)
            + "\n\n"
            "══ CLINICAL REASONING ENGINE ══\n"
            "You are a senior clinical expert writing an evidence-based opinion. "
            "You do NOT list database results. You synthesize evidence and communicate conclusions.\n\n"
            "REQUIRED RESPONSE STRUCTURE (in this exact order):\n\n"
            "## 🩺 Clinical Conclusion\n"
            "3–5 sentences that directly state what the evidence shows about the user's question. "
            "This is your overall clinical judgment based on the retrieved articles. "
            "Write it as a clinician would: assertive, grounded, acknowledging uncertainty where it exists. "
            "NEVER open with search metadata ('X articles retrieved', 'A search was performed', "
            "'Based on the retrieved articles', 'The following studies'). "
            "The section must begin with a clinical statement about the topic itself.\n\n"
            "## Confidence Level\n"
            "State exactly ONE of the following on its own line, then a single sentence explaining why:\n"
            "  🟢 High confidence — multiple consistent RCTs or systematic reviews/meta-analyses\n"
            "  🟡 Moderate confidence — some consistent evidence but limited by study quality, size, or number\n"
            "  🔴 Low confidence — few studies, conflicting results, or only observational/case data\n"
            "Base confidence solely on the retrieved articles above. Never inflate it.\n\n"
            "## 🔬 Evidence Summary\n"
            "Synthesize — do NOT enumerate every article. Write 3–5 sentences that explain:\n"
            "  • Where studies agree\n"
            "  • Where studies disagree or conflict\n"
            "  • Important limitations (sample size, follow-up, study design)\n"
            "Every claim must be traceable to the listed articles. "
            "Do not introduce facts not supported by the abstracts above.\n\n"
            "## 📚 Key Studies\n"
            "Cite 3–5 of the most relevant studies. For each, provide on a single line:\n"
            "  [Source] Full study title — PMID XXXXXXXX | DOI xxx | NCT XXXXXXX (use whichever identifier is available from the article data above)\n"
            "Do NOT reference articles as 'Article 1', 'Article 2', etc. Use their actual titles. "
            "Omit studies that are not relevant to the specific question.\n\n"
            "## ℹ️ General Medical Knowledge (ONLY if necessary)\n"
            "Include this section ONLY when the retrieved articles leave a clinically important gap. "
            "If included, open with exactly this sentence: "
            "'The following information is not derived from the retrieved evidence but from general medical knowledge.' "
            "If the retrieved articles fully answer the question, OMIT this section entirely — do not include a placeholder.\n\n"
            "ABSOLUTE RULES:\n"
            "- Synthesis before articles. Conclusion before evidence. Never the reverse.\n"
            "- Never enumerate all retrieved articles; mention only the most relevant ones in Key Studies.\n"
            "- Never repeat the same information across sections.\n"
            "- Every claim in Evidence Summary and Key Studies must trace back to a listed article.\n"
            "- The response must read like a clinician's written opinion, not a database printout.\n"
            "══════════════════════════════════════════\n"
        )
    elif sc and sc.rwe_evidence:
        # No scientific articles, but RWE items are present — RWE drives the answer.
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH: no scientific articles retrieved "
            f"(query: '{sc.original_query}'). Real-world evidence is available below. ══\n"
            "══════════════════════════════════════════\n"
        )
    elif sc:
        # Search was performed but returned no articles and no RWE
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH: no articles retrieved "
            f"(query: '{sc.original_query}') ══\n"
            "Answer the user's question as directly as possible using general medical knowledge. "
            "If you must rely on general knowledge, state explicitly at the END of your response:\n"
            "  'Note: the current search returned no results. "
            "This answer is based on general medical knowledge.'\n"
            "══════════════════════════════════════════\n"
        )

    # ── RWE evidence block (kept strictly separate from scientific) ──
    # Built ONLY when the frontend forwarded real RWE items. When rwe_evidence
    # is empty, the prompt must NOT list or hint at RWE sources as available —
    # the Assistant may only cite sources whose records are actually in context.
    rwe_block = ""
    has_rwe = bool(sc and sc.rwe_evidence)
    # Source-label map is reused only to format items that are ACTUALLY present.
    _RWE_SRC_LABELS = {
        "reddit": "Reddit", "openfda_faers": "openFDA FAERS",
        "calvizie": "Calvizie.net", "hairlosstalk": "HairLossTalk",
        "hairlossexperiences": "HairLossExperiences",
        "maladiesrares": "MaladiesRaresInfo (FR)",
    }
    if has_rwe:
        rwe_items = sc.rwe_evidence
        rwe_lines = []
        for i, it in enumerate(rwe_items, 1):
            tier_label = {
                "anecdotal": "anecdotal (community)",
                "spontaneous_report": "spontaneous report (pharmacovigilance)",
                "survey": "survey",
            }.get(it.evidence_tier, it.evidence_tier)
            src_label = _RWE_SRC_LABELS.get(it.source, it.source)
            date_part = f" ({it.date})" if it.date else ""
            treat_part = f" | Treatment: {it.treatment}" if it.treatment else ""
            text_excerpt = (it.text or "").strip()
            if len(text_excerpt) > 350:
                text_excerpt = text_excerpt[:340] + "…"
            prov_parts = []
            if it.matched_query:
                prov_parts.append(f"matched_query='{it.matched_query}'")
            if it.matched_query_type:
                prov_parts.append(f"type={it.matched_query_type}")
            if it.source_language:
                prov_parts.append(f"lang={it.source_language}")
            if it.match_reason:
                prov_parts.append(f"match={it.match_reason}")
            if it.relevance_score:
                prov_parts.append(f"score={it.relevance_score:.2f}")
            prov_part = f" | {' '.join(prov_parts)}" if prov_parts else ""
            rwe_lines.append(
                f"  [{i}] [{src_label} — {tier_label}]{date_part}{treat_part}{prov_part}\n"
                f"      {it.title or '(no title)'}\n"
                f"      Text: {text_excerpt or 'N/A'}"
            )
        rwe_source_labels = sorted(set(
            _RWE_SRC_LABELS.get(i.source, i.source) for i in rwe_items
        ))
        rwe_block = (
            f"══ REAL-WORLD EVIDENCE (RWE) — {len(rwe_items)} item(s) from: "
            f"{', '.join(rwe_source_labels)} ══\n"
            "These are patient-reported experiences, community discussions, and "
            "spontaneous adverse-event reports. They are NOT clinical studies and must "
            "NOT be presented as scientific proof. Cite each as '[RWE: <Source>]'.\n\n"
            + "\n\n".join(rwe_lines)
            + "\n\n"
            "RWE RESPONSE RULES:\n"
            "- RWE is supplementary and explicitly lower-evidence than scientific studies.\n"
            "- NEVER present a testimonial or a FAERS report as established clinical fact.\n"
            "- If scientific articles are ALSO present (Priority 1 above), structure your "
            "answer with BOTH: a scientific section AND a 'Testimonianze e discussioni' "
            "section, plus a 'Confronto' (comparison) section noting where RWE agrees or "
            "diverges from the scientific evidence.\n"
            "- If ONLY RWE is present (no scientific articles), answer using the RWE items "
            "under a 'Testimonianze e discussioni' heading, and explicitly state that "
            "these are patient-reported experiences, not clinical evidence, and recommend "
            "consulting the scientific literature / a clinician.\n"
            "- Preserve provenance: every RWE claim must cite its [RWE: <Source>] origin "
            "using ONLY the sources listed above.\n"
            "- Respect privacy_status: never attribute statements to named individuals.\n"
            "══════════════════════════════════════════\n"
        )
    else:
        # No RWE items in context — the Assistant MUST NOT invent or cite any
        # RWE source. This block makes the absence explicit.
        rwe_block = (
            "══ REAL-WORLD EVIDENCE (RWE) — 0 item(s) ══\n"
            "No real-world evidence (patient experiences, community discussions, or "
            "pharmacovigilance reports) is available in the current session.\n"
            "You MUST NOT cite, mention, or reference any RWE source (e.g. Reddit, "
            "openFDA FAERS, community forums) as if it had been consulted. You may only "
            "cite sources whose records appear in the context above.\n"
            "If the user asks about patient experiences / RWE, state explicitly: "
            "'No RWE records are available in the current session.' "
            "Do not fabricate testimonials, forum quotes, or adverse-event reports.\n"
            "══════════════════════════════════════════\n"
        )

    # ── Priority 2: HLEO database (RAG) ────────────────────────────────
    # context_snippets is empty when search articles are present (RAG was skipped above).
    if has_search_articles:
        # RAG was not run — no DB content to show.
        db_header = (
            "══ PRIORITY 2 — HLEO DATABASE (not used — search articles present) ══\n"
            "══════════════════════════════════════════"
        )
    else:
        db_block = (
            "\n\n".join(context_snippets)
            if context_snippets
            else "No matching records in the HLEO database for this query."
        )
        db_header = (
            f"══ PRIORITY 2 — HLEO DATABASE ({len(context_snippets)} records) ══\n"
            f"{db_block}\n"
            "══════════════════════════════════════════"
        )

    # ── Priority 3: general knowledge note ─────────────────────────────
    p3_note = (
        "══ PRIORITY 3 — GENERAL KNOWLEDGE ══\n"
        "Use general medical knowledge ONLY if neither the current search results "
        "nor the HLEO database contain sufficient information to answer the question. "
        "When doing so, explicitly state that the answer is based on general knowledge.\n"
        "══════════════════════════════════════════"
    )

    # Dynamic citation guidance: list only RWE sources that are ACTUALLY present.
    if has_rwe:
        _present_rwe_sources = sorted(set(
            _RWE_SRC_LABELS.get(i.source, i.source) for i in sc.rwe_evidence
        ))
        _rwe_cite_examples = ", ".join(f"'[RWE: {s}]'" for s in _present_rwe_sources)
        _rwe_rule_block = (
            "- When RWE items are present in the REAL-WORLD EVIDENCE block below, you may "
            f"cite them inline using ONLY these labels: {_rwe_cite_examples}. "
            "Do not invent or cite any RWE source not listed there.\n"
            "- Keep scientific evidence and RWE strictly separate: never present a testimonial "
            "or a spontaneous adverse-event report as established clinical fact.\n"
            "- When only RWE is present (no scientific articles), answer under a "
            "'Testimonianze e discussioni' heading using the RWE items, and explicitly state they "
            "are patient-reported experiences, not clinical evidence.\n"
            "- When both scientific and RWE are present, include BOTH a scientific section AND a "
            "'Testimonianze e discussioni' section, plus a 'Confronto' (comparison) section.\n"
        )
    else:
        _rwe_rule_block = (
            "- No RWE is present in this session. You MUST NOT cite, mention, or reference "
            "any RWE source (Reddit, openFDA FAERS, community forums, etc.) as if it had been "
            "consulted. If asked about patient experiences / RWE, state: 'No RWE records are "
            "available in the current session.' Do not fabricate testimonials or reports.\n"
            "- Keep scientific evidence and RWE strictly separate: never present a testimonial "
            "or a spontaneous adverse-event report as established clinical fact.\n"
        )

    system_prompt = (
        "You are HLEO Clinical Assistant, an AI research assistant for clinicians and "
        "researchers. You synthesize scientific evidence and communicate clinical conclusions.\n\n"
        "SOURCE PRIORITY — strictly enforced:\n"
        "  1. CURRENT SEARCH RESULTS — the retrieved scientific articles are the ONLY valid "
        "primary source when present. Build every primary section exclusively from these.\n"
        "  2. REAL-WORLD EVIDENCE (RWE) — patient experiences, community discussions, and "
        "pharmacovigilance reports. Lower-evidence than scientific studies; never presented "
        "as clinical proof. Always kept in a distinct, labelled section.\n"
        "  3. HLEO DATABASE — supplementary only; never used as primary evidence when search "
        "articles are available.\n"
        "  4. GENERAL KNOWLEDGE — only when (1), (2) and (3) are all insufficient.\n\n"
        "You always:\n"
        f"- Cite sources inline (e.g. '[PubMed]', '[Europe PMC]', {_rwe_cite_examples if has_rwe else ''} "
        "'[HLEO DB]'). Only cite a source whose record actually appears in the context below.\n"
        f"{_rwe_rule_block}"
        "- Acknowledge uncertainty; never overstate evidence strength.\n"
        "- Recommend consulting a qualified clinician for personal medical decisions.\n"
        "- When scientific search results are present, follow the CLINICAL REASONING ENGINE "
        "in Priority 1 exactly — synthesis and clinical judgment first, individual studies last.\n"
        "- When no search results are present at all, answer concisely (3–6 sentences) using "
        "HLEO DB then general knowledge.\n\n"
        f"{search_block}"
        f"{rwe_block}"
        f"{db_header}\n\n"
        f"{p3_note}"
        f"{_lang_note}"
    )

    # ── Call LLM ───────────────────────────────────────────────────────────
    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(messages_history)
    llm_messages.append({"role": "user", "content": body.message})

    from core.llm_guard import call_llm, LLMCallError, QuotaExhaustedError
    try:
        assistant_text = call_llm(
            client,
            messages=llm_messages,
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1800 if (has_search_articles or has_rwe) else 800,
            operation="assistant_chat",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/assistant/chat LLM failed — {exc}")
        return {"error": f"Assistant LLM call failed after retries: {exc}"}

    # ── Persist messages ───────────────────────────────────────────
    db.add(ChatMessage(
        session_id=session_id,
        role="user",
        content=body.message,
        context_used=[],
    ))
    db.add(ChatMessage(
        session_id=session_id,
        role="assistant",
        content=assistant_text,
        context_used=context_episode_ids,
    ))
    db.commit()

    return {
        "session_id":          session_id,
        "response":            assistant_text,
        "context_used_count":  len(context_snippets),
        "context_episode_ids": context_episode_ids,
    }


# ── Level 3: Scientific Synthesis ─────────────────────────────────────────────

_SYNTHESIS_PROMPT = """You are a senior clinical evidence synthesizer. You are given:
1) a CLINICAL RELATION the user is investigating, and
2) a set of ALREADY-RETRIEVED, JUDGED-RELEVANT scientific articles (their titles, abstracts, and bibliographic IDs).

Your job: synthesize the evidence ACROSS articles into a structured scientific answer.
You must anchor every claim to the provided articles (cite by index and identifier). Do NOT invent articles, PMIDs, DOIs, or findings not present in the inputs. If the evidence is insufficient or contradictory, say so explicitly.

Return ONLY a JSON object with this exact schema:
{{
  "conclusion": "3-5 sentence clinical conclusion synthesizing what the evidence says about the relation",
  "evidence_summary": "where the studies agree/disagree, sample/population, study design limitations, gaps",
  "claims": [
    {{
      "article_index": <int, the [i] index>,
      "claim": "the specific finding/claim this article contributes regarding the relation",
      "evidence_type": "direct" | "indirect",
      "confidence": "high" | "moderate" | "low",
      "citation": "PMID xxx | DOI xxx | NCT xxx"
    }}
  ],
  "agreements": ["points where >=2 articles concur"],
  "contradictions": ["points where articles disagree"],
  "confidence": {{"level": "high" | "moderate" | "low", "rationale": "why this confidence level (n. articles, consistency, design)"}},
  "key_studies": ["3-5 most relevant citations"]
}}

Direct evidence = the article explicitly studies the relation (agent->manifestation).
Indirect evidence = the relation is inferable or only partially addressed.
"""

def _describe_synthesis_relation(rel: Optional[SynthesisRelation]) -> str:
    if rel is None:
        return "(no structured relation extracted — synthesize from the query and articles)"
    a = rel.agent.get("normalized", "") or rel.agent.get("term", "")
    m = rel.manifestation.get("normalized", "") or rel.manifestation.get("term", "")
    ev = rel.event.get("normalized", "") or rel.event.get("term", "")
    rt = rel.relation_type
    parts = [f"Original query: {rel.original_query}"]
    if rt == "adverse_effect":
        parts.append(f"Relation: {a} (via {ev}) may CAUSE/TRIGGER {m} as an adverse effect")
    elif rt == "efficacy":
        parts.append(f"Relation: {a} used to TREAT {m}")
    elif rt == "exposure_outcome":
        parts.append(f"Relation: {a} (via {ev}) leads to / is associated with {m}")
    else:
        parts.append(f"Relation: {a} -> {m}")
    if rel.temporal:
        parts.append(f"Temporal context: {rel.temporal}")
    if rel.relation_phrases:
        parts.append("Relation phrases: " + "; ".join(rel.relation_phrases))
    return "\n".join(parts)


@app.post("/synthesis")
def synthesize(body: SynthesisRequest):
    """Level 3: synthesize a structured scientific answer from Level 2 results.

    Receives the query, the ClinicalRelation extracted by /search (scientific),
    and the articles already judged relevant — it does NOT re-run retrieval.
    Returns conclusion, evidence summary, per-article claims (direct/indirect),
    agreements/contradictions, confidence, and key study citations.
    """
    import os, json as _json

    from core.llm_provider import build_provider
    client = build_provider()
    if client is None:
        raise HTTPException(status_code=503, detail="Scientific synthesis requires OPENAI_API_KEY.")

    if not body.articles:
        return {"error": "No relevant articles provided for synthesis."}

    # Format articles as a numbered block with identifiers for citation.
    art_lines = []
    for i, a in enumerate(body.articles):
        ident = a.pmid or a.doi or a.nct_id or a.url or ""
        ident_str = f"PMID {a.pmid}" if a.pmid else (f"DOI {a.doi}" if a.doi else (f"NCT {a.nct_id}" if a.nct_id else ident))
        meta_bits = [x for x in [a.journal, a.year] if x]
        meta_str = f" ({', '.join(meta_bits)})" if meta_bits else ""
        rel_str = ""
        if a.relevance_label:
            rel_str = f" | relevance: {a.relevance_label}"
            if a.relevance_score is not None:
                rel_str += f" ({a.relevance_score})"
        abstract = (a.abstract or "")[:1200]
        art_lines.append(
            f"[{i}] [{a.source}] {ident_str}{meta_str} — {a.title}\n    ABSTRACT: {abstract}{rel_str}"
        )
    articles_block = "\n".join(art_lines)

    relation_desc = _describe_synthesis_relation(body.relation)

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: You must respond entirely in {_resp_lang}. "
        f"All your answers, labels, and explanations must be written in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    prompt = (
        f"{_SYNTHESIS_PROMPT}\n\n"
        f"=== CLINICAL RELATION ===\n{relation_desc}\n\n"
        f"=== RELEVANT ARTICLES ({len(body.articles)}) ===\n{articles_block}\n"
        f"{_lang_note}"
    )

    try:
        from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
        parsed = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=2200,
            operation="scientific_synthesis",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/synthesis failed — {exc}")
        return {"error": f"Synthesis failed after retries: {exc}"}
    except _json.JSONDecodeError as exc:
        logger.exception(f"/synthesis: JSON parse error — {exc}")
        return {"error": f"Synthesis returned malformed JSON: {exc}"}
    except Exception as exc:
        logger.exception(f"/synthesis failed — {exc}")
        return {"error": f"Synthesis failed: {exc}"}

    return {
        **parsed,
        "relation": body.relation.model_dump() if body.relation else None,
        "query":    body.query,
        "language": body.language,
    }


# ── FASE 8: On-demand per-card synthesis ────────────────────────────────────
_CARD_SYNTHESIS_PROMPT = """You are a clinical evidence synthesiser. Produce a CONCISE on-demand synthesis of the SINGLE source below, in the context of the user's query.

Return ONLY a JSON object with these keys:
{
  "summary": "2-4 sentence summary of what THIS source says about the query",
  "key_points": ["point 1", "point 2", ...],
  "relevance_to_query": "direct|indirect|background",
  "evidence_type": "scientific|experiential|spontaneous_report",
  "confidence": {"level": "high|moderate|low", "rationale": "..."},
  "limitations": ["limitation 1", ...]
}

Rules:
- Summarise ONLY this one source. Do NOT invent data from other sources.
- For scientific articles: cite the study design and sample size if stated.
- For RWE/community posts: clearly label as patient-reported anecdotal evidence.
- For FAERS records: label as spontaneous adverse-event report, NOT clinical proof.
- key_points: 2-5 bullet points, each ≤120 chars.
- relevance_to_query: how directly this source addresses the user's question.
- Be neutral and evidence-based. Do NOT give medical advice.
"""


@app.post("/synthesis/card")
def synthesize_card(body: CardSynthesisRequest):
    """FASE 8: on-demand synthesis of a SINGLE article/RWE record.

    No auto-synthesis, no batch. The user explicitly requests synthesis for one
    card. Returns a compact structured summary with provenance (source id/type).
    """
    import os

    from core.llm_provider import llm_available
    if not llm_available():
        raise HTTPException(status_code=503,
                            detail="Card synthesis requires OPENAI_API_KEY.")

    content_text = (body.abstract or body.text or "").strip()
    if not content_text:
        return {"error": "No content to synthesise (abstract/text required)."}

    # Build provenance identifier
    ident = (body.pmid or body.doi or body.nct_id
             or body.external_id or body.url or body.source or "")
    if body.pmid:
        ident_str = f"PMID {body.pmid}"
    elif body.doi:
        ident_str = f"DOI {body.doi}"
    elif body.nct_id:
        ident_str = f"NCT {body.nct_id}"
    elif body.external_id:
        ident_str = f"ID {body.external_id}"
    else:
        ident_str = ident
    source_label = body.source or (body.source_type or "unknown")
    tier_label = body.evidence_tier or ""

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: respond entirely in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    meta_bits = []
    if body.treatment:
        meta_bits.append(f"Treatment: {body.treatment}")
    if body.condition:
        meta_bits.append(f"Condition: {body.condition}")
    if tier_label:
        meta_bits.append(f"Evidence tier: {tier_label}")
    meta_str = "\n".join(meta_bits)

    prompt = (
        f"{_CARD_SYNTHESIS_PROMPT}\n\n"
        f"=== USER QUERY ===\n{body.query}\n\n"
        f"=== SOURCE ===\n"
        f"[{source_label}] {ident_str}\n"
        f"Title: {body.title}\n"
        + (f"{meta_str}\n" if meta_str else "")
        + f"Content:\n{content_text[:2000]}\n"
        f"{_lang_note}"
    )

    from core.llm_provider import build_provider
    from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
    client = build_provider()
    try:
        parsed = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=800,
            operation="card_synthesis",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/synthesis/card failed — {exc}")
        return {"error": f"Card synthesis failed after retries: {exc}"}
    except Exception as exc:
        logger.exception(f"/synthesis/card failed — {exc}")
        return {"error": f"Card synthesis failed: {exc}"}

    # Attach provenance
    parsed["source"] = source_label
    parsed["source_id"] = ident_str
    parsed["evidence_tier"] = tier_label
    parsed["query"] = body.query
    parsed["language"] = body.language
    return parsed


# ── FASE 15: AI Assistant — Scientific vs RWE structured comparison ────────

class CompareRequest(BaseModel):
    """FASE 15: structured comparison of scientific evidence vs RWE for a query."""
    query: str
    search_query: str = ""
    detected_language: str = "en"
    language: str = "en"
    scientific_articles: List[SearchArticleCtx] = []
    rwe_evidence: List[RWEItemCtx] = []
    # Optional episode-id lists: backend will fetch full profiles when provided
    clinical_profile_episode_ids: List[str] = []
    rwe_profile_episode_ids: List[str] = []


@app.post("/assistant/compare")
def assistant_compare(body: CompareRequest, db: Session = Depends(get_db)):
    """FASE 15: structured Scientific vs RWE comparison.

    Produces a structured comparison showing where scientific evidence and RWE
    agree, diverge, or where RWE fills gaps the literature doesn't cover.
    RWE is NEVER presented as clinical proof — always labelled as anecdotal /
    spontaneous report.
    """
    import os

    from core.llm_provider import build_provider
    client = build_provider()
    if client is None:
        return {"error": "LLM configuration is not set or incomplete."}

    if not (body.scientific_articles or body.rwe_evidence or getattr(body, 'clinical_profile_episode_ids', None) or getattr(body, 'rwe_profile_episode_ids', None)):
        return {"error": "No evidence provided for comparison."}

    from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError

    # Build SCIENTIFIC and RWE blocks from the provided live search_context in the request.
    # Under the new architecture we DO NOT fetch profiles from the DB; the frontend
    # must supply the relevant scientific_articles and rwe_evidence arrays.
    sci_lines = []
    if body.scientific_articles:
        for i, a in enumerate(body.scientific_articles[:10]):
            ident = a.pmid or a.doi or a.nct_id or a.url or ""
            sci_lines.append(
                f"[S{i}] {a.source} | {ident} — {a.title}\n"
                f"    {(a.abstract or '')[:600]}"
            )
        cp_count = len(body.scientific_articles or [])
    else:
        # Try to load clinical profiles from DB if episode ids are provided
        cp_count = 0
        eps = getattr(body, 'clinical_profile_episode_ids', []) or []
        if eps:
            cp_rows = db.execute(
                select(ClinicalProfile).where(ClinicalProfile.episode_id.in_(eps))
            ).scalars().all()
            for i, cp in enumerate(cp_rows[:10]):
                vp = cp.validation_payload or {}
                payload = cp.extracted_payload or {}
                title = vp.get('title', '')
                diag = ", ".join(payload.get('diagnosis', [])[:3])
                treats = ", ".join(payload.get('treatments', [])[:4])
                snippet = (
                    f"[S{i}] clinical_profile | {cp.episode_id} — {title}\n"
                    f"    Diagnosis: {diag}\n"
                    f"    Treatments: {treats}"
                )
                sci_lines.append(snippet)
            cp_count = len(cp_rows)

    sci_block = "\n".join(sci_lines) or "(no scientific articles)"

    rwe_lines = []
    if body.rwe_evidence:
        for i, r in enumerate(body.rwe_evidence[:10]):
            rwe_lines.append(
                f"[R{i}] {r.source} ({r.evidence_tier}) | {r.external_id or ''} — {r.title}\n"
                f"    treatment={r.treatment or '?'}; {(r.text or '')[:500]}"
            )
        rp_count = len(body.rwe_evidence or [])
    else:
        rp_count = 0
        eps = getattr(body, 'rwe_profile_episode_ids', []) or []
        if eps:
            rp_rows = db.execute(
                select(RWEProfile).where(RWEProfile.episode_id.in_(eps))
            ).scalars().all()
            for i, rp in enumerate(rp_rows[:10]):
                rwe_lines.append(
                    f"[R{i}] {rp.source} ({rp.evidence_tier}) | {rp.external_id or ''} — {rp.title}\n"
                    f"    treatment={rp.treatment or '?'}; {(rp.raw_text or '')[:500]}"
                )
            rp_count = len(rp_rows)

    rwe_block = "\n".join(rwe_lines) or "(no RWE items)"

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: respond entirely in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    prompt = (
        "You are a clinical evidence analyst. Compare the SCIENTIFIC evidence "
        "with the REAL-WORLD EVIDENCE (RWE) for the user's query.\n\n"
        "Return ONLY a JSON object with these keys:\n"
        "{\n"
        '  "scientific_consensus": "1-3 sentences: what does the literature conclude?",\n'
        '  "rwe_consensus": "1-3 sentences: what do patients/reports describe?",\n'
        '  "agreements": ["point where RWE supports the literature", ...],\n'
        '  "divergences": ["point where RWE differs from the literature", ...],\n'
        '  "gaps_filled_by_rwe": ["aspect RWE covers that the literature does not", ...],\n'
        '  "evidence_quality_note": "reminder: RWE is anecdotal/spontaneous, not proof",\n'
        '  "practical_takeaway": "1-2 sentences: what this means for the patient",\n'
        '  "confidence": {"level": "high|moderate|low", "rationale": "..."}\n'
        "}\n\n"
        "Rules:\n"
        "- NEVER present RWE as clinical proof. Always label it as anecdotal or "
        "spontaneous report.\n"
        "- Be neutral and evidence-based. Do NOT give medical advice.\n"
        "- If one side is empty, state that explicitly.\n"
        f"{_lang_note}\n\n"
        f"=== USER QUERY ===\n{body.query}\n\n"
        f"=== SCIENTIFIC EVIDENCE ({len(body.scientific_articles)}) ===\n{sci_block}\n\n"
        f"=== REAL-WORLD EVIDENCE ({len(body.rwe_evidence)}) ===\n{rwe_block}\n"
    )

    try:
        parsed = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1500,
            operation="assistant_compare",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/assistant/compare failed — {exc}")
        return {"error": f"Comparison failed after retries: {exc}"}
    except Exception as exc:
        logger.exception(f"/assistant/compare failed — {exc}")
        return {"error": f"Comparison failed: {exc}"}

    parsed["query"] = body.query
    parsed["language"] = body.language
    parsed["scientific_count"] = cp_count if 'cp_count' in locals() and isinstance(cp_count, int) else len(body.scientific_articles)
    parsed["rwe_count"] = rp_count if 'rp_count' in locals() and isinstance(rp_count, int) else len(body.rwe_evidence)
    return parsed


@app.get("/assistant/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Return full message history + stored search context for a single session."""
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    ).scalars().all()

    return {
        "session_id":     session_id,
        "title":          session.title,
        "status":         session.status,
        "search_query":   session.search_query,
        "search_context": session.search_context,   # restored on session open
        "messages": [
            {
                "role":         m.role,
                "content":      m.content,
                "context_used": m.context_used,
                "created_at":   m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }


# ── Translation endpoint ──────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "it"
    content_type: str = "clinical_article"   # clinical_article | patient_experience | general


@app.post("/translate")
async def translate_text(body: TranslateRequest):
    """
    AI translation endpoint used by the frontend language switcher.
    Accepts {text, target_lang, content_type}.
    Returns  {translation, summary, target_lang, content_type}.
    Results are cached in _translate_cache (process lifetime) to avoid re-calling the LLM.
    """
    import os, json as _json, hashlib

    from core.llm_provider import build_provider
    client = build_provider()
    if client is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="LLM configuration is not configured.")

    cache_key = hashlib.md5(
        f"{body.target_lang}:{body.content_type}:{body.text[:500]}".encode()
    ).hexdigest()
    if cache_key in _translate_cache:
        return _translate_cache[cache_key]

    LANG_NAMES = {
        "it": "Italian",    "en": "English",   "fr": "French",
        "de": "German",     "es": "Spanish",   "pt": "Portuguese",
        "zh": "Chinese",    "ja": "Japanese",  "ar": "Arabic",
        "ru": "Russian",    "nl": "Dutch",     "pl": "Polish",
    }
    lang_name     = LANG_NAMES.get(body.target_lang, body.target_lang)
    content_label = body.content_type.replace("_", " ")

    prompt = (
        f"You are a professional medical translator and summariser.\n\n"
        f"Translate the following {content_label} text to {lang_name}, "
        f"preserving all medical terminology accurately. "
        f"Also provide a concise summary in {lang_name} (2-3 sentences max).\n\n"
        f"TEXT:\n{body.text[:3000]}\n\n"
        "Respond with JSON only:\n"
        '{"translation": "...", "summary": "..."}'
    )

    from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
    try:
        result = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1500,
            operation="translate",
        )
    except (LLMCallError, QuotaExhaustedError) as exc:
        raise HTTPException(status_code=503, detail=f"Translation failed: {exc}")

    out = {
        "translation":  result.get("translation", ""),
        "summary":      result.get("summary", ""),
        "target_lang":  body.target_lang,
        "content_type": body.content_type,
    }
    # Evict oldest entry (FIFO) when at capacity before inserting the new one.
    if _TRANSLATE_CACHE_MAXSIZE > 0:
        while len(_translate_cache) >= _TRANSLATE_CACHE_MAXSIZE:
            _translate_cache.popitem(last=False)
    _translate_cache[cache_key] = out
    logger.info(f"Translated {len(body.text)} chars to {lang_name} ({body.content_type})")
    return out


# ── Session management constants ──────────────────────────────────────────────
_MAX_ACTIVE  = 10
_MAX_CLOSED  = 20

class SessionPatch(BaseModel):
    action: str          # "rename" | "close" | "reopen"
    title: Optional[str] = None


@app.get("/assistant/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """Return active and closed sessions with message counts."""
    all_sessions = db.execute(
        select(ChatSession).order_by(desc(ChatSession.updated_at))
    ).scalars().all()

    # Count messages per session in one query
    msg_counts_raw = db.execute(
        select(ChatMessage.session_id, func.count(ChatMessage.id).label("cnt"))
        .group_by(ChatMessage.session_id)
    ).all()
    msg_counts = {row.session_id: row.cnt for row in msg_counts_raw}

    def _fmt(s: ChatSession) -> dict:
        return {
            "session_id":   s.session_id,
            "title":        s.title or "Untitled",
            "search_query": s.search_query,
            "message_count": msg_counts.get(s.session_id, 0),
            "created_at":   s.created_at.isoformat() if s.created_at else None,
            "updated_at":   s.updated_at.isoformat() if s.updated_at else None,
            "closed_at":    s.closed_at.isoformat() if s.closed_at else None,
        }

    active = [_fmt(s) for s in all_sessions if s.status == "active"]
    closed = [_fmt(s) for s in all_sessions if s.status == "closed"]
    return {"active": active, "closed": closed}


@app.post("/assistant/sessions")
def create_session(db: Session = Depends(get_db)):
    """Create a new active session, enforcing the 10-active limit."""
    active_count = db.execute(
        select(func.count(ChatSession.id)).where(ChatSession.status == "active")
    ).scalar_one()
    if active_count >= _MAX_ACTIVE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"You have reached the maximum number of active sessions ({_MAX_ACTIVE}). "
                "Close or delete an existing session before creating a new one."
            ),
        )
    now = datetime.now(timezone.utc)
    session = ChatSession(
        session_id=str(uuid.uuid4()),
        title="New session",
        status="active",
        created_at=now,
        updated_at=now,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "session_id": session.session_id,
        "title":      session.title,
        "status":     session.status,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@app.patch("/assistant/sessions/{session_id}")
def patch_session(session_id: str, body: SessionPatch, db: Session = Depends(get_db)):
    """Rename, close, or reopen a session with limit enforcement."""
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    now = datetime.now(timezone.utc)

    if body.action == "rename":
        if not body.title or not body.title.strip():
            raise HTTPException(status_code=422, detail="Title cannot be empty.")
        session.title      = body.title.strip()[:120]
        session.updated_at = now

    elif body.action == "close":
        if session.status == "closed":
            raise HTTPException(status_code=409, detail="Session is already closed.")
        closed_count = db.execute(
            select(func.count(ChatSession.id)).where(ChatSession.status == "closed")
        ).scalar_one()
        if closed_count >= _MAX_CLOSED:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"You have reached the maximum number of archived sessions ({_MAX_CLOSED}). "
                    "Delete one or more archived sessions before closing another session."
                ),
            )
        session.status     = "closed"
        session.closed_at  = now
        session.updated_at = now

    elif body.action == "reopen":
        if session.status == "active":
            raise HTTPException(status_code=409, detail="Session is already active.")
        active_count = db.execute(
            select(func.count(ChatSession.id)).where(ChatSession.status == "active")
        ).scalar_one()
        if active_count >= _MAX_ACTIVE:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Maximum active sessions reached. "
                    "Close or delete an active session before reopening this one."
                ),
            )
        session.status     = "active"
        session.closed_at  = None
        session.updated_at = now

    else:
        raise HTTPException(status_code=422, detail=f"Unknown action: {body.action!r}")

    db.commit()
    return {"ok": True, "session_id": session_id, "action": body.action}


@app.delete("/assistant/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Permanently delete a session and all its messages."""
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    db.execute(
        ChatMessage.__table__.delete().where(ChatMessage.session_id == session_id)
    )
    db.delete(session)
    db.commit()
    return {"ok": True, "deleted": session_id}
