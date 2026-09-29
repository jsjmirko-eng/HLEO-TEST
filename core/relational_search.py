"""
HLEO — Relational Search (Level 2)
==================================
Transforms the AI-extracted clinical *relation* into precise per-source
retrieval, a tolerant hard filter, and a batched LLM relational judge that
re-ranks candidates so articles discussing the REQUESTED RELATIONSHIP surface
on top.

Pipeline
--------
    user query
      → (1) ClinicalRelation extraction        [1 LLM call, cached]
      → (2) per-source structured query build   [no LLM]
      → (3) retrieval via existing collectors   [PubMed/EuropePMC/ClinicalTrials]
      → (4) clinical_rank pre-sort              [deterministic, no LLM]
      → (5) LLM relational judge on top-N pool  [batched, ~2 calls/source]
      → (6) final re-rank by judge score        [no LLM]

Design notes
------------
- ClinicalRelation is the sole semantic authority. Query generation uses only
  the relation extracted from the user request; terminology providers are not
  consulted by this chain.
- Fallback: if OPENAI_API_KEY is missing, or relation extraction / judge fail
  (incl. 429), the caller falls back to the existing keyword pipeline. This
  module never raises for those conditions — it returns None / degraded
  rankings so /search keeps working.
- Collectors are reused unchanged: PubMed accepts [tiab]/hasabstract syntax
  in `term`; EuropePMC accepts TITLE:/ABSTRACT: field syntax in `query`.
- Output shape is identical to core.pipeline.collect() (dict of SearchResult
  lists) so /search can swap it in with no frontend changes. Each article's
  `score` is set to the judge-driven combined score so the existing frontend
  sort (by data.score desc) immediately benefits.
"""
from __future__ import annotations

import collections
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from aggregator import HLEOAggregator

from collectors.pubmed import PubMedCollector
from collectors.europepmc import EuropePMCCollector
from collectors.clinicaltrials import ClinicalTrialsCollector
from collectors.scientific_registry import NEW_SCIENTIFIC_COLLECTORS
from core.search_page import collect_search_pages
from core.search_result import SearchResult

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
# JUDGE_BATCH and JUDGE_POOL_PER_SOURCE are now read at runtime from
# core.llm_limits so they are configurable via the Admin UI.
# The module-level names below are kept for backward compat / documentation only.
JUDGE_BATCH = 5          # default (real value: get_limits().judge_batch_size)
JUDGE_POOL_PER_SOURCE = 10  # default (real value: get_limits().judge_pool_per_source)


def _limits():
    """Return current HLEOLimits without raising."""
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


# ── Clinical relation model ──────────────────────────────────────────────────

@dataclass
class ClinicalRelation:
    """Structured clinical relationship extracted from the user query."""
    original_query: str
    agent: dict = field(default_factory=dict)           # {term,normalized,role,identified}
    event: dict = field(default_factory=dict)           # {term,normalized}
    manifestation: dict = field(default_factory=dict)   # {term,normalized,role,site}
    anatomical_site: dict = field(default_factory=dict)  # backwards-compatible site view
    formulation: dict = field(default_factory=dict)      # {term,normalized}
    temporal: str = ""
    relation_type: str = "unknown"
    scientific_query: str = ""
    relation_phrases: list = field(default_factory=list)
    fallback_needed: bool = False
    canonical_query: str = ""
    expanded_queries: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "agent": self.agent,
            "event": self.event,
            "manifestation": self.manifestation,
            "anatomical_site": self.anatomical_site,
            "formulation": self.formulation,
            "temporal": self.temporal,
            "relation_type": self.relation_type,
            "scientific_query": self.scientific_query,
            "relation_phrases": self.relation_phrases,
            "fallback_needed": self.fallback_needed,
            "canonical_query": self.canonical_query,
            "expanded_queries": self.expanded_queries,
        }


# ── Prompt: relation extraction (refined, validated in /tmp prototypes) ──────

_RELATION_PROMPT = """You are a biomedical clinical-NLP system for a scientific literature search engine.

Interpret the CLINICAL RELATIONSHIP the user is asking about, so the engine can retrieve
articles that discuss THAT RELATIONSHIP (not mere co-mention of words).

The agent is NOT assumed to be a drug: it can be a drug, chemical, physical exposure, activity,
procedure, device, or generic/unspecified.

Return ONLY a JSON object (no markdown, no commentary) with this schema:
{
  "query_original": str,
  "agent": {"term": str, "normalized": str, "role": str, "identified": bool},
  "event": {"term": str, "normalized": str},
  "manifestation": {"term": str, "normalized": str, "role": str,
                    "site": {"term": str, "normalized": str}},
  "formulation": {"term": str, "normalized": str},
  "temporal": str,
  "relation_type": str,
  "scientific_query": str,
  "relation_phrases": [str],
  "fallback_needed": bool
}

EVENT/MANIFESTATION SEMANTIC RULE (mandatory):
- Preserve the clinical event, manifestation, site, formulation, temporal
  qualifier and relation expressed by the user. Do not add facts or terms that
  are absent from the request.
- event.normalized MUST describe the complete event when a modifier changes
  the manifestation; manifestation.normalized MUST remain the underlying event.
- If the query names an anatomical site, preserve it as manifestation.site
  nested under the same manifestation. Do not replace the manifestation with
  its site.
- Omit manifestation.site when the user did not specify a site.
- Preserve an explicit formulation or route in the formulation field and in
  scientific_query when it is part of the request.
- Do not create synonym, alias, related-concept or equivalent-term lists.

relation_type STRICT ENUM (choose one):
- adverse_effect   : a DRUG/CHEMICAL/PROCEDURE may CAUSE/TRIGGER the manifestation (harm after/with the agent).
- efficacy         : the agent is used TO TREAT/FOR a condition, or asks if effective/regrowth.
- drug_condition   : agent studied vs condition, neutral.
- exposure_outcome : a NON-PHARMACOLOGICAL exposure/activity -> outcome (running->knee pain, sun->erythema). NOT adverse_effect.
- association      : a statistical/epidemiological link, no causal direction.
- diagnostic       : agent used to diagnose.
- prevention       : agent used to prevent a condition.
- unknown          : relationship unclear.

ROUTING (apply in order):
1. activity/physical_exposure/device -> outcome = exposure_outcome (NOT adverse_effect).
2. "to treat/for/effective/regrowth" -> efficacy.
3. drug/chemical/procedure followed by a harmful manifestation the user suspects caused -> adverse_effect.
Never default all negatives to adverse_effect.

scientific_query: express the same relation in concise scientific English. Preserve every
explicit semantic component of the user's request, including the named agent, manifestation,
site, formulation, temporal qualifier and relation. Do not add synonyms, aliases, related
concepts, alternative terms, causal cues or recovery concepts that the user did not express.
Do not build OR groups or substitute a provider concept for the user's wording.

relation_phrases: optional short paraphrases of the same relation for the Judge only. They must
not introduce any new clinical concept and must never be used to create retrieval queries.

VAGUE-AGENT HANDLING: if the agent is not specifically named (e.g. "un nuovo farmaco"):
agent.identified=false, agent.role="generic_unspecified", agent.normalized="" (do NOT invent a
name), fallback_needed=true.

Normalize terminology only when required to express the user's meaning in
scientific English. Do not use a fixed translation table or add alternative terms.

Query: __QUERY__"""


# ── Prompt: relational LLM judge (batched) ───────────────────────────────────

_JUDGE_PROMPT = """You are a strict biomedical relevance judge.

The original user question is:
  {original_query}

The structured CLINICAL RELATION extracted from it is:
  agent: {agent} ({arole})
  event: {event}
  manifestation: {manifest} ({mrole})
  manifestation site: {site}
  formulation: {formulation}
  temporal: {temporal}
  relation_type: {rtype}
  scientific query: {scientific_query}
  relation phrases: {relation_phrases}
  relation description: {desc}

Judge the meaning of the original question, not isolated keyword overlap. Direct evidence
must address the agent, manifestation, relation and any temporal qualifier that the user
actually expressed. Do not require a literal phrase when the article expresses the same
meaning in different wording.

Assign exactly one semantic tier for EACH article:
  A = directly pertinent: the requested agent/manifestation relationship is central,
      including the temporal or recovery aspect when the article addresses it.
  B = pertinent but incomplete: the agent and manifestation relationship is relevant,
      but an important element (for example recovery, timing, or causality) is absent
      or only indirect.
  C = useful contextual evidence: clinically related background or general condition
      evidence, but not evidence of the requested relationship itself.
  D = not pertinent: unrelated or only accidental word co-occurrence.

Also return a normalized label for compatibility:
  A -> relevant, B -> partial, C -> contextual, D -> not_relevant.

The numeric score is explanatory only: use 0.0-1.0 for semantic confidence and do
not use it to override the tier. Give one short reason grounded in the article.

Return ONLY JSON:
{{"results":[{{"i":int,"tier":"A|B|C|D","label":str,"score":float,"reason":str}}]}}

Articles:
{arts}"""


def _describe(rel: ClinicalRelation) -> str:
    a = rel.agent.get("normalized", "")
    m = rel.manifestation.get("normalized", "")
    ev = rel.event.get("normalized", "")
    rt = rel.relation_type
    if rt == "adverse_effect":
        return f"{a} (via {ev}) may CAUSE/TRIGGER {m} as an adverse effect"
    if rt == "efficacy":
        return f"{a} used to TREAT {m}"
    if rt == "exposure_outcome":
        return f"{a} (via {ev}) leads to / is associated with {m}"
    return f"{a} -> {m}"


# ── RelationalSearch ─────────────────────────────────────────────────────────

class RelationalSearch:
    """
    Relational retrieval + LLM judge re-ranking.

    Usage
    -----
        rs = RelationalSearch()
        out = rs.search("rossore dopo utilizzo di minoxidil")
        # out -> {"pubmed":[SearchResult], "europepmc":[...], "clinicaltrials":[...],
        #         "reddit":[], "relation": ClinicalRelation, "stats": {...}}
        # out is None when relational mode is unavailable (no key / extraction failed)
        # so the caller falls back to the existing keyword pipeline.
    """

    # Process-lifetime cache for relation extraction (identical query → same relation).
    # Bounded to 512 entries (FIFO eviction) to prevent unbounded memory growth.
    _rel_cache: collections.OrderedDict = collections.OrderedDict()
    _rel_cache_maxsize: int = 512

    def __init__(self) -> None:
        self._client = None
        try:
            from core.llm_provider import build_provider
            self._client = build_provider()
        except Exception as exc:
            logger.warning("RelationalSearch: LLM provider init failed — %s", exc)
        self.pubmed = PubMedCollector()
        self.europepmc = EuropePMCCollector()
        self.clinicaltrials = ClinicalTrialsCollector()
        self._new_collectors = {
            source: collector()
            for source, collector in NEW_SCIENTIFIC_COLLECTORS.items()
        }
        # Per-source semaphores: limit concurrent HTTP tasks per source to
        # avoid violating each API's rate limit, even when collector_max_workers
        # would allow more parallel tasks overall.
        lim = _limits()
        self._source_sems: dict[str, threading.Semaphore] = {
            "pubmed":        threading.Semaphore(lim.pubmed_max_concurrent),
            "europepmc":     threading.Semaphore(lim.epmc_max_concurrent),
            "clinicaltrials": threading.Semaphore(lim.ct_max_concurrent),
            **{
                source: threading.Semaphore(1)
                for source in NEW_SCIENTIFIC_COLLECTORS
            },
        }

    def _scientific_collectors(self) -> dict:
        collectors = {
            "pubmed": (self.pubmed, self._build_pubmed_query),
            "europepmc": (self.europepmc, self._build_epmc_query),
            "clinicaltrials": (self.clinicaltrials, self._build_ct_query),
        }
        for source, collector in getattr(self, "_new_collectors", {}).items():
            collectors[source] = (collector, self._build_scientific_query)
        return collectors

    # ── Public API ───────────────────────────────────────────────────────────

    def search(self, raw_query: str) -> Optional[dict]:
        """Run relation extraction, pre-retrieval expansion, and global ranking."""
        if self._client is None:
            return None
        t0 = time.perf_counter()
        source_names = list(self._scientific_collectors())
        stats: dict[str, Any] = {
            "openai_calls": 0, "judge_errors": [], "judge_used": True,
            "query_calls": 0,
            "collector_requests": {source: [] for source in source_names},
            "collector_errors": [],
            "judge_evaluated": 0,
            "judge_invalid": 0,
            "judge_tiers": {"A": 0, "B": 0, "C": 0, "D": 0},
        }
        rel = self._extract_relation(raw_query)
        stats["openai_calls"] += 1
        if rel is None:
            return None

        expanded = self._expand_relation(rel, raw_query)
        rel.expanded_queries = [provenance for _variant, provenance in expanded]
        stats["expanded_queries"] = len(expanded)
        stats["semantic_query_count"] = len(expanded)
        collectors = self._scientific_collectors()
        raw: dict[str, list] = {source: [] for source in collectors}

        # ── Parallel bounded retrieval ────────────────────────────────────────
        # Build one task per (variant, source) pair — independent HTTP calls.
        # Bounded by collector_max_workers (default 6) from Global Limits.
        # Each task is fully isolated: an exception in one does not affect others.
        # Metadata mutation (match_provenance, matched_queries) happens in the
        # main thread after all futures complete — no shared-state races.

        # Capture semaphore map from the instance for use inside threads.
        # Use getattr fallback so tests that create instances via __new__
        # (without calling __init__) get functional semaphores automatically.
        source_sems = getattr(self, "_source_sems", None)
        if source_sems is None:
            lim_now = _limits()
            source_sems = {
                "pubmed":         threading.Semaphore(lim_now.pubmed_max_concurrent),
                "europepmc":      threading.Semaphore(lim_now.epmc_max_concurrent),
                "clinicaltrials": threading.Semaphore(lim_now.ct_max_concurrent),
                **{
                    source: threading.Semaphore(1)
                    for source in getattr(self, "_new_collectors", {})
                },
            }

        def _collect_one(
            source: str,
            collector,
            query: str,
            provenance: str,
        ) -> tuple[str, str, str, list, str]:
            """Return (source, provenance, query, items, error). Never raises.

            Acquires the per-source semaphore before starting HTTP calls so
            that concurrent tasks for the same source stay within the
            configured rate-limit budget (pubmed_max_concurrent etc.), even
            when collector_max_workers allows more total tasks.
            """
            sem = source_sems.get(source)
            with sem:
                try:
                    items = collect_search_pages(collector, query, limit=None)
                    return source, provenance, query, items, ""
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Scientific %s retrieval failed (query=%r): %s",
                        source, query, exc,
                    )
                    return source, provenance, query, [], error

        max_workers = _limits().collector_max_workers
        tasks = []
        seen_tasks = set()
        for variant, provenance in expanded:
            for source, (collector, builder) in collectors.items():
                query = builder(variant)
                task_key = (source, query)
                if task_key in seen_tasks:
                    continue
                seen_tasks.add(task_key)
                tasks.append((source, collector, query, provenance))
        stats["collector_task_count"] = len(tasks)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_collect_one, source, collector, query, provenance): (
                    source, provenance, query
                )
                for source, collector, query, provenance in tasks
            }
            for fut in as_completed(futures):
                source, provenance, query, items, error = fut.result()
                stats["collector_requests"][source].append({
                    "query": query,
                    "raw_results": len(items),
                    "error": error,
                })
                if error:
                    stats["collector_errors"].append({
                        "source": source, "query": query, "error": error,
                    })
                if items:
                    stats["query_calls"] += 1
                for item in items:
                    item.metadata = dict(item.metadata or {})
                    item.metadata.setdefault("match_provenance", []).append(provenance)
                    item.metadata.setdefault("matched_queries", []).append(query)
                    raw[source].append(item)

        stats["candidates_raw"] = {k: len(v) for k, v in raw.items()}
        stats["candidates_raw_total"] = sum(stats["candidates_raw"].values())
        candidates = self._deduplicate_scientific(raw)
        stats["candidates_deduped"] = len(candidates)
        from core.content_search_experiment import experiment_enabled, run_content_search
        if experiment_enabled():
            stats["content_search"] = run_content_search(candidates, rel)
        grouped = self._split_scientific(candidates)
        candidates = [item for items in grouped.values() for item in items]
        stats["after_soft_relation_pass"] = len(candidates)
        stats["after_hard_filter"] = len(candidates)
        stats["hard_filter_applied"] = False
        stats["funnel"] = {
            "raw": stats["candidates_raw_total"],
            "dedup": stats["candidates_deduped"],
            "soft": len(candidates),
            "hard": len(candidates),
        }

        from core.ranker import clinical_rank
        tier_order = {"A": 3, "B": 2, "C": 1, "D": 0}
        candidates.sort(key=clinical_rank, reverse=True)
        judged_count = 0
        remaining = candidates
        while remaining and judged_count < len(candidates) and judged_count < 400:
            pool = remaining[:300]
            judgements = self._judge_batched(pool, rel, stats)
            stats["judge_evaluated"] += len(pool)
            for item, judgement in zip(pool, judgements):
                item.metadata = dict(item.metadata or {})
                if not judgement.get("valid", False):
                    stats["judge_invalid"] += 1
                    item.metadata.update({
                        "judge_status": "invalid",
                        "semantic_tier": "INVALID",
                        "relevance_label": None,
                        "relevance_score": None,
                        "relevance_reason": judgement.get("reason", "invalid judge decision"),
                        "final_score": None,
                        "judge_score_raw": None,
                    })
                    item.score = clinical_rank(item)
                    continue

                raw_score = judgement["score"]
                tier = judgement["tier"]
                stats["judge_tiers"][tier] += 1
                item.metadata.update({
                    "judge_status": "valid",
                    "semantic_tier": tier,
                    "relevance_label": judgement["label"],
                    "relevance_score": raw_score,
                    "relevance_reason": judgement["reason"],
                    "final_score": raw_score,
                    "judge_score_raw": raw_score,
                })
                # Tier is the semantic decision. Numeric confidence and source
                # quality only order articles within the same tier.
                item.score = round(
                    tier_order[tier] * 1_000_000.0
                    + raw_score * 1_000.0
                    + clinical_rank(item), 2)
            judged_count += len(pool)
            remaining = candidates[judged_count:]
            if not remaining:
                break
        stats["funnel"]["judge"] = stats["judge_evaluated"]
        candidates.sort(key=lambda item: item.score, reverse=True)
        final = [item for item in candidates
                 if (item.metadata or {}).get("semantic_tier") in {"A", "B", "C"}][:400]
        stats["funnel"]["final"] = len(final)
        stats["funnel"]["excluded_by_judge"] = len(candidates) - len(final)
        out = {source: [] for source in source_names}
        out["reddit"] = []
        for item in final:
            key = self._source_key(item)
            if key in out:
                out[key].append(item)
        stats["judge_pool"] = judged_count
        stats["final_count"] = len(final)
        stats["final"] = {k: len(out[k]) for k in source_names}
        stats["elapsed_s"] = round(time.perf_counter() - t0, 2)
        return {**out, "relation": rel, "stats": stats}

    # ── (1) Relation extraction ──────────────────────────────────────────────

    @staticmethod
    def _clean_relation_part(value: Any, fields: tuple[str, ...], nested_site: bool = False) -> dict:
        if not isinstance(value, dict):
            return {}
        cleaned = {field: value[field] for field in fields if field in value}
        if nested_site and isinstance(value.get("site"), dict):
            cleaned["site"] = {
                field: value["site"][field]
                for field in ("term", "normalized")
                if field in value["site"]
            }
        return cleaned

    @staticmethod
    def _replace_term(text: str, old: str, new: str) -> str:
        if not old or not new:
            return text
        return re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, text,
                      count=1, flags=re.IGNORECASE)

    def _expand_relation(self, rel: ClinicalRelation, original_query: str):
        """Return the single semantic query produced by ClinicalRelation."""
        agent = (rel.agent.get("normalized") or rel.agent.get("term") or "").strip()
        manifestation = (rel.manifestation.get("normalized") or rel.manifestation.get("term") or "").strip()
        base = rel.scientific_query or " ".join(x for x in (agent, manifestation) if x)
        if not rel.canonical_query:
            rel.canonical_query = base

        provenance = {
            "query": self._build_pubmed_query(rel),
            "original_term": original_query,
            "expanded_term": base,
            "match_kind": "translation" if original_query.strip().lower() != base.lower() else "canonical",
            "tier": 0.85 if original_query.strip().lower() != base.lower() else 1.0,
            "provider": None,
            "source_entity": None,
            "query_origin": "translation" if original_query.strip().lower() != base.lower() else "canonicalization",
            "source_language": "en",
            "matched_entities": [x for x in (agent, manifestation) if x],
        }
        return [(rel, provenance)]

    @staticmethod
    def _source_key(item) -> str:
        source = re.sub(r"[^a-z0-9]", "", str(getattr(item, "source", "")).lower())
        aliases = {
            "pubmed": "pubmed",
            "europepmc": "europepmc",
            "europe": "europepmc",
            "clinicaltrials": "clinicaltrials",
            "openalex": "openalex",
            "crossref": "crossref",
            "doaj": "doaj",
            "biorxiv": "biorxiv",
            "medrxiv": "medrxiv",
            "openaire": "openaire",
            "hal": "hal",
            "cinii": "cinii",
            "jstage": "jstage",
            "base": "base",
        }
        for marker, key in aliases.items():
            if marker in source:
                return key
        return ""

    @classmethod
    def _split_scientific(cls, items: list) -> dict:
        out = {}
        for item in items:
            key = cls._source_key(item)
            if key:
                out.setdefault(key, []).append(item)
        return out

    @staticmethod
    def _relation_bonus(item, rel: ClinicalRelation) -> Tuple[float, list[str]]:
        """Small additive bonus that keeps scientific ranking relation-aware."""
        text = f"{getattr(item, 'title', '') or ''} {getattr(item, 'abstract', '') or ''}".lower()
        relation_type = str(getattr(rel, "relation_type", "") or "").lower().strip()
        bonus = 0.0
        reasons: list[str] = []

        def _hits(terms: list[str]) -> list[str]:
            return sorted({t for t in terms if t and t.lower() in text})[:6]

        agent_terms = []
        manifest_terms = []
        for side, target in ((rel.agent, agent_terms), (rel.manifestation, manifest_terms)):
            if isinstance(side, dict):
                for key in ("term", "normalized"):
                    val = str(side.get(key) or "").strip().lower()
                    if val:
                        target.append(val)

        agent_hits = _hits(agent_terms)
        if agent_hits:
            bonus += min(0.05, 0.02 * len(agent_hits))
            reasons.append(f"agent={agent_hits[:3]}")

        manifest_hits = _hits(manifest_terms)
        if manifest_hits:
            bonus += min(0.08, 0.03 * len(manifest_hits))
            reasons.append(f"manifestation={manifest_hits[:3]}")

        phrase_hits = _hits([str(p).lower() for p in (rel.relation_phrases or [])])
        if phrase_hits:
            bonus += min(0.05, 0.02 * len(phrase_hits))
            reasons.append(f"phrase={phrase_hits[:2]}")

        if relation_type in {"adverse_effect", "efficacy", "comparison", "exposure_outcome"}:
            if agent_hits and manifest_hits:
                bonus += 0.03
                reasons.append("agent+relation")

        # Relation-specificity: prefer articles containing the extracted
        # manifestation or relation phrase over accidental co-occurrences.
        if relation_type == "adverse_effect":
            manifest_normalized = str(
                (rel.manifestation or {}).get("normalized") or "").lower().strip()
            if manifest_normalized and manifest_normalized in text:
                bonus += 0.03
                reasons.append(f"specific_manifestation={manifest_normalized}")
            specificity_phrases = [
                p for p in (rel.relation_phrases or [])
                if str(p).lower().strip() and str(p).lower().strip() in text
            ]
            if specificity_phrases:
                bonus += 0.02
                reasons.append(f"specific_phrase={str(specificity_phrases[0]).lower()}")

        return round(min(0.20, bonus), 3), reasons


    def _deduplicate_scientific(self, raw: dict) -> list:
        aggregator = HLEOAggregator()
        deduped, _stats = aggregator.deduplicate_across_sources(raw)
        return [
            item
            for source in raw
            if source != "reddit"
            for item in deduped.get(source, [])
        ]


    def _extract_relation(self, query: str) -> Optional[ClinicalRelation]:
        import hashlib
        ck = hashlib.md5(query.lower().strip().encode()).hexdigest()
        if ck in self._rel_cache:
            return self._rel_cache[ck]
        try:
            data = self._llm_json(_RELATION_PROMPT.replace("__QUERY__", query), max_tokens=700)
        except Exception as exc:
            logger.warning("RelationalSearch: relation extraction failed — %s", exc)
            return None
        manifestation = self._clean_relation_part(
            data.get("manifestation"),
            ("term", "normalized", "role"),
            nested_site=True,
        )
        anatomical_site = manifestation.get("site", {}) or {}
        rel = ClinicalRelation(
            original_query=data.get("query_original", query),
            agent=self._clean_relation_part(
                data.get("agent"), ("term", "normalized", "role", "identified")
            ),
            event=self._clean_relation_part(data.get("event"), ("term", "normalized")),
            manifestation=manifestation,
            anatomical_site=anatomical_site,
            formulation=self._clean_relation_part(
                data.get("formulation"), ("term", "normalized")
            ),
            temporal=data.get("temporal", ""),
            relation_type=data.get("relation_type", "unknown"),
            scientific_query=data.get("scientific_query", ""),
            relation_phrases=data.get("relation_phrases", []) or [],
            fallback_needed=bool(data.get("fallback_needed", False)),
        )
        if len(self._rel_cache) >= self._rel_cache_maxsize:
            self._rel_cache.popitem(last=False)  # evict oldest (FIFO)
        self._rel_cache[ck] = rel
        return rel

    # ── (2) Per-source query builders ────────────────────────────────────────

    @staticmethod
    def _retrieval_query(rel: ClinicalRelation) -> str:
        """Return the semantic query produced by the ClinicalRelation unchanged."""
        return rel.scientific_query

    def _build_pubmed_query(self, rel: ClinicalRelation) -> str:
        return self._retrieval_query(rel)

    def _build_epmc_query(self, rel: ClinicalRelation) -> str:
        return self._retrieval_query(rel)

    def _build_ct_query(self, rel: ClinicalRelation) -> str:
        return self._retrieval_query(rel)

    def _build_scientific_query(self, rel: ClinicalRelation) -> str:
        return self._retrieval_query(rel)

    # ── (4) Hard filter (canonical relation terms only) ─────────────────────

    def _hard_filter(self, items: list[SearchResult], rel: ClinicalRelation) -> list[SearchResult]:
        agent_terms = []
        if rel.agent.get("normalized"):
            agent_terms.append(rel.agent["normalized"].lower())
        elif rel.agent.get("term"):
            agent_terms.append(rel.agent["term"].lower())
        mani_terms = []
        if rel.manifestation.get("normalized"):
            mani_terms.append(rel.manifestation["normalized"].lower())
        elif rel.manifestation.get("term"):
            mani_terms.append(rel.manifestation["term"].lower())
        agent_terms = list(dict.fromkeys(agent_terms))
        mani_terms = list(dict.fromkeys(mani_terms))

        kept = []
        for it in items:
            blob = ((getattr(it, "title", "") or "") + " " + (getattr(it, "abstract", "") or "")).lower()
            has_agent = any(t and t in blob for t in agent_terms)
            has_mani = any(t and t in blob for t in mani_terms)
            # If we have no terms to match on, keep the item (don't over-filter).
            if not agent_terms and not mani_terms:
                kept.append(it)
            elif has_agent and has_mani:
                kept.append(it)
            elif has_agent and not mani_terms:
                kept.append(it)
            elif has_mani and not agent_terms:
                kept.append(it)
            # else: drop — clearly incompatible (neither agent nor manifestation present)
        return kept

    # ── (6) LLM judge + (7) re-rank ──────────────────────────────────────────

    def _judge_and_rank(self, items: list[SearchResult], rel: ClinicalRelation, stats: dict) -> list[SearchResult]:
        from core.ranker import clinical_rank
        if not items:
            return items
        _pool_size = _limits().judge_pool_per_source
        pool = items[:_pool_size]
        tail = items[_pool_size:]

        judgements = self._judge_batched(pool, rel, stats)

        # Combine: judge score dominates; clinical_rank breaks ties.
        # Combined score = score*1000 + clinical_rank so judge ordering wins.
        for art, j in zip(pool, judgements):
            raw_score = float(j.get("score", 0.0))
            relation_bonus, relation_reasons = self._relation_bonus(art, rel)
            final_score = min(1.0, max(0.0, raw_score + relation_bonus))
            combined = final_score * 1000.0 + clinical_rank(art) + (relation_bonus * 100.0)
            art.score = round(combined, 2)
            art.metadata = dict(art.metadata or {})
            art.metadata["relevance_label"] = j.get("label", "not_relevant")
            art.metadata["relevance_score"] = final_score
            art.metadata["judge_score_raw"] = raw_score
            art.metadata["relation_bonus"] = relation_bonus
            art.metadata["relation_reasons"] = relation_reasons
            art.metadata["relevance_reason"] = j.get("reason", "")

        pool.sort(key=lambda a: a.score, reverse=True)
        # tail keeps its clinical_rank score (already set), ranked after pool
        return pool + tail

    @staticmethod
    def _invalid_judgement(reason: str) -> dict:
        return {
            "valid": False,
            "tier": None,
            "label": None,
            "score": None,
            "reason": reason,
        }

    @staticmethod
    def _validate_judgement(judgement: object) -> dict:
        if not isinstance(judgement, dict):
            return RelationalSearch._invalid_judgement("judge result is not an object")

        tier = str(judgement.get("tier", "")).strip().upper()
        labels = {"A": "relevant", "B": "partial", "C": "contextual", "D": "not_relevant"}
        label = str(judgement.get("label", "")).strip().lower()
        reason = judgement.get("reason")
        score = judgement.get("score")
        if tier not in labels:
            return RelationalSearch._invalid_judgement("judge tier is missing or invalid")
        if label != labels[tier]:
            return RelationalSearch._invalid_judgement("judge label is missing or inconsistent with tier")
        if not isinstance(reason, str) or not reason.strip():
            return RelationalSearch._invalid_judgement("judge reason is missing")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return RelationalSearch._invalid_judgement("judge score is missing or invalid")
        if not 0.0 <= float(score) <= 1.0:
            return RelationalSearch._invalid_judgement("judge score is outside the supported range")
        return {
            "valid": True,
            "tier": tier,
            "label": label,
            "score": float(score),
            "reason": reason.strip(),
        }

    def _judge_batched(self, pool: list[SearchResult], rel: ClinicalRelation, stats: dict) -> list[dict]:
        if not pool or self._client is None:
            stats["judge_used"] = False
            return [self._invalid_judgement("judge unavailable") for _ in pool]
        out: list[dict] = []
        _batch_sz = _limits().judge_batch_size
        for i in range(0, len(pool), _batch_sz):
            batch = pool[i:i + _batch_sz]
            try:
                res = self._llm_judge(batch, rel)
                stats["openai_calls"] += 1
                if not isinstance(res, list):
                    res = []
                # Align by the explicit article index; missing entries stay invalid.
                idx_map = {r.get("i"): r for r in res if isinstance(r, dict)}
                for j, _art in enumerate(batch):
                    raw = idx_map.get(j)
                    out.append(self._validate_judgement(raw))
            except Exception as exc:
                stats["judge_used"] = False
                stats["judge_errors"].append(str(exc))
                logger.warning("RelationalSearch: judge batch failed — %s", exc)
                for _art in batch:
                    out.append(self._invalid_judgement(f"judge error: {exc}"))
            time.sleep(1.0)  # be gentle with rate limits between batches
        return out

    def _llm_judge(self, batch: list[SearchResult], rel: ClinicalRelation) -> list[dict]:
        arts_txt = "\n".join(
            f"[{i}] TITLE: {getattr(a,'title','')}\n    ABSTRACT: {(getattr(a,'abstract','') or '')[:2000]}"
            for i, a in enumerate(batch)
        )
        prompt = _JUDGE_PROMPT.format(
            original_query=rel.original_query,
            agent=rel.agent.get("normalized", ""), arole=rel.agent.get("role", ""),
            event=rel.event.get("normalized", ""), manifest=rel.manifestation.get("normalized", ""),
            mrole=rel.manifestation.get("role", ""),
            site=rel.anatomical_site or rel.manifestation.get("site", {}),
            formulation=rel.formulation,
            temporal=rel.temporal,
            rtype=rel.relation_type, scientific_query=rel.scientific_query,
            relation_phrases=", ".join(str(p) for p in (rel.relation_phrases or [])),
            desc=_describe(rel), arts=arts_txt,
        )
        data = self._llm_json(prompt, max_tokens=900)
        return data.get("results", []) or []

    # ── LLM helper (delegates retry to the central llm_guard) ─────────────────
    #
    # The guard is the ONLY retry boundary: MAX_TOTAL_ATTEMPTS=5, no nested
    # retry. quota exhaustion raises QuotaExhaustedError (no retry); transient
    # 429s and JSON/parse errors retry up to the cap. This method must NOT add
    # its own retry loop on top — that would breach the absolute cap.

    def _llm_json(self, prompt: str, max_tokens: int = 700) -> dict:
        from core.llm_guard import call_llm_json
        return call_llm_json(
            self._client,
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0,
            max_tokens=max_tokens,
            operation="relational_search_llm",
        )
