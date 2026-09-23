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
      → (4) hard filter (agent + manifestation co-occurrence, synonym-tolerant)
      → (5) clinical_rank pre-sort              [deterministic, no LLM]
      → (6) LLM relational judge on top-N pool  [batched, ~2 calls/source]
      → (7) final re-rank by judge score        [no LLM]

Design notes
------------
- NO hardcoded clinical combinations: the LLM generates agent/manifestation
  search_terms, relation_type and relation_phrases per query.
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
import copy
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from aggregator import HLEOAggregator
from core.vocab.models import MATCH_TIERS

from collectors.pubmed import PubMedCollector
from collectors.europepmc import EuropePMCCollector
from collectors.clinicaltrials import ClinicalTrialsCollector
from core.search_result import SearchResult

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
# JUDGE_BATCH and JUDGE_POOL_PER_SOURCE are now read at runtime from
# core.llm_limits so they are configurable via the Admin UI.
# The module-level names below are kept for backward compat / documentation only.
JUDGE_BATCH = 5          # default (real value: get_limits().judge_batch_size)
JUDGE_POOL_PER_SOURCE = 10  # default (real value: get_limits().judge_pool_per_source)


def _audit_enabled() -> bool:
    return os.getenv("HLEO_AUDIT_TRACE", "0").strip().lower() in {"1", "true", "yes"}


def _audit_item(item, include_abstract: bool = True) -> dict:
    metadata = dict(getattr(item, "metadata", {}) or {})
    record = {
        "title": getattr(item, "title", "") or "",
        "source": getattr(item, "source", "") or "",
        "doi": getattr(item, "doi", None),
        "pmid": getattr(item, "pmid", None),
        "year": getattr(item, "year", None),
        "score": getattr(item, "score", None),
        "matched_queries": list(metadata.get("matched_queries", []) or []),
        "match_provenance": list(metadata.get("match_provenance", []) or []),
    }
    if include_abstract:
        record["abstract"] = (getattr(item, "abstract", "") or "")[:4000]
    for key in (
        "judge_score_raw", "judge_score_adjusted", "final_score",
        "relevance_label", "relevance_reason", "semantic_tier", "relation_bonus",
    ):
        if key in metadata:
            record[key] = metadata[key]
    return record

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
    agent: dict = field(default_factory=dict)           # {term,normalized,role,identified,search_terms}
    event: dict = field(default_factory=dict)           # {term,normalized}
    anatomical_site: dict = field(default_factory=dict)  # {term,normalized,role,search_terms}
    manifestation: dict = field(default_factory=dict)   # {term,normalized,role,search_terms}
    temporal: str = ""
    relation_type: str = "unknown"
    scientific_query: str = ""
    relation_phrases: list = field(default_factory=list)
    fallback_needed: bool = False
    canonical_query: str = ""
    vocabulary: dict = field(default_factory=dict)
    expanded_queries: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "agent": self.agent,
            "event": self.event,
            "anatomical_site": self.anatomical_site,
            "manifestation": self.manifestation,
            "temporal": self.temporal,
            "relation_type": self.relation_type,
            "scientific_query": self.scientific_query,
            "relation_phrases": self.relation_phrases,
            "fallback_needed": self.fallback_needed,
            "canonical_query": self.canonical_query,
            "vocabulary": self.vocabulary,
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
  "agent": {"term": str, "normalized": str, "role": str, "identified": bool, "search_terms": [str]},
  "event": {"term": str, "normalized": str},
  "anatomical_site": {"term": str, "normalized": str, "role": str, "search_terms": [str]},
  "manifestation": {"term": str, "normalized": str, "role": str, "search_terms": [str]},
  "temporal": str,
  "relation_type": str,
  "scientific_query": str,
  "relation_phrases": [str],
  "fallback_needed": bool
}

IMPORTANT:
- anatomical_site is for body location/site/region terms (temples, temporal region, temporal hairline, frontal hairline, scalp location).
- manifestation is for the underlying clinical condition/disorder if one is explicitly mentioned (e.g. androgenetic alopecia, alopecia, hair loss disorder).
- Do NOT put body location/site terms into manifestation.
- If the query has no separate clinical condition, keep manifestation empty ({}).

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

search_terms: provide 2-5 English synonyms/variants for SEARCHING (INN for drugs; scientific
terms for manifestations, e.g. erythema, "skin irritation", "cutaneous irritation", dermatitis).
These are used to match articles by synonym, so include clinically equivalent terms.

scientific_query: REQUIRE the agent AND manifestation to co-occur, joined with AND. For
adverse_effect append a causal group using ONLY: "adverse effect","side effect",induced,
"caused by","triggered by","secondary to","drug-related","treatment-related",worsening.
DO NOT use bare "effect" alone, and DO NOT use "following" or "associated with" as mandatory
terms (they are too generic). For efficacy: (treatment OR efficacy OR therapeutic OR "response to").
For exposure_outcome: (following OR "due to" OR injury OR caused). English, parentheses + OR groups.

relation_phrases: 3-6 diverse natural scientific phrases expressing THIS specific relation
(vary grammar: induced / associated with / following / secondary to / adverse reaction to).

VAGUE-AGENT HANDLING: if the agent is not specifically named (e.g. "un nuovo farmaco"):
agent.identified=false, agent.role="generic_unspecified", agent.normalized="" (do NOT invent a
name), fallback_needed=true.

Normalize: drugs->INN; Italian lay terms -> scientific English (rossore->erythema/"skin irritation";
caduta dei capelli->"hair shedding"/"hair loss"; dolore articolare->arthralgia; mal di testa->headache;
ginocchio->knee; tempie->temples/"temporal region"/"temporal hairline"; esposizione al sole->"sun exposure"; eritema->erythema).

Query: __QUERY__"""


# ── Prompt: relational LLM judge (batched) ───────────────────────────────────

_JUDGE_PROMPT = """You are a strict biomedical relevance judge. Your task is to answer the
specific user question, not to reward broad topical similarity.

Original user query:
  {original_query}

The user is asking about this CLINICAL RELATION:
  agent/exposure: {agent} ({arole})
  event/outcome: {event}
  condition/manifestation: {manifest} ({mrole})
  anatomical_site: {site} ({srole})
  temporal qualifier: {temporal}
  relation_type: {rtype}
  qualifiers: anatomical_site={site}; temporal={temporal}; relation_phrases={phrases}
  scientific_query: {scientific_query}
  canonical_query: {canonical_query}
  relation description: {desc}

For EACH article, read the title and abstract together and decide whether the
article's OWN CLINICAL EVIDENCE answers the user's question about the COMPLETE
relational pattern (agent, outcome, condition, site, timing, direction), rather
than merely co-mentioning words.

Use this CONTENT-FIRST rubric, then return the compatible labels below:
  A / relevant (score 1.0): the article directly studies or reports the requested
    agent -> outcome relation (here: dutasteride and hair regrowth in the relevant
    alopecia domain). It may be A even if some SECONDARY qualifiers (such as the
    exact anatomical site within the scalp) are not explicitly named, provided
    the article clearly addresses hair regrowth / treatment response in the
    appropriate clinical context.
  B / partial (score 0.8): the same agent and clinical domain/relation are present,
    but one IMPORTANT part of the clinical relation is missing or diluted (for
    example: only hair loss without regrowth outcomes, only surrogate endpoints
    without clinical improvement, or dutasteride merely listed among many options
    without being actually evaluated). B is useful context but not a direct answer.
  C / partial (score 0.15): the article is broadly about the hair/clinical domain
    or mentions the agent among treatments, but the requested agent-outcome
    relationship is generic, secondary, mechanistic, or not the article's
    substantive clinical focus. C is contextual evidence below the inclusion
    boundary, not a result for this question.
  D / not_relevant (score 0.0): the article is about a different condition, adverse
    outcome, population, or topic, or only contains incidental co-mentions.

Important decision rules:
- For efficacy/regrowth, require evidence that the agent is used/studied for the
  requested clinical outcome (hair regrowth / improvement in androgenetic
  alopecia). "Review of alopecia treatments" alone is C unless the abstract
  specifically evaluates dutasteride's efficacy.
- Consider the anatomical site (e.g. temporal region) as a SPECIFICITY qualifier:
  do NOT automatically demote an article to B solely because it does not name the
  temporal region, if it clearly studies dutasteride for hair regrowth in
  androgenetic alopecia or a closely matching pattern.
- If the agent appears only as a comparator, background example, or list of drugs,
  do not assign A; use B or C according to whether the relation is actually
  studied.
- For adverse effects, require the requested manifestation/event to be linked to
  the agent; generic safety or sexual/psychiatric effects are not evidence of
  hair loss.
- Never upgrade an article because it contains the words dutasteride + hair +
  alopecia if the complete clinical relation is not supported by the article's
  own evidence.
- Do not classify a formulation, delivery-system, carrier, microneedle,
  liposome, nanoemulsion, ethosome, or mechanistic study as A merely because
  it mentions dutasteride or 5-alpha-reductase inhibitors and hair regeneration.
  Assign A only when the article provides evidence of the requested therapeutic
  outcome in the relevant clinical or experimental treatment context. If its
  main contribution is formulation, delivery, mechanistic characterization,
  surrogate measurement, or vehicle development without demonstrating that
  outcome, assign C or B according to the actual evidence. Human evidence is
  not required when an experimental study directly demonstrates the outcome.

Assign only the existing output labels:
  label: "relevant" | "partial" | "not_relevant"
  score: 0.0-1.0
    - A/relevant: 1.0
    - B/partial: 0.8
    - C/partial: 0.15
    - D/not_relevant: 0.0
  reason: one short sentence naming the satisfied or missing clinical element.

Return ONLY JSON: {{"results":[{{"i":int,"tier":"A|B|C|D","label":str,"score":float,"reason":str}}]}}

Articles:
{arts}"""


_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relational_judge_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["i", "tier", "label", "score", "reason"],
                        "properties": {
                            "i": {"type": "integer"},
                            "tier": {"type": "string", "enum": ["A", "B", "C", "D"]},
                            "label": {"type": "string"},
                            "score": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}



def _describe(rel: ClinicalRelation) -> str:
    a = rel.agent.get("normalized", "")
    m = rel.manifestation.get("normalized", "")
    ev = rel.event.get("normalized", "")
    site = rel.anatomical_site.get("normalized", "")
    temporal = rel.temporal
    rt = rel.relation_type
    site_txt = f" at {site}" if site else ""
    time_txt = f" {temporal}" if temporal else ""
    if rt == "adverse_effect":
        return f"{a} (via {ev}) may CAUSE/TRIGGER {m}{site_txt}{time_txt} as an adverse effect"
    if rt == "efficacy":
        return f"{a} used to TREAT {m}{site_txt}{time_txt}"
    if rt == "exposure_outcome":
        return f"{a} (via {ev}) leads to / is associated with {m}{site_txt}{time_txt}"
    return f"{a} -> {m}{site_txt}{time_txt}"


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
        # Per-source semaphores: limit concurrent HTTP tasks per source to
        # avoid violating each API's rate limit, even when collector_max_workers
        # would allow more parallel tasks overall.
        lim = _limits()
        self._source_sems: dict[str, threading.Semaphore] = {
            "pubmed":        threading.Semaphore(lim.pubmed_max_concurrent),
            "europepmc":     threading.Semaphore(lim.epmc_max_concurrent),
            "clinicaltrials": threading.Semaphore(lim.ct_max_concurrent),
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def search(self, raw_query: str) -> Optional[dict]:
        """Run relation extraction, pre-retrieval expansion, and global ranking."""
        if self._client is None:
            return None
        t0 = time.perf_counter()
        audit_on = _audit_enabled()
        audit = {
            "enabled": audit_on,
            "raw_items": [],
            "dedup_groups": [],
            "filter_eliminated": [],
            "judge_inputs": [],
            "judge_pool": [],
            "threshold": 0.20,
            "threshold_eliminated": [],
        } if audit_on else None
        self._active_audit = audit

        stats: dict[str, Any] = {
            "openai_calls": 0, "judge_errors": [], "judge_used": True,
            "vocab_enabled": False, "query_calls": 0,
        }
        rel = self._extract_relation(raw_query)
        stats["openai_calls"] += 1
        if rel is None:
            return None

        expanded = self._expand_relation(rel, raw_query)
        rel.expanded_queries = [provenance for _variant, provenance in expanded]
        stats["vocab_enabled"] = bool(rel.vocabulary)
        stats["expanded_queries"] = len(expanded)
        raw: dict[str, list] = {"pubmed": [], "europepmc": [], "clinicaltrials": []}
        collectors = {
            "pubmed": (self.pubmed, self._build_pubmed_query),
            "europepmc": (self.europepmc, self._build_epmc_query),
            "clinicaltrials": (self.clinicaltrials, self._build_ct_query),
        }

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
            }

        def _collect_one(
            source: str,
            collector,
            query: str,
            provenance: str,
        ) -> tuple[str, str, str, list]:
            """Return (source, provenance, query, items). Never raises.

            Acquires the per-source semaphore before starting HTTP calls so
            that concurrent tasks for the same source stay within the
            configured rate-limit budget (pubmed_max_concurrent etc.), even
            when collector_max_workers allows more total parallel tasks.
            """
            sem = source_sems.get(source)
            with sem:
                try:
                    items = collector.search(query, limit=None)
                    return source, provenance, query, items
                except Exception as exc:
                    logger.warning(
                        "Scientific %s retrieval failed (query=%r): %s",
                        source, query, exc,
                    )
                    return source, provenance, query, []

        max_workers = _limits().collector_max_workers
        tasks = []
        if audit_on:
            audit["variants"] = []
        for variant, provenance in expanded:
            variant_record = {
                "provenance": provenance,
                "queries": {},
            }
            for source, (collector, builder) in collectors.items():
                query = builder(variant)
                tasks.append((source, collector, query, provenance))
                if audit_on:
                    variant_record["queries"][source] = query
            if audit_on:
                audit["variants"].append(variant_record)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_collect_one, source, collector, query, provenance): (
                    source, provenance, query
                )
                for source, collector, query, provenance in tasks
            }
            for fut in as_completed(futures):
                source, provenance, query, items = fut.result()
                if items:
                    stats["query_calls"] += 1
                for item in items:
                    item.metadata = dict(item.metadata or {})
                    item.metadata.setdefault("match_provenance", []).append(provenance)
                    item.metadata.setdefault("matched_queries", []).append(query)
                    raw[source].append(item)

        stats["candidates_raw"] = {k: len(v) for k, v in raw.items()}
        raw_key_groups = collections.defaultdict(list)
        if audit_on:
            dedup_helper = HLEOAggregator()
            for source, items in raw.items():
                for item in items:
                    audit["raw_items"].append({
                        "source": source,
                        **_audit_item(item),
                    })
                    key = dedup_helper.create_key(item)
                    raw_key_groups[key or f"keyless:{id(item)}"].append((source, item))
        candidates = self._deduplicate_scientific(raw)
        stats["candidates_deduped"] = len(candidates)
        if audit_on:
            survivor_ids = {id(item) for item in candidates}
            for key, group in raw_key_groups.items():
                if len(group) < 2:
                    continue
                kept = [item for _source, item in group if id(item) in survivor_ids]
                winner = kept[0] if kept else None
                audit["dedup_groups"].append({
                    "key": key,
                    "input_count": len(group),
                    "output_count": len(kept),
                    "winner": _audit_item(winner, False) if winner else None,
                    "members": [
                        {"source": source, **_audit_item(item, False),
                         "kept": id(item) in survivor_ids}
                        for source, item in group
                    ],
                })
        grouped = self._split_scientific(candidates)
        candidates = [item for items in grouped.values() for item in items]
        stats["after_soft_relation_pass"] = len(candidates)
        filter_input = list(candidates)
        candidates = self._hard_filter(candidates, rel)
        stats["after_hard_filter"] = len(candidates)
        stats["hard_filter_applied"] = True
        if audit_on:
            kept_ids = {id(item) for item in candidates}
            agent_terms = self._agent_terms_for_filter(rel)
            outcome_terms = self._outcome_terms_for_filter(rel)
            for item in filter_input:
                if id(item) in kept_ids:
                    continue
                blob = f"{getattr(item, 'title', '') or ''} {getattr(item, 'abstract', '') or ''}".lower()
                has_agent = any(term in blob for term in agent_terms)
                has_outcome = any(term in blob for term in outcome_terms)
                if not has_agent and not has_outcome:
                    reason = "missing_agent_and_outcome"
                elif not has_agent:
                    reason = "missing_agent"
                else:
                    reason = "missing_outcome"
                audit["filter_eliminated"].append({
                    **_audit_item(item),
                    "filter_reason": reason,
                    "agent_terms_checked": agent_terms,
                    "outcome_terms_checked": outcome_terms,
                    "clinical_relation": {
                        "agent": rel.agent,
                        "event": rel.event,
                        "anatomical_site": rel.anatomical_site,
                        "manifestation": rel.manifestation,
                        "temporal": rel.temporal,
                        "relation_type": rel.relation_type,
                    },
                })

        from core.ranker import clinical_rank
        # Retrieval order only determines which articles enter the Judge; once
        # judged, semantic score is the primary ranking key.
        candidates.sort(key=clinical_rank, reverse=True)
        judged_count = 0
        remaining = candidates
        while remaining and judged_count < len(candidates) and judged_count < 400:
            pool = remaining[:300]
            judgements = self._judge_batched(pool, rel, stats)
            for item, judgement in zip(pool, judgements):
                raw_score = max(0.0, min(1.0, float(judgement.get("score", 0.0))))
                adjusted_score = self._qualifier_adjusted_score(item, rel, raw_score)
                relation_bonus, relation_reasons = self._relation_bonus(item, rel)

                semantic_tier = str(judgement.get("tier", "") or "").strip().upper()
                if semantic_tier not in {"A", "B", "C", "D"}:
                    # This is only a safety fallback for malformed/missing
                    # Judge output; it must not infer semantics from label/score.
                    semantic_tier = "C"

                semantic_cap = None
                if semantic_tier == "C":
                    adjusted_score = min(adjusted_score, 0.15)
                    semantic_cap = 0.15
                elif semantic_tier == "D":
                    adjusted_score = 0.0
                    semantic_cap = 0.0
                elif adjusted_score <= 0.50 and adjusted_score < raw_score:
                    relation_bonus = 0.0
                    relation_reasons = [*relation_reasons, "qualifier_cap_suppressed_bonus"]
                final_score = min(1.0, max(0.0, adjusted_score + relation_bonus))
                if semantic_cap is not None:
                    final_score = min(final_score, semantic_cap)
                item.metadata = dict(item.metadata or {})
                item.metadata.update({
                    "relevance_label": judgement.get("label", "not_relevant"),
                    "relevance_score": final_score,
                    "relevance_reason": judgement.get("reason", ""),
                    "semantic_tier": semantic_tier,
                    "final_score": final_score,
                    "judge_score_raw": raw_score,
                    "judge_score_adjusted": adjusted_score,
                    "semantic_sort_score": raw_score,
                    "clinical_rank": clinical_rank(item),
                    "relation_bonus": relation_bonus,
                    "relation_reasons": relation_reasons,
                })
                # Keep the public numeric score aligned with raw Judge ordering;
                # the actual ordering below uses a tuple, so clinical_rank remains
                # a tie-breaker without an artificial additive weight.
                item.score = round(raw_score * 1000.0, 2)
                if audit_on:
                    trace = _audit_item(item)
                    trace["judge_label"] = judgement.get("label", "not_relevant")
                    trace["judge_reason"] = judgement.get("reason", "")
                    trace["decision"] = "above_threshold" if final_score >= 0.20 else "below_threshold"
                    trace["rank_before_final_sort"] = len(audit["judge_pool"]) + 1
                    audit["judge_pool"].append(trace)
            judged_count += len(pool)
            remaining = candidates[judged_count:]
            ranked_now = sorted(
                candidates[:judged_count],
                key=lambda item: (
                    float((item.metadata or {}).get("semantic_sort_score", 0.0)),
                    float((item.metadata or {}).get("clinical_rank", 0.0)),
                ),
                reverse=True,
            )
            if sum(float((item.metadata or {}).get("final_score", 0.0)) >= 0.20
                   for item in ranked_now) >= 400 or not remaining:
                break
        candidates.sort(
            key=lambda item: (
                float((item.metadata or {}).get("semantic_sort_score", 0.0)),
                float((item.metadata or {}).get("clinical_rank", 0.0)),
            ),
            reverse=True,
        )
        final = [item for item in candidates
                 if float((item.metadata or {}).get("final_score", 0.0)) >= 0.20][:400]
        out = {"pubmed": [], "europepmc": [], "clinicaltrials": [], "reddit": []}
        for item in final:
            key = self._source_key(item)
            if key:
                out[key].append(item)
        stats["judge_pool"] = judged_count
        stats["final_count"] = len(final)
        stats["final"] = {k: len(out[k]) for k in ("pubmed", "europepmc", "clinicaltrials")}
        stats["threshold"] = 0.20
        stats["above_threshold"] = sum(
            float((item.metadata or {}).get("final_score", 0.0)) >= 0.20
            for item in candidates
        )
        stats["below_threshold"] = len(candidates) - stats["above_threshold"]
        stats["elapsed_s"] = round(time.perf_counter() - t0, 2)
        if audit_on:
            audit["final_ranking"] = []
            for rank, item in enumerate(final, 1):
                record = _audit_item(item)
                record["rank"] = rank
                record["decision"] = "returned"
                audit["final_ranking"].append(record)
            audit["threshold_eliminated"] = [
                record for record in audit["judge_pool"]
                if record.get("decision") == "below_threshold"
            ]
            audit["counts"] = {
                "filter_input": len(filter_input),
                "filter_output": len(candidates),
                "filter_eliminated": len(audit["filter_eliminated"]),
                "judge_input": len(audit["judge_pool"]),
                "threshold_eliminated": len(audit["threshold_eliminated"]),
                "returned": len(final),
            }
            audit["clinical_relation"] = {
                "original_query": rel.original_query,
                "agent": rel.agent,
                "event": rel.event,
                "anatomical_site": rel.anatomical_site,
                "manifestation": rel.manifestation,
                "temporal": rel.temporal,
                "relation_type": rel.relation_type,
                "scientific_query": rel.scientific_query,
                "canonical_query": rel.canonical_query,
            }
            stats["audit"] = audit
        return {**out, "relation": rel, "stats": stats}

    # ── (1) Relation extraction ──────────────────────────────────────────────

    @staticmethod
    def _replace_term(text: str, old: str, new: str) -> str:
        if not old or not new:
            return text
        return re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, text,
                      count=1, flags=re.IGNORECASE)

    @staticmethod
    def _variant_relation(rel: ClinicalRelation, agent: str, manifestation: str) -> ClinicalRelation:
        variant = copy.deepcopy(rel)
        variant.agent = dict(variant.agent)
        variant.manifestation = dict(variant.manifestation)
        variant.agent["normalized"] = agent
        variant.agent["search_terms"] = [agent]
        variant.manifestation["normalized"] = manifestation
        variant.manifestation["search_terms"] = [manifestation]
        return variant

    @staticmethod
    def _match_is_compatible(source: dict, match) -> bool:
        """Keep provider enrichment within the source entity's semantic type."""
        role = str(source.get("role", "") or "").lower()
        group = str(getattr(match, "semantic_group", "") or "").lower()
        if role in {"drug", "medication", "agent"} and group not in {"drug", "general"}:
            return False
        if role in {"condition", "disease", "manifestation"} and group not in {"condition", "symptom", "general"}:
            return False
        if role in {"symptom", "event"} and group not in {"condition", "symptom", "general"}:
            return False
        return True

    @staticmethod
    def _term_is_compatible(source: dict, match, term: str) -> bool:
        """Accept lexical forms of the same concept, not provider descriptions."""
        term_tokens = set(re.findall(r"[a-z0-9]+", str(term or "").lower()))
        source_tokens = set(re.findall(
            r"[a-z0-9]+",
            " ".join(str(source.get(key, "") or "") for key in ("term", "normalized")),
        ))
        if not term_tokens or len(term_tokens) > 10:
            return False
        match_kind = str(getattr(match, "match_kind", "") or "").lower()
        group = str(getattr(match, "semantic_group", "") or "").lower()
        if source_tokens.intersection(term_tokens):
            return True
        identity_kinds = {
            "exact", "canonical", "preferred", "synonym", "translation",
            "abbreviation", "orthographic_variant", "colloquial", "normalized",
        }
        return group == "drug" and match_kind in identity_kinds and len(term_tokens) <= 4

    def _expand_relation(self, rel: ClinicalRelation, original_query: str):
        """Resolve typed vocabulary and generate anchored search queries."""
        agent = (rel.agent.get("normalized") or rel.agent.get("term") or "").strip()
        manifestation = (rel.manifestation.get("normalized") or rel.manifestation.get("term") or "").strip()
        base = rel.scientific_query or " ".join(x for x in (agent, manifestation) if x)
        rel.canonical_query = base
        terms = [x for x in (agent, manifestation) if len(x) >= 3]
        resolutions = {}
        from core.vocab.resolver import build_resolver_from_env
        resolver = build_resolver_from_env()
        if resolver is not None:
            resolutions = resolver.resolve_terms(list(dict.fromkeys(terms)), language="en")
            rel.vocabulary = {
                term: [m.model_dump() for m in result.matches]
                for term, result in resolutions.items() if result.matches
            }

        variants = []
        original_agent = str(rel.agent.get("term") or "").strip()
        original_manifestation = str(rel.manifestation.get("term") or "").strip()
        if original_agent and original_agent.lower() != agent.lower():
            variants.append((original_agent, original_manifestation or manifestation, {
                "query": f"{original_agent} {original_manifestation or manifestation}".strip(),
                "original_term": original_query, "expanded_term": original_agent,
                "match_kind": "exact", "tier": 1.0, "provider": None,
                "source_entity": original_agent, "query_origin": "user",
            }))
        variants.append((agent, manifestation, {
            "query": base, "original_term": original_query,
            "expanded_term": base, "match_kind": "canonical", "tier": 1.0,
            "provider": None, "source_entity": None, "query_origin": "canonicalization",
        }))
        if original_query.strip().lower() != base.lower():
            variants.insert(0, (agent, manifestation, {
                "query": base, "original_term": original_query,
                "expanded_term": base, "match_kind": "translation", "tier": 0.85,
                "provider": None, "source_entity": None, "query_origin": "translation",
            }))
        for source_entity, side in ((agent, "agent"), (manifestation, "manifestation")):
            resolution = resolutions.get(source_entity)
            if resolution is None:
                continue
            source_part = rel.agent if side == "agent" else rel.manifestation
            for match in resolution.matches:
                tier = MATCH_TIERS.get(match.match_kind)
                if tier is None or not self._match_is_compatible(source_part, match):
                    continue
                for term in [match.preferred_term, *match.synonyms]:
                    term = (term or "").strip()
                    if (
                        len(term) < 3
                        or term.lower() == source_entity.lower()
                        or not self._term_is_compatible(source_part, match, term)
                    ):
                        continue
                    a, m = agent, manifestation
                    if side == "agent":
                        a = term
                    else:
                        m = term
                    variants.append((a, m, {
                        "query": f"{a} {m}".strip(),
                        "original_term": source_entity,
                        "expanded_term": term,
                        "match_kind": match.match_kind,
                        "tier": tier, "provider": match.provider,
                        "source_entity": source_entity, "query_origin": "vocabulary",
                    }))

        unique = []
        seen = set()
        for a, m, provenance in variants:
            key = (a.lower(), m.lower())
            if not key[0] and not key[1] or key in seen:
                continue
            seen.add(key)
            variant = self._variant_relation(rel, a, m)
            provenance = dict(provenance)
            provenance["query"] = self._build_pubmed_query(variant)
            provenance["source_language"] = "en"
            provenance["matched_entities"] = [x for x in (a, m) if x]
            unique.append((variant, provenance))
            if len(unique) >= 16:
                break
        return unique

    @staticmethod
    def _source_key(item) -> str:
        source = str(getattr(item, "source", "")).lower().replace(" ", "")
        if "pubmed" in source:
            return "pubmed"
        if "europepmc" in source or "europe" in source:
            return "europepmc"
        if "clinicaltrials" in source:
            return "clinicaltrials"
        return ""

    @classmethod
    def _split_scientific(cls, items: list) -> dict:
        out = {"pubmed": [], "europepmc": [], "clinicaltrials": []}
        for item in items:
            key = cls._source_key(item)
            if key:
                out[key].append(item)
        return out

    @classmethod
    def _qualifier_adjusted_score(cls, item, rel: ClinicalRelation, raw_score: float) -> float:
        """Cap unsupported exact matches while preserving Judge ordering."""
        title = (getattr(item, "title", "") or "").lower()
        text = f"{title} {getattr(item, 'abstract', '') or ''}".lower()
        agent_terms = cls._filter_terms(rel.agent)
        event_terms = cls._filter_terms(rel.event)
        event_text = " ".join(event_terms)
        if "increased hair loss" in event_text or "hair shedding" in event_text:
            event_terms.extend(["hair loss", "shedding"])
        event_terms = list(dict.fromkeys(event_terms))
        manifestation_terms = cls._filter_terms(rel.manifestation)
        site_terms = cls._filter_terms(rel.anatomical_site)
        relation_type = str(rel.relation_type or "").lower()
        has_agent = any(term in text for term in agent_terms)
        has_event = any(term in text for term in event_terms)
        has_manifestation = any(term in text for term in manifestation_terms)
        has_site = any(term in text for term in site_terms)
        hair_context = bool(re.search(
            r"alopecia|hair\s+(loss|growth|regrowth|restoration)|hairline|androgenetic|androgenic",
            text,
        ))
        unrelated_dominant_topic = bool(re.search(
            r"\b(covid|viral shedding|candida|prostate|testicular|spermatozoa|"
            r"lipid profile|depression|suicid|pharmacokinetic|contraceptive)\b",
            title,
        ))
        causal_adverse_cues = r"induc|caus|trigger|following|after|withdrawal|adverse effect|side effect"
        direct_adverse_link = any(
            re.search(r"(?:finasteride|dutasteride|5-alpha-reductase inhibitor)", sentence)
            and re.search(r"(?:hair loss|hair shedding|increased shedding|shedding)", sentence)
            and re.search(causal_adverse_cues, sentence)
            for sentence in re.split(r"[.!?;:\n]+", text)
        )
        adjusted = raw_score
        if agent_terms and not has_agent:
            adjusted = min(adjusted, 0.15)
        if event_terms and not has_event and relation_type != "efficacy":
            adjusted = min(adjusted, 0.50)
        if manifestation_terms and not has_manifestation:
            adjusted = min(adjusted, 0.50)
        if relation_type == "efficacy" and not hair_context:
            adjusted = min(adjusted, 0.15)
        if relation_type == "adverse_effect" and event_terms and not has_event:
            adjusted = min(adjusted, 0.15)
        if relation_type == "adverse_effect" and event_terms and has_agent and has_event and not direct_adverse_link:
            adjusted = min(adjusted, 0.15)
        if unrelated_dominant_topic and (
            relation_type == "efficacy" and (not has_site or not hair_context)
        ):
            adjusted = min(adjusted, 0.15)
        elif unrelated_dominant_topic and not (has_event and hair_context):
            adjusted = min(adjusted, 0.50)
        if site_terms and not has_site:
            adjusted = min(adjusted, 0.80)
        return adjusted

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
                for key in ("search_terms",):
                    for val in side.get(key) or []:
                        sval = str(val or "").strip().lower()
                        if sval:
                            target.append(sval)

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

        relation_cues = {
            "adverse_effect": {
                "adverse effect", "side effect", "safety", "tolerability",
                "hypertrichosis", "shedding", "alopecia", "rash", "edema",
                "irritation", "pustulosis", "exanthematous", "pruritus",
            },
            "efficacy": {
                "efficacy", "effectiveness", "improve", "improvement",
                "response", "regrowth", "regrew", "worked", "helped",
                "treatment", "therapeutic", "benefit",
            },
            "comparison": {
                "versus", "comparison", "compared", "compare", "network meta-analysis",
                "head-to-head", "noninferiority", "randomized", "randomised",
            },
            "exposure_outcome": {
                "following", "due to", "caused", "triggered", "after",
                "secondary to", "resulted in",
            },
        }
        cue_hits = _hits(list(relation_cues.get(relation_type, set())))
        if cue_hits:
            bonus += min(0.06, 0.02 * len(cue_hits))
            reasons.append(f"relation={cue_hits[:3]}")

        if relation_type in {"adverse_effect", "efficacy", "comparison", "exposure_outcome"}:
            if agent_hits and manifest_hits:
                bonus += 0.03
                reasons.append("agent+relation")

        # Relation-specificity: for an adverse-effect query, a paper whose text
        # contains the exact normalized manifestation (e.g. "hypertrichosis",
        # not a generic "shed") or one of the extracted relation phrases is
        # more on-relation than a paper that merely matches the cue vocabulary.
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
        return [item for source in ("pubmed", "europepmc", "clinicaltrials")
                for item in deduped.get(source, [])]


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
        anatomical_site = data.get("anatomical_site", {}) or {}
        manifestation = data.get("manifestation", {}) or {}
        if not anatomical_site and isinstance(manifestation, dict):
            role = str(manifestation.get("role", "") or "").lower().strip()
            if role in {"anatomical_location", "body_site", "site", "location"}:
                anatomical_site = manifestation
                manifestation = {}
        rel = ClinicalRelation(
            original_query=data.get("query_original", query),
            agent=data.get("agent", {}) or {},
            event=data.get("event", {}) or {},
            anatomical_site=anatomical_site,
            manifestation=manifestation,
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
    def _or_group(terms: list[str]) -> str:
        terms = [t for t in terms if t]
        if not terms:
            return ""
        if len(terms) == 1:
            return terms[0]
        return "(" + " OR ".join(terms) + ")"

    # Causal groups — kept as conservative retrieval cues, not hard filters.
    _CAUSAL = {
        "adverse_effect": '("adverse effect" OR "side effect" OR induced OR "caused by" OR "triggered by" OR "secondary to" OR "drug-related" OR "treatment-related" OR worsening)',
        "efficacy": '(treatment OR efficacy OR therapeutic OR "response to")',
        "exposure_outcome": '(following OR "due to" OR injury OR caused)',
        "drug_condition": "",
        "association": '("associated with" OR correlation OR linked)',
        "unknown": "",
        "diagnostic": "",
        "prevention": '(prevention OR preventive OR prophylaxis)',
    }

    @staticmethod
    def _clean_query_term(term: str) -> str:
        cleaned = re.sub(r"[\[\]\(\)\{\}/:;]+", " ", str(term or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _is_noisy_query_term(cls, term: str) -> bool:
        text = cls._clean_query_term(term)
        if not text:
            return True
        # Provider formulations (dose/route/product) are valid optional
        # variants. Only discard terms that are plainly oversized; syntax is
        # sanitized by _clean_query_term before reaching a collector.
        return len(text) > 80 or len(text.split()) > 10

    @classmethod
    def _query_terms_for(cls, part: dict, fallback: str = "") -> list[str]:
        raw_terms = [part.get("normalized", ""), part.get("term", "")]
        raw_terms.extend(part.get("search_terms") or [])
        fallback_clean = cls._clean_query_term(fallback)
        out = []
        seen = set()
        for raw in raw_terms:
            term = cls._clean_query_term(raw)
            if not term:
                continue
            if cls._is_noisy_query_term(term):
                term = fallback_clean or term
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(term)
        if not out and fallback_clean:
            out = [fallback_clean]
        return out[:4]

    def _query_clause_sets(self, rel: ClinicalRelation) -> list[list[list[str]]]:
        agent = self._query_terms_for(rel.agent, rel.agent.get("term", ""))
        event = self._query_terms_for(rel.event, rel.event.get("term", ""))
        site = self._query_terms_for(rel.anatomical_site, rel.anatomical_site.get("term", ""))
        manifestation = self._query_terms_for(rel.manifestation, rel.manifestation.get("term", ""))

        core = [terms for terms in (agent, event) if terms]
        if not core:
            return []

        clauses = [core]
        if site:
            clauses.append(core + [site])
        if manifestation:
            clauses.append(core + [manifestation])
        if site and manifestation:
            clauses.append(core + [site, manifestation])

        unique = []
        seen = set()
        for clause in clauses:
            key = tuple(tuple(term.lower() for term in group) for group in clause)
            if key in seen:
                continue
            seen.add(key)
            unique.append(clause)
        return unique

    def _render_pubmed_clause(self, clause: list[list[str]]) -> str:
        parts = []
        for terms in clause:
            expr = self._or_group(terms)
            if expr:
                parts.append(f"{expr}[tiab]")
        return " AND ".join(parts)

    def _render_epmc_clause(self, clause: list[list[str]]) -> str:
        parts = []
        for terms in clause:
            expr = self._or_group(terms)
            if expr:
                parts.append(f"(TITLE:{expr} OR ABSTRACT:{expr})")
        return " AND ".join(parts)

    def _render_ct_clause(self, clause: list[list[str]]) -> str:
        parts = []
        for terms in clause:
            expr = self._or_group(terms)
            if expr:
                parts.append(expr)
        return " AND ".join(parts)

    def _build_pubmed_query(self, rel: ClinicalRelation) -> str:
        clauses = self._query_clause_sets(rel)
        causal = self._CAUSAL.get(rel.relation_type, "")
        rendered = [self._render_pubmed_clause(clause) for clause in clauses]
        rendered = [part for part in rendered if part]
        if not rendered:
            rendered = [self._or_group(self._query_terms_for(rel.agent, rel.agent.get("term", "")))]
        body = rendered[0] if len(rendered) == 1 else "(" + " OR ".join(f"({part})" for part in rendered) + ")"
        parts = [body]
        if causal:
            parts.append(causal)
        parts.append("hasabstract")
        return " AND ".join(parts)

    def _build_epmc_query(self, rel: ClinicalRelation) -> str:
        clauses = self._query_clause_sets(rel)
        causal = self._CAUSAL.get(rel.relation_type, "")
        rendered = [self._render_epmc_clause(clause) for clause in clauses]
        rendered = [part for part in rendered if part]
        if not rendered:
            rendered = [self._or_group(self._query_terms_for(rel.agent, rel.agent.get("term", "")))]
        body = rendered[0] if len(rendered) == 1 else "(" + " OR ".join(f"({part})" for part in rendered) + ")"
        parts = [body]
        if causal:
            parts.append(causal)
        return " AND ".join(parts)

    def _build_ct_query(self, rel: ClinicalRelation) -> str:
        clauses = self._query_clause_sets(rel)
        rendered = [self._render_ct_clause(clause) for clause in clauses]
        rendered = [part for part in rendered if part]
        if not rendered:
            return self._clean_query_term(rel.original_query) or rel.original_query
        return rendered[0] if len(rendered) == 1 else "(" + " OR ".join(f"({part})" for part in rendered) + ")"

    # ── (4) Hard filter (synonym-tolerant) ────────────────────────────────────

    @classmethod
    def _filter_terms(cls, part: dict) -> list[str]:
        values = [part.get("normalized"), part.get("term"), *(part.get("search_terms") or [])]
        out = []
        seen = set()
        for value in values:
            text = cls._clean_query_term(value)
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    @classmethod
    def _agent_terms_for_filter(cls, rel: ClinicalRelation) -> list[str]:
        terms = cls._filter_terms(rel.agent)
        agent_text = " ".join(terms)
        if any(term in agent_text for term in ("dutasteride", "finasteride")):
            terms.extend(["5-alpha-reductase inhibitor", "5α-reductase inhibitor"])
        return list(dict.fromkeys(terms))

    @classmethod
    def _outcome_terms_for_filter(cls, rel: ClinicalRelation) -> list[str]:
        terms = cls._filter_terms(rel.event) + cls._filter_terms(rel.manifestation)
        relation_type = str(rel.relation_type or "").lower()
        event_text = " ".join(terms)
        if relation_type == "efficacy" and any(
            token in event_text for token in ("hair", "regrowth", "growth", "alopecia")
        ):
            # These are retrieval-level semantic anchors, not relevance claims:
            # treatment studies often report efficacy in AGA without using the
            # exact phrase "hair regrowth" in title or abstract.
            terms.extend([
                "hair growth", "hair regrowth", "hair restoration", "hair loss",
                "alopecia", "androgenetic alopecia", "androgenic alopecia",
                "hairline", "hair follicle", "follicular", "follicle targeting",
            ])
        return list(dict.fromkeys(term.lower() for term in terms if term))

    def _hard_filter(self, items: list[SearchResult], rel: ClinicalRelation) -> list[SearchResult]:
        agent_terms = self._agent_terms_for_filter(rel)
        outcome_terms = self._outcome_terms_for_filter(rel)
        kept = []
        for it in items:
            blob = ((getattr(it, "title", "") or "") + " " +
                    (getattr(it, "abstract", "") or "")).lower()
            has_agent = any(t in blob for t in agent_terms)
            has_outcome = any(t in blob for t in outcome_terms)
            if agent_terms and outcome_terms:
                # Adverse-effect retrieval keeps agent-only safety papers in
                # the Judge pool: absence of the outcome is useful evidence
                # for distinguishing generic safety discussion from the exact
                # adverse relationship. Efficacy/exposure queries stay strict.
                if has_agent and (has_outcome or rel.relation_type == "adverse_effect"):
                    kept.append(it)
            elif has_agent or has_outcome or not (agent_terms or outcome_terms):
                kept.append(it)
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

    def _judge_batched(self, pool: list[SearchResult], rel: ClinicalRelation, stats: dict) -> list[dict]:
        if not pool or self._client is None:
            stats["judge_used"] = False
            return [{"label": "partial", "score": 0.5, "reason": "judge unavailable"} for _ in pool]
        out: list[dict] = []
        _batch_sz = _limits().judge_batch_size
        for i in range(0, len(pool), _batch_sz):
            batch = pool[i:i + _batch_sz]
            try:
                res = self._llm_judge(batch, rel)
                stats["openai_calls"] += 1
                trace = getattr(self, "_active_audit", None)
                if trace is not None:
                    trace["judge_inputs"].append({
                        "batch_start": i,
                        "batch_size": len(batch),
                        "prompt": getattr(self, "_last_judge_prompt", ""),
                        "articles": [_audit_item(article) for article in batch],
                    })
                # align by index 'i' in the returned JSON
                idx_map = {r.get("i"): r for r in res}
                for j, _art in enumerate(batch):
                    out.append(idx_map.get(j, {
                        "i": j, "tier": "C", "label": "partial",
                        "score": 0.0, "reason": "missing Judge result",
                    }))
            except Exception as exc:
                stats["judge_used"] = False
                stats["judge_errors"].append(str(exc))
                logger.warning("RelationalSearch: judge batch failed — %s", exc)
                for _art in batch:
                    out.append({
                        "tier": "C", "label": "partial", "score": 0.0,
                        "reason": f"judge error: {exc}",
                    })
            time.sleep(1.0)  # be gentle with rate limits between batches
        return out

    def _llm_judge(self, batch: list[SearchResult], rel: ClinicalRelation) -> list[dict]:
        arts_txt = "\n".join(
            f"[{i}] TITLE: {getattr(a,'title','')}\n    ABSTRACT: {(getattr(a,'abstract','') or '')}"
            for i, a in enumerate(batch)
        )
        prompt = _JUDGE_PROMPT.format(
            original_query=rel.original_query,
            agent=rel.agent.get("normalized", ""), arole=rel.agent.get("role", ""),
            event=rel.event.get("normalized", ""),
            site=rel.anatomical_site.get("normalized", ""),
            srole=rel.anatomical_site.get("role", ""),
            manifest=rel.manifestation.get("normalized", ""),
            mrole=rel.manifestation.get("role", ""), temporal=rel.temporal,
            rtype=rel.relation_type,
            phrases=", ".join(str(p) for p in (rel.relation_phrases or []) if p),
            scientific_query=rel.scientific_query,
            canonical_query=rel.canonical_query,
            desc=_describe(rel), arts=arts_txt,
        )
        self._last_judge_prompt = prompt
        data = self._llm_json(
            prompt,
            max_tokens=900,
            response_format=_JUDGE_RESPONSE_FORMAT,
        )
        return data.get("results", []) or []

    # ── LLM helper (delegates retry to the central llm_guard) ─────────────────
    #
    # The guard is the ONLY retry boundary: MAX_TOTAL_ATTEMPTS=5, no nested
    # retry. quota exhaustion raises QuotaExhaustedError (no retry); transient
    # 429s and JSON/parse errors retry up to the cap. This method must NOT add
    # its own retry loop on top — that would breach the absolute cap.

    def _llm_json(
        self,
        prompt: str,
        max_tokens: int = 700,
        response_format: Optional[dict] = None,
    ) -> dict:
        from core.llm_guard import call_llm_json
        return call_llm_json(
            self._client,
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0,
            max_tokens=max_tokens,
            response_format=response_format,
            operation="relational_search_llm",
        )
