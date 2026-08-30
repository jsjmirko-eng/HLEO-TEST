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

import copy
import json
import logging
import re
import time
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
JUDGE_BATCH = 5          # articles per judge LLM call
JUDGE_POOL_PER_SOURCE = 10  # top-N candidates judged per source


# ── Clinical relation model ──────────────────────────────────────────────────

@dataclass
class ClinicalRelation:
    """Structured clinical relationship extracted from the user query."""
    original_query: str
    agent: dict = field(default_factory=dict)           # {term,normalized,role,identified,search_terms}
    event: dict = field(default_factory=dict)           # {term,normalized}
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
  "manifestation": {"term": str, "normalized": str, "role": str, "search_terms": [str]},
  "temporal": str,
  "relation_type": str,
  "scientific_query": str,
  "relation_phrases": [str],
  "fallback_needed": bool
}

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
ginocchio->knee; esposizione al sole->"sun exposure"; eritema->erythema).

Query: __QUERY__"""


# ── Prompt: relational LLM judge (batched) ───────────────────────────────────

_JUDGE_PROMPT = """You are a strict biomedical relevance judge.

The user is asking about this CLINICAL RELATION:
  agent: {agent} ({arole})
  event: {event}
  manifestation: {manifest} ({mrole})
  temporal: {temporal}
  relation_type: {rtype}
  relation description: {desc}

For EACH article below, decide whether it substantively discusses THIS RELATIONSHIP
(the agent causing/being associated with the manifestation as an adverse effect / exposure
outcome, or the agent treating the condition if efficacy), versus merely mentioning the
words separately or discussing a different topic.

Assign:
  label: "relevant" | "partial" | "not_relevant"
  score: 0.0-1.0  (1.0 = the relation is a central focus; 0.5 = both terms present but
                   relation not the focus; 0.0 = no real relation)
  reason: one short sentence

Return ONLY JSON: {{"results":[{{"i":int,"label":str,"score":float,"reason":str}}]}}

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

    # Process-lifetime cache for relation extraction (identical query → same relation)
    _rel_cache: dict[str, ClinicalRelation] = {}

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

    # ── Public API ───────────────────────────────────────────────────────────

    def search(self, raw_query: str) -> Optional[dict]:
        """Run relation extraction, pre-retrieval expansion, and global ranking."""
        if self._client is None:
            return None
        t0 = time.perf_counter()
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
        for variant, provenance in expanded:
            for source, (collector, builder) in collectors.items():
                query = builder(variant)
                try:
                    items = collector.search(query, limit=None)
                    stats["query_calls"] += 1
                except Exception as exc:
                    logger.warning("Scientific %s retrieval failed: %s", source, exc)
                    items = []
                for item in items:
                    item.metadata = dict(item.metadata or {})
                    item.metadata.setdefault("match_provenance", []).append(provenance)
                    item.metadata.setdefault("matched_queries", []).append(query)
                    raw[source].append(item)

        stats["candidates_raw"] = {k: len(v) for k, v in raw.items()}
        candidates = self._deduplicate_scientific(raw)
        stats["candidates_deduped"] = len(candidates)
        grouped = self._split_scientific(candidates)
        candidates = [item for items in grouped.values() for item in items]
        stats["after_soft_relation_pass"] = len(candidates)
        stats["after_hard_filter"] = len(candidates)
        stats["hard_filter_applied"] = False

        from core.ranker import clinical_rank
        candidates.sort(
            key=lambda item: clinical_rank(item) + (self._relation_bonus(item, rel)[0] * 100.0),
            reverse=True,
        )
        judged_count = 0
        remaining = candidates
        while remaining and judged_count < len(candidates) and judged_count < 400:
            pool = remaining[:300]
            judgements = self._judge_batched(pool, rel, stats)
            for item, judgement in zip(pool, judgements):
                raw_score = max(0.0, min(1.0, float(judgement.get("score", 0.0))))
                relation_bonus, relation_reasons = self._relation_bonus(item, rel)
                final_score = min(1.0, max(0.0, raw_score + relation_bonus))
                # final_score * 1000 dominates (judge ordering wins);
                # relation_bonus * 50 breaks ties in favour of relation-specific
                # articles; clinical_rank * 0.5 is the last tie-break.
                item.score = round(
                    final_score * 1000.0 + (relation_bonus * 50.0)
                    + (clinical_rank(item) * 0.5), 2)
                item.metadata = dict(item.metadata or {})
                item.metadata.update({
                    "relevance_label": judgement.get("label", "not_relevant"),
                    "relevance_score": final_score,
                    "relevance_reason": judgement.get("reason", ""),
                    "final_score": final_score,
                    "judge_score_raw": raw_score,
                    "relation_bonus": relation_bonus,
                    "relation_reasons": relation_reasons,
                })
            judged_count += len(pool)
            remaining = candidates[judged_count:]
            ranked_now = sorted(candidates[:judged_count], key=lambda item: item.score,
                                reverse=True)
            if sum(float((item.metadata or {}).get("final_score", 0.0)) >= 0.20
                   for item in ranked_now) >= 400 or not remaining:
                break
        candidates.sort(key=lambda item: item.score, reverse=True)
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
        stats["elapsed_s"] = round(time.perf_counter() - t0, 2)
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
            for match in resolution.matches:
                tier = MATCH_TIERS.get(match.match_kind)
                if tier is None:
                    continue
                for term in [match.preferred_term, *match.synonyms]:
                    term = (term or "").strip()
                    if len(term) < 3 or term.lower() == source_entity.lower():
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
        rel = ClinicalRelation(
            original_query=data.get("query_original", query),
            agent=data.get("agent", {}) or {},
            event=data.get("event", {}) or {},
            manifestation=data.get("manifestation", {}) or {},
            temporal=data.get("temporal", ""),
            relation_type=data.get("relation_type", "unknown"),
            scientific_query=data.get("scientific_query", ""),
            relation_phrases=data.get("relation_phrases", []) or [],
            fallback_needed=bool(data.get("fallback_needed", False)),
        )
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

    # Causal groups — tightened: no bare "effect", no "following"/"associated with" mandatory
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

    def _build_pubmed_query(self, rel: ClinicalRelation) -> str:
        ag = self._or_group(rel.agent.get("search_terms") or [rel.agent.get("normalized", "")])
        mn = self._or_group(rel.manifestation.get("search_terms") or [rel.manifestation.get("normalized", "")])
        causal = self._CAUSAL.get(rel.relation_type, "")
        # [tiab] restricts to title/abstract; hasabstract ensures an abstract exists.
        parts = []
        if ag:
            parts.append(f"{ag}[tiab]")
        if mn:
            parts.append(f"{mn}[tiab]")
        if causal:
            parts.append(causal)
        parts.append("hasabstract")
        return " AND ".join(parts)

    def _build_epmc_query(self, rel: ClinicalRelation) -> str:
        ag = self._or_group(rel.agent.get("search_terms") or [rel.agent.get("normalized", "")])
        mn = self._or_group(rel.manifestation.get("search_terms") or [rel.manifestation.get("normalized", "")])
        causal = self._CAUSAL.get(rel.relation_type, "")
        # Force the agent into TITLE or ABSTRACT (measured ~100% agent presence),
        # require manifestation, optional causal group.
        parts = []
        if ag:
            parts.append(f"(TITLE:{ag} OR ABSTRACT:{ag})")
        if mn:
            parts.append(mn)
        if causal:
            parts.append(causal)
        return " AND ".join(parts)

    def _build_ct_query(self, rel: ClinicalRelation) -> str:
        # ClinicalTrials v2 query.term is free-text; use agent + manifestation.
        ag = rel.agent.get("normalized", "")
        mn = rel.manifestation.get("normalized", "")
        return f"{ag} {mn}".strip() or rel.original_query

    # ── (4) Hard filter (synonym-tolerant) ────────────────────────────────────

    def _hard_filter(self, items: list[SearchResult], rel: ClinicalRelation) -> list[SearchResult]:
        agent_terms = [t.lower() for t in (rel.agent.get("search_terms") or []) if t]
        if rel.agent.get("normalized"):
            agent_terms.append(rel.agent["normalized"].lower())
        mani_terms = [t.lower() for t in (rel.manifestation.get("search_terms") or []) if t]
        if rel.manifestation.get("normalized"):
            mani_terms.append(rel.manifestation["normalized"].lower())
        for entity, target in ((rel.agent.get("normalized"), agent_terms),
                               (rel.manifestation.get("normalized"), mani_terms)):
            for entry in rel.vocabulary.get(entity, []):
                if entry.get("match_kind") == "related_concept":
                    continue
                target.extend(
                    str(term).lower() for term in
                    [entry.get("preferred_term"), *(entry.get("synonyms") or [])]
                    if term
                )
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
        pool = items[:JUDGE_POOL_PER_SOURCE]
        tail = items[JUDGE_POOL_PER_SOURCE:]

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
        for i in range(0, len(pool), JUDGE_BATCH):
            batch = pool[i:i + JUDGE_BATCH]
            try:
                res = self._llm_judge(batch, rel)
                stats["openai_calls"] += 1
                # align by index 'i' in the returned JSON
                idx_map = {r.get("i"): r for r in res}
                for j, _art in enumerate(batch):
                    out.append(idx_map.get(j, {"label": "partial", "score": 0.5, "reason": "missing"}))
            except Exception as exc:
                stats["judge_used"] = False
                stats["judge_errors"].append(str(exc))
                logger.warning("RelationalSearch: judge batch failed — %s", exc)
                for _art in batch:
                    out.append({"label": "partial", "score": 0.5, "reason": f"judge error: {exc}"})
            time.sleep(1.0)  # be gentle with rate limits between batches
        return out

    def _llm_judge(self, batch: list[SearchResult], rel: ClinicalRelation) -> list[dict]:
        arts_txt = "\n".join(
            f"[{i}] TITLE: {getattr(a,'title','')}\n    ABSTRACT: {(getattr(a,'abstract','') or '')[:700]}"
            for i, a in enumerate(batch)
        )
        prompt = _JUDGE_PROMPT.format(
            agent=rel.agent.get("normalized", ""), arole=rel.agent.get("role", ""),
            event=rel.event.get("normalized", ""), manifest=rel.manifestation.get("normalized", ""),
            mrole=rel.manifestation.get("role", ""), temporal=rel.temporal,
            rtype=rel.relation_type, desc=_describe(rel), arts=arts_txt,
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
