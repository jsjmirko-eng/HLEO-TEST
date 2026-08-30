"""
RWE relation-precision filter — Levels A/B/C + structured profile extraction.

Precision-first gate applied to RWE results AFTER the semi-semantic relevance
filter. An item is pertinent to an agent→manifestation query only when ALL
three levels hold:

  Level A (AGENTE)        — the item actually refers to the queried agent
                            (canonical + provider-verified identity variants,
                            e.g. finasteride / propecia / proscar).
  Level B (MANIFESTAZIONE)— the item actually expresses the queried
                            manifestation via a specific clinical phrase
                            ("erectile dysfunction", "libido loss", ...). An
                            isolated generic word ("sexual", "dysfunction",
                            "effects") is NOT sufficient.
  Level C (RELAZIONE)     — agent and manifestation are LINKED in the
                            experience: same sentence, close text window,
                            title co-occurrence, or a structured
                            pharmacovigilance record (drug field + reaction
                            field). Mere co-presence in the same document is
                            not enough.

The gate is deterministic (no LLM, no network) and activates ONLY when the
query plan exposes a structured agent→manifestation relation; otherwise the
pipeline behaviour is unchanged. No terminology is hardcoded: agent and
manifestation term sets come from the plan's provider-recognised entities,
their query surfaces, and their Catena C provider variants. RWE items stay
patient-reported evidence: the gate never upgrades a testimony to a causal
claim — it only certifies that the testimony is ON-TOPIC for the queried
relation.

Each kept item is re-ranked (relation-aware boost / other-agent penalty) and
stamped with:
  metadata["relation_gate"] — per-level verdicts + matched terms;
  metadata["rwe_profile"]   — best-effort structured profile (empty fields
                              when the information is absent — never invented).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.rwe.intent import GENERIC_EVENT_TERMS
from core.rwe.models import RWEItem

# Max character distance between an agent mention and a manifestation mention
# for Level C to consider them linked in free text.
_RELATION_WINDOW = 300

_WORD_RE = re.compile(r"[a-zà-öø-ÿ0-9]+")
_SENTENCE_SPLIT = re.compile(r"[.!?\n\r;]+")

# Tokens of the generic any-event vocabulary (existing RWE canonical set) can
# never, alone, prove a specific manifestation.
_GENERIC_EVENT_TOKENS = {
    tok for term in GENERIC_EVENT_TERMS for tok in _WORD_RE.findall(term.lower())
}

# Isolated ambiguous modifiers: explicitly insufficient evidence of the
# manifestation when they appear ALONE (not part of a clinical phrase).
_AMBIGUOUS_LONE_TOKENS = _GENERIC_EVENT_TOKENS | {
    "sexual", "dysfunction", "dysfunctions",
}

# First-person / testimonial cues reused for the profile relation strength
# (aligned with the existing pipeline cue style, EN/IT/FR/ES).
_TESTIMONIAL_CUES = {
    "i started", "since i started", "since starting", "after taking",
    "after using", "when i started", "i got", "i had", "i noticed",
    "it gave me", "gave me", "made me", "i developed", "i experienced",
    "ho iniziato", "dopo aver", "da quando", "mi ha dato", "ho avuto",
    "mi è venuto", "mi è comparso", "j'ai", "après avoir", "me dio",
}

_DISCONTINUED_CUES = {
    "stopped", "came off", "quit", "discontinued", "gave up on",
    "sospeso", "interrotto", "sospensione", "ho smesso", "arrêté",
    "dejé", "lo dejé",
}
_IMPROVED_CUES = {
    "improved", "recovered", "went away", "resolved", "back to normal",
    "recuperato", "migliorato", "scomparsi", "scomparso", "résolu",
}
_WORSENED_CUES = {
    "worse", "worsened", "worsening", "got worse", "peggiorato",
    "peggioramento", "empiré", "empeorado",
}
_PERSISTENT_CUES = {
    "persistent", "still no", "still have", "hasn't come back",
    "never came back", "persiste", "persistente", "ancora",
}
_RECHALLENGE_CUES = {
    "restarted", "started again", "tried again", "rechallenge",
    "retried", "ripreso", "riprovato", "recommencé", "reintenté",
}

_DOSE_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s?(mg|mcg|µg|g)\b", re.IGNORECASE)
_FREQ_RE = re.compile(
    r"\b(\d+\s?(?:times?|volte)\s+(?:a|per)\s+week|every other day|eod|"
    r"\d+\s?(?:times?|volte)\s+(?:a|per)\s+(?:day|giorno))\b", re.IGNORECASE)
_ONSET_RE = re.compile(
    r"\b(?:after|since|within|dopo|da quando|après|después)\s+"
    r"(\d+(?:[.,]\d+)?)\s?"
    r"(years?|yrs?|months?|weeks?|days?|anni|mesi|settimane|giorni|mois|"
    r"semaines|jours)\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\b(?:for|da|depuis|desde|per)?\s*(\d+(?:[.,]\d+)?)\s?"
    r"(years?|yrs?|months?|weeks?|anni|mesi|settimane|mois|semaines)\b",
    re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _occurrences(body: str, term: str) -> List[int]:
    """Start offsets of phrase ``term`` in ``body`` (word-boundary match)."""
    if not term:
        return []
    return [m.start() for m in re.finditer(
        rf"(?<!\w){re.escape(term)}(?!\w)", body)]


def _sentences(body: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(body or "") if s.strip()]


def _is_generic_surface(surface: str) -> bool:
    """True when the query surface is itself a generic any-event phrase
    ("side effects", "adverse effects", ...): the user asked about ANY event,
    so no specific manifestation gate can be derived from it."""
    toks = _tokens(surface)
    if not toks:
        return True
    joined = " ".join(toks)
    return joined in GENERIC_EVENT_TERMS or all(
        t in _GENERIC_EVENT_TOKENS for t in toks)


def _is_specific_term(term: str) -> bool:
    """A manifestation term proves Level B only if it is clinically specific:
    a multi-token phrase (not itself a generic any-event phrase) or a
    non-ambiguous single token ("impotence" yes; "sexual"/"dysfunction" no).
    """
    toks = _tokens(term)
    if not toks:
        return False
    if len(toks) == 1:
        return len(toks[0]) >= 5 and toks[0] not in _AMBIGUOUS_LONE_TOKENS
    return " ".join(toks) not in GENERIC_EVENT_TERMS


def _canonical_base(canonical: str) -> str:
    """Canonical without provider qualifier suffixes: 'sexual dysfunction,
    physiological' → 'sexual dysfunction'."""
    base = (canonical or "").split(",")[0].strip().lower()
    return base


def _provider_terms(vocabulary: Optional[dict], canonical: str) -> List[str]:
    """Identity-tier provider variants of a canonical from the plan's slim
    vocabulary (Catena C evidence; related_concept never scored)."""
    entries = (vocabulary or {}).get(canonical.lower())
    if not entries:
        return []
    from core.vocab.models import VocabularyResolution
    return list(VocabularyResolution.from_slim(canonical, entries).scored_terms())


def _related_concept_terms(vocabulary: Optional[dict], canonical: str) -> List[str]:
    """Provider related_concept terms of the anchor drug (evidence only):
    used to detect items that are primarily about ANOTHER agent."""
    out: List[str] = []
    for e in (vocabulary or {}).get(canonical.lower(), []):
        if e.get("match_kind") != "related_concept":
            continue
        # Only provider-typed DRUG concepts are credible "other agent"
        # signals — general-purpose related concepts ("medicine", "drug",
        # "pharmaceutical") would penalise arbitrary items.
        if e.get("semantic_group") not in {"drug", "active_ingredient"}:
            continue
        for t in [e.get("preferred_term"), *(e.get("synonyms") or [])]:
            t = (t or "").lower().strip()
            if len(t) >= 3 and "/" not in t:
                out.append(t)
    return out


@dataclass
class RelationContext:
    """Agent→manifestation term sets derived from the query plan."""
    agent_terms: List[str] = field(default_factory=list)
    manifestation_terms: List[str] = field(default_factory=list)
    manifestation_tokens: set = field(default_factory=set)
    other_agent_terms: List[str] = field(default_factory=list)
    agent_canonical: str = ""


def build_relation_context(plan) -> Optional[RelationContext]:
    """Build the RelationContext from a RWEQueryPlan, or None when the query
    has no structured agent→manifestation relation (gate stays OFF and the
    pipeline keeps its previous behaviour)."""
    entities = getattr(plan, "entities", None) or []
    vocabulary = getattr(plan, "vocabulary", None) or {}
    surfaces = getattr(plan, "surfaces", None) or {}

    ctx = RelationContext()
    manifestation_surfaces: List[str] = []
    intent_pre = getattr(plan, "intent", None)
    _intent_outcomes_multi = [
        str(o).lower() for o in
        (getattr(intent_pre, "outcomes", None) or [])
        if len(_tokens(str(o))) >= 2] if intent_pre is not None else []
    for etype, canonical, _conf in entities:
        canon = (canonical or "").lower()
        surface = (surfaces.get(canonical) or canon).lower()
        if etype in ("drug", "active_ingredient"):
            ctx.agent_canonical = ctx.agent_canonical or canon
            ctx.agent_terms.append(canon)
            ctx.agent_terms.append(surface)
            ctx.agent_terms.extend(_provider_terms(vocabulary, canon))
            ctx.other_agent_terms.extend(_related_concept_terms(vocabulary, canon))
        elif etype in ("symptom", "adverse_effect"):
            if _is_generic_surface(surface):
                continue  # generic any-event side: no specific gate
            # Generic-hypernym guard (demonstrated precision bug): an event
            # entity that is an IDENTITY match on a single generic token
            # (canonical == surface, e.g. "pain" resolved from the "pain"
            # token inside "joint pain") is a HYPERNYM of the specific event
            # the query asks about. Its provider variants ("abdominal pain",
            # "back pain") would broaden Level B beyond the requested event.
            # Conceptual mappings (canonical != surface, e.g. "shedding" →
            # "alopecia") are kept — they are the intended provider value.
            canon_base = _canonical_base(canon)
            if (_intent_outcomes_multi
                    and canon_base == surface
                    and len(_tokens(canon_base)) == 1
                    and any(set(_tokens(canon_base)) < set(_tokens(o))
                            for o in _intent_outcomes_multi)):
                continue
            manifestation_surfaces.append(surface)
            ctx.manifestation_terms.append(canon_base)
            ctx.manifestation_terms.append(surface)
            ctx.manifestation_terms.extend(_provider_terms(vocabulary, canon))

    # QU intent sides (LLM-structured, feature-flagged upstream): outcomes and
    # their QU synonyms join the manifestation side, interventions the agent
    # side — mirroring merged_sides in the V3 scorer.
    intent = getattr(plan, "intent", None)
    if intent is not None:
        synonyms = getattr(intent, "synonyms", None) or {}
        interventions = [str(t).lower() for t in
                         (getattr(intent, "interventions", None) or [])]
        outcomes = [str(t).lower() for t in
                    (getattr(intent, "outcomes", None) or [])]
        for t in interventions:
            ctx.agent_terms.append(t)
            ctx.agent_terms.extend(
                str(a).lower() for a in synonyms.get(t, []) or [])
        for t in outcomes:
            if _is_generic_surface(t):
                continue
            ctx.manifestation_terms.append(t)
            ctx.manifestation_terms.extend(
                str(a).lower() for a in synonyms.get(t, []) or [])
            manifestation_surfaces.append(t)

    # Specificity guard + dedup (order preserved).
    ctx.agent_terms = [t for t in dict.fromkeys(ctx.agent_terms) if len(t) >= 3]
    ctx.manifestation_terms = [
        t for t in dict.fromkeys(ctx.manifestation_terms) if _is_specific_term(t)]
    ctx.other_agent_terms = [
        t for t in dict.fromkeys(ctx.other_agent_terms)
        if t not in set(ctx.agent_terms)]
    ctx.manifestation_tokens = {
        tok for s in manifestation_surfaces for tok in _tokens(s)
        if tok not in _GENERIC_EVENT_TOKENS
    }

    if not ctx.agent_terms or not ctx.manifestation_terms:
        return None
    return ctx


def extract_rwe_profile(
    item: RWEItem,
    agent_hit: str,
    manifestation_hit: str,
    relation_kind: str,
) -> Dict[str, object]:
    """Best-effort structured RWE profile (deterministic, text-cue based).

    Fields stay empty when the information is absent — nothing is invented.
    The profile describes a patient-reported experience; it is NOT a clinical
    causal assertion.
    """
    body = f"{item.title or ''} {item.text or ''}"
    lower = body.lower()

    onset = _ONSET_RE.search(lower)
    duration = _DURATION_RE.search(lower)
    dose = _DOSE_RE.search(lower)
    freq = _FREQ_RE.search(lower)

    outcome = ""
    for label, cues in (("improved", _IMPROVED_CUES),
                        ("worsened", _WORSENED_CUES),
                        ("persistent", _PERSISTENT_CUES)):
        if any(c in lower for c in cues):
            outcome = label
            break

    return {
        "agent": agent_hit or "",
        "manifestation": manifestation_hit or "",
        "relation": relation_kind or "",
        "onset": onset.group(0) if onset else "",
        "duration": duration.group(0).strip() if duration else "",
        "dose": (f"{dose.group(1)}{dose.group(2).lower()}" if dose else "")
                + (f" {freq.group(0)}" if freq else ""),
        "discontinued": any(c in lower for c in _DISCONTINUED_CUES),
        "outcome": outcome,
        "rechallenge": any(c in lower for c in _RECHALLENGE_CUES),
        "context": "personal_testimony" if any(c in lower for c in _TESTIMONIAL_CUES)
                   else "discussion",
    }


# Negation cues: a manifestation mention preceded shortly by a negation
# ("never gave me any sexual side effects") is a NEGATED relation — the
# agent explicitly did NOT cause the event — not evidence for it.
_NEGATION_TOKENS = {
    "never", "no", "not", "didn't", "didnt", "don't", "dont", "without",
    "none", "non", "nessun", "nessuna", "nessuno", "jamais", "aucun",
    "aucune", "kein", "keine", "keinen", "nunca", "jamás",
}


def _negated_before(text: str, pos: int, lookback: int = 60) -> bool:
    """True when a negation cue appears shortly BEFORE the mention at pos."""
    seg = (text or "")[max(0, pos - lookback):pos]
    toks = [t.strip(".,;:!?()\"'«»") for t in seg.split()]
    return any(t in _NEGATION_TOKENS for t in toks)


def _level_c_relation(
    body: str,
    title: str,
    agent_hits: List[str],
    manifestation_hits: List[str],
    window: int,
    manifestation_sentences: Optional[List[str]] = None,
) -> Tuple[bool, str, Optional[int]]:
    """Verify the agent↔manifestation LINK in free text.

    Returns (verified, kind, min_char_distance). Kinds, strongest first:
    'title' (both in the title), 'sentence' (same sentence), 'window'
    (≤ window chars apart); kind 'negated' (never verified) means every
    candidate link was negated in context ("never gave me any …"). Mere
    document co-presence is NOT a relation.
    ``manifestation_sentences`` covers the Level-B fallback (phrase tokens
    co-present in one sentence): the agent must share that sentence or be
    within ``window`` chars of it.
    """
    saw_negated = False

    for a in agent_hits:
        if not _occurrences(title, a):
            continue
        for m in manifestation_hits:
            for pm in _occurrences(title, m):
                if _negated_before(title, pm):
                    saw_negated = True
                    continue
                return True, "title", 0

    for sent in _sentences(body):
        for a in agent_hits:
            if not _occurrences(sent, a):
                continue
            for m in manifestation_hits:
                for pm in _occurrences(sent, m):
                    if _negated_before(sent, pm):
                        saw_negated = True
                        continue
                    return True, "sentence", None

    for sent in manifestation_sentences or []:
        for a in agent_hits:
            if _occurrences(sent, a):
                return True, "sentence", None
        pos = body.find(sent)
        if pos >= 0:
            span = (pos, pos + len(sent))
            for a in agent_hits:
                for pa in _occurrences(body, a):
                    d = min(abs(pa - span[0]), abs(pa - span[1]))
                    if span[0] <= pa <= span[1] or d <= window:
                        return True, "window", 0 if span[0] <= pa <= span[1] else d

    best: Optional[int] = None
    for a in agent_hits:
        for m in manifestation_hits:
            for pa in _occurrences(body, a):
                for pm in _occurrences(body, m):
                    d = abs(pa - pm)
                    if d <= window and _negated_before(body, pm):
                        saw_negated = True
                        continue
                    if best is None or d < best:
                        best = d
    if best is not None and best <= window:
        return True, "window", best
    return False, ("negated" if saw_negated else ""), best


def apply_relation_gate(
    items: List[RWEItem],
    plan,
    window: int = _RELATION_WINDOW,
) -> Tuple[List[RWEItem], Optional[dict]]:
    """Apply the Level A/B/C precision gate + relation-aware re-ranking.

    Returns (kept_items, stats). When the query has no structured
    agent→manifestation relation, returns (items, None) unchanged.
    """
    ctx = build_relation_context(plan)
    if ctx is None:
        return items, None

    stats = {
        "input": len(items),
        "after_agent": 0,
        "after_manifestation": 0,
        "after_relation": 0,
        "final": 0,
        "dropped_agent": 0,
        "dropped_manifestation": 0,
        "dropped_relation": 0,
        "dropped_relation_negated": 0,
    }
    kept: List[RWEItem] = []

    for it in items:
        title = (it.title or "").lower()
        body = f"{title}\n{(it.text or '').lower()}"
        treatment = (it.treatment or "").lower()
        condition = (it.condition or "").lower()
        is_authoritative = (it.source_type or "").lower() == "pharmacovigilance"

        it.metadata = dict(it.metadata or {})

        # ── Level A: agent ───────────────────────────────────────────────────
        agent_hits = sorted({t for t in ctx.agent_terms
                             if _occurrences(body, t)
                             or _occurrences(treatment, t)})
        if not agent_hits:
            stats["dropped_agent"] += 1
            it.metadata["relation_gate"] = {
                "level_a": False, "level_b": None, "level_c": None,
                "drop_reason": "agent_absent",
            }
            continue
        stats["after_agent"] += 1

        # ── Level B: manifestation ───────────────────────────────────────────
        manifestation_hits = sorted({
            t for t in ctx.manifestation_terms
            if _occurrences(body, t) or _occurrences(condition, t)})
        manifestation_sentences: List[str] = []
        if not manifestation_hits and len(ctx.manifestation_tokens) >= 2:
            # Phrase-tokens co-present in ONE sentence (not isolated words).
            for sent in _sentences(body):
                if ctx.manifestation_tokens <= set(_tokens(sent)):
                    manifestation_sentences.append(sent)
            if manifestation_sentences:
                manifestation_hits = [
                    " ".join(sorted(ctx.manifestation_tokens)) + " (sentence)"]
        if not manifestation_hits:
            stats["dropped_manifestation"] += 1
            it.metadata["relation_gate"] = {
                "level_a": True, "level_b": False, "level_c": None,
                "matched_agent": agent_hits[:3],
                "drop_reason": "manifestation_absent",
            }
            continue
        stats["after_manifestation"] += 1

        # ── Level C: agent↔manifestation relation ────────────────────────────
        if is_authoritative:
            # Structured pharmacovigilance record: drug field + reaction field
            # ARE the link.
            verified = True
            kind, dist = "structured_record", None
        else:
            verified, kind, dist = _level_c_relation(
                body, title, agent_hits,
                [h for h in manifestation_hits if not h.endswith(" (sentence)")],
                window,
                manifestation_sentences=manifestation_sentences)
        if not verified:
            negated = kind == "negated"
            stats["dropped_relation"] += 1
            if negated:
                stats["dropped_relation_negated"] += 1
            it.metadata["relation_gate"] = {
                "level_a": True, "level_b": True, "level_c": False,
                "matched_agent": agent_hits[:3],
                "matched_manifestation": manifestation_hits[:3],
                "min_agent_manifestation_distance": dist,
                "drop_reason": ("relation_negated" if negated
                                else "relation_not_verified"),
            }
            continue
        stats["after_relation"] += 1

        # ── Re-ranking: relation-aware boost / other-agent penalty ──────────
        base = it.relevance_score or 0.0
        score = base
        score += 0.10 if kind in {"sentence", "title", "window"} else 0.05
        profile = extract_rwe_profile(it, agent_hits[0], manifestation_hits[0], kind)
        if profile["context"] == "personal_testimony":
            score += 0.05
        if profile["onset"] or profile["duration"]:
            score += 0.05
        title_has_agent = any(_occurrences(title, t) for t in ctx.agent_terms)
        other_agent = [t for t in ctx.other_agent_terms if _occurrences(title, t)]
        other_agent_flag = bool(other_agent) and not title_has_agent
        if other_agent_flag:
            score -= 0.25

        it.metadata["base_relevance_score"] = base
        it.relevance_score = round(max(0.0, min(1.0, score)), 3)
        it.match_reason = (
            f"{it.match_reason or ''} | gate:A+B+C({kind})"
            f"{' other-agent-title:' + other_agent[0] if other_agent_flag else ''}"
        ).strip(" |")
        it.metadata["relation_gate"] = {
            "level_a": True, "level_b": True, "level_c": True,
            "matched_agent": agent_hits[:3],
            "matched_manifestation": manifestation_hits[:3],
            "relation_kind": kind,
            "min_agent_manifestation_distance": dist,
            "other_agent_in_title": other_agent[:2] if other_agent_flag else [],
        }
        it.metadata["rwe_profile"] = profile
        kept.append(it)

    stats["final"] = len(kept)
    return kept, stats
