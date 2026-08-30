#!/usr/bin/env python3
"""
RELEVANCE SCORER V2 — comparative diagnostic (NON-production).

Question: can a relevance scorer that exploits the structured QU output
(interventions / outcomes / conditions / synonyms) select RWE results better
than the current production relevance_filter?

Method
------
- SAME corpus as the previous benchmarks: PRE corpora are rebuilt from
  corpus_snapshot.json in CACHE-ONLY mode (live=False, no network, snapshot
  not rewritten), exhaustive collection, cap=100/source — identical to the
  exhaustive ab_pre_post run (355 items/query).
- SAME 4 benchmark queries, SAME independent judge (substring on QU+KB terms).
- V1 = production relevance_filter executed UNCHANGED (min_score=0.20).
- V2 = scorer defined ONLY in this file. It consumes:
    * QU structured extraction (benchmarks/qu_cache.json): interventions,
      outcomes, conditions, synonyms — fixes the KB entity gap (e.g.
      "shedding" not recognised by the KB gets an event side from QU).
    * production plan fields: original_query, translated_query, entities,
      matched_query_type provenance.
    * production helpers reused read-only: _event_match, _tokens,
      _AUTHORITATIVE_SOURCES.
- Nothing in production is imported for modification; relevance_filter is
  called as-is. No commits.

Known limitation (disclosed): the judge shares vocabulary with V2 (both read
QU terms), so judge-precision can favour V2. To compensate we also report
V1-only / V2-only item verdicts and print disagreement samples for manual
inspection.

Usage: python benchmarks/relevance_scorer_v2.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from qu_ab_benchmark import (  # noqa: E402  (benchmark harness, not production)
    build_judge_terms, build_plans, get_collectors, judge_item,
)
from ab_pre_post_relevance import (  # noqa: E402  (diagnostic harness)
    BENCH_QUERIES, collect,
)
from core.rwe.models import RWEItem  # noqa: E402
from core.rwe.pipeline import (  # noqa: E402  (read-only reuse, unchanged)
    _AUTHORITATIVE_SOURCES, _event_match, _tokens, deduplicate,
    relevance_filter,
)
SNAPSHOT = Path(__file__).resolve().parent / "corpus_snapshot.json"
QU_CACHE = Path(__file__).resolve().parent / "qu_cache.json"
OUT_JSON = Path(__file__).resolve().parent / "relevance_scorer_v2_report.json"

MIN_SCORE = 0.20  # same threshold as production, for comparability


# ─── V2 term sets (QU-driven) ────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.lower().strip()


def build_v2_terms(qu: dict, entities: list, vocabulary: dict = None) -> dict:
    """Term sets for V2: QU interventions/outcomes/conditions + QU synonyms,
    merged with the plan's provider-recognised entities and the Catena C
    provider vocabulary (slim resolutions). No local alias dictionaries."""
    iv = {_norm(t) for t in qu.get("interventions", []) if t}
    oc = {_norm(t) for t in qu.get("outcomes", []) if t}
    cd = {_norm(t) for t in qu.get("conditions", []) if t}
    for canonical, aliases in (qu.get("synonyms") or {}).items():
        c = _norm(canonical)
        bucket = iv if c in iv else oc if c in oc else cd if c in cd else None
        if bucket is not None:
            bucket.update(_norm(a) for a in aliases
                          if isinstance(a, str) and len(a) > 3)
    # Provider-recognised entities (production plan.entities)
    etype_buckets = {"drug": iv, "active_ingredient": iv,
                     "symptom": oc, "adverse_effect": oc,
                     "condition": cd, "disease": cd}
    for etype, canonical, _conf in entities or []:
        bucket = etype_buckets.get(etype, oc)
        bucket.add(_norm(canonical))
    # Catena C provider variants (slim vocabulary attached to the plan)
    if vocabulary:
        from core.vocab.models import VocabularyResolution
        for canonical, entries in vocabulary.items():
            c = _norm(canonical)
            bucket = iv if c in iv else oc if c in oc else cd if c in cd else None
            if bucket is None:
                continue
            bucket.update(_norm(t) for t in
                          VocabularyResolution.from_slim(canonical, entries).scored_terms())
    return {"iv": iv, "oc": oc, "cd": cd}


# ─── V2 scorer (diagnostic only) ─────────────────────────────────────────────

def score_v2(it: RWEItem, plan, terms: dict) -> tuple[float, str]:
    """QU-aware relevance score (0–1). Differences vs production V1:
    1. event/outcome side comes from QU outcomes+conditions+synonyms, so it
       exists even when the KB fails to recognise the concept ("shedding");
    2. event semantics reuse the production _event_match synonym groups;
    3. structured fields (treatment/condition) and expansion-tier provenance
       give small bounded boosts;
    4. authoritative (openFDA) drug side trusted, event side still required
       when the query HAS an event side — same strictness principle as V1,
       but with a correct event side.
    """
    body = f"{it.title} {it.text}".lower()
    treat = (it.treatment or "").lower()
    condf = (it.condition or "").lower()
    iv, oc, cd = terms["iv"], terms["oc"], terms["cd"]
    event_side = oc | cd  # outcome side = outcomes + conditions (judge-consistent)

    q_tokens = _tokens(plan.translated_query or "") | _tokens(plan.original_query or "")
    body_tokens = _tokens(f"{it.title} {it.text} {treat} {condf}")
    token_score = len(q_tokens & body_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0

    is_auth = it.source in _AUTHORITATIVE_SOURCES

    # ── anchor side = interventions ∪ conditions (judge semantics) ──
    if is_auth and iv:
        iv_score, iv_hits = 1.0, ["(authoritative)"]
    else:
        hay = f"{body} {treat}"
        iv_hits = sorted({t for t in iv if t and t in hay})
        iv_score = min(1.0, 0.6 + 0.2 * len(iv_hits)) if iv_hits else 0.0
    cd_hits = sorted({t for t in cd if t and (t in body or t in condf)})
    cd_score = min(1.0, 0.6 + 0.2 * len(cd_hits)) if cd_hits else 0.0
    anchor_score = max(iv_score, cd_score)
    anchor_side = iv | cd

    # ── event/outcome side (QU-driven, production semantic groups) ──
    if event_side:
        ev_score, ev_hits = _event_match(body, list(event_side))
        if not ev_hits:  # also try the structured condition field
            ev_score2, ev_hits2 = _event_match(condf, list(event_side))
            if ev_hits2:
                ev_score, ev_hits = max(ev_score, ev_score2), ev_hits2
    else:
        ev_score, ev_hits = 0.5, []  # neutral, as in production

    # ── structured-field + provenance boosts (small, bounded) ──
    boost = 0.0
    if treat and any(t in treat for t in iv):
        boost += 0.05
    if condf and any(t in condf for t in event_side):
        boost += 0.05
    if it.matched_query_type in ("colloquial", "qu", "synonym", "mesh", "combo"):
        boost += 0.03

    # ── combine ──
    if anchor_side and event_side:
        if ev_score == 0.0:
            base = 0.15 * anchor_score + 0.05 * token_score
            reason = f"v2 anchor_match_only (iv={iv_hits[:2]}; event_missing)"
        elif anchor_score == 0.0:
            # Symmetric: no intervention/condition anchor → low relevance.
            base = 0.15 * ev_score + 0.05 * token_score
            reason = f"v2 event_match_only (ev={ev_hits[:3]}; anchor_missing)"
        else:
            base = (0.40 * anchor_score + 0.40 * ev_score
                    + 0.10 * token_score + boost)
            reason = (f"v2 anchor+event (iv={iv_hits[:2]}; ev={ev_hits[:3]}; "
                      f"tokens={len(q_tokens & body_tokens)})")
    elif anchor_side:
        base = 0.55 * anchor_score + 0.25 * token_score + boost
        reason = f"v2 anchor_only (iv={iv_hits[:2]}; tokens={len(q_tokens & body_tokens)})"
    elif event_side:
        base = 0.55 * ev_score + 0.30 * token_score + boost
        reason = f"v2 event_only (ev={ev_hits[:3]})"
    else:
        base = 0.8 * token_score
        reason = "v2 token_fallback"
    return round(min(1.0, base), 3), reason


def apply_v2(items: list[RWEItem], plan, terms: dict, min_score: float = MIN_SCORE):
    kept = []
    for it in items:
        s, r = score_v2(it, plan, terms)
        it.relevance_score = s
        it.match_reason = r
        it.relevance = "relevant" if s >= min_score else "irrelevant"
        if s >= min_score:
            kept.append(it)
    kept.sort(key=lambda x: x.relevance_score, reverse=True)
    return kept


# ─── Evaluation helpers ──────────────────────────────────────────────────────

def key(it):
    return (it.source, it.external_id or it.source_url)


def judge_set(items, terms):
    out = {}
    for it in items:
        rel, relation = judge_item(f"{it.title} {it.text}", terms)
        out[key(it)] = {"relevant": rel, "relation": relation}
    return out


def stats(items, judged):
    n = len(items)
    rel = sum(1 for it in items if judged[key(it)]["relevant"])
    return {
        "n": n,
        "by_source": dict(Counter(it.source for it in items)),
        "judge_relevant": rel,
        "judge_precision": round(rel / n, 4) if n else 0.0,
        "score_range": [round(min((it.relevance_score for it in items), default=0), 3),
                        round(max((it.relevance_score for it in items), default=0), 3)],
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    qu_cache = json.loads(QU_CACHE.read_text())
    collectors = get_collectors()
    report = {"config": {
        "queries": BENCH_QUERIES, "cap_per_source": 100,
        "corpus": "cache-only rebuild of exhaustive PRE corpus (unchanged)",
        "min_score": MIN_SCORE,
        "v1": "production relevance_filter (unchanged)",
        "v2": "diagnostic scorer in this file (QU-driven event side)",
    }, "queries": {}}

    for q in BENCH_QUERIES:
        qu = qu_cache[q]
        plan, _plan_b = build_plans(q, qu)  # production plan (CURRENT arm)
        judge_terms = build_judge_terms(qu, plan.entities)
        v2_terms = build_v2_terms(qu, plan.entities,
                                  getattr(plan, "vocabulary", None))

        # ── SAME corpus as the exhaustive run: cache-only, no live fetch ──
        per_source, _trace, _fs = collect(collectors, snapshot, plan, 100,
                                          live=False, exhaustive=True)
        raw_dicts = []
        for items in per_source.values():
            raw_dicts.extend(it.model_dump() for it in items)

        # V1 on fresh objects (relevance_filter mutates items)
        pre_v1 = deduplicate([RWEItem(**d) for d in raw_dicts])
        v1_kept = relevance_filter(
            pre_v1, plan.translated_query or plan.original_query,
            entities=plan.entities)
        v1_kept.sort(key=lambda it: it.relevance_score, reverse=True)

        # V2 on fresh objects
        pre_v2 = deduplicate([RWEItem(**d) for d in raw_dicts])
        v2_kept = apply_v2(pre_v2, plan, v2_terms)

        pre_pool = deduplicate([RWEItem(**d) for d in raw_dicts])
        judged_pre = judge_set(pre_pool, judge_terms)
        rel_pre_total = sum(1 for v in judged_pre.values() if v["relevant"]) or 1

        k1, k2 = {key(it) for it in v1_kept}, {key(it) for it in v2_kept}
        only_v1 = [it for it in v1_kept if key(it) not in k2]
        only_v2 = [it for it in v2_kept if key(it) not in k1]

        s_pre = stats(pre_pool, judged_pre)
        s_v1 = stats(v1_kept, judged_pre)
        s_v2 = stats(v2_kept, judged_pre)
        s_v1["judge_recall"] = round(s_v1["judge_relevant"] / rel_pre_total, 4)
        s_v2["judge_recall"] = round(s_v2["judge_relevant"] / rel_pre_total, 4)

        only_v1_rel = sum(1 for it in only_v1 if judged_pre[key(it)]["relevant"])
        only_v2_rel = sum(1 for it in only_v2 if judged_pre[key(it)]["relevant"])

        print(f"\n{'=' * 100}\nQUERY: {q}")
        print(f"  V2 terms: iv={sorted(v2_terms['iv'])[:6]}")
        print(f"            oc={sorted(v2_terms['oc'])[:8]}")
        print(f"            cd={sorted(v2_terms['cd'])[:8]}")
        print(f"  PRE : n={s_pre['n']} judge_relevant={s_pre['judge_relevant']} "
              f"(P={s_pre['judge_precision']:.3f})")
        print(f"  V1  : n={s_v1['n']:>3} P={s_v1['judge_precision']:.3f} "
              f"R={s_v1['judge_recall']:.3f} {str(s_v1['by_source'])[:60]}")
        print(f"  V2  : n={s_v2['n']:>3} P={s_v2['judge_precision']:.3f} "
              f"R={s_v2['judge_recall']:.3f} {str(s_v2['by_source'])[:60]}")
        print(f"  Δ   : V2-only kept={len(only_v2)} (judge-relevant {only_v2_rel}) | "
              f"V1-only kept={len(only_v1)} (judge-relevant {only_v1_rel}) | "
              f"common={len(k1 & k2)}")

        samples = []
        for label, pool in (("V2_only", only_v2), ("V1_only", only_v1)):
            for it in pool[:6]:
                samples.append({
                    "set": label, "source": it.source,
                    "judge_relevant": judged_pre[key(it)]["relevant"],
                    "score": round(it.relevance_score, 3),
                    "title": (it.title or "")[:80],
                    "reason": (it.match_reason or "")[:80],
                })
        for s in samples:
            print(f"    [{s['set']}] {s['source']:<20} rel={s['judge_relevant']} "
                  f"score={s['score']:.2f} {s['title'][:58]!r}")

        report["queries"][q] = {
            "v2_terms": {k: sorted(v) for k, v in v2_terms.items()},
            "pre": s_pre, "v1": s_v1, "v2": s_v2,
            "overlap": {"common": len(k1 & k2),
                        "v2_only": len(only_v2), "v2_only_judge_relevant": only_v2_rel,
                        "v1_only": len(only_v1), "v1_only_judge_relevant": only_v1_rel},
            "samples": samples,
        }

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved {OUT_JSON.name}")


if __name__ == "__main__":
    main()
