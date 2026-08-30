#!/usr/bin/env python3
"""
QU A/B Benchmark — experimental, read-only with respect to production code.

Compares:
  A = CURRENT  : the committed RWE pipeline (RWEQueryEngine KB expansion).
  B = QU-enhanced (SIMULATED): the same KB expansion PLUS Query-Understanding
     expansions (conditions / interventions / outcomes / relation / synonyms /
     search_concepts extracted by an LLM), merged, deduplicated and capped,
     with provenance tag expansion_type="qu".

Nothing in core/ or api/ is imported for modification — only read/executed.

Reproducibility:
  - LLM QU extractions are cached in qu_cache.json (keyed by query text).
  - Every collector response is cached in corpus_snapshot.json
    (keyed by source + query string). Re-running with the caches present
    performs zero network calls and yields identical numbers.

Query set: the exact set used in the lost pre-transfer session is
unrecoverable (it was never committed). This set is reconstructed from the
canonical queries of tests/test_rwe_search.py, which encode the intended
coverage: canonical Italian query, EN/IT variants, brand->generic,
multi-drug, condition-only, colloquial.

Metrics per query (top-30 of each arm, ranked by pipeline relevance_score):
  - unique results (deduped, before filter) / relevant (filter kept)
  - precision@30  : share of top-30 judged relevant by an INDEPENDENT
                    deterministic judge (QU-extracted term sets), not by the
                    pipeline's own filter.
  - relation@30   : stricter — share of top-30 where an intervention term
                    AND an outcome/condition term co-occur.
  - coverage      : unique/relevant items found only by B / only by A.

Usage:  python benchmarks/qu_ab_benchmark.py [--limit N] [--no-llm]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # so .env is found by production modules

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

BENCH_DIR = Path(__file__).resolve().parent
QU_CACHE = BENCH_DIR / "qu_cache.json"
CORPUS_SNAPSHOT = BENCH_DIR / "corpus_snapshot.json"
REPORT_JSON = BENCH_DIR / "qu_ab_report.json"
DETAILS_JSON = BENCH_DIR / "qu_ab_details.json"

# ── Benchmark query set (reconstructed from tests/test_rwe_search.py) ────────
QUERIES = [
    "La finasteride può causare shedding iniziale?",  # canonical IT full pipeline
    "finasteride shedding",
    "finasteride hair shedding",                       # HANDOFF-verified RWE query
    "caduta capelli finasteride",
    "finasteride initial shedding",
    "propecia side effects",                           # brand → generic
    "minoxidil hair loss",
    "finasteride dutasteride minoxidil",               # multi-drug
    "androgenetic alopecia treatment",                 # condition-only
    "caduta capelli",                                  # colloquial IT, no drug
]

# Sources usable without OAuth credentials (reddit excluded: no creds here;
# identical for both arms, so the comparison is unaffected).
SOURCES = ["openfda_faers", "calvizie", "hairlosstalk",
           "hairlossexperiences", "maladiesrares"]

PER_SOURCE_CAP = 15          # same default as production pipeline
TOP_K = 30
MAX_EXPANDED = 16            # production cap
MAX_QU_ADDITIONS = 8
QU_MODEL = "gpt-4o-mini"

# ── Caches ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ── QU extraction (LLM, cached) ──────────────────────────────────────────────

QU_PROMPT = """You are a biomedical query-understanding component for a
hair-loss real-world-evidence search engine.

Analyse the user query and return STRICT JSON with these keys:
  "conditions":      [str]  medical conditions mentioned or implied (canonical English)
  "interventions":   [str]  drugs/treatments mentioned (generic names)
  "outcomes":        [str]  symptoms/side-effects/outcomes mentioned or implied
  "relation":        object with keys "type" and "description"; type is one of
                     causation | side_effect | treatment | prevention |
                     comparison | monitoring | unknown
  "synonyms":        object mapping canonical_term -> [alias, ...] with
                     patient/colloquial AND medical synonyms
  "search_concepts": [str]  2-5 short search phrases (2-5 words each) that
                     would surface PATIENT EXPERIENCES of this exact relation
                     (e.g. "finasteride initial shedding stories")

Rules: canonical English medical terms; keep everything specific to the
query; never broaden to generic hair-loss chatter; empty lists when absent.

Query: __QUERY__
JSON:"""


def extract_qu(query: str, cache: dict, use_llm: bool) -> dict:
    if query in cache:
        return cache[query]
    empty = {"conditions": [], "interventions": [], "outcomes": [],
             "relation": {"type": "unknown", "description": ""},
             "synonyms": {}, "search_concepts": []}
    if not use_llm:
        cache[query] = empty
        return empty
    from openai import OpenAI
    client = OpenAI()  # key from env (.env loaded by production modules)
    try:
        resp = client.chat.completions.create(
            model=QU_MODEL,
            messages=[{"role": "user", "content": QU_PROMPT.replace("__QUERY__", query)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        for k, v in empty.items():
            data.setdefault(k, v)
    except Exception as exc:  # noqa: BLE001 - benchmark must never crash on LLM
        print(f"  [QU] LLM extraction failed for '{query}': {exc}")
        data = empty
    cache[query] = data
    return data


def qu_queries(qu: dict, kb_strings: set[str]) -> list[str]:
    """Build QU expansion queries: intervention×outcome combos, search
    concepts, synonyms not already covered by the KB expansion."""
    out: list[str] = []
    interv = qu.get("interventions", [])[:2]
    outcomes = (qu.get("outcomes", []) + qu.get("conditions", []))[:3]
    for iv in interv:
        for oc in outcomes:
            out.append(f"{iv} {oc}")
    out.extend(qu.get("search_concepts", [])[:4])
    for canonical, aliases in (qu.get("synonyms") or {}).items():
        for alias in aliases[:2]:
            if isinstance(alias, str) and 1 <= len(alias.split()) <= 4:
                out.append(alias)
    seen = {s.lower().strip() for s in kb_strings}
    deduped = []
    for q in out:
        k = q.lower().strip()
        if k and k not in seen and len(k) > 2:
            seen.add(k)
            deduped.append(q)
    return deduped[:MAX_QU_ADDITIONS]


# ── Plan construction (A and B) ──────────────────────────────────────────────

def build_plans(query: str, qu: dict):
    """A = production plan. B = A + QU expansions (merged, deduped, capped)."""
    from core.rwe.query_engine import (
        EXP_COLLOQUIAL, EXP_MESH, EXP_NEIGHBOR, EXP_ORIGINAL, EXP_SYNONYM,
        EXP_TRANSLATED, EXP_COMBO, ExpandedQuery, RWEQueryEngine,
    )
    engine = RWEQueryEngine()
    plan_a = engine.plan(query)

    kb_strings = [eq.query for eq in plan_a.expanded_queries]
    additions = [
        ExpandedQuery(query=q, expansion_type="qu",
                      source_language="en",
                      matched_entities=[c for _, c, _ in plan_a.entities])
        for q in qu_queries(qu, kb_strings)
    ]
    merged = list(plan_a.expanded_queries) + additions
    # dedup + cap with tier ordering; QU sits with the specific tiers (2)
    tier = {EXP_ORIGINAL: 0, EXP_TRANSLATED: 1, EXP_SYNONYM: 2, EXP_MESH: 2,
            EXP_COMBO: 2, "qu": 2, EXP_COLLOQUIAL: 3, EXP_NEIGHBOR: 4}
    seen, unique = set(), []
    for eq in merged:
        k = eq.query.lower().strip()
        if k and k not in seen:
            seen.add(k)
            unique.append(eq)
    ent_lower = {c.lower() for _, c, _ in plan_a.entities}
    unique.sort(key=lambda eq: (
        tier.get(eq.expansion_type, 9),
        -sum(1 for n in ent_lower if n in eq.query.lower()),
    ))
    plan_b = type(plan_a)(
        original_query=plan_a.original_query,
        detected_language=plan_a.detected_language,
        translated_query=plan_a.translated_query,
        translation_applied=plan_a.translation_applied,
        entities=plan_a.entities,
        expanded_queries=unique[:MAX_EXPANDED],
    )
    return plan_a, plan_b


# ── Collection (production-faithful, cached) ─────────────────────────────────

def get_collectors():
    from core.rwe.openfda_collector import OpenFDACollector
    from core.rwe.calvizie_collector import CalvizieCollector
    from core.rwe.hairlosstalk_collector import HairLossTalkCollector
    from core.rwe.hairlossexperiences_collector import HairLossExperiencesCollector
    from core.rwe.maladiesrares_collector import MaladiesRaresCollector
    return {
        "openfda_faers": OpenFDACollector(),
        "calvizie": CalvizieCollector(),
        "hairlosstalk": HairLossTalkCollector(),
        "hairlossexperiences": HairLossExperiencesCollector(),
        "maladiesrares": MaladiesRaresCollector(),
    }


def cached_search(collectors, snapshot: dict, source: str, qstr: str, limit: int):
    key = f"{source}||{qstr.lower().strip()}||{limit}"
    if key in snapshot:
        return snapshot[key]
    try:
        items, status, _reason = collectors[source].search_with_status(qstr, limit=limit)
        entry = {"status": status, "items": [it.model_dump() for it in items]}
    except Exception as exc:  # noqa: BLE001
        entry = {"status": "network_error", "items": [], "error": str(exc)}
    snapshot[key] = entry
    return entry


def collect_arm(collectors, snapshot: dict, plan, source: str, cap: int):
    """Faithful reimplementation of RWEPipeline._collect_source for one source
    (same expansion order, same early stop, same provenance stamping)."""
    from core.rwe.models import RWEItem
    collected, seen_ids = [], set()
    for eq in plan.expanded_queries:
        if len(collected) >= cap:
            break
        entry = cached_search(collectors, snapshot, source, eq.query,
                              max(1, cap - len(collected)))
        if entry["status"] != "ok":
            continue
        for raw in entry["items"]:
            it = RWEItem(**raw)
            k = (it.source, it.external_id or it.source_url)
            if k in seen_ids:
                continue
            seen_ids.add(k)
            it.matched_query = eq.query
            it.matched_query_type = eq.expansion_type
            it.source_language = eq.source_language
            collected.append(it)
            if len(collected) >= cap:
                break
    return collected


# ── Independent relevance judge ──────────────────────────────────────────────

def build_judge_terms(qu: dict, entities: list) -> dict:
    """Term sets for the deterministic judge, from QU output + KB entities."""
    def norm(s):
        return s.lower().strip()

    interventions = {norm(t) for t in qu.get("interventions", [])}
    outcomes = {norm(t) for t in qu.get("outcomes", [])}
    conditions = {norm(t) for t in qu.get("conditions", [])}
    for canonical, aliases in (qu.get("synonyms") or {}).items():
        target = norm(canonical)
        bucket = interventions if target in interventions else outcomes
        bucket.update(norm(a) for a in aliases if isinstance(a, str))
    # Entity canonicals from the plan (provider-recognised, Catena C).
    # Provider variants already reach the judge through the QU synonyms above;
    # no local alias dictionaries are used anymore.
    etype_buckets = {"drug": interventions, "active_ingredient": interventions,
                     "condition": conditions, "disease": conditions,
                     "symptom": outcomes, "adverse_effect": outcomes}
    for etype, canonical, _conf in entities:
        bucket = etype_buckets.get(etype, outcomes)
        bucket.add(norm(canonical))
    return {"interventions": interventions, "outcomes": outcomes,
            "conditions": conditions}


def judge_item(text: str, terms: dict) -> tuple[bool, bool]:
    """Returns (relevant, relation_hit). Independent of the pipeline filter."""
    t = text.lower()
    has_iv = any(term in t for term in terms["interventions"])
    has_oc = any(term in t for term in terms["outcomes"] | terms["conditions"])
    anchors = terms["interventions"] | terms["conditions"]
    has_anchor = any(term in t for term in anchors)
    relevant = (has_anchor and has_oc) if anchors else has_oc
    relation = has_iv and has_oc
    return relevant, relation


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_arm(collectors, snapshot, plan, source_list, judge_terms, cap):
    from core.rwe.pipeline import deduplicate, relevance_filter
    raw = []
    for src in source_list:
        raw.extend(collect_arm(collectors, snapshot, plan, src, cap))
    unique = deduplicate(raw)
    relevance_query = plan.translated_query or plan.original_query
    kept = relevance_filter(unique, relevance_query, entities=plan.entities)
    kept.sort(key=lambda it: it.relevance_score, reverse=True)
    top = kept[:TOP_K]
    judged = []
    for it in top:
        rel, relation = judge_item(f"{it.title} {it.text}", judge_terms)
        judged.append({"url": it.source_url, "title": it.title,
                       "source": it.source,
                       "matched_query": it.matched_query,
                       "matched_query_type": it.matched_query_type,
                       "score": round(it.relevance_score, 3),
                       "judge_relevant": rel, "judge_relation": relation})
    n = len(top) or 1
    return {
        "retrieved": len(raw),
        "unique": len(unique),
        "relevant": len(kept),
        "unique_keys": sorted({(it.source, it.external_id or it.source_url)
                               for it in unique}),
        "relevant_keys": sorted({(it.source, it.external_id or it.source_url)
                                 for it in kept}),
        "precision_at_30": sum(1 for j in judged if j["judge_relevant"]) / n,
        "relation_at_30": sum(1 for j in judged if j["judge_relation"]) / n,
        "top30": judged,
        "n_top": len(top),
        "expansions": [{"q": eq.query, "type": eq.expansion_type}
                       for eq in plan.expanded_queries],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N queries (smoke mode)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM QU extraction (B == A control run)")
    ap.add_argument("--cap", type=int, default=PER_SOURCE_CAP,
                    help="per-source item cap (production default: 15; raise "
                         "to let expansions beyond the first contribute)")
    args = ap.parse_args()

    qu_cache = _load_json(QU_CACHE)
    snapshot = _load_json(CORPUS_SNAPSHOT)
    collectors = get_collectors()

    queries = QUERIES[: args.limit] if args.limit else QUERIES
    report, details = {"config": {
        "queries": queries, "sources": SOURCES,
        "per_source_cap": args.cap, "top_k": TOP_K,
        "max_expanded": MAX_EXPANDED, "qu_model": QU_MODEL,
        "llm": not args.no_llm, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "arms": {}}, {}

    for qi, query in enumerate(queries, 1):
        print(f"[{qi}/{len(queries)}] {query}")
        qu = extract_qu(query, qu_cache, use_llm=not args.no_llm)
        _save_json(QU_CACHE, qu_cache)
        plan_a, plan_b = build_plans(query, qu)
        terms = build_judge_terms(qu, plan_a.entities)

        res_a = evaluate_arm(collectors, snapshot, plan_a, SOURCES, terms, args.cap)
        res_b = evaluate_arm(collectors, snapshot, plan_b, SOURCES, terms, args.cap)
        _save_json(CORPUS_SNAPSHOT, snapshot)

        keys_a, keys_b = set(res_a["relevant_keys"]), set(res_b["relevant_keys"])
        row = {
            "query": query,
            "relation_type": qu["relation"].get("type"),
            "expansions_a": len(res_a["expansions"]),
            "expansions_b": len(res_b["expansions"]),
            "qu_added": sum(1 for e in res_b["expansions"] if e["type"] == "qu"),
            "A": {k: res_a[k] for k in ("retrieved", "unique", "relevant",
                                        "precision_at_30", "relation_at_30")},
            "B": {k: res_b[k] for k in ("retrieved", "unique", "relevant",
                                        "precision_at_30", "relation_at_30")},
            "relevant_only_b": len(keys_b - keys_a),
            "relevant_only_a": len(keys_a - keys_b),
        }
        report["arms"][query] = row
        details[query] = {"A": res_a, "B": res_b, "qu": qu}
        print(f"    A: uniq={row['A']['unique']} rel={row['A']['relevant']} "
              f"P@30={row['A']['precision_at_30']:.2f} R@30={row['A']['relation_at_30']:.2f}")
        print(f"    B: uniq={row['B']['unique']} rel={row['B']['relevant']} "
              f"P@30={row['B']['precision_at_30']:.2f} R@30={row['B']['relation_at_30']:.2f} "
              f"(+{row['qu_added']} qu, only-B={row['relevant_only_b']}, only-A={row['relevant_only_a']})")

    # Aggregate
    n = len(report["arms"]) or 1
    agg = {}
    for arm in ("A", "B"):
        agg[arm] = {
            m: round(sum(r[arm][m] for r in report["arms"].values()) / n, 4)
            for m in ("precision_at_30", "relation_at_30")
        }
        agg[arm]["unique_total"] = sum(r[arm]["unique"] for r in report["arms"].values())
        agg[arm]["relevant_total"] = sum(r[arm]["relevant"] for r in report["arms"].values())
    report["aggregate"] = agg
    suffix = "" if args.cap == PER_SOURCE_CAP else f"_cap{args.cap}"
    report_path = BENCH_DIR / f"qu_ab_report{suffix}.json"
    details_path = BENCH_DIR / f"qu_ab_details{suffix}.json"
    _save_json(report_path, report)
    _save_json(details_path, details)
    print(f"\nSaved: {report_path.name}, {details_path.name}, "
          f"{QU_CACHE.name}, {CORPUS_SNAPSHOT.name}")


if __name__ == "__main__":
    main()
