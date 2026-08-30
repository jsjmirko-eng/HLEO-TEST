#!/usr/bin/env python3
"""
Vocabulary-layer benchmark — THREE arms on the SAME corpus per query:

  V1        production relevance_filter (intent=None)
  V3        QU-aware V3 (HLEO_RWE_INTENT_SCORING=1), no vocabulary
  V3+VOCAB  V3 + external vocabulary evidence (HLEO_VOCAB_ENABLED=1)

Sections:
  1. A–G benchmark queries (imported from v3_ab_benchmark, same judge).
  2. Vocabulary-focused queries: Italian colloquial, EN variants,
     abbreviations (minox/dut/fin/rogaine), forum slang.
  3. Vocabulary resolution probe: which providers matched the required term
     list, with match kinds, external-call count and cache hit-rate.

Anti-circularity: judge term sets are hand-written here/in v3_ab_benchmark
and never derived from QU/KB/provider output.

Usage: python benchmarks/vocab_ab_benchmark.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

os.environ["HLEO_RWE_INTENT_SCORING"] = "1"
os.environ["HLEO_VOCAB_ENABLED"] = "1"

import core.rwe.intent as intent_mod  # noqa: E402
from ab_pre_post_relevance import collect  # noqa: E402
from qu_ab_benchmark import get_collectors  # noqa: E402
from core.rwe.models import RWEItem  # noqa: E402
from core.rwe.pipeline import deduplicate, relevance_filter  # noqa: E402
from core.rwe.query_engine import RWEQueryEngine  # noqa: E402
from core.vocab.base import VocabularyProvider  # noqa: E402
from core.vocab.resolver import VocabularyResolver  # noqa: E402

from v3_ab_benchmark import (  # noqa: E402  (shared judge + A–G queries)
    QUERIES as AG_QUERIES, FIN_IV, MIN_IV, DUT_IV, SHED_EV, HL_COND, judge,
)

SNAPSHOT = Path(__file__).resolve().parent / "corpus_snapshot.json"
OUT_JSON = Path(__file__).resolve().parent / "vocab_ab_report.json"

# ── Vocabulary-focused queries (corpus collected live once, shared by arms) ──
HAIRLINE = ["hairline", "receding", "stempiatura", "tempie", "hair line"]
CROWN = ["crown", "vertex", "chierica", "whorl"]
SLANG_SHED = ["shedding", "shed", "falling out", "clumps", "caduta"]

VOCAB_QUERIES = [
    # ITALIANO colloquiale
    ("IT", "caduta capelli", HL_COND, []),
    ("IT", "mi stanno cadendo i capelli", HL_COND, []),
    ("IT", "mi si stanno diradando i capelli", HL_COND, []),
    ("IT", "stempiatura", HL_COND + HAIRLINE, []),
    ("IT", "perdo un sacco di capelli", HL_COND, []),
    # ENGLISH variants
    ("EN", "hair fall", HL_COND, []),
    ("EN", "hair thinning", HL_COND, []),
    # FARMACI / ABBREVIAZIONI
    ("AB", "minox", MIN_IV, []),
    ("AB", "dut", DUT_IV + ["dut"], []),
    ("AB", "fin", FIN_IV, []),
    ("AB", "rogaine", MIN_IV, []),
    # SLANG / COLLOQUIALE (realistic forum phrasings)
    ("SL", "shedding like crazy", SLANG_SHED, []),
    ("SL", "my hairline keeps receding", HAIRLINE + HL_COND, []),
    ("SL", "my crown is thinning bad", CROWN + HL_COND, []),
    ("SL", "losing hair on fin", FIN_IV, SHED_EV),
    ("SL", "minox not working", MIN_IV, []),
]

# Terms probed in the vocabulary-resolution section (from the requirements).
PROBE_TERMS = [
    ("finasteride", "en"), ("minoxidil", "en"), ("dutasteride", "en"),
    ("rogaine", "en"), ("minox", "en"), ("fin", "en"), ("dut", "en"),
    ("hair loss", "en"), ("hair shedding", "en"), ("alopecia", "en"),
    ("side effects", "en"), ("hair fall", "en"), ("hair thinning", "en"),
    ("ferritin", "en"), ("DHT", "en"),
    ("caduta capelli", "it"), ("stempiatura", "it"),
    ("mi stanno cadendo i capelli", "it"), ("perdita di capelli", "it"),
    ("pelade", "fr"),
]


def _arm(items, judge_fn, denom):
    n = len(items)
    rel = sum(1 for it in items if judge_fn(it))
    return {"n": n, "judge_relevant": rel,
            "precision": round(rel / n, 4) if n else 0.0,
            "recall": round(rel / denom, 4),
            "false_positives": n - rel}


def run_queries(engine, collectors, snapshot, queries, llm, ext, report,
                totals):
    for group, query, anchors, events in queries:
        calls_before = llm["calls"]
        ext_before = ext["calls"]
        t0 = time.time()
        plan = engine.plan(query)      # intent (+vocabulary when flag on)
        plan_s = time.time() - t0

        per_source, _t, _fs = collect(collectors, snapshot, plan, 100,
                                      live=True, exhaustive=True)
        raw = [it.model_dump() for items in per_source.values() for it in items]
        pre = deduplicate([RWEItem(**d) for d in raw])
        rq = plan.translated_query or plan.original_query

        def j(it):
            return judge(f"{it.title} {it.text}", anchors, events)

        pre_rel = sum(1 for it in pre if j(it))
        denom = pre_rel or 1
        keys_pre_rel = {(i.source, i.external_id or i.source_url)
                        for i in pre if j(i)}

        arms = {}
        for name, use_intent in (("v1", None), ("v3", "plain"),
                                 ("v3vocab", "vocab")):
            if use_intent is None:
                intent = None
            elif use_intent == "plain":
                # V3 without vocabulary evidence
                intent = plan.intent.model_copy() if plan.intent else None
                if intent is not None:
                    intent.vocabulary = {}
            else:
                intent = plan.intent
            t = time.time()
            kept = relevance_filter([RWEItem(**d) for d in raw], rq,
                                    entities=plan.entities, intent=intent)
            ms = (time.time() - t) * 1000
            s = _arm(kept, j, denom)
            got = {(i.source, i.external_id or i.source_url) for i in kept}
            s["lost_relevant"] = len(keys_pre_rel - got)
            s["ms"] = round(ms, 1)
            arms[name] = s

        vocab_n = sum(len(v) for v in (plan.intent.vocabulary or {}).values()
                      ) if plan.intent else 0
        line = (f"[{group}] {query}\n"
                f"     PRE n={len(pre)} (judge-rel {pre_rel}) "
                f"| vocab_evidence={vocab_n} "
                f"ext_calls={ext['calls'] - ext_before}")
        for a in ("v1", "v3", "v3vocab"):
            s = arms[a]
            line += (f"\n     {a:8s} n={s['n']:3d} P={s['precision']:.3f} "
                     f"R={s['recall']:.3f} FP={s['false_positives']} "
                     f"lost={s['lost_relevant']} ({s['ms']}ms)")
        print(line, flush=True)
        report["queries"][query] = {
            "group": group, "plan_seconds": round(plan_s, 2),
            "pre_n": len(pre), "pre_relevant": pre_rel,
            "llm_calls": llm["calls"] - calls_before,
            "ext_vocab_calls": ext["calls"] - ext_before,
            "arms": arms}
        for a in arms:
            totals[f"{a}_n"] += arms[a]["n"]
            totals[f"{a}_fp"] += arms[a]["false_positives"]
            totals[f"{a}_lost"] += arms[a]["lost_relevant"]
        totals["pre"] += len(pre)
        totals["pre_rel"] += pre_rel


def vocabulary_probe(ext):
    print("\n── Vocabulary resolution probe ──", flush=True)
    resolver = VocabularyResolver()
    print(f"active providers: {resolver.active_providers()}")
    rows = {}
    for term, lang in PROBE_TERMS:
        before = ext["calls"]
        res = resolver.resolve_term(term, language=lang)
        rows[f"{term} ({lang})"] = {
            "queried": res.providers_queried,
            "failed": res.providers_failed,
            "ext_calls": ext["calls"] - before,
            "matches": [{"provider": m.provider, "kind": m.match_kind,
                         "concept": m.concept_id,
                         "preferred": m.preferred_term,
                         "synonyms": m.synonyms[:4],
                         "confidence": m.confidence,
                         "lang": m.language}
                        for m in res.matches],
        }
        top = ", ".join(f"{m['provider']}:{m['kind']}" for m in
                        rows[f"{term} ({lang})"]["matches"][:4]) or "—"
        print(f"  {term} ({lang}) → {top}", flush=True)
    # cache hit-rate: second identical pass must cost 0 external calls
    before = ext["calls"]
    for term, lang in PROBE_TERMS:
        resolver.resolve_term(term, language=lang)
    hits_cost = ext["calls"] - before
    print(f"cache: 2nd pass external calls = {hits_cost} "
          f"(0 = 100% hit-rate) | entries={len(resolver.cache)}")
    return rows, {"second_pass_ext_calls": hits_cost,
                  "cache_entries": len(resolver.cache)}


def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    collectors = get_collectors()
    engine = RWEQueryEngine()

    llm = {"calls": 0, "seconds": 0.0}
    orig_extract = intent_mod.extract_intent_llm

    def counting_extract(*a, **k):
        t0 = time.time()
        try:
            return orig_extract(*a, **k)
        finally:
            llm["calls"] += 1
            llm["seconds"] += time.time() - t0

    intent_mod.extract_intent_llm = counting_extract

    # Count external vocabulary HTTP calls (any provider).
    ext = {"calls": 0}
    orig_get = VocabularyProvider._get_json

    def counting_get(self, *a, **k):
        ext["calls"] += 1
        return orig_get(self, *a, **k)

    VocabularyProvider._get_json = counting_get

    report = {"config": {
        "min_score": 0.20,
        "flags": ["HLEO_RWE_INTENT_SCORING=1", "HLEO_VOCAB_ENABLED=1"],
        "providers": os.getenv("HLEO_VOCAB_PROVIDERS",
                               "rxnorm,mesh,conceptnet,wikidata"),
        "judge": "hand-written term sets (anti-circular)",
        "n_queries": len(AG_QUERIES) + len(VOCAB_QUERIES),
    }, "queries": {}, "probe": {}}

    totals = Counter()
    print("── A–G queries (3 arms) ──", flush=True)
    run_queries(engine, collectors, snapshot, AG_QUERIES, llm, ext,
                report, totals)
    print("\n── Vocabulary-focused queries (3 arms) ──", flush=True)
    run_queries(engine, collectors, snapshot, VOCAB_QUERIES, llm, ext,
                report, totals)

    probe_rows, cache_stats = vocabulary_probe(ext)
    report["probe"] = {"terms": probe_rows, "cache": cache_stats}

    report["totals"] = dict(totals)
    report["llm"] = llm
    report["external_vocab_calls"] = ext["calls"]
    print(f"\nTOTALS: {dict(totals)}")
    print(f"LLM calls={llm['calls']} ({llm['seconds']:.1f}s) | "
          f"external vocab HTTP calls={ext['calls']}")
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved {OUT_JSON.name}")


if __name__ == "__main__":
    main()
