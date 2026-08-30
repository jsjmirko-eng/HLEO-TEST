#!/usr/bin/env python3
"""Catena C benchmark — V3 vs V3+VOCAB on the RWE chain.

Measures SEPARATELY:
  - raw candidates            (retrieval, pre-dedup)
  - retrieval gain            (new candidates obtained via vocabulary expansions)
  - relevance gain            (kept-set after relevance filter)
  - ranking gain              (NDCG-style: relevant items in top-30)
  - false positives           (kept items judged off-topic)
  - collector/query calls
  - latency
  - cache hit/miss (vocabulary resolver)

Arms:
  V3        HLEO_VOCAB_ENABLED=0 (plan without provider evidence)
  V3+VOCAB  HLEO_VOCAB_ENABLED=1 (vocabulary resolution feeds expansions)

Usage: python benchmarks/catena_c_benchmark.py [--live]
       default: offline synthetic corpus per source (deterministic).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

os.environ.setdefault("HLEO_RWE_INTENT_SCORING", "1")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from core.database import Base, engine  # noqa: E402
import core.models  # noqa: E402,F401  (register models)
Base.metadata.create_all(engine)

from core.rwe.models import RWEItem  # noqa: E402
from core.rwe.pipeline import deduplicate, relevance_filter  # noqa: E402
from core.rwe.query_engine import RWEQueryEngine  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "catena_c_report.json"

# Hand-written judge term sets (never derived from KB/providers).
JUDGE = {
    "finasteride shedding": (["finasteride", "propecia"],
                             ["shedding", "shed", "hair fall", "hair loss", "caduta"]),
    "minoxidil erythema": (["minoxidil", "rogaine"],
                           ["erythema", "redness", "skin redness", "irritation"]),
    "dutasteride brain fog": (["dutasteride", "avodart"],
                              ["brain fog", "fog", "cognitive", "memory"]),
}
QUERIES = list(JUDGE.keys())


def _mk(source, i, agent_terms, event_terms, hit=True, tag=""):
    a = agent_terms[0] if hit else "unrelated compound"
    e = event_terms[0] if hit else "placebo tolerability"
    return RWEItem(
        source=source, source_type="community_forum",
        evidence_tier="anecdotal", collection_method="official_rss_feed",
        source_url=f"https://example.org/{source}/{tag}{i}",
        title=f"{a} and {e} — user report {tag}{i}",
        text=f"I used {a} and experienced {e}. " * 3,
        language="en",
    )


class SyntheticCollector:
    """Deterministic per-source corpus; query match = any query token in text."""

    def __init__(self, source, corpus):
        self.source = source
        self.corpus = corpus
        self.calls = 0

    def search_with_status(self, query, limit=None):
        self.calls += 1
        toks = {t for t in query.lower().split() if len(t) > 2}
        items = [it for it in self.corpus
                 if toks & set((it.title + " " + it.text).lower().split())]
        return items, "ok" if items else "no_results", f"synthetic {len(items)}"


def _corpus():
    """Each source has items reachable via canonical terms and items reachable
    ONLY via vocabulary variants (Rogaine, Avodart, Propecia)."""
    corpus = {}
    for src in ("openfda_faers", "calvizie", "hairlosstalk"):
        items = []
        for q, (agents, events) in JUDGE.items():
            for i in range(12):
                items.append(_mk(src, i, agents, events, hit=True, tag=f"can{i}"))
            # synonym-only reachable items
            for i in range(6):
                alias = {"finasteride shedding": "Propecia",
                         "minoxidil erythema": "Rogaine",
                         "dutasteride brain fog": "Avodart"}[q]
                it = _mk(src, i, agents, events, hit=True, tag=f"syn{i}")
                it.title = it.title.replace(agents[0], alias)
                it.text = it.text.replace(agents[0], alias)
                items.append(it)
            # off-topic noise
            for i in range(10):
                items.append(_mk(src, i, agents, events, hit=False, tag=f"off{i}"))
        corpus[src] = items
    return corpus


def run_arm(vocab: bool, corpus):
    os.environ["HLEO_VOCAB_ENABLED"] = "1" if vocab else "0"
    import importlib
    import core.vocab.resolver as resolver_mod
    importlib.reload(resolver_mod)

    collectors = {s: SyntheticCollector(s, c) for s, c in corpus.items()}
    engine = RWEQueryEngine()
    rows = []
    for query in QUERIES:
        agents, events = JUDGE[query]
        t0 = time.perf_counter()
        plan = engine.plan(query)
        raw, calls = [], 0
        cache_stats = Counter()
        for src, col in collectors.items():
            # mirror production: feed-like collectors get primary queries only
            feed_like = src != "openfda_faers"
            if feed_like:
                qs = [e for e in plan.expanded_queries
                      if e.expansion_type in {"original", "translated", "canonical"}]
            else:
                qs = plan.expanded_queries[:6]
            seen = set()
            for eq in qs:
                items, status, _ = col.search_with_status(eq.query, limit=None)
                calls += 1
                for it in items:
                    k = (it.source, it.source_url)
                    if k in seen:
                        continue
                    seen.add(k)
                    it.matched_query = eq.query
                    it.matched_query_type = eq.expansion_type
                    raw.append(it)
        n_raw = len(raw)
        deduped = deduplicate(raw)
        kept = relevance_filter(deduped, plan.translated_query or query,
                                entities=plan.entities,
                                intent=getattr(plan, "intent", None),
                                min_score=0.20)
        kept.sort(key=lambda x: x.relevance_score, reverse=True)

        def is_rel(it):
            blob = (it.title + " " + it.text).lower()
            return any(a in blob for a in agents) and any(e in blob for e in events)

        kept_rel = [it for it in kept if is_rel(it)]
        fp = len(kept) - len(kept_rel)
        top30 = kept[:30]
        rank_rel = sum(1 for it in top30 if is_rel(it))
        # retrieval gain: raw items first reached via a non-primary expansion
        gain = sum(1 for it in raw
                   if it.matched_query_type not in
                   {"original", "translated", "canonical"})
        vocab_cache = getattr(engine, "_resolver", None)
        if vocab_cache is not None and hasattr(vocab_cache, "cache"):
            st = vocab_cache.cache.stats() if hasattr(vocab_cache.cache, "stats") else {}
            cache_stats.update(st)
        rows.append({
            "query": query,
            "raw_candidates": n_raw,
            "retrieval_gain_items": gain,
            "deduped": len(deduped),
            "kept": len(kept),
            "kept_relevant": len(kept_rel),
            "false_positives": fp,
            "top30_relevant": rank_rel,
            "expanded_queries": len(plan.expanded_queries),
            "collector_calls": calls,
            "latency_s": round(time.perf_counter() - t0, 3),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="use real collectors instead of the synthetic corpus")
    args = ap.parse_args()
    if args.live:
        print("Live mode not implemented in this harness; use vocab_ab_benchmark.py")
        sys.exit(2)

    corpus = _corpus()
    v3 = run_arm(vocab=False, corpus=corpus)
    vv = run_arm(vocab=True, corpus=corpus)

    report = {"arms": {}}
    for name, rows in (("v3", v3), ("v3_vocab", vv)):
        agg = {
            "raw_candidates": sum(r["raw_candidates"] for r in rows),
            "retrieval_gain_items": sum(r["retrieval_gain_items"] for r in rows),
            "deduped": sum(r["deduped"] for r in rows),
            "kept": sum(r["kept"] for r in rows),
            "kept_relevant": sum(r["kept_relevant"] for r in rows),
            "false_positives": sum(r["false_positives"] for r in rows),
            "top30_relevant": sum(r["top30_relevant"] for r in rows),
            "collector_calls": sum(r["collector_calls"] for r in rows),
            "latency_s": round(sum(r["latency_s"] for r in rows), 3),
        }
        report["arms"][name] = {"per_query": rows, "aggregate": agg}

    a, b = report["arms"]["v3"]["aggregate"], report["arms"]["v3_vocab"]["aggregate"]
    report["delta"] = {
        "retrieval_gain": b["raw_candidates"] - a["raw_candidates"],
        "relevance_gain": b["kept_relevant"] - a["kept_relevant"],
        "ranking_gain_top30": b["top30_relevant"] - a["top30_relevant"],
        "false_positive_delta": b["false_positives"] - a["false_positives"],
        "collector_calls_delta": b["collector_calls"] - a["collector_calls"],
        "latency_delta_s": round(b["latency_s"] - a["latency_s"], 3),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["delta"], indent=2))
    print(f"\nV3      : {a}")
    print(f"V3+VOCAB: {b}")
    print(f"\nreport → {OUT_JSON}")


if __name__ == "__main__":
    main()
