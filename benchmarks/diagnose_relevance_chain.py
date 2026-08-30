#!/usr/bin/env python3
"""
RWE relevance-chain DIAGNOSTICS — read-only, non-productive.

Reconstructs the exact production path for one query:

  user query
    → RWEQueryEngine.plan() (language/translation/entities/expansion)
    → per collector: expanded queries actually sent
    → raw results
    → per-item normalisation audit (RWEItem field population)
    → cross-query dedup + provenance stamping
    → global dedup
    → relevance_filter(query=translated, entities=plan.entities)
    → final ranked list

For every stage it reports items in / items out, the fields that are
populated or lost, the exact drop reason of every dropped item, and a
per-item diagnostic table (score + reason + which text fields existed).

It uses ONLY the committed production code (collectors, RWEQueryEngine,
deduplicate, relevance_filter, _score_item) and the benchmark's
corpus_snapshot.json cache — no production file is modified, no new
sources are introduced, no network is required when the cache is warm.

Usage: python benchmarks/diagnose_relevance_chain.py [query ...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from core.rwe.models import RWEItem  # noqa: E402
from core.rwe.pipeline import (  # noqa: E402
    _AUTHORITATIVE_SOURCES, _event_match, _split_entities, _score_item,
    deduplicate, relevance_filter,
)
from core.rwe.query_engine import RWEQueryEngine  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent / "corpus_snapshot.json"
PER_SOURCE_CAP = 15  # production default used in the benchmark

DEFAULT_QUERIES = [
    "finasteride shedding",
    "finasteride hair shedding",
    "minoxidil hair loss",
    "finasteride dutasteride minoxidil",
]


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


def field_audit(items):
    """Normalisation audit: which RWEItem fields are actually populated."""
    fields = ["title", "text", "source_url", "external_id", "treatment",
              "condition", "date", "topic", "metadata"]
    pop = Counter()
    for it in items:
        for f in fields:
            v = getattr(it, f, None)
            if v not in (None, "", [], {}):
                pop[f] += 1
    n = max(1, len(items))
    return {f: f"{pop[f]}/{len(items)} ({100*pop[f]/n:.0f}%)" for f in fields}


def diagnose_query(query, collectors, snapshot, sources):
    engine = RWEQueryEngine()
    plan = engine.plan(query)
    relevance_query = plan.translated_query or plan.original_query

    print("=" * 100)
    print(f"QUERY: {query}")
    print(f"  plan: lang={plan.detected_language} translated='{plan.translated_query}' "
          f"entities={[(t, c) for t, c, _ in plan.entities]}")
    print(f"  expansions ({len(plan.expanded_queries)}): "
          + "; ".join(f"[{eq.expansion_type}] {eq.query}" for eq in plan.expanded_queries))
    drugs, events, ctx = _split_entities(plan.entities)
    print(f"  relevance view: drug_terms={drugs[:6]}")
    print(f"                  event_terms={events[:8]}")
    print(f"  relevance filter query = '{relevance_query}'  threshold min_score=0.20")

    global_pool, per_source = [], {}
    for src in sources:
        print(f"\n── Collector: {src} " + "─" * (84 - len(src)))
        col = collectors[src]
        collected, seen_ids, rows_trace = [], set(), []
        for eq in plan.expanded_queries:
            if len(collected) >= PER_SOURCE_CAP:
                rows_trace.append((eq.query, eq.expansion_type, "SKIPPED (cap reached)", 0, 0))
                continue
            key = f"{src}||{eq.query.lower().strip()}||{max(1, PER_SOURCE_CAP - len(collected))}"
            entry = snapshot.get(key)
            if entry is None:
                rows_trace.append((eq.query, eq.expansion_type, "NOT IN CACHE", 0, 0))
                continue
            items = [RWEItem(**r) for r in entry["items"]]
            kept_n = 0
            for it in items:
                k = (it.source, it.external_id or it.source_url)
                if k in seen_ids:
                    continue
                seen_ids.add(k)
                it.matched_query = eq.query
                it.matched_query_type = eq.expansion_type
                it.source_language = eq.source_language
                collected.append(it)
                kept_n += 1
                if len(collected) >= PER_SOURCE_CAP:
                    break
            rows_trace.append((eq.query, eq.expansion_type, entry["status"], len(items), kept_n))

        for q, t, st, n_raw, n_kept in rows_trace:
            print(f"  [{t:<11}] '{q[:52]:<52}' → status={st:<11} raw={n_raw:>3} kept(after in-source dedup)={n_kept:>3}")

        audit = field_audit(collected)
        print(f"  NORMALISATION AUDIT ({len(collected)} items): " +
              ", ".join(f"{k}={v}" for k, v in audit.items()))

        # per-source relevance outcome (production does it globally; shown here
        # per source for diagnosis, using the identical scoring function)
        drops = Counter()
        survivors = []
        for it in collected:
            is_auth = it.source in _AUTHORITATIVE_SOURCES
            score, reason = _score_item(it, relevance_query, plan.entities, is_auth)
            it.relevance_score = score
            it.match_reason = reason
            (survivors.append(it) if score >= 0.20 else drops.update({reason.split("(")[0].strip(): 1}))
        per_source[src] = {
            "raw": sum(r[3] for r in rows_trace),
            "collected": len(collected),
            "survivors": survivors,
            "drops": drops,
        }
        print(f"  PER-SOURCE RELEVANCE: {len(collected)} in → {len(survivors)} out; "
              f"drop reasons: {dict(drops) or '-'}")
        global_pool.extend(collected)

    print("\n── GLOBAL STAGES " + "─" * 82)
    pre_dedup = len(global_pool)
    deduped = deduplicate(global_pool)
    print(f"  cross-source pool: {pre_dedup} → deduplicate() → {len(deduped)}")
    kept = relevance_filter(deduped, relevance_query, entities=plan.entities)
    kept.sort(key=lambda it: it.relevance_score, reverse=True)
    by_src = Counter(it.source for it in kept)
    print(f"  relevance_filter(global): {len(deduped)} in → {len(kept)} out; survivors by source: {dict(by_src)}")

    print("\n── ITEM DIAGNOSTICS (top-5 KEPT + first 5 DROPPED) " + "─" * 48)
    hdr = f"{'#':>2} {'source':<19} {'title':<44} {'text?':<6}{'treat?':<7}{'meta?':<6}{'score':>6}  verdict/reason"
    print(hdr)
    dropped = [it for it in deduped if (it.relevance_score or 0.0) < 0.20]
    sample = (kept[:5] + dropped[:5])[:10]
    for i, it in enumerate(sample, 1):
        title = (it.title or "")[:42]
        verdict = "KEPT " if it.relevance_score >= 0.20 else "DROP "
        print(f"{i:>2} {it.source:<19} {title:<44} "
              f"{'yes' if it.text else 'NO':<6}{'yes' if it.treatment else '-':<7}"
              f"{'yes' if it.metadata else '-':<6}{it.relevance_score:>6.3f}  "
              f"{verdict}{it.match_reason[:70]}")
    print()


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES
    snapshot = json.loads(SNAPSHOT.read_text())
    collectors = get_collectors()
    sources = list(collectors)
    print(f"snapshot entries: {len(snapshot)} | sources: {sources} | cap/source={PER_SOURCE_CAP}")
    for q in queries:
        diagnose_query(q, collectors, snapshot, sources)


if __name__ == "__main__":
    main()
