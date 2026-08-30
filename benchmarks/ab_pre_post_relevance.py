#!/usr/bin/env python3
"""
A/B DIAGNOSTIC pre/post relevance_filter — non-productive, read-only.

Verifies whether the production relevance_filter is HIDING a QU advantage,
by comparing CURRENT vs QU-enhanced at TWO checkpoints:

  PRE  = after collect → normalise → dedup        (raw corpus, no filter)
  POST = after production relevance_filter        (min_score=0.20)

For each query and each arm it reports, at both checkpoints:
  - item counts (total + per source)
  - independent-judge precision/relation (substring match on QU+KB terms)
  - how many PRE-stage QU-only items are dropped vs kept by the filter
  - per-source traces of which expanded queries actually contributed items

CURRENT arm = production plan (KB expansion).
QU arm      = production plan + simulated QU expansions (cached LLM
              extractions from the earlier benchmark), merged, deduped,
              capped at 16 — same simulation as qu_ab_benchmark.py.

Nothing in production is modified; the production relevance_filter and
deduplicate are executed unchanged. Live fetches happen only for cache
misses and are written back into corpus_snapshot.json (RSS collectors are
forum-scoped/query-insensitive, so results are identical to what the
previous benchmark runs would have produced).

Usage: python benchmarks/ab_pre_post_relevance.py [--cap 100]
"""
from __future__ import annotations

import argparse
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
    QUERIES, SOURCES, build_judge_terms, build_plans, get_collectors,
    judge_item,
)
from core.rwe.models import RWEItem  # noqa: E402
from core.rwe.pipeline import deduplicate, relevance_filter  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent / "corpus_snapshot.json"
QU_CACHE = Path(__file__).resolve().parent / "qu_cache.json"
OUT_JSON = Path(__file__).resolve().parent / "ab_pre_post_report.json"

BENCH_QUERIES = [
    "finasteride shedding",
    "finasteride hair shedding",
    "minoxidil hair loss",
    "finasteride dutasteride minoxidil",
]


def cached_search(collectors, snapshot, source, qstr, limit, live):
    key = f"{source}||{qstr.lower().strip()}||{limit}"
    if key in snapshot:
        return snapshot[key], True
    if not live:
        return {"status": "not_cached", "items": []}, False
    try:
        items, status, _ = collectors[source].search_with_status(qstr, limit=limit)
        entry = {"status": status, "items": [it.model_dump() for it in items]}
    except Exception as exc:  # noqa: BLE001
        entry = {"status": "network_error", "items": [], "error": str(exc)}
    snapshot[key] = entry
    return entry, False


def collect(collectors, snapshot, plan, cap, live, exhaustive=False):
    """Per-source collection with contribution tracing.

    Default mode is production-faithful (early stop at cap). In exhaustive
    mode every expansion is fetched first and only then the pool is trimmed
    to cap — this is the diagnostic mode that lets expansions positioned
    after the first (QU, colloquial) contribute to the PRE corpus.
    """
    per_source_items, trace, fetch_stats = {}, [], Counter()
    for src in SOURCES:
        collected, seen = [], set()
        for eq in plan.expanded_queries:
            if not exhaustive and len(collected) >= cap:
                break
            fetch_n = cap if exhaustive else max(1, cap - len(collected))
            entry, was_cached = cached_search(
                collectors, snapshot, src, eq.query, fetch_n, live)
            fetch_stats["cached" if was_cached else "live"] += 1
            contributed = 0
            if entry["status"] == "ok":
                for raw in entry["items"]:
                    k = (raw["source"], raw.get("external_id") or raw.get("source_url"))
                    if k in seen:
                        continue
                    seen.add(k)
                    it = RWEItem(**raw)
                    it.matched_query = eq.query
                    it.matched_query_type = eq.expansion_type
                    collected.append(it)
                    contributed += 1
                    if not exhaustive and len(collected) >= cap:
                        break
            trace.append({"source": src, "q": eq.query, "type": eq.expansion_type,
                          "status": entry["status"], "raw": len(entry["items"]),
                          "contributed": contributed})
        per_source_items[src] = collected[:cap] if exhaustive else collected
    return per_source_items, trace, fetch_stats


def evaluate(per_source_items, plan, terms):
    """PRE (dedup only) and POST (production relevance_filter) checkpoints."""
    raw = [it for items in per_source_items.values() for it in items]
    pre = deduplicate(raw)
    relevance_query = plan.translated_query or plan.original_query
    post = relevance_filter(pre, relevance_query, entities=plan.entities)
    post.sort(key=lambda it: it.relevance_score, reverse=True)

    def stats(items):
        judged = [judge_item(f"{it.title} {it.text}", terms) for it in items]
        n = len(items) or 1
        return {
            "n": len(items),
            "by_source": dict(Counter(it.source for it in items)),
            "judge_relevant": sum(1 for r, _ in judged if r),
            "judge_relation": sum(1 for _, r in judged if r),
            "judge_precision": round(sum(1 for r, _ in judged if r) / n, 4),
            "judge_relation_rate": round(sum(1 for _, r in judged if r) / n, 4),
        }

    keys_post = {(it.source, it.external_id or it.source_url) for it in post}
    qu_pre = [it for it in pre if it.matched_query_type == "qu"]
    qu_dropped = [it for it in qu_pre
                  if (it.source, it.external_id or it.source_url) not in keys_post]
    return {
        "pre": stats(pre),
        "post": stats(post),
        "qu_items_pre": len(qu_pre),
        "qu_items_kept_post": len(qu_pre) - len(qu_dropped),
        "qu_items_dropped": len(qu_dropped),
        "qu_drop_reasons": dict(Counter(
            (it.match_reason or "").split("(")[0].strip() for it in qu_dropped)),
        "pre_keys": sorted({(it.source, it.external_id or it.source_url) for it in pre}),
        "post_keys": sorted(keys_post),
        "post_items": [
            {"source": it.source, "title": (it.title or "")[:90],
             "matched_query_type": it.matched_query_type,
             "matched_query": it.matched_query,
             "score": round(it.relevance_score, 3),
             "reason": (it.match_reason or "")[:90]}
            for it in post
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--exhaustive", action="store_true",
                    help="fetch EVERY expansion of every source (no early cap "
                         "stop) before trimming to cap — lets late expansions "
                         "(QU, colloquial) contribute to the PRE corpus. "
                         "Required to test whether the filter hides QU value.")
    args = ap.parse_args()
    cap = args.cap

    snapshot = json.loads(SNAPSHOT.read_text())
    qu_cache = json.loads(QU_CACHE.read_text())
    collectors = get_collectors()
    report = {"config": {"queries": BENCH_QUERIES, "cap_per_source": cap,
                         "sources": SOURCES, "filter": "production min_score=0.20",
                         "exhaustive": args.exhaustive},
              "queries": {}}

    for q in BENCH_QUERIES:
        print(f"\n{'='*96}\nQUERY: {q}")
        qu = qu_cache[q]
        plan_a, plan_b = build_plans(q, qu)
        terms = build_judge_terms(qu, plan_a.entities)

        res = {}
        for arm, plan in (("CURRENT", plan_a), ("QU", plan_b)):
            items, trace, fstats = collect(collectors, snapshot, plan, cap,
                                           live=True, exhaustive=args.exhaustive)
            ev = evaluate(items, plan, terms)
            contributed_by_type = Counter()
            for t in trace:
                if t["contributed"]:
                    contributed_by_type[t["type"]] += t["contributed"]
            ev["contributions_by_expansion_type"] = dict(contributed_by_type)
            ev["fetches"] = dict(fstats)
            res[arm] = ev
            print(f"  {arm:<8} PRE : n={ev['pre']['n']:>3} {str(ev['pre']['by_source'])[:70]}")
            print(f"  {arm:<8} POST: n={ev['post']['n']:>3} {str(ev['post']['by_source'])[:70]}")
            print(f"           judge P(pre)={ev['pre']['judge_precision']:.3f} → P(post)={ev['post']['judge_precision']:.3f} | "
                  f"R(pre)={ev['pre']['judge_relation_rate']:.3f} → R(post)={ev['post']['judge_relation_rate']:.3f}")
            print(f"           contributions: {ev['contributions_by_expansion_type']}  fetches: {ev['fetches']}")
            if arm == "QU":
                print(f"           QU-matched items: pre={ev['qu_items_pre']} kept_post={ev['qu_items_kept_post']} "
                      f"dropped={ev['qu_items_dropped']} reasons={ev['qu_drop_reasons']}")

        pre_only_qu = set(res["QU"]["pre_keys"]) - set(res["CURRENT"]["pre_keys"])
        post_only_qu = set(res["QU"]["post_keys"]) - set(res["CURRENT"]["post_keys"])
        post_lost = set(res["QU"]["pre_keys"]) - set(res["QU"]["post_keys"])
        res["delta"] = {
            "pre_items_only_in_QU": len(pre_only_qu),
            "post_items_only_in_QU": len(post_only_qu),
            "pre_QU_items_lost_by_filter": len(post_lost),
        }
        print(f"  Δ PRE-stage advantage of QU: {len(pre_only_qu)} items | kept POST: {len(post_only_qu)} | "
              f"filter killed {len(post_lost)} of QU's PRE corpus")
        report["queries"][q] = res
        SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False))

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved {OUT_JSON.name} | snapshot entries now: {len(snapshot)}")


if __name__ == "__main__":
    main()
