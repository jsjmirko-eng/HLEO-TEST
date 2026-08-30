#!/usr/bin/env python3
"""
V1 vs V3 A/B benchmark — production relevance_filter (intent=None) vs the
QU-aware V3 branch (feature flag HLEO_RWE_INTENT_SCORING=1), on the SAME
corpus per query.

Method
------
- For every query the plan is built ONCE by the production RWEQueryEngine
  with the flag ON; the intent comes from the production extractor
  (ONE LLM call per query, counted; KB fallback on failure).
- The corpus is collected ONCE per query (exhaustive, cap=100/source,
  cache-first; live fetches only for cache misses, written back to
  corpus_snapshot.json). V3 does not change retrieval, so both arms share
  the identical PRE corpus by construction.
- V1 = relevance_filter(..., intent=None). V3 = relevance_filter(...,
  intent=plan.intent). Both use the production code path.
- Judge: HAND-WRITTEN term sets per query (see JUDGE below) — NOT derived
  from the QU output, the KB expansion, or the scorer. This is the
  anti-circularity guarantee: the judge cannot influence the intent
  construction (LLM prompt is fixed, judge-independent) nor the V3 scoring
  (deterministic given the intent; unit test
  test_no_benchmark_or_judge_dependency_in_production_modules guards the
  import boundary).

Metrics per query: PRE n, V1/V3 kept n, judge precision, judge recall
(vs judge-relevant items in PRE), judge-relevant lost, false positives
kept, deltas, latency (plan+LLM, scoring), LLM calls.

Usage: python benchmarks/v3_ab_benchmark.py
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

os.environ["HLEO_RWE_INTENT_SCORING"] = "1"  # enable V3 intent in the engine

import core.rwe.intent as intent_mod  # noqa: E402
from ab_pre_post_relevance import collect  # noqa: E402  (diagnostic harness)
from qu_ab_benchmark import get_collectors  # noqa: E402
from core.rwe.models import RWEItem  # noqa: E402
from core.rwe.pipeline import deduplicate, relevance_filter  # noqa: E402
from core.rwe.query_engine import RWEQueryEngine  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent / "corpus_snapshot.json"
OUT_JSON = Path(__file__).resolve().parent / "v3_ab_report.json"

# ── Hand-written judge (anti-circularity: no QU/KB-scorer derivation) ────────

FIN_IV = ["finasteride", "propecia", "proscar", "finpecia", "fincar",
          "finasterida"]
MIN_IV = ["minoxidil", "rogaine", "regaine", "minox"]
DUT_IV = ["dutasteride", "avodart"]
SHED_EV = ["shedding", "shed", "hair loss", "hair fall", "hairfall",
           "falling out", "fell out", "falls out", "caduta", "perdita di capelli",
           "perdita capelli", "perdita dei capelli", "telogen", "effluvium",
           "alopecia", "thinning", "diradamento"]
DEP_EV = ["depression", "depressed", "depressive", "depresso", "depressi",
          "deprimiert", "dépress", "mood"]
SEX_EV = ["libido", "sex drive", "sexual", "sessuale", "erectile", "erection",
          "erezione", "impotenz", "desiderio"]
SIDE_EV = SHED_EV + DEP_EV + SEX_EV + ["gynecomastia", "ginecomastia",
          "brain fog", "anxiety", "ansia", "nausea", "dizziness", "vertigini",
          "side effect", "effetti collaterali", "effetto collaterale",
          "adverse"]
HL_COND = ["hair loss", "hair fall", "alopecia", "caduta", "perdita di capelli",
           "perdita capelli", "perdita dei capelli", "thinning", "diradamento",
           "baldness", "calvizie", "chute de cheveux", "perte de cheveux"]
AGA_COND = ["androgenetic alopecia", "alopecia androgenetica", "androgenetica",
            "male pattern", "female pattern", "pattern hair loss",
            "pattern baldness"]

# (group, query, anchors, events). relevant = anchor AND event; if no events,
# relevant = anchor.
QUERIES = [
    ("A", "finasteride shedding", FIN_IV, SHED_EV),
    ("A", "finasteride hair shedding", FIN_IV, SHED_EV),
    ("A", "minoxidil hair loss", MIN_IV, SHED_EV),
    ("B", "finasteride", FIN_IV, []),
    ("B", "minoxidil", MIN_IV, []),
    ("C", "androgenetic alopecia", AGA_COND, []),
    ("C", "hair loss", HL_COND, []),
    ("D", "finasteride can cause depression", FIN_IV, DEP_EV),
    ("D", "finasteride side effects", FIN_IV, SIDE_EV),
    ("E", "finasteride dutasteride minoxidil", FIN_IV + DUT_IV + MIN_IV, []),
    ("F", "does finasteride make your hair fall out at first?", FIN_IV, SHED_EV),
    ("F", "since I started minoxidil my hair is falling out more", MIN_IV, SHED_EV),
    ("F", "I lost my libido since taking finasteride", FIN_IV, SEX_EV),
    ("G", "caduta capelli finasteride", FIN_IV, HL_COND + SHED_EV),
    ("G", "la finasteride può causare depressione?", FIN_IV, DEP_EV),
    ("G", "minoxidil shedding", MIN_IV, SHED_EV),
    ("G", "finasteride initial shedding", FIN_IV, SHED_EV),
]


def judge(text: str, anchors: list, events: list) -> bool:
    t = text.lower()
    has_anchor = any(a in t for a in anchors)
    if not events:
        return has_anchor
    return has_anchor and any(e in t for e in events)


def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    collectors = get_collectors()
    engine = RWEQueryEngine()

    # Count + time LLM calls made by the production intent extractor.
    llm = {"calls": 0, "seconds": 0.0}
    orig_extract = intent_mod.extract_intent_llm

    def counting_extract(*args, **kwargs):
        t0 = time.time()
        try:
            return orig_extract(*args, **kwargs)
        finally:
            llm["calls"] += 1
            llm["seconds"] += time.time() - t0

    intent_mod.extract_intent_llm = counting_extract

    report = {"config": {
        "min_score": 0.20, "cap_per_source": 100, "exhaustive": True,
        "flag": "HLEO_RWE_INTENT_SCORING=1",
        "judge": "hand-written term sets (anti-circular)",
        "n_queries": len(QUERIES),
    }, "queries": {}}

    totals = Counter()
    for group, query, anchors, events in QUERIES:
        calls_before = llm["calls"]
        t0 = time.time()
        plan = engine.plan(query)          # includes intent (1 LLM call max)
        plan_seconds = time.time() - t0

        per_source, _trace, _fs = collect(collectors, snapshot, plan, 100,
                                          live=True, exhaustive=True)
        raw_dicts = [it.model_dump() for items in per_source.values() for it in items]
        pre = deduplicate([RWEItem(**d) for d in raw_dicts])
        relevance_query = plan.translated_query or plan.original_query

        t1 = time.time()
        v1 = relevance_filter([RWEItem(**d) for d in raw_dicts],
                              relevance_query, entities=plan.entities,
                              intent=None)
        v1_ms = (time.time() - t1) * 1000
        t2 = time.time()
        v3 = relevance_filter([RWEItem(**d) for d in raw_dicts],
                              relevance_query, entities=plan.entities,
                              intent=plan.intent)
        v3_ms = (time.time() - t2) * 1000

        def j(it):
            return judge(f"{it.title} {it.text}", anchors, events)

        pre_rel = sum(1 for it in pre if j(it)) or 0
        denom = pre_rel or 1

        def arm(items):
            n = len(items)
            rel = sum(1 for it in items if j(it))
            return {"n": n, "judge_relevant": rel,
                    "precision": round(rel / n, 4) if n else 0.0,
                    "recall": round(rel / denom, 4),
                    "false_positives": n - rel}

        s1, s3 = arm(v1), arm(v3)
        # judge-relevant items present in PRE but NOT kept by the arm
        k1 = {(i.source, i.external_id or i.source_url) for i in v1}
        k3 = {(i.source, i.external_id or i.source_url) for i in v3}
        s1["lost_relevant"] = sum(1 for it in pre
                                  if j(it) and (it.source, it.external_id or it.source_url) not in k1)
        s3["lost_relevant"] = sum(1 for it in pre
                                  if j(it) and (it.source, it.external_id or it.source_url) not in k3)

        intent = plan.intent
        entry = {
            "group": group,
            "intent_source": getattr(intent, "source", None),
            "intent_iv": getattr(intent, "interventions", None),
            "intent_oc": getattr(intent, "outcomes", None),
            "intent_cd": getattr(intent, "conditions", None),
            "llm_calls": llm["calls"] - calls_before,
            "plan_seconds": round(plan_seconds, 2),
            "scoring_ms": {"v1": round(v1_ms, 1), "v3": round(v3_ms, 1)},
            "pre_n": len(pre), "pre_judge_relevant": pre_rel,
            "v1": s1, "v3": s3,
            "delta": {
                "precision": round(s3["precision"] - s1["precision"], 4),
                "recall": round(s3["recall"] - s1["recall"], 4),
                "kept": s3["n"] - s1["n"],
                "lost_relevant": s3["lost_relevant"] - s1["lost_relevant"],
                "false_positives": s3["false_positives"] - s1["false_positives"],
            },
        }
        report["queries"][query] = entry
        totals["pre"] += len(pre)
        totals["v1_n"] += s1["n"]; totals["v3_n"] += s3["n"]
        totals["v1_fp"] += s1["false_positives"]; totals["v3_fp"] += s3["false_positives"]
        totals["v1_lost"] += s1["lost_relevant"]; totals["v3_lost"] += s3["lost_relevant"]

        print(f"[{group}] {query}")
        print(f"     intent({entry['intent_source']}, llm_calls={entry['llm_calls']}): "
              f"iv={entry['intent_iv']} oc={entry['intent_oc']} cd={entry['intent_cd']}")
        print(f"     PRE n={len(pre)} (judge-rel {pre_rel}) | "
              f"V1 n={s1['n']} P={s1['precision']:.3f} R={s1['recall']:.3f} FP={s1['false_positives']} lost={s1['lost_relevant']} | "
              f"V3 n={s3['n']} P={s3['precision']:.3f} R={s3['recall']:.3f} FP={s3['false_positives']} lost={s3['lost_relevant']}")
        print(f"     Δ P={entry['delta']['precision']:+.3f} R={entry['delta']['recall']:+.3f} "
              f"kept={entry['delta']['kept']:+d} lost_rel={entry['delta']['lost_relevant']:+d} "
              f"FP={entry['delta']['false_positives']:+d} | "
              f"plan={entry['plan_seconds']}s score v1={entry['scoring_ms']['v1']}ms v3={entry['scoring_ms']['v3']}ms")
        SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False))

    report["totals"] = {
        **dict(totals),
        "llm_calls_total": llm["calls"],
        "llm_seconds_total": round(llm["seconds"], 2),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nTOTALS: {dict(totals)} | LLM calls={llm['calls']} ({llm['seconds']:.1f}s)")
    print(f"Saved {OUT_JSON.name}")


if __name__ == "__main__":
    main()
