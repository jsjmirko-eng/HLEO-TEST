"""
HLEO — UMLS live acceptance test (trichology Italian query).

Run by the GitHub Actions workflow. Read-only w.r.t. the repository; uses a
temporary in-memory SQLite DB. Prints a full end-to-end report and exits
non-zero on a real provider failure or a gate regression. The API key value
is NEVER printed — only "UMLS_API_KEY configured: yes|no".
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("HLEO_RWE_INTENT_SCORING", "1")

QUERY = os.getenv("HLEO_LIVE_QUERY") or "perdita di capelli dopo finasteride, esperienze dei pazienti"

from core.database import engine  # noqa: E402
from core.models import Base  # noqa: E402

Base.metadata.create_all(bind=engine)


def _model_remap():
    """Route the OpenAI-compatible client to the configured endpoint/model
    without touching repo code (env-driven)."""
    base = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("HLEO_LLM_MODEL", "").strip()
    if not (base or model):
        return
    from openai.resources.chat.completions import Completions
    _orig = Completions.create

    def _patched(self, *args, **kwargs):
        if model:
            kwargs["model"] = model
        return _orig(self, *args, **kwargs)

    Completions.create = _patched


_model_remap()

umls_configured = bool((os.getenv("UMLS_API_KEY") or "").strip())
print(f"UMLS_API_KEY configured: {'yes' if umls_configured else 'no'}")

report = {"query": QUERY, "umls_configured": umls_configured}
t_start = time.time()

# ── 1. Query plan: translation, entities, UMLS concepts, expansions ─────────
from core.rwe.query_engine import RWEQueryEngine  # noqa: E402
from core.rwe.relation_filter import build_relation_context  # noqa: E402

plan = RWEQueryEngine().plan(QUERY)
report["detected_language"] = plan.detected_language
report["translated_query"] = plan.translated_query
report["translation_applied"] = plan.translation_applied
report["translation_method"] = plan.translation_method
report["canonical_query"] = plan.canonical_query
report["entities"] = [{"type": t, "canonical": c, "conf": cf}
                      for t, c, cf in plan.entities]
report["intent"] = (plan.intent.model_dump() if plan.intent else None)
report["expanded_queries"] = [eq.query for eq in plan.expanded_queries]

umls_concepts = []
for canonical, entries in (plan.vocabulary or {}).items():
    for e in entries:
        if e.get("provider") == "umls":
            umls_concepts.append({
                "canonical": canonical, "cui": e.get("concept_id"),
                "preferred_term": e.get("preferred_term"),
                "synonyms": (e.get("synonyms") or [])[:8],
                "match_kind": e.get("match_kind"),
                "semantic_group": e.get("semantic_group"),
            })
report["umls_concepts"] = umls_concepts

ctx = build_relation_context(plan)
report["relation_context"] = ({
    "agent_terms": ctx.agent_terms[:12],
    "manifestation_terms": ctx.manifestation_terms[:16],
    "other_agent_terms": ctx.other_agent_terms[:6],
} if ctx else None)

print(f"[A] lang={plan.detected_language} translation={plan.translation_method} "
      f"-> {plan.translated_query}")
print(f"[A] canonical: {plan.canonical_query}")
print(f"[A] entities: {[e['canonical'] for e in report['entities']]}")
print(f"[A] UMLS concepts: {len(umls_concepts)} "
      f"{[c['cui'] for c in umls_concepts]}")
print(f"[A] expanded ({len(report['expanded_queries'])}): "
      + " | ".join(q[:36] for q in report["expanded_queries"][:6]))

# ── 2. RWE search ────────────────────────────────────────────────────────────
import core.rwe.pipeline as pipe_mod  # noqa: E402

captured = {}
_orig_gate = pipe_mod.apply_relation_gate


def _spy(items, plan2):
    captured["items"] = list(items)
    return _orig_gate(items, plan2)


pipe_mod.apply_relation_gate = _spy
from core.rwe.pipeline import RWEPipeline  # noqa: E402

t0 = time.time()
res = RWEPipeline().search(QUERY, limit=400)
report["rwe"] = {
    "source_status": res.source_status,
    "totals": res.totals,
    "n_pre_gate": len(captured.get("items", [])),
    "elapsed_s": round(time.time() - t0, 1),
    "final_items": [{
        "title": it.title, "source": it.source, "url": it.source_url,
        "score": it.relevance_score,
        "gate": (it.metadata or {}).get("relation_gate"),
        "profile": (it.metadata or {}).get("rwe_profile"),
        "text": (it.text or "")[:300],
    } for it in res.items],
    "dropped": [{
        "title": it.title, "source": it.source,
        "drop_reason": ((it.metadata or {}).get("relation_gate") or {}).get("drop_reason"),
    } for it in captured.get("items", [])
    if ((it.metadata or {}).get("relation_gate") or {}).get("drop_reason")],
}
print(f"[C] RWE: pre_gate={report['rwe']['n_pre_gate']} "
      f"final={len(report['rwe']['final_items'])} totals={res.totals}")

# ── 3. Scientific search ─────────────────────────────────────────────────────
report["scientific"] = {"note": "skipped (no OPENAI_API_KEY)"}
if os.getenv("OPENAI_API_KEY"):
    t0 = time.time()
    try:
        from core.relational_search import RelationalSearch  # noqa: E402
        out = RelationalSearch().search(QUERY)
        if out:
            rel = out["relation"]
            items = []
            for src in ("pubmed", "europepmc", "clinicaltrials"):
                for it in out.get(src, []):
                    md = it.metadata or {}
                    items.append({
                        "source": it.source, "title": it.title,
                        "year": it.year, "url": it.url, "score": it.score,
                        "final_score": md.get("final_score"),
                        "relevance_reason": md.get("relevance_reason"),
                    })
            report["scientific"] = {
                "relation": rel.to_dict() if hasattr(rel, "to_dict") else str(rel),
                "stats": out["stats"],
                "items": sorted(items, key=lambda x: -(x["score"] or 0))[:15],
                "elapsed_s": round(time.time() - t0, 1),
            }
        else:
            report["scientific"] = {"note": "RelationalSearch returned None"}
    except Exception as exc:  # noqa: BLE001
        report["scientific"] = {"error": f"{type(exc).__name__}: {exc}"}
    sci_n = len(report["scientific"].get("items", []))
    print(f"[B] SCI: final items={sci_n}")
else:
    print("[B] SCI: skipped (no OPENAI_API_KEY)")

# ── 4. Synthesis ─────────────────────────────────────────────────────────────
report["synthesis"] = {"note": "skipped"}
sci_items = report["scientific"].get("items", [])
if os.getenv("OPENAI_API_KEY") and sci_items:
    try:
        from fastapi.testclient import TestClient  # noqa: E402
        from api.main import app  # noqa: E402
        client = TestClient(app)
        art = [{
            "id": it.get("url") or it.get("title", "")[:40],
            "source": it.get("source", ""),
            "title": it.get("title", ""),
            "abstract": "",
            "year": it.get("year"),
        } for it in sci_items[:10]]
        r = client.post("/synthesis", json={
            "query": QUERY,
            "relation": report["scientific"].get("relation"),
            "articles": art,
        })
        report["synthesis"] = r.json() if r.status_code == 200 else {
            "error": f"HTTP {r.status_code}"}
        print("[G] synthesis: HTTP", r.status_code)
    except Exception as exc:  # noqa: BLE001
        report["synthesis"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("[G] synthesis error:", type(exc).__name__)
else:
    print("[G] synthesis skipped")

report["total_elapsed_s"] = round(time.time() - t_start, 1)

with open("/tmp/umls_live_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False, default=str)

print("REPORT saved to /tmp/umls_live_report.json")
print("TOTAL elapsed:", report["total_elapsed_s"], "s")

fail = False
if umls_configured and not umls_concepts and not report["expanded_queries"]:
    print("FAIL: UMLS configured but no concepts and no expansions produced")
    fail = True
if res.totals.get("final", 0) == 0 and res.totals.get("relevant", 0) == 0:
    print("FAIL: pipeline produced zero relevant AND zero final results")
    fail = True
sys.exit(1 if fail else 0)
