"""
MeSH provider — official NLM MeSH RDF REST services (id.nlm.nih.gov/mesh).

Docs: https://id.nlm.nih.gov/mesh/  (lookup + RDF record endpoints)

Motivation for HLEO: conditions, symptoms and biomedical concepts with their
Entry Terms (the official scientific synonyms) — the event/condition side of
the QU intent, aligned with the biomedical literature.

Uses the lookup REST API (descriptor?label=...&match=exact|contains) and the
RDF JSON record (…/mesh/{UI}.json) for Entry Terms. No scraping.
"""
from __future__ import annotations

from typing import List, Optional

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch

LOOKUP = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
RECORD = "https://id.nlm.nih.gov/mesh/{}.json"


class MeSHProvider(VocabularyProvider):
    name = "mesh"
    semantic_group_hint = "condition"

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        headers = {"Accept": "application/json"}
        out: List[VocabularyMatch] = []
        seen = set()
        for match_mode, kind in (("exact", "exact"), ("contains", "normalized")):
            if len(out) >= limit:
                break
            data = self._get_json(LOOKUP, params={"label": term,
                                                  "match": match_mode,
                                                  "limit": limit},
                                  headers=headers)
            for row in (data or []):
                ui = (row.get("resource") or "").rstrip("/").split("/")[-1]
                if not ui or ui in seen:
                    continue
                seen.add(ui)
                concept = self._record(ui, term)
                if concept:
                    if kind == "normalized" and concept.match_kind == "exact":
                        pass  # keep exact kind from the record comparison
                    else:
                        concept.match_kind = kind
                        if kind == "normalized":
                            concept.confidence = min(concept.confidence, 0.7)
                    out.append(concept)
                if len(out) >= limit:
                    break
        return out[:limit]

    @staticmethod
    def _label_of(node) -> str:
        if isinstance(node, dict):
            return (node.get("@value") or "").strip()
        if isinstance(node, str):
            return node.strip()
        return ""

    def _term_label(self, term_uri: str) -> str:
        """Fetch a MeSH Term record and return its lexical label.

        Term records expose `prefLabel` (not `label`)."""
        ui = term_uri.rstrip("/").split("/")[-1]
        try:
            data = self._get_json(RECORD.format(ui),
                                  headers={"Accept": "application/json"})
        except Exception:  # noqa: BLE001 — a dead term record is just a miss
            return ""
        if not isinstance(data, dict):
            return ""
        return self._label_of(data.get("prefLabel") or data.get("label"))

    def _record(self, ui: str, queried_term: str) -> Optional[VocabularyMatch]:
        data = self._get_json(RECORD.format(ui),
                              headers={"Accept": "application/json"})
        if not isinstance(data, dict):
            return None
        preferred = self._label_of(data.get("label"))
        if not preferred:
            return None
        # Entry Terms: descriptor → concepts (cap 3) → terms (cap 8 each).
        # One-time cost, then served by the cache.
        synonyms: List[str] = []
        concepts = data.get("concept") or []
        if isinstance(concepts, str):
            concepts = [concepts]
        for concept_uri in concepts[:3]:
            cui = concept_uri.rstrip("/").split("/")[-1]
            try:
                cdata = self._get_json(RECORD.format(cui),
                                       headers={"Accept": "application/json"})
            except Exception:  # noqa: BLE001
                continue
            terms = (cdata or {}).get("term") or [] if isinstance(cdata, dict) else []
            if isinstance(terms, str):
                terms = [terms]
            for term_uri in terms[:8]:
                label = self._term_label(term_uri)
                if (label and label.lower() != preferred.lower()
                        and label not in synonyms):
                    synonyms.append(label)
        kind = ("exact" if queried_term.lower() == preferred.lower()
                else "synonym" if queried_term.lower()
                in {s.lower() for s in synonyms} else "concept")
        trees = data.get("treeNumber") or []
        if isinstance(trees, str):
            trees = [trees]
        trees = [t.rsplit("/", 1)[-1] for t in trees]
        # MeSH tree C23 = "Pathological Conditions, Signs and Symptoms":
        # the provider's own tree numbers type the concept — no local mapping.
        semantic_group = ("symptom" if any(t.startswith("C23") for t in trees)
                          else "condition")
        return VocabularyMatch(
            provider=self.name,
            concept_id=ui,
            preferred_term=preferred,
            synonyms=synonyms[:15],
            semantic_group=semantic_group,
            language="en",
            confidence=1.0 if kind == "exact" else 0.9,
            match_kind=kind,
            source_url=f"https://id.nlm.nih.gov/mesh/{ui}.html",
            metadata={"tree_numbers": trees[:5]},
        )

    def _get_synonyms(self, concept_id: str) -> List[str]:
        match = self._record(concept_id, queried_term="")
        return match.synonyms if match else []

    def _get_concept(self, concept_id: str) -> Optional[VocabularyMatch]:
        return self._record(concept_id, queried_term="")
