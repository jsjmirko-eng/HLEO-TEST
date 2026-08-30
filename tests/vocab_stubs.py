"""Shared test doubles for the Catena C vocabulary layer (offline).

These fakes simulate the RESPONSES of the external providers (RxNorm, MeSH,
ConceptNet) for a small set of well-known concepts. They exist because the
test suite is hermetic: production providers are network services and the
resolver/provider contract is already covered offline in tests/test_vocab.py.
No production code reads from this module — it is test scaffolding only.

FakeResolver mimics ``VocabularyResolver.resolve_terms(terms, language)``:
mapping keys are (term, language) with "*" as the any-language fallback, and
every call is logged so tests can assert WHICH language the Catena C used.
"""
from __future__ import annotations


class FakeMatch:
    def __init__(self, preferred_term, synonyms=(), match_kind="exact",
                 provider="rxnorm", concept_id="C1", semantic_group="drug",
                 language="en", confidence=0.9):
        self.preferred_term = preferred_term
        self.synonyms = list(synonyms)
        self.match_kind = match_kind
        self.provider = provider
        self.concept_id = concept_id
        self.semantic_group = semantic_group
        self.language = language
        self.confidence = confidence
        self.source_url = ""
        self.metadata = {}

    def model_dump(self):
        return {"preferred_term": self.preferred_term,
                "synonyms": list(self.synonyms),
                "match_kind": self.match_kind,
                "provider": self.provider,
                "confidence": self.confidence,
                "concept_id": self.concept_id,
                "semantic_group": self.semantic_group,
                "language": self.language,
                "source_url": "",
                "metadata": {}}


class FakeResolution:
    def __init__(self, matches):
        self.matches = list(matches)


class FakeResolver:
    """Drop-in for VocabularyResolver in tests (offline, deterministic)."""

    def __init__(self, mapping=None):
        # mapping: {(term_lower, language|"*"): FakeResolution}
        self._mapping = dict(mapping or {})
        self.calls = []  # [(terms_tuple, language)]

    def resolve_terms(self, terms, language="en"):
        lang = (language or "en").lower()
        self.calls.append((tuple(terms), lang))
        out = {}
        for t in terms or []:
            key = (t.lower().strip(), lang)
            if key in self._mapping:
                out[t] = self._mapping[key]
            elif (t.lower().strip(), "*") in self._mapping:
                out[t] = self._mapping[(t.lower().strip(), "*")]
        return out

    def languages_used(self):
        return {lang for _terms, lang in self.calls}


def patch_resolver(monkeypatch, resolver):
    """Point BOTH Catena C consumers (RWE query engine + scientific
    relational search) at the same FakeResolver instance."""
    monkeypatch.setattr("core.vocab.resolver.build_resolver_from_env",
                        lambda: resolver)
    monkeypatch.setattr("core.rwe.query_engine.build_resolver_from_env",
                        lambda: resolver)
    return resolver


def slim(*resolutions):
    """Slim serialisable view {term: [match_dicts]} like the query plan's."""
    out = {}
    for term, res in resolutions:
        out[term] = [m.model_dump() for m in res.matches]
    return out


# ── Standard concept fixtures (what the real providers would return) ─────────

def rxnorm(preferred, synonyms, concept_id, kind="exact", conf=0.95):
    return FakeMatch(preferred, synonyms, kind, "rxnorm", concept_id,
                     "drug", "en", conf)


def mesh(preferred, synonyms, concept_id, group="symptom", kind="exact",
         conf=0.95):
    return FakeMatch(preferred, synonyms, kind, "mesh", concept_id,
                     group, "en", conf)


def conceptnet(preferred, synonyms, kind, language, concept_id=None,
               group="general", conf=0.8):
    return FakeMatch(preferred, synonyms, kind, "conceptnet",
                     concept_id or f"/c/{language}/{preferred.replace(' ', '_')}",
                     group, language, conf)


def default_mapping():
    """Concepts used across the RWE/Catena C test suite."""
    m = {
        ("finasteride", "*"): FakeResolution([
            rxnorm("finasteride", ["propecia", "proscar", "finpecia"], "RX156"),
            mesh("Finasteride", ["Propecia", "Proscar"], "D015751", "drug"),
            conceptnet("sexual dysfunction",
                       ["libido", "depression", "depressed", "gynecomastia"],
                       "related_concept", "en", group="general", conf=0.7),
        ]),
        ("propecia", "*"): FakeResolution([
            rxnorm("finasteride", ["propecia", "proscar"], "RX156", "synonym", 0.9),
        ]),
        ("dutasteride", "*"): FakeResolution([
            rxnorm("dutasteride", ["avodart"], "RX157"),
        ]),
        ("minoxidil", "*"): FakeResolution([
            rxnorm("minoxidil", ["rogaine", "regaine", "loniten"], "RX210"),
            mesh("Minoxidil", ["Rogaine"], "D008914", "drug"),
        ]),
        ("rogaine", "*"): FakeResolution([
            rxnorm("minoxidil", ["rogaine", "regaine"], "RX210", "synonym", 0.9),
        ]),
        ("hypertrichosis", "*"): FakeResolution([
            mesh("Hypertrichosis",
                 ["excessive hair growth", "excess hair", "unwanted hair",
                  "excessive body hair", "excess facial hair"],
                 "D006983", "symptom"),
            conceptnet("hypertrichosis",
                       ["ipertricosi", "peli in eccesso"],
                       "colloquial", "it", group="general", conf=0.7),
        ]),
        ("hirsutism", "*"): FakeResolution([
            mesh("Hirsutism", ["androgen-dependent hair growth"],
                 "D053628", "symptom"),
        ]),
        ("hair loss", "*"): FakeResolution([
            mesh("Alopecia",
                 ["hair loss", "hair shedding", "baldness", "hair fall",
                  "caduta dei capelli", "caduta capelli"],
                 "D000505", "symptom"),
            conceptnet("hair loss", ["initial shedding", "hair fall out"],
                       "colloquial", "en", group="general", conf=0.7),
        ]),
        ("hair shedding", "*"): FakeResolution([
            mesh("Alopecia", ["hair loss", "hair shedding", "alopecia"],
                 "D000505", "symptom", "synonym", 0.9),
        ]),
        ("shedding", "*"): FakeResolution([
            mesh("Alopecia", ["hair loss", "hair shedding"],
                 "D000505", "symptom", "normalized", 0.7),
        ]),
        ("initial shedding", "*"): FakeResolution([
            mesh("Alopecia", ["hair loss", "hair shedding",
                              "caduta dei capelli"],
                 "D000505", "symptom", "normalized", 0.7),
        ]),
        ("androgenetic alopecia", "*"): FakeResolution([
            mesh("Androgenetic alopecia", ["male pattern baldness"],
                 "D000506", "condition"),
        ]),
        ("side effects", "*"): FakeResolution([
            mesh("Drug-Related Side Effects and Adverse Reactions",
                 ["adverse effects", "adverse drug reactions", "side effect"],
                 "D064420", "symptom"),
        ]),
        ("sexual dysfunction", "*"): FakeResolution([
            mesh("Sexual Dysfunction, Physiological",
                 ["erectile dysfunction", "libido loss", "impotence"],
                 "D012784", "symptom"),
        ]),
        ("erythema", "*"): FakeResolution([
            mesh("Erythema", ["redness", "skin redness"], "D004890", "symptom"),
        ]),
        # ── multilingual: translations into English concepts ────────────────
        ("caduta capelli", "it"): FakeResolution([
            conceptnet("hair loss", ["alopecia", "hair shedding"],
                       "translation", "it", group="symptom", conf=0.85),
        ]),
        ("ipertricosi", "it"): FakeResolution([
            conceptnet("hypertrichosis", ["excessive hair growth"],
                       "translation", "it", group="symptom", conf=0.85),
        ]),
        ("hipertricose", "pt"): FakeResolution([
            conceptnet("hypertrichosis", ["excessive hair growth"],
                       "translation", "pt", group="symptom", conf=0.85),
        ]),
        ("多毛症", "ja"): FakeResolution([
            conceptnet("hypertrichosis", ["excessive hair growth"],
                       "translation", "ja", group="symptom", conf=0.85),
        ]),
        ("ミノキシジル", "ja"): FakeResolution([
            conceptnet("minoxidil", ["rogaine"],
                       "translation", "ja", group="drug", conf=0.85),
        ]),
        ("haaruitval", "nl"): FakeResolution([
            conceptnet("hair loss", ["alopecia"],
                       "translation", "nl", group="symptom", conf=0.85),
        ]),
    }
    return m


def default_resolver():
    return FakeResolver(default_mapping())
