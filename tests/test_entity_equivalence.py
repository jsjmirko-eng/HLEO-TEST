from core.vocab.entities import recognize
from tests.vocab_stubs import FakeResolution, FakeResolver, default_resolver, mesh


def _recognized(surface, match):
    resolver = FakeResolver({
        (surface, "*"): FakeResolution([match]),
    })
    return recognize(surface, "en", resolver)


def test_normalized_unrelated_terms_are_rejected():
    assert not _recognized(
        "cutaneo", mesh("Administration, Cutaneous", [], "M1", kind="normalized")
    ).entities
    assert not _recognized(
        "mal", mesh("Citrate", [], "M2", kind="normalized", group="condition")
    ).entities
    assert not _recognized(
        "alla", mesh("Allantoin", [], "M3", kind="normalized", group="condition")
    ).entities


def test_exact_edema_match_is_preserved():
    rec = _recognized("edema", mesh("Edema", [], "M4", group="symptom", conf=1.0))
    assert rec.entities == [("symptom", "edema", 1.0)]


def test_existing_clinical_synonym_is_preserved():
    rec = recognize("hair shedding", "en", default_resolver())
    assert any(canonical == "alopecia" for _etype, canonical, _conf in rec.entities)


def test_existing_drug_brand_is_preserved():
    rec = recognize("propecia", "en", default_resolver())
    assert any(canonical == "finasteride" for _etype, canonical, _conf in rec.entities)
