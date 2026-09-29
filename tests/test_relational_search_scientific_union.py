from core.relational_search import RelationalSearch
from core.search_result import SearchResult


def _article(source, title, *, doi=None, source_id=None, pmid=None):
    return SearchResult(
        title=title,
        source=source,
        doi=doi,
        pmid=pmid,
        source_id=source_id,
        authors=["Author"],
        abstract="A scientific abstract with enough metadata.",
        metadata={"journal": "Journal", "year": 2024},
        year=2024,
    )


def _dedup(raw):
    instance = RelationalSearch.__new__(RelationalSearch)
    return instance._deduplicate_scientific(raw)


def test_exclusive_registered_sources_survive_dedup():
    raw = {
        "pubmed": [],
        "europepmc": [],
        "clinicaltrials": [],
        "openaire": [_article("OpenAIRE", "OpenAIRE-only", doi="10.1000/openaire")],
        "hal": [_article("HAL", "HAL-only", source_id="hal-0001")],
        "cinii": [_article("CiNii Research", "CiNii-only", source_id="cinii-0001")],
        "jstage": [_article("J-STAGE", "J-STAGE-only", doi="10.1000/jstage")],
        "crossref": [_article("Crossref", "Crossref-only", doi="10.1000/crossref")],
    }

    results = _dedup(raw)

    assert {item.title for item in results} == {
        "OpenAIRE-only", "HAL-only", "CiNii-only", "J-STAGE-only", "Crossref-only",
    }


def test_duplicate_doi_keeps_one_record_and_all_sources():
    raw = {
        "pubmed": [_article("PubMed", "Shared article", doi="10.1000/shared", pmid="123")],
        "europepmc": [],
        "clinicaltrials": [],
        "openaire": [_article("OpenAIRE", "Shared article", doi="10.1000/shared")],
    }

    results = _dedup(raw)

    assert len(results) == 1
    assert results[0].doi == "10.1000/shared"
    assert results[0].sources == ["PubMed", "OpenAIRE"]
    assert results[0].metadata["sources"] == ["PubMed", "OpenAIRE"]


def test_source_id_without_doi_survives():
    raw = {
        "pubmed": [],
        "europepmc": [],
        "clinicaltrials": [],
        "hal": [_article("HAL", "No DOI article", source_id="hal-record-42")],
    }

    results = _dedup(raw)

    assert len(results) == 1
    assert results[0].source_id == "hal-record-42"


def test_legacy_sources_continue_to_survive():
    raw = {
        "pubmed": [_article("PubMed", "Legacy PubMed", pmid="100")],
        "europepmc": [_article("Europe PMC", "Legacy Europe PMC", source_id="PMC100")],
        "clinicaltrials": [_article("ClinicalTrials.gov", "Legacy trial", source_id="NCT000100")],
    }

    results = _dedup(raw)

    assert {item.title for item in results} == {
        "Legacy PubMed", "Legacy Europe PMC", "Legacy trial",
    }
