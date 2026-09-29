from types import SimpleNamespace

from core.content_search_experiment import (
    _parse_full_text,
    _scan_text,
    derive_content_terms,
    experiment_enabled,
    run_content_search,
)
from core.relational_search import ClinicalRelation


def test_terms_come_from_relation_without_proprietary_vocabulary():
    relation = ClinicalRelation(
        original_query="dolore ai testicoli con finasteride",
        agent={"normalized": "finasteride"},
        event={"normalized": "testicular pain"},
        manifestation={"normalized": "testis"},
        scientific_query="finasteride may cause testicular pain",
        relation_phrases=["testicular pain associated with finasteride"],
    )

    terms = derive_content_terms(relation)
    texts = {term.text for term in terms}

    assert "finasteride" in texts
    assert "testicular pain" in texts
    assert all(term.source != "hleo_medical_vocabulary" for term in terms)


def test_full_text_parser_preserves_paragraph_and_section_context():
    text, paragraphs = _parse_full_text(
        """
        <article><body><sec><title>Results</title>
        <p>Finasteride was observed near testicular pain.</p>
        </sec></body></article>
        """
    )

    assert text == "Finasteride was observed near testicular pain."
    assert paragraphs == [{
        "text": "Finasteride was observed near testicular pain.",
        "section": "Results",
    }]


def test_scan_reports_occurrences_snippets_and_relation_distance():
    relation = ClinicalRelation(
        original_query="test",
        agent={"normalized": "finasteride"},
        event={"normalized": "testicular pain"},
        scientific_query="finasteride may cause testicular pain",
    )
    terms = derive_content_terms(relation)
    result = _scan_text(
        "A long prefix. FINASTERIDE may cause testicular pain in this paragraph.",
        terms,
        context_chars=20,
    )

    assert "finasteride" in result["matched_terms"]
    assert "testicular pain" in result["matched_terms"]
    assert result["match_count"] >= 2
    assert result["relation_proximity"]["minimum_distance_characters"] is not None
    assert result["relation_proximity"]["same_paragraph"] is True
    assert any("FINASTERIDE" in snippet or "testicular pain" in snippet
               for snippet in result["contexts"])


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("CONTENT_SEARCH_EXPERIMENT", raising=False)
    assert experiment_enabled() is False
    monkeypatch.setenv("CONTENT_SEARCH_EXPERIMENT", "true")
    assert experiment_enabled() is True


def test_missing_identifier_is_not_available():
    relation = ClinicalRelation(
        original_query="test",
        agent={"normalized": "finasteride"},
        event={"normalized": "testicular pain"},
        scientific_query="finasteride may cause testicular pain",
    )
    result = run_content_search(
        [SimpleNamespace(title="No ID", source="Unknown", metadata={})],
        relation,
    )

    assert result["total_results"] == 1
    assert result["full_text_available"] == 0
    assert result["full_text_not_available"] == 1
    assert result["articles"][0]["error"] == "no_public_identifier"
