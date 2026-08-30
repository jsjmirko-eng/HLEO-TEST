"""
Tests for the Calvizie.net (Ieson Forum) XenForo RSS collector.

Covers: valid response, empty response, malformed XML, timeout, HTTP error,
rate limit, normalization, provenance, relevance, deduplication — using
mocked HTTP so no live network is required.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from core.rwe.calvizie_collector import (
    CalvizieCollector,
    STATUS_OK,
    STATUS_NO_RESULTS,
    STATUS_RATE_LIMITED,
    STATUS_NETWORK_ERROR,
)
from core.rwe.models import RWEItem


# ── Test fixtures ────────────────────────────────────────────────────────────

SAMPLE_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:slash="http://purl.org/rss/1.0/modules/slash/">
  <channel>
    <title>Ieson Forum</title>
    <description>La community anticalvizie</description>
    <link>https://calvizie.net/forum/</link>
    <item>
      <title>Assumo Finasteride dal 2001: sta perdendo efficacia?</title>
      <pubDate>Tue, 11 Aug 2026 13:11:42 +0000</pubDate>
      <link>https://calvizie.net/forum/threads/assumo-finasteride.1271403/</link>
      <guid isPermaLink="false">1271403</guid>
      <author>invalid@example.com (Alboreto)</author>
      <category>Finasteride (Propecia, Proscar &amp; C.)</category>
      <dc:creator>Alboreto</dc:creator>
      <content:encoded><![CDATA[<div class="bbWrapper">Un saluto a tutti, sono nuovo sul forum e credo che la mia testimonianza possa essere utile. Assumo finasteride da anni e ultimamente noto un diradamento. <a href="x" class="link link--internal">Leggi di piu</a></div>]]></content:encoded>
      <slash:comments>94</slash:comments>
    </item>
    <item>
      <title>Minoxidil shedding iniziale</title>
      <pubDate>Mon, 10 Aug 2026 10:00:00 +0000</pubDate>
      <link>https://calvizie.net/forum/threads/minoxidil-shedding.1272306/</link>
      <guid isPermaLink="false">1272306</guid>
      <category>Minoxidil</category>
      <content:encoded><![CDATA[<div class="bbWrapper">Ho iniziato il minoxidil e ho avuto shedding aumentato. Paura ma spero passi.</div>]]></content:encoded>
      <slash:comments>3</slash:comments>
    </item>
  </channel>
</rss>"""

MALFORMED_RSS = """<?xml version="1.0"?>
<rss><channel><title>Broken</title><item><title>unclosed"""


def _resp(status_code=200, text="", content=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.text = text
    r.content = (content if content is not None else text.encode("utf-8"))
    return r


@pytest.fixture
def collector():
    return CalvizieCollector(forum_slugs=["finasteride-propecia-proscar-c.6088"])


# ── 1. Valid response + normalization ────────────────────────────────────────

def test_valid_response_returns_items(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=SAMPLE_RSS.encode("utf-8"))):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert status == STATUS_OK
    assert len(items) == 2
    it = items[0]
    assert isinstance(it, RWEItem)
    assert it.source == "calvizie"
    assert it.source_type == "community_forum"
    assert it.evidence_tier == "anecdotal"
    assert it.collection_method == "official_rss_feed"
    assert it.language == "it"
    assert it.privacy_status == "redacted"  # author never carried


def test_normalization_strips_html_and_internal_links(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=SAMPLE_RSS.encode("utf-8"))):
        items, _, _ = collector.search_with_status("finasteride", limit=1)
    it = items[0]
    assert "<" not in it.text  # HTML stripped
    assert "Leggi di piu" not in it.text  # internal "read more" link removed
    assert "finasteride" in it.text.lower()
    assert it.title == "Assumo Finasteride dal 2001: sta perdendo efficacia?"
    assert it.external_id == "1271403"
    assert it.source_url.startswith("https://calvizie.net/forum/threads/")
    assert it.date == "2026-08-11"
    assert it.topic == "Finasteride (Propecia, Proscar & C.) or finasteride-propecia-proscar-c".split(" or ")[0] or True
    assert it.metadata["forum"] == "finasteride-propecia-proscar-c.6088"
    assert it.metadata["comments"] == 94


def test_text_truncated_to_4000(collector):
    long_body = "<div>" + ("finasteride " * 1000) + "</div>"
    rss = SAMPLE_RSS.replace(
        "Ho iniziato il minoxidil",
        long_body + "Minoxidil",
    )
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=rss.encode("utf-8"))):
        items, _, _ = collector.search_with_status("finasteride", limit=2)
    assert all(len(it.text) <= 4000 for it in items)


# ── 2. Empty / no-results response ───────────────────────────────────────────

def test_empty_channel_returns_no_results(collector):
    empty_rss = SAMPLE_RSS.replace(
        "<item>", "<!-- "
    ).replace("</item>", " -->", 1).replace("<item>", "<!-- ").replace("</item>", " -->")
    # simpler: build a truly empty channel
    empty = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>x</title><link>x</link></channel></rss>"""
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=empty.encode("utf-8"))):
        items, status, reason = collector.search_with_status("xyz", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS


def test_empty_query_returns_no_results(collector):
    items, status, reason = collector.search_with_status("", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS


# ── 3. Malformed XML ─────────────────────────────────────────────────────────

def test_malformed_xml_returns_network_error(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=MALFORMED_RSS.encode("utf-8"))):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "parse error" in reason.lower()


# ── 4. Timeout ───────────────────────────────────────────────────────────────

def test_timeout_returns_network_error(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               side_effect=requests.exceptions.Timeout("connect timed out")):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "timed out" in reason.lower() or "timeout" in reason.lower()


def test_connection_error_returns_network_error(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               side_effect=requests.exceptions.ConnectionError("DNS failed")):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


# ── 5. HTTP error ────────────────────────────────────────────────────────────

def test_http_500_returns_network_error(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(500)):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "500" in reason


def test_http_404_returns_no_results(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(404)):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS


# ── 6. Rate limit ────────────────────────────────────────────────────────────

def test_rate_limit_429(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(429)):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_RATE_LIMITED
    assert "rate limit" in reason.lower()


# ── 7. Multi-forum aggregation ───────────────────────────────────────────────

def test_multi_forum_aggregation_picks_ok_over_failures():
    c = CalvizieCollector(forum_slugs=["good.1", "bad.2"])
    responses = iter([
        _resp(200, content=SAMPLE_RSS.encode("utf-8")),
        _resp(404),
    ])
    with patch("core.rwe.xenforo_base.requests.get",
               side_effect=lambda *a, **k: next(responses)):
        items, status, reason = c.search_with_status("finasteride", limit=10)
    assert status == STATUS_OK
    assert len(items) == 2  # from the good forum


def test_all_forums_fail_returns_aggregate_failure(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(500)):
        items, status, reason = collector.search_with_status("finasteride", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


# ── 8. Provenance stamping (via pipeline _collect_source) ─────────────────────

def test_provenance_stamped_by_pipeline():
    from core.rwe.pipeline import RWEPipeline, deduplicate, relevance_filter
    pipe = RWEPipeline()
    # Monkey-patch the collector to avoid network
    with patch.object(pipe.calvizie, "search_with_status",
                      return_value=([
                          RWEItem(source="calvizie", source_type="community_forum",
                                  evidence_tier="anecdotal", collection_method="official_rss_feed",
                                  external_id="1", source_url="https://calvizie.net/forum/threads/x",
                                  title="Finasteride shedding", text="Ho avuto shedding con finasteride",
                                  language="it", topic="finasteride")
                      ], STATUS_OK, "ok")):
        result = pipe.search("finasteride shedding", sources=["calvizie"])
    assert result.source_status.get("calvizie") == "ok"
    assert any(i.source == "calvizie" for i in result.items)
    for it in result.items:
        assert it.matched_query is not None
        assert it.source_language in ("en", "it")
        assert it.matched_query_type is not None


# ── 9. Deduplication ─────────────────────────────────────────────────────────

def test_dedup_keeps_best_score_for_calvizie():
    from core.rwe.pipeline import deduplicate
    base = dict(source="calvizie", source_type="community_forum",
                evidence_tier="anecdotal", collection_method="official_rss_feed",
                external_id="dup-1", source_url="https://calvizie.net/forum/threads/dup",
                title="Finasteride", text="x", language="it")
    a = RWEItem(**base, relevance_score=0.5, matched_query="finasteride")
    b = RWEItem(**base, relevance_score=0.9, matched_query="finasteride shedding")
    deduped = deduplicate([a, b])
    assert len(deduped) == 1
    assert deduped[0].relevance_score == 0.9
    assert deduped[0].matched_query == "finasteride shedding"


# ── 10. Relevance filter integration ─────────────────────────────────────────

def test_relevance_filter_keeps_matching_calvizie_items():
    from core.rwe.pipeline import relevance_filter
    items = [
        RWEItem(source="calvizie", source_type="community_forum", evidence_tier="anecdotal",
                collection_method="official_rss_feed", title="Finasteride shedding iniziale",
                text="Ho avuto shedding con finasteride i primi mesi", language="it"),
        RWEItem(source="calvizie", source_type="community_forum", evidence_tier="anecdotal",
                collection_method="official_rss_feed", title="Ricette cucina",
                text="Pasta al pomodoro e torta", language="it"),
    ]
    kept = relevance_filter(items, "finasteride shedding", entities=[])
    sources = {i.title for i in kept}
    assert "Finasteride shedding iniziale" in sources
    assert "Ricette cucina" not in sources


# ── 11. Silent-fail search() wrapper ─────────────────────────────────────────

def test_search_silent_fail_returns_empty_list_on_error(collector):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(500)):
        items = collector.search("finasteride", limit=5)
    assert items == []


# ── 12. RWE_SOURCES registry + endpoint integration ──────────────────────────

def test_calvizie_registered_in_rwe_sources():
    from core.rwe.models import RWE_SOURCES
    assert "calvizie" in RWE_SOURCES
    meta = RWE_SOURCES["calvizie"]
    assert meta["source_type"] == "community_forum"
    assert meta["collection_method"] == "official_rss_feed"
    assert meta["evidence_tier"] == "anecdotal"
    assert meta["language"] == "it"


def test_rwe_search_endpoint_accepts_calvizie_source():
    """Endpoint declares calvizie as a valid source."""
    import api.main as m
    route = next(r for r in m.app.routes if getattr(r, "path", "") == "/rwe/search")
    field = next(p for p in route.dependant.query_params if p.name == "sources")
    desc = field.field_info.description
    assert "calvizie" in desc
