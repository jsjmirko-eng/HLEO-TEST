"""
Tests for the MaladiesRaresInfo.org phpBB Atom collector (alopecia areata, FR).

Covers: valid response, empty feed, malformed XML, timeout, HTTP error, rate
limit, normalization (FR language, ISO-8601 dates, Atom entry parsing),
provenance, multi-forum aggregation — using mocked HTTP so no live network is
required.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from core.rwe.maladiesrares_collector import (
    MaladiesRaresCollector,
    STATUS_OK,
    STATUS_NO_RESULTS,
    STATUS_RATE_LIMITED,
    STATUS_NETWORK_ERROR,
)
from core.rwe.models import RWEItem


# ── Test fixtures (phpBB Atom feed) ───────────────────────────────────────────

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="fr">
  <title>Le Forum maladies rares</title>
  <link href="https://forums.maladiesraresinfo.org/index.php" />
  <updated>2026-03-28T12:57:56+02:00</updated>
  <id>https://forums.maladiesraresinfo.org/app.php/feed/forum/173</id>
  <entry>
    <author><name><![CDATA[Moha]]></name></author>
    <updated>2026-03-28T12:57:56+02:00</updated>
    <published>2026-03-28T12:57:56+02:00</published>
    <id>https://forums.maladiesraresinfo.org/viewtopic.php?p=46948#p46948</id>
    <link href="https://forums.maladiesraresinfo.org/viewtopic.php?p=46948#p46948"/>
    <title type="html"><![CDATA[Pelade universelle • Re: Pelade, comment je l'ai vécu]]></title>
    <category term="Pelade universelle" label="Pelade universelle"/>
    <content type="html"><![CDATA[Bonjour,<br>J&#8217;ai traîné une pelade sur la barbe pendant 10 ans.]]></content>
  </entry>
  <entry>
    <author><name><![CDATA[Anonymous]]></name></author>
    <updated>2026-01-07T09:00:00+02:00</updated>
    <published>2026-01-07T09:00:00+02:00</published>
    <id>https://forums.maladiesraresinfo.org/viewtopic.php?p=46404#p46404</id>
    <link href="https://forums.maladiesraresinfo.org/viewtopic.php?p=46404#p46404"/>
    <title type="html"><![CDATA[Pelade universelle • Re: Mon expérience]]></title>
    <content type="html"><![CDATA[Merci pour votre témoignage. Pelade chronique depuis l&#8217;enfance.]]></content>
  </entry>
</feed>"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Empty</title></feed>"""

MALFORMED_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>unclosed"""


def _resp(status_code=200, text="", content=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.text = text
    r.content = (content if content is not None else text.encode("utf-8"))
    return r


@pytest.fixture
def collector():
    return MaladiesRaresCollector(forum_ids=[173])


# ── 1. Valid response + normalization ─────────────────────────────────────────

def test_valid_response_returns_items(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=SAMPLE_ATOM.encode("utf-8"))):
        items, status, reason = collector.search_with_status("pelade", limit=5)
    assert status == STATUS_OK
    assert len(items) == 2
    it = items[0]
    assert it.source == "maladiesrares"
    assert it.source_type == "community_forum"
    assert it.evidence_tier == "anecdotal"
    assert it.language == "fr"
    assert it.privacy_status == "redacted"
    assert it.date == "2026-03-28"
    assert it.title.startswith("Pelade universelle")
    # HTML entities decoded & br tags stripped to plain text
    assert "traîné une pelade" in it.text
    assert "<br>" not in it.text
    assert "bonjour" in it.text.lower()
    # external_id from <id>; source_url from <link href>
    assert it.external_id.endswith("p=46948#p46948")
    assert it.source_url.endswith("p=46948#p46948")
    # topic carries the forum id
    assert "173" in it.topic
    assert it.metadata["forum_id"] == 173


def test_second_entry_parsed(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=SAMPLE_ATOM.encode("utf-8"))):
        items, _, _ = collector.search_with_status("pelade", limit=5)
    assert items[1].date == "2026-01-07"
    assert "témoignage" in items[1].text.lower()


def test_text_truncated_to_4000(collector):
    big = "<content type=\"html\"><![CDATA[" + ("x" * 5000) + "]]></content>"
    atom = SAMPLE_ATOM.replace(
        "Bonjour,<br>J&#8217;ai traîné une pelade sur la barbe pendant 10 ans.",
        "x" * 5000,
    )
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=atom.encode("utf-8"))):
        items, status, _ = collector.search_with_status("pelade", limit=1)
    assert status == STATUS_OK
    assert len(items[0].text) <= 4000


# ── 2. Edge cases: empty / malformed / no title-body ──────────────────────────

def test_empty_feed_returns_no_results(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=EMPTY_ATOM.encode("utf-8"))):
        items, status, reason = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS


def test_malformed_xml_returns_network_error(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=MALFORMED_ATOM.encode("utf-8"))):
        items, status, reason = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "parse" in reason.lower()


def test_entry_without_title_and_body_is_skipped(collector):
    atom = SAMPLE_ATOM.replace(
        "<title type=\"html\"><![CDATA[Pelade universelle • Re: Mon expérience]]></title>",
        "",
    ).replace(
        "Merci pour votre témoignage. Pelade chronique depuis l&#8217;enfance.",
        "",
    )
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=atom.encode("utf-8"))):
        items, status, _ = collector.search_with_status("x", limit=5)
    assert status == STATUS_OK
    assert len(items) == 1  # second entry skipped


# ── 3. HTTP / network errors ─────────────────────────────────────────────────

def test_http_500_returns_network_error(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(500)):
        items, status, reason = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "500" in reason


def test_http_404_returns_no_results(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(404)):
        items, status, reason = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS
    assert "404" in reason


def test_rate_limit_429(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(429)):
        items, status, reason = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_RATE_LIMITED
    assert "rate" in reason.lower()


def test_timeout_returns_network_error(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               side_effect=requests.exceptions.Timeout("connect timed out")):
        items, status, _ = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


def test_connection_error_returns_network_error(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               side_effect=requests.exceptions.ConnectionError("DNS failed")):
        items, status, _ = collector.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


# ── 4. Query / aggregation ────────────────────────────────────────────────────

def test_empty_query_returns_no_results(collector):
    items, status, reason = collector.search_with_status("   ", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS
    assert "empty" in reason.lower()


def test_search_silent_fail_returns_empty_list_on_error(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(500)):
        assert collector.search("x", limit=5) == []


def test_multi_forum_aggregation_picks_ok_over_failures():
    c = MaladiesRaresCollector(forum_ids=[173, 999])
    def fake_get(url, *a, **k):
        if "999" in url:
            return _resp(404)
        return _resp(200, content=SAMPLE_ATOM.encode("utf-8"))
    with patch("core.rwe.maladiesrares_collector.requests.get", side_effect=fake_get):
        items, status, _ = c.search_with_status("pelade", limit=10)
    assert status == STATUS_OK
    assert len(items) == 2


def test_all_forums_fail_returns_aggregate_failure():
    c = MaladiesRaresCollector(forum_ids=[173])
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(500)):
        items, status, reason = c.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


def test_limit_caps_total(collector):
    with patch("core.rwe.maladiesrares_collector.requests.get",
               return_value=_resp(200, content=SAMPLE_ATOM.encode("utf-8"))):
        items, status, _ = collector.search_with_status("x", limit=1)
    assert status == STATUS_OK
    assert len(items) == 1


def test_default_forum_ids_includes_pelade_universelle():
    c = MaladiesRaresCollector()
    assert 173 in c.forum_ids
