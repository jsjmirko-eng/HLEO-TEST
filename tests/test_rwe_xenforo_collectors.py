"""
Tests for the HairLossTalk.com and HairLossExperiences.com XenForo RSS
collectors (both share the XenForo base in ``core.rwe.xenforo_base``).

Covers: valid response, empty response, malformed XML, timeout, HTTP error,
rate limit, normalization, provenance, multi-forum aggregation — using
mocked HTTP so no live network is required.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from core.rwe.hairlosstalk_collector import HairLossTalkCollector
from core.rwe.hairlossexperiences_collector import HairLossExperiencesCollector
from core.rwe.xenforo_base import (
    STATUS_OK,
    STATUS_NO_RESULTS,
    STATUS_RATE_LIMITED,
    STATUS_NETWORK_ERROR,
)
from core.rwe.models import RWEItem


# ── Test fixtures (XenForo RSS — same shape as Calvizie) ─────────────────────

SAMPLE_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:slash="http://purl.org/rss/1.0/modules/slash/">
  <channel>
    <title>Antiandrogens - Propecia, Dutasteride, etc.</title>
    <link>https://www.hairlosstalk.com/interact/</link>
    <item>
      <title>Topical finasteride - Support for a Beginner</title>
      <pubDate>Sun, 09 Aug 2026 01:50:41 +0000</pubDate>
      <link>https://www.hairlosstalk.com/interact/threads/topical-finasteride.130714/</link>
      <guid isPermaLink="false">130714</guid>
      <author>invalid@example.com (Timmy_17)</author>
      <category>Antiandrogens</category>
      <content:encoded><![CDATA[<div class="bbWrapper">Hello, I was hoping for help with topical finasteride. <a href="x" class="link link--internal">read more</a></div>]]></content:encoded>
      <slash:comments>12</slash:comments>
    </item>
    <item>
      <title>Minoxidil oral results</title>
      <pubDate>Mon, 04 Aug 2026 12:00:00 +0000</pubDate>
      <link>https://www.hairlosstalk.com/interact/threads/oral-minoxidil.130800/</link>
      <guid isPermaLink="false">130800</guid>
      <category>Growth Stimulants</category>
      <content:encoded><![CDATA[<div class="bbWrapper">Anyone tried oral minoxidil? Shedding increased then regrowth.</div>]]></content:encoded>
      <slash:comments>5</slash:comments>
    </item>
  </channel>
</rss>"""

EMPTY_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>Empty</title><link>x</link></channel></rss>"""

MALFORMED_RSS = """<?xml version="1.0"?>
<rss><channel><title>Broken</title><item><title>unclosed"""


def _resp(status_code=200, text="", content=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.text = text
    r.content = (content if content is not None else text.encode("utf-8"))
    return r


# ── HairLossTalk ──────────────────────────────────────────────────────────────

@pytest.fixture
def hlt():
    return HairLossTalkCollector(forum_slugs=["antiandrogens-propecia-dutasteride-etc.35"])


def test_hlt_valid_response_returns_items(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=SAMPLE_RSS.encode("utf-8"))):
        items, status, reason = hlt.search_with_status("finasteride", limit=5)
    assert status == STATUS_OK
    assert len(items) == 2
    assert items[0].source == "hairlosstalk"
    assert items[0].language == "en"
    assert items[0].source_type == "community_forum"
    assert items[0].evidence_tier == "anecdotal"
    assert items[0].privacy_status == "redacted"
    assert items[0].title == "Topical finasteride - Support for a Beginner"
    assert items[0].date == "2026-08-09"
    # internal link wrapper stripped, not in plain text
    assert "read more" not in items[0].text
    assert "topical finasteride" in items[0].text
    # external_id from guid; url from link
    assert items[0].external_id == "130714"
    assert items[0].source_url.endswith("130714/")
    # provenance metadata: forum slug + comment count carried
    assert items[0].metadata["forum"] == "antiandrogens-propecia-dutasteride-etc.35"
    assert items[0].metadata["comments"] == 12
    assert "hairlosstalk" in reason.lower()


def test_hlt_empty_feed_returns_no_results(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=EMPTY_RSS.encode("utf-8"))):
        items, status, reason = hlt.search_with_status("x", limit=5)
    assert items == []
    # all forums fetched OK but empty → no_results
    assert status == STATUS_NO_RESULTS


def test_hlt_malformed_xml_returns_network_error(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=MALFORMED_RSS.encode("utf-8"))):
        items, status, reason = hlt.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "parse" in reason.lower()


def test_hlt_http_500_returns_network_error(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(500)):
        items, status, reason = hlt.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "500" in reason


def test_hlt_http_404_returns_no_results(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(404)):
        items, status, reason = hlt.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS
    assert "404" in reason


def test_hlt_rate_limit_429(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(429)):
        items, status, reason = hlt.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_RATE_LIMITED
    assert "rate" in reason.lower()


def test_hlt_timeout_returns_network_error(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               side_effect=requests.exceptions.Timeout("connect timed out")):
        items, status, reason = hlt.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


def test_hlt_empty_query_returns_no_results(hlt):
    items, status, reason = hlt.search_with_status("   ", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS
    assert "empty" in reason.lower()


def test_hlt_search_silent_fail_returns_empty_list_on_error(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(500)):
        assert hlt.search("x", limit=5) == []


def test_hlt_multi_forum_aggregation_picks_ok_over_failures():
    c = HairLossTalkCollector(forum_slugs=["antiandrogens-propecia-dutasteride-etc.35", "missing.999"])
    def fake_get(url, *a, **k):
        if "missing.999" in url:
            return _resp(404)
        return _resp(200, content=SAMPLE_RSS.encode("utf-8"))
    with patch("core.rwe.xenforo_base.requests.get", side_effect=fake_get):
        items, status, reason = c.search_with_status("finasteride", limit=10)
    assert status == STATUS_OK
    assert len(items) == 2


def test_hlt_all_forums_fail_returns_aggregate_failure():
    c = HairLossTalkCollector(forum_slugs=["a.1", "b.2"])
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(500)):
        items, status, reason = c.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


def test_hlt_limit_caps_per_forum_and_total(hlt):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=SAMPLE_RSS.encode("utf-8"))):
        items, status, _ = hlt.search_with_status("x", limit=1)
    assert status == STATUS_OK
    assert len(items) == 1


# ── HairLossExperiences ───────────────────────────────────────────────────────

@pytest.fixture
def hle():
    return HairLossExperiencesCollector(forum_slugs=["hair-loss-medications.15"])


def test_hle_valid_response_returns_items(hle):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=SAMPLE_RSS.encode("utf-8"))):
        items, status, reason = hle.search_with_status("minoxidil", limit=5)
    assert status == STATUS_OK
    assert len(items) == 2
    assert items[0].source == "hairlossexperiences"
    assert items[0].language == "en"
    assert items[0].source_type == "community_forum"
    assert items[0].evidence_tier == "anecdotal"
    assert items[0].privacy_status == "redacted"
    assert items[0].metadata["forum"] == "hair-loss-medications.15"
    assert "hairlossexperiences" in reason.lower()


def test_hle_rate_limit_429(hle):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(429)):
        items, status, reason = hle.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_RATE_LIMITED


def test_hle_empty_feed_returns_no_results(hle):
    with patch("core.rwe.xenforo_base.requests.get",
               return_value=_resp(200, content=EMPTY_RSS.encode("utf-8"))):
        items, status, _ = hle.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NO_RESULTS


def test_hle_connection_error_returns_network_error(hle):
    with patch("core.rwe.xenforo_base.requests.get",
               side_effect=requests.exceptions.ConnectionError("DNS failed")):
        items, status, _ = hle.search_with_status("x", limit=5)
    assert items == []
    assert status == STATUS_NETWORK_ERROR


def test_hle_default_forum_slugs_excludes_transplant_sections():
    c = HairLossExperiencesCollector()
    # transplant-heavy forums must NOT be in the curated set
    slugs = " ".join(c.forum_slugs).lower()
    assert "transplant" not in slugs
    assert "scalp-micropigmentation" not in slugs
    assert "wigs" not in slugs
    assert "hair-loss-medications.15" in c.forum_slugs
    assert "general-hair-loss-forum.46" in c.forum_slugs


def test_hle_default_forum_slugs_exclude_transplant_and_cosmetic_sections():
    """Regression: sections removed during the 2026-08 scope audit must stay out.

    female-hair-loss-forum.14 and frequently-asked-hair-loss-questions.9 were
    removed because live audit showed their threads are FUT/FUE transplant
    surgery content, not non-surgical RWE. hair-loss-products.48 is cosmetic
    (hair fibres / wigs / styling spray).
    """
    c = HairLossExperiencesCollector()
    slugs = set(c.forum_slugs)
    assert "female-hair-loss-forum.14" not in slugs
    assert "frequently-asked-hair-loss-questions.9" not in slugs
    assert "hair-loss-products.48" not in slugs
    # only the two genuine non-surgical RWE forums remain
    assert slugs == {"hair-loss-medications.15", "general-hair-loss-forum.46"}


def test_hlt_default_forum_slugs_excludes_transplant_sections():
    c = HairLossTalkCollector()
    slugs = " ".join(c.forum_slugs).lower()
    assert "hair-transplant" not in slugs
    assert "hair-replacement" not in slugs
    assert "antiandrogens-propecia-dutasteride-etc.35" in c.forum_slugs
    assert "alopecia-areata.19" in c.forum_slugs


def test_hlt_default_forum_slugs_exclude_cosmetic_and_low_signal_sections():
    """Regression: sections removed during the 2026-08 scope audit must stay out."""
    c = HairLossTalkCollector()
    slugs = set(c.forum_slugs)
    # concealers / styling / fibres = cosmetic, not RWE
    assert "concealers-styling-products.27" not in slugs
    # womens-hair-loss.16 is actually "Wigs, Toppers, Extensions" = cosmetic
    assert "womens-hair-loss.16" not in slugs
    # general-discussions.63 = spam/off-topic noise
    assert "general-discussions-presentations-and-interviews.63" not in slugs
    # hair-loss-products.48 RSS is empty
    assert "hair-loss-products.48" not in slugs
    # RWE sections added in the audit are present
    for s in [
        "shedding-shedding-shedding.30",
        "success-stories.23",
        "womens-hair-loss-treatments.14",
        "mens-general-hair-loss-discussions.11",
    ]:
        assert s in slugs, f"expected RWE section {s} in defaults"
