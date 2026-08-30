"""
Calvizie.net (Ieson Forum) collector — official XenForo RSS feed.

Source: https://calvizie.net/forum/forums/-/index.rss
Forum:  XenForo (https://xenforo.com) — "Community platform by XenForo"
ToS:    RSS is the official, documented XenForo content-syndication channel;
        each forum exposes ``<forum-slug>.<id>/index.rss``. robots.txt points
        to the sitemap; no anti-bot directive targets the RSS endpoint. The
        feed is intended for syndication (the homepage links the RSS link).
Auth:   None (public RSS). No API key, no OAuth.

Calvizie.net is an Italian, hair-loss-specific community ("La community
anticalvizie") with dedicated sub-forums for finasteride, dutasteride,
minoxidil, alopecia areata, telogen effluvium, female/male pattern loss,
LLLT, anti-DHT, etc. Threads are patient experiences / treatment discussions
— classified evidence_tier="anecdotal", source_type="community_forum".

Hair-transplant / cosmetic-concealer sub-forums are deliberately EXCLUDED
(see ``_FORUM_SLUGS``) so the collector never surfaces transplant content.
"""
from __future__ import annotations

from typing import List, Optional

from core.rwe.xenforo_base import (
    XenForoRSSCollector,
    STATUS_OK,
    STATUS_NO_RESULTS,
    STATUS_RATE_LIMITED,
    STATUS_NETWORK_ERROR,
)

BASE_URL = "https://calvizie.net/forum/forums"

# Hair-loss sub-forums included in collection. Each is a (slug, forum_id) pair
# for a non-surgical, hair-loss-relevant forum. Transplant / cosmetic-concealer
# forums (autotrapianto, tricopigmentazione, protesi, k-max fibers) are
# intentionally absent — they fall outside the hair-loss domain (surgical /
# cosmetic), per project exclusion rules.
_FORUM_SLUGS: List[str] = [
    # ── Drugs / treatments (core RWE) ──────────────────────────────────────
    "finasteride-propecia-proscar-c.6088",
    "dutasteride.2",
    "minoxidil-capelli-a-cosa-serve-e-tipologie.6090",
    "ormoni-estrogeni-idrocortisone-c.6103",
    "prodotti-anti-dht-revivogen-nioxin-hair-genesis.6099",
    "integratori-estratti-naturali-vitamine-co.6101",
    "nuovi-farmaci-molecole-e-tecniche-per-la-calvizie.6086",
    "clonazione-genetica-farmaci-sperimentali.6123",
    "rame-peptidi-e-antinfiammatori-topici.6097",
    # ── Conditions ──────────────────────────────────────────────────────────
    "alopecia-areata.6130",
    "effluvio-stagionale-capelli-e-tipi-di-calvizie.6132",
    "per-la-donna-tutto-sulla-calvizie-femminile.6128",
    "patologie-del-cuoio-capelluto-e-cure.6134",
    # ── Non-surgical treatments / care ──────────────────────────────────────
    "fototerapia-e-lllt.6095",
    "per-i-nuovi-scegliere-la-terapia-anticalvizie.6077",
    "i-migliori-shampoo-anticaduta-dht-capelli-fini.6105",
    "cura-e-igiene-per-la-bellezza-dei-capelli.6136",
    "psicologia-e-perdita-dei-capelli.6080",
]


class CalvizieCollector(XenForoRSSCollector):
    """Read-only Calvizie.net (Ieson Forum) XenForo RSS collector."""

    source = "calvizie"
    base_url = BASE_URL
    language = "it"
    forum_slugs: List[str] = _FORUM_SLUGS

    def __init__(self, forum_slugs: Optional[List[str]] = None) -> None:
        if forum_slugs is not None:
            self.forum_slugs = forum_slugs
