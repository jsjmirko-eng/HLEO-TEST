"""
HairLossTalk.com collector — official XenForo RSS feed.

Source: https://www.hairlosstalk.com/interact/forums/<slug>/index.rss
Forum:  XenForo (https://xenforo.com) — "HairLossTalk Forums"
ToS:    RSS is the official, documented XenForo content-syndication channel;
        each forum exposes ``<forum-slug>.<id>/index.rss``. robots.txt
        disallows only ``/shop/``; the ``/interact/`` (forums) tree and its RSS
        endpoints are not disallowed. The feed is linked from the forum
        homepage and intended for syndication. No API key, no OAuth.
Auth:   None (public RSS).

HairLossTalk is a long-running, English-language, hair-loss-specific community
with dedicated sub-forums for antiandrogens (Propecia/Dutasteride), growth
stimulants (Rogaine/Minoxidil), alopecia areata, dealing with side effects,
alternative treatments, shedding, success stories and general discussion.
Threads are patient experiences / treatment discussions — classified
evidence_tier="anecdotal", source_type="community_forum".

Hair-transplant sub-forums (Dr. Bernstein before/after photos, FUE/FUT
discussions, hair-transplant doctor reviews), hair-replacement/wig systems,
and concealer/cosmetic sub-forums (Toppik, fibres, styling products,
extensions) are deliberately EXCLUDED (see ``_FORUM_SLUGS``) so the collector
never surfaces transplant or cosmetic content — they fall outside the
non-surgical hair-loss RWE domain, per project rules.
"""
from __future__ import annotations

from typing import List, Optional

from core.rwe.xenforo_base import XenForoRSSCollector

BASE_URL = "https://www.hairlosstalk.com/interact/forums"

# Hair-loss sub-forums included in collection (non-surgical, RWE-relevant).
# Transplant / hair-replacement / wig / concealer / cosmetic sub-forums are
# intentionally absent — they fall outside the non-surgical hair-loss domain.
# Section scope verified 2026-08 against the live forums index.
_FORUM_SLUGS: List[str] = [
    # ── Drugs / treatments (core RWE) ──────────────────────────────────────
    "antiandrogens-propecia-dutasteride-etc.35",
    "growth-stimulants-rogaine-minoxidil-tricomin.29",
    "alternative-treatments.17",
    # ── Conditions ──────────────────────────────────────────────────────────
    "alopecia-areata.19",
    "alopecia-totalis-and-universalis-support.20",
    "alopecia-general-discussions.37",
    # ── Side effects / shedding / outcomes ──────────────────────────────────
    "dealing-with-side-effects.31",
    "shedding-shedding-shedding.30",
    "success-stories.23",
    # ── General patient discussion (men's & women's treatment tracks) ───────
    "mens-general-hair-loss-discussions.11",
    "womens-hair-loss-treatments.14",
]


class HairLossTalkCollector(XenForoRSSCollector):
    """Read-only HairLossTalk.com XenForo RSS collector."""

    source = "hairlosstalk"
    base_url = BASE_URL
    language = "en"
    forum_slugs: List[str] = _FORUM_SLUGS

    def __init__(self, forum_slugs: Optional[List[str]] = None) -> None:
        if forum_slugs is not None:
            self.forum_slugs = forum_slugs
