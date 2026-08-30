"""
HairLossExperiences.com collector — official XenForo RSS feed.

Source: https://www.hairlossexperiences.com/forums/<slug>/index.rss
Forum:  XenForo (https://xenforo.com) — "Hair loss Forum"
ToS:    RSS is the official, documented XenForo content-syndication channel;
        each forum exposes ``<forum-slug>.<id>/index.rss``. robots.txt carries
        EU content-signals (search / ai-input / ai-train) but does NOT set any
        to ``no`` (case (c): neither grants nor restricts permission via
        content signal). The feed is linked from the forum homepage and is
        intended for syndication. No API key, no OAuth. No CAPTCHA / login /
        anti-bot challenge on the RSS endpoint.
Auth:   None (public RSS).

HairLossExperiences is an English-language hair-loss community. The site is
transplant-oriented overall (the vast majority of sub-forums are
Dr.-specific transplant clinics, before/after photo galleries, and
hair-transplant patient logs), so per-section inclusion is essential. Only
two sub-forums carry genuine non-surgical RWE: hair-loss-medications
(finasteride, oral minoxidil, dutasteride, clascoterone, PP405, treatment
duration) and general-hair-loss (spermidine, apigenin vs minoxidil, peptonix,
patient discussion). Threads are classified evidence_tier="anecdotal",
source_type="community_forum".

Per the per-section relevance rule, the transplant-heavy majority of the site
(Dr.-specific sub-forums, Hair-Transplant-Patients, Hair-Transplant clinics,
cosmetic-surgery, scalp-micropigmentation, hair-loss-wigs-and-toupees,
clinic-announcements) is deliberately EXCLUDED. The female-hair-loss-forum
and frequently-asked-hair-loss-questions sub-forums are also EXCLUDED because
their actual thread content is dominated by FUT/FUE transplant surgery and
post-transplant recovery, not non-surgical RWE. hair-loss-products is EXCLUDED
as cosmetic (hair fibres, wigs, styling spray). See ``_FORUM_SLUGS``.
"""
from __future__ import annotations

from typing import List, Optional

from core.rwe.xenforo_base import XenForoRSSCollector

BASE_URL = "https://www.hairlossexperiences.com/forums"

# Non-surgical, hair-loss-relevant sub-forums only. The transplant / clinic /
# SMP / wig / cosmetic-surgery sub-forums that dominate this site are
# intentionally absent — they fall outside the non-surgical hair-loss domain,
# per project exclusion rules. female-hair-loss-forum.14 and
# frequently-asked-hair-loss-questions.9 were removed after live audit showed
# their content is transplant-surgery dominated. hair-loss-products.48 is
# cosmetic (fibres/wigs). Section scope verified 2026-08.
_FORUM_SLUGS: List[str] = [
    "hair-loss-medications.15",
    "general-hair-loss-forum.46",
]


class HairLossExperiencesCollector(XenForoRSSCollector):
    """Read-only HairLossExperiences.com XenForo RSS collector."""

    source = "hairlossexperiences"
    base_url = BASE_URL
    language = "en"
    forum_slugs: List[str] = _FORUM_SLUGS

    def __init__(self, forum_slugs: Optional[List[str]] = None) -> None:
        if forum_slugs is not None:
            self.forum_slugs = forum_slugs
