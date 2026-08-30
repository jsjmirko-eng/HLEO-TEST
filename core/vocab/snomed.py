"""
SNOMED CT International provider — INTERFACE/STUB ONLY.

SNOMED CT requires a license (affiliate or national). Until the appropriate
authorization is in place this provider is INACTIVE:

- available() is True ONLY when HLEO_SNOMED_API_URL + HLEO_SNOMED_API_KEY
  are configured (e.g. a licensed Snowstorm/FHIR terminology server);
- no SNOMED CT content is bundled, downloaded or scraped;
- until then every operation degrades to [] / None.

The resolver treats it exactly like any other provider, so enabling it later
is a configuration change, not a code change.
"""
from __future__ import annotations

import os
from typing import List, Optional

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch


class SNOMEDCTProvider(VocabularyProvider):
    name = "snomed_ct"
    semantic_group_hint = "general"

    def available(self) -> bool:
        return bool(
            os.getenv("HLEO_SNOMED_API_URL", "").strip()
            and os.getenv("HLEO_SNOMED_API_KEY", "").strip()
        )

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        # License pending — intentionally not implemented.
        return []
