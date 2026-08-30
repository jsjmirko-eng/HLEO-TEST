"""External vocabulary provider layer for HLEO (feature-flagged)."""
from core.vocab.base import VocabularyProvider
from core.vocab.cache import VocabCache
from core.vocab.models import (
    MATCH_KINDS,
    MATCH_TIERS,
    VocabularyMatch,
    VocabularyResolution,
)
from core.vocab.resolver import (
    VocabularyResolver,
    build_resolver_from_env,
    vocab_enabled,
)

__all__ = [
    "VocabularyProvider",
    "VocabCache",
    "VocabularyMatch",
    "VocabularyResolution",
    "VocabularyResolver",
    "MATCH_KINDS",
    "MATCH_TIERS",
    "build_resolver_from_env",
    "vocab_enabled",
]
