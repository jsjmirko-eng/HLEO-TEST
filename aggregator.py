"""
HLEO Article Aggregator
=======================
Responsible for collecting and deduplicating scientific articles across all
sources (PubMed, Europe PMC, ClinicalTrials.gov).

Key entry points
----------------
- HLEOAggregator.deduplicate_across_sources(sources)
    Accepts the raw per-source dict returned by HLEOPipeline.collect(),
    removes cross-source duplicates, and returns a cleaned dict + stats.
    Used by HLEOPipeline.collect() before any AI work.

- HLEOAggregator.search(query, limit)
    Legacy two-source search (PubMed + Europe PMC) used by SearchEngine.
    Preserved for backward compatibility.

Deduplication key priority
--------------------------
1. DOI  (canonical cross-database identifier)
2. PMID (PubMed-indexed papers)
3. NCT ID (ClinicalTrials registry number)
4. Europe PMC internal ID
5. Normalised title (last resort; not unique across journals)

When two records share a key, the one with the higher completeness score
is kept and the duplicate is discarded from the losing source list.
"""
import hashlib
import logging
import re
import unicodedata

from collectors.pubmed import PubMedCollector
from collectors.europepmc import EuropePMCCollector

logger = logging.getLogger(__name__)

# Source preference order used as a tiebreaker when completeness scores are equal.
# PubMed is preferred because it carries full PMID + abstract; ClinicalTrials last
# because its "abstract" is a protocol description, not a study result.
_SOURCE_PRIORITY = {"pubmed": 0, "europepmc": 1, "clinicaltrials": 2}


class HLEOAggregator:

    def __init__(self):
        self.pubmed = PubMedCollector()
        self.europepmc = EuropePMCCollector()

    # ── Deduplication key ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_doi(value: object) -> str:
        value = str(value or "").strip().lower()
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
        value = re.sub(r"^doi:\s*", "", value)
        return value.rstrip(". ")

    @staticmethod
    def _normalize_title(value: object) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    @classmethod
    def _document_fingerprint(cls, article) -> str | None:
        title = cls._normalize_title(getattr(article, "title", ""))
        abstract = " ".join((getattr(article, "abstract", "") or "").lower().split())
        year = str(getattr(article, "year", "") or "").strip()
        if not title or not abstract:
            return None
        payload = f"{title}|{year}|{abstract}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _identity_keys(cls, article) -> list[str]:
        """Return strong identifiers plus a conservative content identity."""
        meta = getattr(article, "metadata", {}) or {}
        keys: list[str] = []

        pmid = getattr(article, "pmid", None)
        if not pmid:
            epmc_id = str(meta.get("id", "")).strip()
            if epmc_id.isdigit():
                pmid = epmc_id
        if pmid:
            keys.append(f"pmid:{str(pmid).strip()}")

        doi = cls._normalize_doi(getattr(article, "doi", None))
        if doi:
            keys.append(f"doi:{doi}")

        nct = str(meta.get("nct_id", "") or "").strip()
        if nct and nct.lower() != "unknown":
            keys.append(f"nct:{nct.upper()}")

        epmc_id = str(meta.get("id", "")).strip()
        if epmc_id:
            keys.append(f"epmcid:{epmc_id}")

        fingerprint = cls._document_fingerprint(article)
        if fingerprint:
            keys.append(f"docfp:{fingerprint}")
        elif getattr(article, "title", None):
            keys.append(f"title:{cls._normalize_title(article.title)}")
        return keys

    @classmethod
    def create_key(cls, article) -> str | None:
        """Return the strongest canonical identity available for an article."""
        keys = cls._identity_keys(article)
        return keys[0] if keys else None

    # ── Completeness scoring ──────────────────────────────────────────────────

    @staticmethod
    def completeness_score(article) -> float:
        """
        Score a SearchResult by how much usable data it carries.
        Higher is better; used to pick the winner when two records are duplicates.

        Weights
        -------
        Abstract length    up to 5.0  (most important for AI reasoning)
        Authors present    2.0        (signals peer-reviewed publication)
        Author count       up to 1.0  (each author adds 0.2, capped at 5)
        Has DOI            1.0
        Has PMID           1.0
        Has journal name   0.5
        Has publication year 0.5
        """
        score = 0.0
        meta = getattr(article, "metadata", {}) or {}

        abstract = getattr(article, "abstract", None) or ""
        score += min(len(abstract) / 200.0, 5.0)

        authors = getattr(article, "authors", []) or []
        if authors:
            score += 2.0
            score += min(len(authors) * 0.2, 1.0)

        if getattr(article, "doi", None):
            score += 1.0
        if getattr(article, "pmid", None):
            score += 1.0
        if meta.get("journal"):
            score += 0.5
        if getattr(article, "year", None):
            score += 0.5

        return score

    # ── Cross-source deduplication ────────────────────────────────────────────

    def deduplicate_across_sources(
        self, sources: dict
    ) -> tuple[dict, dict]:
        """
        Deduplicate scientific articles across pubmed, europepmc, clinicaltrials.
        Reddit posts are never touched.

        Algorithm
        ---------
        1. Flatten all scientific articles into a single list, tagging each
           with its source name.
        2. For every article compute its dedup key.
        3. If the key has been seen before, compare completeness scores.
           - New score > existing score  → replace the winner, mark the old as dup.
           - New score <= existing score → discard the new one as dup.
           - Equal scores → source priority order (PubMed > EuropePMC > ClinicalTrials).
        4. Rebuild per-source lists from the survivors.

        Parameters
        ----------
        sources : dict
            Output of HLEOPipeline.collect():
            {"pubmed": [...], "europepmc": [...], "clinicaltrials": [...], "reddit": [...]}

        Returns
        -------
        (cleaned_sources, stats)
            cleaned_sources  — same structure as input with duplicates removed
            stats            — {"retrieved": int, "removed": int, "unique": int,
                                "duplicate_keys": [(key, loser_source, winner_source), ...]}
        """
        # derive scientific keys dynamically from the provided sources dict (exclude reddit)
        scientific_keys = [k for k in sources.keys() if k != 'reddit']

        # Tag each article with its source name so we can rebuild per-source lists
        tagged: list[tuple[str, object]] = []
        for src in scientific_keys:
            for art in sources.get(src, []):
                tagged.append((src, art))

        retrieved = len(tagged)

        def merge_provenance(winner, duplicate) -> None:
            winner.metadata = dict(getattr(winner, "metadata", {}) or {})
            duplicate_metadata = getattr(duplicate, "metadata", {}) or {}
            queries = winner.metadata.setdefault("matched_queries", [])
            for query in duplicate_metadata.get("matched_queries", []):
                if query not in queries:
                    queries.append(query)
            provenance = winner.metadata.setdefault("match_provenance", [])
            for entry in duplicate_metadata.get("match_provenance", []):
                if entry not in provenance:
                    provenance.append(entry)

        # canonical key → (source_name, article, score, identity_keys)
        winners: dict[str, tuple[str, object, float, set[str]]] = {}
        identity_index: dict[str, str] = {}
        duplicate_keys: list[tuple[str, str, str]] = []  # (key, loser_src, winner_src)
        keyless_survivors: list[tuple[str, object]] = []

        for src, art in tagged:
            identity_keys = self._identity_keys(art)
            if not identity_keys:
                keyless_survivors.append((src, art))
                continue

            strong_keys = [key for key in identity_keys
                           if not key.startswith("docfp:") and not key.startswith("title:")]
            canonical_key = next(
                (identity_index[key] for key in strong_keys if key in identity_index),
                None,
            )
            if canonical_key is None:
                fingerprint = next((key for key in identity_keys if key.startswith("docfp:")), None)
                candidate = identity_index.get(fingerprint) if fingerprint else None
                if candidate is not None:
                    existing_strong = winners[candidate][3]
                    new_dois = {key for key in strong_keys if key.startswith("doi:")}
                    old_dois = {key for key in existing_strong if key.startswith("doi:")}
                    if not (new_dois and old_dois and new_dois != old_dois):
                        canonical_key = candidate

            score = self.completeness_score(art)
            src_priority = _SOURCE_PRIORITY.get(src, 99)

            if canonical_key is None:
                canonical_key = strong_keys[0] if strong_keys else identity_keys[0]
                winners[canonical_key] = (src, art, score, set(identity_keys))
                for key in identity_keys:
                    identity_index[key] = canonical_key
                continue

            win_src, win_art, win_score, win_identity_keys = winners[canonical_key]
            win_priority = _SOURCE_PRIORITY.get(win_src, 99)

            # Replace if strictly better score, or equal score but higher-priority source.
            if score > win_score or (score == win_score and src_priority < win_priority):
                merge_provenance(art, win_art)
                duplicate_keys.append((canonical_key, win_src, src))
                merged_keys = win_identity_keys | set(identity_keys)
                winners[canonical_key] = (src, art, score, merged_keys)
                for key in merged_keys:
                    identity_index[key] = canonical_key
            else:
                merge_provenance(win_art, art)
                duplicate_keys.append((canonical_key, src, win_src))
                win_identity_keys.update(identity_keys)
                for key in win_identity_keys:
                    identity_index[key] = canonical_key

        # Rebuild per-source lists (winners + keyless survivors)
        result: dict[str, list] = {src: [] for src in scientific_keys}
        for _key, (src, art, _, _identity_keys) in winners.items():
            result[src].append(art)
        for src, art in keyless_survivors:
            result[src].append(art)

        # Preserve Reddit untouched
        result["reddit"] = sources.get("reddit", [])

        unique = sum(len(result[s]) for s in scientific_keys)
        removed = retrieved - unique

        stats = {
            "retrieved":     retrieved,
            "removed":       removed,
            "unique":        unique,
            "duplicate_keys": duplicate_keys,
        }

        return result, stats

    # ── Legacy two-source search (SearchEngine compatibility) ─────────────────

    def search(self, query: str, limit: int = 5):
        """
        Collect from PubMed + Europe PMC and return a deduplicated flat list.
        Used by SearchEngine; not called by the live API endpoints.
        """
        all_results = []

        try:
            all_results.extend(self.pubmed.search(query, limit))
        except Exception as e:
            logger.warning(f"PubMed error: {e}")

        try:
            all_results.extend(self.europepmc.search(query, limit))
        except Exception as e:
            logger.warning(f"Europe PMC error: {e}")

        seen: dict[str, bool] = {}
        final: list = []
        for article in all_results:
            key = self.create_key(article)
            if key not in seen:
                seen[key] = True
                final.append(article)

        return final
