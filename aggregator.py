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
import logging

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

    def create_key(self, article) -> str | None:
        """
        Return a canonical deduplication key for a SearchResult.

        Priority: PMID → DOI → NCT ID → EuropePMC non-numeric ID → normalised title.

        PMID is checked **before** DOI because it is the only identifier that is
        guaranteed to be identical across PubMed and Europe PMC for the same paper.
        Europe PMC returns the PMID of MEDLINE-indexed papers in its ``id`` field as
        a plain integer string, so we promote any all-digit ``metadata["id"]`` to a
        PMID key.  This catches the common case where PubMed returns a paper without
        a DOI while Europe PMC returns the same paper with a DOI — without this rule
        the two keys would never match and the duplicate would slip through.
        """
        meta = getattr(article, "metadata", {}) or {}

        # 1. PMID — stable across PubMed and Europe PMC (MEDLINE-indexed papers)
        pmid = getattr(article, "pmid", None)
        if not pmid:
            # Europe PMC stores the PMID in metadata["id"] as a numeric string
            epmc_id = str(meta.get("id", "")).strip()
            if epmc_id.isdigit():
                pmid = epmc_id
        if pmid:
            return f"pmid:{str(pmid).strip()}"

        # 2. DOI — reliable for non-MEDLINE papers that have one
        if getattr(article, "doi", None):
            return f"doi:{article.doi.strip().lower()}"

        # 3. NCT ID — canonical identifier for ClinicalTrials entries
        nct = meta.get("nct_id", "")
        if nct and nct.lower() not in ("", "unknown"):
            return f"nct:{nct.strip().upper()}"

        # 4. Europe PMC non-numeric internal ID (preprint servers, etc.)
        epmc_id = str(meta.get("id", "")).strip()
        if epmc_id:
            return f"epmcid:{epmc_id}"

        # 5. Normalised title (last resort — not unique across journals)
        if getattr(article, "title", None):
            return "title:" + " ".join(article.title.lower().split())

        return None

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

        # key → (source_name, article, score)
        winners: dict[str, tuple[str, object, float]] = {}
        duplicate_keys: list[tuple[str, str, str]] = []  # (key, loser_src, winner_src)
        keyless_survivors: list[tuple[str, object]] = []

        for src, art in tagged:
            key = self.create_key(art)

            if key is None:
                # No key at all — keep unconditionally, cannot dedup
                keyless_survivors.append((src, art))
                continue

            score = self.completeness_score(art)
            src_priority = _SOURCE_PRIORITY.get(src, 99)

            if key not in winners:
                winners[key] = (src, art, score)
            else:
                win_src, win_art, win_score = winners[key]
                win_priority = _SOURCE_PRIORITY.get(win_src, 99)

                # Replace if strictly better score, or equal score but higher-priority source
                if score > win_score or (
                    score == win_score and src_priority < win_priority
                ):
                    # Current article is better — demote the old winner.
                    merge_provenance(art, win_art)
                    duplicate_keys.append((key, win_src, src))
                    winners[key] = (src, art, score)
                else:
                    # Existing winner is better — discard current.
                    merge_provenance(win_art, art)
                    duplicate_keys.append((key, src, win_src))

        # Rebuild per-source lists (winners + keyless survivors)
        result: dict[str, list] = {src: [] for src in scientific_keys}
        for key, (src, art, _) in winners.items():
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
