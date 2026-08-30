import logging

from aggregator import HLEOAggregator
from collectors.reddit import RedditCollector
from collectors.pubmed import PubMedCollector
from collectors.europepmc import EuropePMCCollector
from collectors.clinicaltrials import ClinicalTrialsCollector
from core.extractor import LLMExtractor
from core.validator import HLEOValidator
from core.judge import HLEOJudge
from search.source_fetcher import SourceFetcher

logger = logging.getLogger(__name__)

_aggregator = HLEOAggregator()


class HLEOPipeline:

    def __init__(self):
        self.collector = RedditCollector()
        self.pubmed = PubMedCollector()
        self.europepmc = EuropePMCCollector()
        self.clinicaltrials = ClinicalTrialsCollector()
        self.extractor = LLMExtractor()
        self.validator = HLEOValidator()
        self.judge = HLEOJudge()
        self.fetcher = SourceFetcher()

    def collect(self, query: str) -> dict:
        """
        Collect raw data from all sources and deduplicate across sources.

        Flow
        ----
        1. Fetch from PubMed, Europe PMC, ClinicalTrials.gov, Reddit in parallel.
        2. Run HLEOAggregator.deduplicate_across_sources() over the three
           scientific lists (Reddit is never touched).
        3. Log Retrieved / Duplicates removed / Final unique papers.
        4. Return the cleaned per-source dict — identical structure to before,
           but with cross-source duplicates removed (winning copy only retained).
        """
        # Dynamic discovery of scientific sources via SourceRegistry (if present),
        # otherwise fall back to the legacy builtin collectors.
        from core.database import SessionLocal
        from core.models import SourceRegistry
        from sqlalchemy import select

        registry_map: Dict[str, SourceRegistry] = {}
        raw: dict = {}

        # Collector map for known built-in collectors
        collector_map = {
            "pubmed": self.pubmed,
            "europepmc": self.europepmc,
            "clinicaltrials": self.clinicaltrials,
        }

        # Discover active scientific sources from registry
        with SessionLocal() as db:
            rows = db.execute(
                select(SourceRegistry).where(
                    SourceRegistry.category == "scientific",
                    SourceRegistry.status == "active",
                )
            ).scalars().all()
        if rows:
            for r in rows:
                registry_map[r.source_id] = r
                collector_key = r.runtime_collector or r.source_id
                # If runtime_collector maps to a known collector, call it; otherwise
                # if it's a generic REST configuration, instantiate GenericRESTCollector
                if collector_key in collector_map:
                    try:
                        raw[r.source_id] = collector_map[collector_key].search(query, limit=None)
                    except Exception:
                        logger.exception("Collector %s failed for query %s", collector_key, query)
                        raw[r.source_id] = []
                elif collector_key == "generic_rest":
                    try:
                        from collectors.generic_rest import GenericRESTCollector
                        gen = GenericRESTCollector(r.connection_spec or {}, source_id=r.source_id, category=r.category)
                        raw[r.source_id] = gen.search(query, limit=None)
                    except Exception:
                        logger.exception("GenericRESTCollector failed for %s", r.source_id)
                        raw[r.source_id] = []
                else:
                    # Unknown runtime_collector — skip
                    logger.warning("Unknown runtime_collector '%s' for source '%s'", collector_key, r.source_id)
                    raw[r.source_id] = []
        else:
            # legacy behaviour — one source failure must not fail the search
            raw = {}
            for key, collector in (
                ("reddit", self.collector),
                ("pubmed", self.pubmed),
                ("europepmc", self.europepmc),
                ("clinicaltrials", self.clinicaltrials),
            ):
                try:
                    raw[key] = collector.search(query, limit=None)
                except Exception:
                    logger.exception("Collector %s failed for query %s", key, query)
                    raw[key] = []

        # include reddit if not already present
        if "reddit" not in raw:
            try:
                raw["reddit"] = self.collector.search(query, limit=None)
            except Exception:
                logger.exception("Collector reddit failed for query %s", query)
                raw["reddit"] = []

        # ── Cross-source deduplication (works over dynamic set of scientific keys) ──
        deduped, stats = _aggregator.deduplicate_across_sources(raw)

        logger.info(
            "Dedup | Retrieved: %d | Duplicates removed: %d | Final unique papers: %d",
            stats["retrieved"],
            stats["removed"],
            stats["unique"],
        )

        if stats["removed"] > 0:
            for key, loser_src, winner_src in stats["duplicate_keys"]:
                logger.info(
                    "  dup removed: [%s] kept from %s, dropped from %s",
                    key, winner_src, loser_src,
                )

        # ── Clinical re-ranking ───────────────────────────────────────────────
        from core.ranker import rank_articles
        rank_articles(deduped)

        return deduped

    def process(self, query: str) -> list:
        """Full pipeline: collect → LLM extract → validate → judge."""
        logger.info("Pipeline avviata")

        data = self.collect(query)

        posts = data.get("reddit", [])

        logger.info(f"Post Reddit: {len(posts)}")

        # Build unified results list from all non-reddit keys in the collected data.
        results = []
        has_scientific = False
        for k, items in data.items():
            if k == "reddit":
                continue
            if not items:
                continue
            has_scientific = True
            if k == "clinicaltrials":
                for trial in items:
                    results.append({"type": "clinicaltrials", "trial": trial})
            else:
                # treat everything else as an article list
                for article in items:
                    results.append({"type": k, "article": article})

        if not posts and not has_scientific:
            logger.warning("Nessun dato trovato")
            return []

        # Process reddit posts (extraction) as before
        for post in posts:
            try:
                raw_sources = self.fetcher.fetch(post.url)
                profile = self.extractor.extract(post.text)
                logger.info("Estrazione completata")

                validation = self.validator.validate(
                    profile,
                    raw_sources,
                    post.created_at,
                )
                logger.info("Validazione completata")

                judge_result = self.judge.evaluate(
                    profile.baseline_status.value,
                    profile.post_treatment_status.value,
                    validation.passed_validation,
                    profile.post_treatment_status.support_strength,
                    profile.conflict_detected,
                    profile.episode_id,
                )
                logger.info("Giudizio completato")

                results.append({
                    "type": "reddit",
                    "post": post,
                    "profile": profile,
                    "validation": validation,
                    "judge": judge_result,
                })

            except Exception as e:
                logger.exception(f"Errore elaborazione post {post.url}: {e}")

        return results
