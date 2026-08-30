"""
hleo_v1/core/ranker.py
======================
Deterministic clinical re-ranking for SearchResult objects.

Called by HLEOPipeline.collect() immediately after cross-source deduplication.
No LLM calls — scoring uses only metadata already present at collection time.

Pipeline position
-----------------
    retrieve → dedup → clinical_rerank → SearchArticleCtx → Assistant

Scoring components
------------------
    study_type_score   0–100   evidence design detected by regex in title + abstract
                               (or ClinicalTrials phase from metadata)
    recency            0–15    +1.5 per year within last 10 years (linear decay)
    abstract_present   +20     any non-empty abstract
    abstract_richness  0–10    +2 per 200 chars of abstract, capped at 10
    author_count       0–5     +0.5 per author, capped at 10 authors
    has_doi            +5
    has_pmid           +5
    ─────────────────────────────────────────────────────
    theoretical max    ≈ 160   (meta-analysis, 2026, long abstract, many authors, both IDs)

Study type priority (first match wins)
---------------------------------------
    Meta-analysis        100
    Systematic review     90
    RCT / Phase III       80
    Phase II              65
    Clinical trial / Ph1  60
    Cohort                50
    Case-control          40
    Cross-sectional       35
    Case series           25
    Case report           15
    Unknown               10
"""
import logging
import re

logger = logging.getLogger(__name__)

# ── Study-type regex table (ordered: first match wins) ────────────────────────
_STUDY_TYPE_RULES: list[tuple[int, re.Pattern]] = [
    (100, re.compile(r"\bmeta[\s\-]?analy",                          re.I)),
    (90,  re.compile(r"\bsystematic[\s\-]?review",                   re.I)),
    (80,  re.compile(r"\brandomis|\brandomiz|\brct\b",               re.I)),
    (65,  re.compile(r"\bphase\s*ii\b|\bphase\s*2\b",               re.I)),
    (60,  re.compile(r"\bclinical[\s\-]?trial|\bphase\s*i\b|\bphase\s*1\b", re.I)),
    (50,  re.compile(r"\bcohort\b",                                  re.I)),
    (40,  re.compile(r"\bcase[\s\-]?control\b",                     re.I)),
    (35,  re.compile(r"\bcross[\s\-]?sectional\b",                  re.I)),
    (25,  re.compile(r"\bcase[\s\-]?series\b",                      re.I)),
    (15,  re.compile(r"\bcase[\s\-]?report\b",                      re.I)),
]
_DEFAULT_STUDY_SCORE = 10   # no recognised design

# Freeze to the current year; update annually or use datetime.now().year
_CURRENT_YEAR = 2026


def _detect_study_type(text: str) -> int:
    """Return the evidence-design score for the first matching pattern in text."""
    for pts, pattern in _STUDY_TYPE_RULES:
        if pattern.search(text):
            return pts
    return _DEFAULT_STUDY_SCORE


def clinical_rank(article) -> float:
    """
    Return a deterministic clinical relevance score for a single SearchResult.

    Parameters
    ----------
    article : SearchResult   (or any object with the same attributes)

    Returns
    -------
    float — raw score (higher = more clinically relevant)
    """
    score = 0.0
    meta  = getattr(article, "metadata", {}) or {}

    # ── Study type ───────────────────────────────────────────────────────────
    # For ClinicalTrials entries the trial phase in metadata is more reliable
    # than a text scan of the protocol description.
    phase_raw = str(meta.get("phase", "")).upper().replace(" ", "").replace("-", "")
    if "PHASE4" in phase_raw or "PHASE3" in phase_raw:
        score += 80.0
    elif "PHASE2" in phase_raw:
        score += 65.0
    elif "PHASE1" in phase_raw:
        score += 60.0
    else:
        text = (
            (getattr(article, "title",    None) or "") + " " +
            (getattr(article, "abstract", None) or "")
        )
        score += _detect_study_type(text)

    # ── Recency ──────────────────────────────────────────────────────────────
    year = getattr(article, "year", None)
    if not year:
        # PubMed stores year inside metadata.pubdate: "YYYY Mon DD" or "YYYY"
        pubdate = str(meta.get("pubdate", "")).strip()
        if len(pubdate) >= 4 and pubdate[:4].isdigit():
            year = int(pubdate[:4])
    if year:
        age   = max(0, _CURRENT_YEAR - int(year))
        score += max(0.0, (10 - age) * 1.5)     # 0-10 years → +15 to 0

    # ── Abstract ─────────────────────────────────────────────────────────────
    abstract = getattr(article, "abstract", None) or ""
    if abstract:
        score += 20.0
        score += min(len(abstract) / 200.0, 5.0) * 2.0   # up to +10

    # ── Authors ──────────────────────────────────────────────────────────────
    authors = getattr(article, "authors", []) or []
    score += min(len(authors), 10) * 0.5                  # up to +5

    # ── Identifiers ──────────────────────────────────────────────────────────
    if getattr(article, "doi",  None):
        score += 5.0
    if getattr(article, "pmid", None):
        score += 5.0

    return round(score, 2)


def rank_articles(sources: dict) -> None:
    """
    Score every scientific article in the pipeline result dict and sort
    each per-source list by clinical relevance (descending).

    - Mutates ``article.score`` in place.
    - Sorts ``sources["pubmed"]``, ``sources["europepmc"]``,
      ``sources["clinicaltrials"]`` independently in descending score order.
    - ``sources["reddit"]`` is never touched.

    Called in HLEOPipeline.collect() after dedup, before the dict is
    returned and consumed by /search or /pipeline/run.
    """
    keys  = ("pubmed", "europepmc", "clinicaltrials")
    total = 0

    for key in keys:
        lst = sources.get(key, [])
        for art in lst:
            art.score = clinical_rank(art)
        lst.sort(key=lambda a: a.score, reverse=True)
        total += len(lst)

    logger.info("Rerank | scored and sorted %d articles (3 sources)", total)
