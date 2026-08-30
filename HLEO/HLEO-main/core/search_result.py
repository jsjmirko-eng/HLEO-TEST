from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    title: str
    source: str

    url: Optional[str] = None

    abstract: Optional[str] = None

    authors: List[str] = field(default_factory=list)

    year: Optional[int] = None

    doi: Optional[str] = None

    pmid: Optional[str] = None

    keywords: List[str] = field(default_factory=list)

    score: float = 0.0

    metadata: dict = field(default_factory=dict)