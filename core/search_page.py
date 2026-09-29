"""Provider-agnostic page contract for scientific collectors."""

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class SearchPage(Generic[T]):
    """One normalized result page returned by a scientific adapter.

    ``next_cursor`` is intentionally opaque to the caller. Each provider
    adapter owns its cursor shape and interprets it on the next call.
    """

    items: list[T]
    next_cursor: Optional[Any] = None
    has_more: bool = False
    total_available: Optional[int] = None


def collect_search_pages(collector, query: str, limit: Optional[int] = None) -> list:
    """Collect adapter pages without knowing provider pagination details."""

    if not hasattr(collector, "search_page"):
        return collector.search(query, limit=limit)

    items = []
    cursor = None
    while True:
        page = collector.search_page(query, cursor=cursor, limit=limit)
        items.extend(page.items)
        if not page.has_more:
            break
        cursor = page.next_cursor

    return items[:limit] if limit is not None else items
