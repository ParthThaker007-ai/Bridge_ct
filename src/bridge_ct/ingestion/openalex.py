"""OpenAlex works ingestion (cursor pagination).

OpenAlex API details (verify against https://docs.openalex.org/ before large runs; schemas can change):
* ``filter`` combines conditions with commas; ``title_and_abstract.search`` does text search.
* ``select`` limits returned fields (smaller, faster responses).
* ``per-page`` max is 200; ``cursor=*`` starts deep pagination, ``meta.next_cursor`` continues it.
* ``mailto`` (polite pool) and/or ``api_key`` identify you; read from env OPENALEX_MAILTO / OPENALEX_API_KEY.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

from ..utils import get_logger
from .http import CachedClient

log = get_logger()
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = ("id,doi,title,publication_date,publication_year,type,authorships,"
                 "abstract_inverted_index,referenced_works,topics,ids,cited_by_count")


def _clean_query(query: str) -> str:
    """Remove characters that have meaning inside an OpenAlex filter expression."""
    return re.sub(r"[,|:\"]", " ", query).strip()


def build_filter(query: str, start_year: int, end_year: int, work_types: Sequence[str]) -> str:
    """Compose the ``filter`` parameter (exact phrase search + date window + work types)."""
    parts = [f'title_and_abstract.search:"{_clean_query(query)}"',
             f"from_publication_date:{start_year}-01-01",
             f"to_publication_date:{end_year}-12-31"]
    if work_types:
        parts.append("type:" + "|".join(work_types))
    return ",".join(parts)


def fetch_works(client: CachedClient, query: str, start_year: int, end_year: int,
                max_records: Optional[int] = None, work_types: Sequence[str] = ("article", "review"),
                per_page: int = 200) -> List[Dict[str, Any]]:
    """Download all works matching ``query`` in the year window.

    Returns the raw OpenAlex work dictionaries (pagination handled). Raises ``requests.HTTPError`` after
    exhausting retries so a failed run is never mistaken for an empty result.
    """
    secret = {}
    if os.environ.get("OPENALEX_MAILTO"):
        secret["mailto"] = os.environ["OPENALEX_MAILTO"]
    if os.environ.get("OPENALEX_API_KEY"):
        secret["api_key"] = os.environ["OPENALEX_API_KEY"]
    flt = build_filter(query, start_year, end_year, work_types)
    results: List[Dict[str, Any]] = []
    cursor: Optional[str] = "*"
    while cursor:
        params = {"filter": flt, "per-page": per_page, "cursor": cursor, "select": SELECT_FIELDS}
        page = client.get_json(OPENALEX_WORKS_URL, params=params, secret_params=secret)
        batch = page.get("results", [])
        results.extend(batch)
        log.info("OpenAlex: %d works fetched", len(results))
        if max_records is not None and len(results) >= max_records:
            return results[:max_records]
        cursor = page.get("meta", {}).get("next_cursor")
        if not batch:
            break
    return results
