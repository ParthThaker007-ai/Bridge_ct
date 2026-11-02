"""Community evolution across yearly snapshots (birth, continue, split, merge, death).

Community ids from Louvain are arbitrary per year, so we align them by member overlap (Jaccard) and give
every aligned chain a persistent ``lineage_id``. Splits and merges are recorded as events - identifiers are
never assumed to be permanent.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd


def align_step(prev: Dict[int, set], cur: Dict[int, set], prev_lineage: Dict[int, int], next_lineage: int,
               min_jaccard: float = 0.3, split_frac: float = 0.25, merge_frac: float = 0.5
               ) -> Tuple[pd.DataFrame, List[dict], int]:
    """Align ``cur`` communities to ``prev``.

    * One-to-one lineage inheritance by greedy highest Jaccard (>= ``min_jaccard``).
    * ``merge``: >=2 previous communities each sent >= ``merge_frac`` of their members into one current community.
    * ``split``: a previous community sent >= ``split_frac`` of its members to each of >=2 current communities.
    * ``death``: previous community with no current community receiving >= ``split_frac`` of it.

    Returns (rows for current communities, death events, next free lineage id).
    """
    prev_of = {n: p for p, members in prev.items() for n in members}
    overlap: Counter = Counter()
    for c, members in cur.items():
        for n in members:
            p = prev_of.get(n)
            if p is not None:
                overlap[(p, c)] += 1

    def jac(p: int, c: int) -> float:
        inter = overlap[(p, c)]
        return inter / (len(prev[p]) + len(cur[c]) - inter)

    cands = sorted(((jac(p, c), p, c) for (p, c) in overlap), key=lambda t: (-t[0], t[1], t[2]))
    parent: Dict[int, Tuple[int, float]] = {}
    used_prev = set()
    for j, p, c in cands:
        if j < min_jaccard:
            break
        if p in used_prev or c in parent:
            continue
        parent[c] = (p, j)
        used_prev.add(p)

    merged_into: Dict[int, int] = {}
    split_children: Dict[int, List[int]] = {}
    for (p, c), inter in overlap.items():
        frac_prev = inter / len(prev[p])
        if frac_prev >= merge_frac:
            merged_into[c] = merged_into.get(c, 0) + 1
        if frac_prev >= split_frac:
            split_children.setdefault(p, []).append(c)
    split_cur = {c for p, kids in split_children.items() if len(kids) >= 2 for c in kids}

    rows, nl = [], next_lineage
    for c in sorted(cur):
        if c in parent:
            lineage, pj, pl = prev_lineage[parent[c][0]], parent[c][1], prev_lineage[parent[c][0]]
        else:
            lineage, pj, pl = nl, 0.0, -1
            nl += 1
        if merged_into.get(c, 0) >= 2:
            event = "merge"
        elif c in split_cur:
            event = "split"
        elif c in parent:
            event = "continue"
        else:
            event = "birth"
        rows.append({"community_id": c, "lineage_id": lineage, "event": event, "parent_jaccard": pj,
                     "parent_lineage": pl, "size": len(cur[c])})
    deaths = [{"lineage_id": prev_lineage[p], "prev_community_id": p, "size": len(prev[p])}
              for p in sorted(prev) if not split_children.get(p)]
    return pd.DataFrame(rows), deaths, nl


def track_evolution(members_by_year: Dict[int, Dict[int, set]], min_size: int = 5, **kwargs
                    ) -> Tuple[pd.DataFrame, Dict[int, Dict[int, int]]]:
    """Track communities (size >= ``min_size``) through all years.

    Returns ``(evolution, lineage_by_year)``. ``evolution`` columns: year, community_id, lineage_id, event,
    size, age (years since lineage birth, 1 = born this year), growth (size / parent size - 1),
    turnover (1 - Jaccard with parent; 1.0 for births), n_deaths_prev (communities that vanished this year).
    """
    years = sorted(members_by_year)
    out: List[pd.DataFrame] = []
    lineage_by_year: Dict[int, Dict[int, int]] = {}
    birth_year: Dict[int, int] = {}
    last_size: Dict[int, int] = {}
    prev: Dict[int, set] = {}
    prev_lineage: Dict[int, int] = {}
    next_lineage = 0
    for y in years:
        cur = {c: m for c, m in members_by_year[y].items() if len(m) >= min_size}
        if not cur:
            lineage_by_year[y] = {}
            prev, prev_lineage = {}, {}
            continue
        if not prev:
            rows = pd.DataFrame([{"community_id": c, "lineage_id": next_lineage + i, "event": "birth",
                                  "parent_jaccard": 0.0, "parent_lineage": -1, "size": len(cur[c])}
                                 for i, c in enumerate(sorted(cur))])
            next_lineage += len(cur)
            deaths: List[dict] = []
        else:
            rows, deaths, next_lineage = align_step(prev, cur, prev_lineage, next_lineage, **kwargs)
        ages, growth, turnover = [], [], []
        for r in rows.itertuples(index=False):
            if r.lineage_id not in birth_year:
                birth_year[r.lineage_id] = y
            ages.append(y - birth_year[r.lineage_id] + 1)
            prev_size = last_size.get(r.lineage_id)
            growth.append((r.size / prev_size - 1.0) if prev_size else 0.0)
            turnover.append(1.0 - r.parent_jaccard if r.parent_lineage >= 0 else 1.0)
            last_size[r.lineage_id] = r.size
        rows = rows.assign(year=y, age=ages, growth=growth, turnover=turnover, n_deaths_prev=len(deaths))
        out.append(rows)
        lineage_by_year[y] = dict(zip(rows["community_id"], rows["lineage_id"]))
        prev, prev_lineage = cur, lineage_by_year[y]
    cols = ["year", "community_id", "lineage_id", "event", "size", "age", "growth", "turnover",
            "parent_jaccard", "n_deaths_prev"]
    evo = pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)
    return evo, lineage_by_year
