"""Collaboration graph construction (the temporal backbone of BRIDGE-CT).

Strategy: build ONE long table of co-authorship pair contributions (u, v, year, work, weight). Any snapshot
G_t is then just ``pair_table[year <= t]`` aggregated - fast, and leakage-proof by construction because the
cutoff filter is the only place future data could enter.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

import networkx as nx
import pandas as pd

PAIR_COLS = ["u", "v", "year", "work_id", "weight"]


def build_pair_table(authorships: pd.DataFrame, max_authors: int = 50,
                     weight_mode: str = "fractional") -> Tuple[pd.DataFrame, int]:
    """Expand each work into author pairs.

    Parameters
    ----------
    authorships : needs ``work_id, author_id, publication_year``.
    max_authors : works with more authors are skipped (n*(n-1)/2 blow-up; consortium papers are weak ties).
    weight_mode : ``fractional`` -> each pair gets 1/(n-1) per paper (blueprint rule); ``binary`` -> 1.0.

    Returns ``(pair_table, n_skipped_works)``. ``u < v`` always (undirected canonical order).
    """
    required = {"work_id", "author_id", "publication_year"}
    missing = required - set(authorships.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if weight_mode not in ("fractional", "binary"):
        raise ValueError("weight_mode must be 'fractional' or 'binary'")
    df = (authorships.dropna(subset=["work_id", "author_id", "publication_year"])
          [["work_id", "author_id", "publication_year"]].drop_duplicates(["work_id", "author_id"]))
    us, vs, ys, ws, wk = [], [], [], [], []
    skipped = 0
    for work_id, grp in df.groupby("work_id", sort=True):
        authors = sorted(set(grp["author_id"].astype(str)))
        n = len(authors)
        if n < 2:
            continue
        if n > max_authors:
            skipped += 1
            continue
        year = int(grp["publication_year"].min())
        contribution = 1.0 / (n - 1) if weight_mode == "fractional" else 1.0
        for u, v in combinations(authors, 2):
            us.append(u); vs.append(v); ys.append(year); ws.append(contribution); wk.append(work_id)
    table = pd.DataFrame({"u": pd.Series(us, dtype="object"), "v": pd.Series(vs, dtype="object"),
                          "year": pd.Series(ys, dtype="int64"), "work_id": pd.Series(wk, dtype="object"),
                          "weight": pd.Series(ws, dtype="float64")})
    return table, skipped


def snapshot_edges(pair_table: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Aggregate pair contributions up to and including ``cutoff``.

    Columns: u, v, weight (sum), first_year, last_year, n_papers.
    """
    sub = pair_table[pair_table["year"] <= cutoff]
    if sub.empty:
        return pd.DataFrame(columns=["u", "v", "weight", "first_year", "last_year", "n_papers"])
    return (sub.groupby(["u", "v"], sort=True)
            .agg(weight=("weight", "sum"), first_year=("year", "min"), last_year=("year", "max"),
                 n_papers=("work_id", "size")).reset_index())


def snapshot_nodes(authorships: pd.DataFrame, cutoff: int) -> list:
    """Authors with at least one eligible paper (published <= cutoff), sorted for determinism."""
    sub = authorships[authorships["publication_year"] <= cutoff]
    return sorted(sub["author_id"].dropna().astype(str).unique())


def graph_from_edges(nodes: list, edges: pd.DataFrame, cutoff: int) -> nx.Graph:
    """Build a networkx graph with deterministic node/edge insertion order (stable community detection)."""
    G = nx.Graph()
    G.add_nodes_from(sorted(nodes))
    G.add_edges_from((r.u, r.v, {"weight": float(r.weight), "first_year": int(r.first_year),
                                  "last_year": int(r.last_year), "n_papers": int(r.n_papers)})
                     for r in edges.itertuples(index=False))
    G.graph["cutoff_year"] = cutoff
    return G


def build_collaboration_graph(authorships: pd.DataFrame, cutoff_year: int, pair_table: Optional[pd.DataFrame] = None,
                              max_authors: int = 50, weight_mode: str = "fractional") -> nx.Graph:
    """Convenience wrapper: ``G_t`` from raw authorships (builds the pair table if not supplied)."""
    if pair_table is None:
        pair_table, _ = build_pair_table(authorships, max_authors, weight_mode)
    return graph_from_edges(snapshot_nodes(authorships, cutoff_year), snapshot_edges(pair_table, cutoff_year), cutoff_year)


def author_activity(authorships: pd.DataFrame, cutoff: int, window: int) -> pd.DataFrame:
    """Per-author activity up to ``cutoff``: total papers, papers in the last ``window`` years, last paper year."""
    sub = authorships[authorships["publication_year"] <= cutoff].drop_duplicates(["work_id", "author_id"])
    g = sub.groupby("author_id")
    out = pd.DataFrame({
        "papers_total": g["work_id"].nunique(),
        "last_year": g["publication_year"].max(),
        "papers_recent": sub[sub["publication_year"] > cutoff - window].groupby("author_id")["work_id"].nunique(),
    })
    out["papers_recent"] = out["papers_recent"].fillna(0).astype(int)
    out["years_since_last"] = cutoff - out["last_year"]
    return out
