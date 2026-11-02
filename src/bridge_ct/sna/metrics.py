"""Descriptive social-network-analysis measures used by the reports and as model features."""
from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd


def graph_summary(G: nx.Graph, clustering_sample: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Size, density, fragmentation and clustering of a snapshot.

    Average clustering is computed on a seeded node sample when the graph is large (exact is O(sum deg^2)).
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "components": 0, "giant_component_size": 0,
                "giant_component_frac": 0.0, "avg_clustering": 0.0, "isolated_frac": 0.0}
    comps = list(nx.connected_components(G))
    giant = max(len(c) for c in comps)
    nodes = list(G.nodes())
    if n > clustering_sample:
        rng = np.random.default_rng(seed)
        nodes = [nodes[i] for i in rng.choice(n, clustering_sample, replace=False)]
    return {"nodes": n, "edges": G.number_of_edges(), "density": nx.density(G), "components": len(comps),
            "giant_component_size": giant, "giant_component_frac": giant / n,
            "avg_clustering": float(nx.average_clustering(G, nodes=nodes)),
            "isolated_frac": sum(1 for _ in nx.isolates(G)) / n}


def node_table(G: nx.Graph, membership: Dict[str, int], betweenness_k: Optional[int] = 200,
               seed: int = 42) -> pd.DataFrame:
    """Per-node measures indexed by author id.

    Columns
    -------
    degree        number of distinct collaborators
    wdeg          weighted degree (collaboration strength)
    betweenness   (sampled) shortest-path betweenness, UNweighted: edge weights are strengths, not distances.
                  ``betweenness_k`` pivots are sampled (``None`` = exact, O(nm), ``0`` = skip); sampling is seeded.
    cross_ratio   share of a node's weighted ties that cross community borders (brokerage proxy, 0..1)
    """
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame(columns=["degree", "wdeg", "betweenness", "cross_ratio"])
    if betweenness_k == 0:  # explicit opt-out (used for the cheap previous-year snapshot)
        btw = {n: 0.0 for n in nodes}
    else:
        k = None if (betweenness_k is None or betweenness_k >= len(nodes)) else int(betweenness_k)
        btw = nx.betweenness_centrality(G, k=k, seed=seed, normalized=True)
    degree = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    cross = {}
    for n in nodes:
        tot = out = 0.0
        for nb, d in G[n].items():
            w = d.get("weight", 1.0)
            tot += w
            if membership.get(nb) != membership.get(n):
                out += w
        cross[n] = out / tot if tot > 0 else 0.0
    return pd.DataFrame({"degree": degree, "wdeg": wdeg, "betweenness": btw, "cross_ratio": cross}).loc[nodes]


def random_walk_with_restart(G: nx.Graph, seeds: list, restart: float = 0.15) -> pd.Series:
    """Personalised PageRank == random walk with restart to ``seeds``.

    ``restart`` is the teleport probability (networkx ``alpha = 1 - restart``). Higher restart keeps scores
    local to the seeds (e.g. authors of disease-specific trial papers).
    """
    present = [s for s in seeds if s in G]
    if not present:
        raise ValueError("none of the seed nodes are in the graph")
    personalization = {n: (1.0 / len(present) if n in present else 0.0) for n in G}
    pr = nx.pagerank(G, alpha=1.0 - restart, personalization=personalization, weight="weight")
    return pd.Series(pr).sort_values(ascending=False)


def degree_distribution(G: nx.Graph) -> pd.DataFrame:
    """Degree histogram (for the SNA report: heavy tails are expected)."""
    counts = pd.Series(dict(G.degree())).value_counts().sort_index()
    return counts.rename_axis("degree").reset_index(name="n_nodes")
