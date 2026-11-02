"""Community detection with diagnostics and a stability check.

Louvain is the default (networkx implementation, seeded). Leiden is available when the optional
``python-igraph`` + ``leidenalg`` packages are installed. Communities are algorithmic groupings, not formal
scientific fields: always inspect topic distributions and example papers.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def _louvain(G: nx.Graph, resolution: float, seed: int) -> List[set]:
    """networkx Louvain. Parameters: weight (edge attr), resolution (>1 smaller comms), threshold, max_level, seed."""
    return [set(c) for c in nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)]


def _leiden(G: nx.Graph, resolution: float, seed: int) -> List[set]:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Leiden needs: pip install python-igraph leidenalg") from exc
    nodes = list(G.nodes())
    pos = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(pos[u], pos[v]) for u, v in G.edges()])
    g.es["weight"] = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed)
    return [{nodes[i] for i in members} for members in part]


def detect_communities(G: nx.Graph, algorithm: str = "louvain", resolution: float = 1.0, seed: int = 42,
                       min_size: int = 5) -> Tuple[Dict[str, int], Dict]:
    """Partition ``G``.

    Returns ``(membership, diagnostics)``. Community ids are 0..k-1 ordered by (-size, smallest node id), so ids
    are deterministic for a given partition. A graph without edges yields singleton communities.
    """
    if G.number_of_nodes() == 0:
        return {}, {"n_communities": 0, "modularity": 0.0, "n_ge_min_size": 0, "max_size": 0, "median_size": 0.0}
    if G.number_of_edges() == 0:
        comms = [{n} for n in G.nodes()]
    elif algorithm == "louvain":
        comms = _louvain(G, resolution, seed)
    elif algorithm == "leiden":
        comms = _leiden(G, resolution, seed)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    comms.sort(key=lambda c: (-len(c), min(c)))
    membership = {n: i for i, c in enumerate(comms) for n in c}
    sizes = np.array([len(c) for c in comms])
    mod = nx.community.modularity(G, comms, weight="weight", resolution=resolution) if G.number_of_edges() else 0.0
    return membership, {"n_communities": len(comms), "modularity": float(mod),
                        "n_ge_min_size": int((sizes >= min_size).sum()), "max_size": int(sizes.max()),
                        "median_size": float(np.median(sizes)), "n_singletons": int((sizes == 1).sum())}


def community_members(membership: Dict[str, int]) -> Dict[int, set]:
    """Invert ``node -> community`` into ``community -> set(nodes)``."""
    out: Dict[int, set] = {}
    for n, c in membership.items():
        out.setdefault(c, set()).add(n)
    return out


def community_stability(G: nx.Graph, seeds: Sequence[int] = (0, 1, 2, 3, 4), algorithm: str = "louvain",
                        resolution: float = 1.0) -> Dict[str, float]:
    """Mean/min pairwise NMI between partitions from different random seeds (1.0 = identical)."""
    nodes = sorted(G.nodes())
    labelings = []
    for s in seeds:
        mem, _ = detect_communities(G, algorithm, resolution, seed=s)
        labelings.append([mem[n] for n in nodes])
    scores = [normalized_mutual_info_score(a, b) for a, b in combinations(labelings, 2)]
    return {"nmi_mean": float(np.mean(scores)) if scores else 1.0, "nmi_min": float(np.min(scores)) if scores else 1.0,
            "n_seeds": len(seeds)}
