"""Counterfactual graph simulation (structural what-if, NOT a causal or clinical claim).

Question: if we add K hypothetical collaboration edges chosen by (a) the model, (b) random cross-community
pairs, (c) degree product, (d) summed betweenness, how much do giant-component size, cross-community
connectivity and simulated diffusion coverage change? Diffusion is an independent-cascade process with a
fixed activation probability (an explicit assumption).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class _DSU:
    """Disjoint-set union with component sizes (used to update the giant component after adding edges)."""

    def __init__(self, labels: np.ndarray) -> None:
        self.parent = {int(c): int(c) for c in np.unique(labels)}
        self.size = {int(c): int((labels == c).sum()) for c in self.parent}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if self.size[ra] < self.size[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            self.size[ra] += self.size[rb]


class GraphSimulator:
    """Holds an integer-indexed copy of one snapshot so many what-if runs are cheap.

    Parameters: ``G`` snapshot graph, ``membership`` node -> community, ``max_nodes`` cap (the giant component's
    highest-degree ``max_nodes`` nodes are kept when larger), ``seed``.
    """

    def __init__(self, G: nx.Graph, membership: Dict[str, int], max_nodes: int = 20000, seed: int = 42) -> None:
        comp = max(nx.connected_components(G), key=len) if G.number_of_nodes() else set()
        sub = G.subgraph(comp)
        if sub.number_of_nodes() > max_nodes:
            keep = sorted(sub.nodes(), key=lambda n: (-sub.degree(n), n))[:max_nodes]
            sub = G.subgraph(keep)
        self.nodes = sorted(sub.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.membership = np.array([membership.get(n, -1) for n in self.nodes])
        self.adj: List[np.ndarray] = [np.array(sorted(self.index[m] for m in sub[n]), dtype=np.int64) for n in self.nodes]
        self.edges = {(min(self.index[u], self.index[v]), max(self.index[u], self.index[v])) for u, v in sub.edges()}
        self.G = sub
        self.rng_seed = seed
        labels = np.zeros(len(self.nodes), dtype=int)
        for k, c in enumerate(nx.connected_components(sub)):
            for n in c:
                labels[self.index[n]] = k
        self.comp_labels = labels

    # ------------------------------------------------------------------ structural metrics
    def structural(self, added: Sequence[Tuple[int, int]]) -> Dict[str, float]:
        """Giant-component fraction, #components, connected community pairs and cross-community edge share."""
        dsu = _DSU(self.comp_labels)
        for a, b in added:
            dsu.union(int(self.comp_labels[a]), int(self.comp_labels[b]))
        roots: Dict[int, int] = {}
        for c, s in dsu.size.items():
            r = dsu.find(c)
            if r == c:
                roots[r] = s
        n = len(self.nodes)
        pairs = {(min(self.membership[a], self.membership[b]), max(self.membership[a], self.membership[b]))
                 for a, b in list(self.edges) + list(added) if self.membership[a] != self.membership[b]}
        all_edges = list(self.edges) + [e for e in added if e not in self.edges]
        cross = sum(1 for a, b in all_edges if self.membership[a] != self.membership[b])
        return {"giant_frac": max(roots.values()) / n, "n_components": len(roots),
                "connected_community_pairs": len(pairs), "cross_edge_share": cross / max(1, len(all_edges))}

    # ------------------------------------------------------------------ diffusion
    def coverage(self, added: Sequence[Tuple[int, int]], seeds: np.ndarray, p: float, n_cascades: int,
                 rng: np.random.Generator) -> float:
        """Mean fraction of nodes reached by an independent cascade started from ``seeds``."""
        adj = list(self.adj)
        for a, b in added:
            adj[a] = np.append(adj[a], b)
            adj[b] = np.append(adj[b], a)
        n, total = len(adj), 0.0
        for _ in range(n_cascades):
            active = np.zeros(n, dtype=bool)
            active[seeds] = True
            frontier = list(seeds)
            while frontier:
                nxt: List[int] = []
                for u in frontier:
                    nb = adj[u]
                    if len(nb) == 0:
                        continue
                    hit = nb[(rng.random(len(nb)) < p) & ~active[nb]]
                    if len(hit):
                        active[hit] = True
                        nxt.extend(hit.tolist())
                frontier = nxt
            total += active.mean()
        return total / n_cascades

    # ------------------------------------------------------------------ strategies
    def pool(self, candidate_pairs: Sequence[Tuple[str, str]]) -> List[Tuple[int, int]]:
        """Map candidate author pairs onto the simulated subgraph (drop pairs outside it or already linked)."""
        out = []
        for u, v in candidate_pairs:
            if u in self.index and v in self.index:
                a, b = sorted((self.index[u], self.index[v]))
                if (a, b) not in self.edges and self.membership[a] != self.membership[b]:
                    out.append((a, b))
        return out

    def run(self, model_pairs: Sequence[Tuple[str, str]], pool_pairs: Sequence[Tuple[str, str]], k: int,
            n_replicates: int, p: float, n_seed_nodes: int, n_cascades: int) -> pd.DataFrame:
        """Compare strategies over ``n_replicates`` random seed sets / random edge draws.

        Strategies: ``baseline`` (no edges), ``model`` (top-k supplied by the model, ranked order),
        ``random`` (k random pool pairs), ``degree`` (top-k by degree product), ``betweenness`` (top-k by summed
        sampled betweenness). Every replicate shares the same diffusion seeds across strategies (paired design).
        """
        rng = np.random.default_rng(self.rng_seed)
        pool = self.pool(pool_pairs)
        if not pool:
            raise ValueError("no usable candidate pairs inside the simulated subgraph")
        model_edges = self.pool(model_pairs)[:k]
        deg = np.array([len(a) for a in self.adj])
        btw = nx.betweenness_centrality(self.G, k=min(200, len(self.nodes)), seed=self.rng_seed)
        btw_arr = np.array([btw[n] for n in self.nodes])
        by_degree = sorted(pool, key=lambda e: (-(deg[e[0]] * deg[e[1]]), e))[:k]
        by_btw = sorted(pool, key=lambda e: (-(btw_arr[e[0]] + btw_arr[e[1]]), e))[:k]
        rows = []
        for r in range(n_replicates):
            seeds = rng.choice(len(self.nodes), size=min(n_seed_nodes, len(self.nodes)), replace=False)
            rand_idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            strategies = {"baseline": [], "model": model_edges, "random": [pool[i] for i in rand_idx],
                          "degree": by_degree, "betweenness": by_btw}
            casc_seed = int(rng.integers(1 << 31))
            for name, edges in strategies.items():
                m = self.structural(edges)
                m["coverage"] = self.coverage(edges, seeds, p, n_cascades, np.random.default_rng(casc_seed))
                rows.append({"replicate": r, "strategy": name, "n_edges": len(edges), **m})
        return pd.DataFrame(rows)


def summarise_simulation(sim: pd.DataFrame) -> pd.DataFrame:
    """Mean/sd per strategy and the paired mean improvement over the no-edge baseline."""
    metrics = ["giant_frac", "connected_community_pairs", "cross_edge_share", "coverage"]
    base = sim[sim["strategy"] == "baseline"].set_index("replicate")[metrics]
    rows = []
    for name, g in sim.groupby("strategy"):
        g = g.set_index("replicate")
        row = {"strategy": name, "n_edges": int(g["n_edges"].iloc[0])}
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=0)
            row[f"{m}_delta"] = float((g[m] - base[m]).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)
