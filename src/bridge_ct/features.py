"""Candidate generation, temporal feature engineering and label construction.

LEAKAGE CONTRACT: every function that builds *features* for cutoff ``t`` reads only rows with
``publication_year <= t`` (works, authorships, citations among such works, links whose paper AND trial are
dated <= t). The *only* function allowed to look past ``t`` is :meth:`FeatureBuilder.label_candidates`.
``tests/test_no_future_leakage.py`` enforces this by recomputing features on data truncated at ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import Config
from .graph.collab import author_activity, graph_from_edges, snapshot_edges, snapshot_nodes
from .graph.communities import community_members, detect_communities
from .sna.metrics import node_table

# ----------------------------------------------------------------------------- feature registry
NODE_LEVEL: Dict[str, str] = {   # per-author (or per-author's-community) features -> group
    "degree": "collaboration", "wdeg": "collaboration",
    "betweenness": "brokerage", "cross_ratio": "brokerage", "comm_broker": "brokerage",
    "degree_delta": "temporal", "cross_ratio_delta": "temporal",
    "comm_size": "evolution", "comm_age": "evolution", "comm_growth": "evolution", "comm_turnover": "evolution",
    "papers_total": "activity", "papers_recent": "activity", "years_since_last": "activity", "trial_papers": "activity",
}
PAIR_LEVEL: Dict[str, str] = {   # per-candidate-pair features -> group
    "common_neighbors": "collaboration", "jaccard": "collaboration", "adamic_adar": "collaboration",
    "resource_alloc": "collaboration", "pref_attach": "collaboration",
    "cross_edge_count": "collaboration", "cross_edge_density": "collaboration",
    "semantic_author": "semantic", "semantic_comm": "semantic", "shared_features": "semantic",
    "citation_pair": "citation", "shared_trials": "activity",
}
GROUPS = ["collaboration", "brokerage", "temporal", "evolution", "semantic", "citation", "activity"]
_NONNEG_COUNTS = {"degree", "wdeg", "comm_size", "comm_age", "papers_total", "papers_recent", "years_since_last",
                  "trial_papers", "degree_delta", "common_neighbors", "adamic_adar", "resource_alloc", "pref_attach",
                  "cross_edge_count", "shared_features", "citation_pair", "shared_trials"}
ID_COLS = ["cutoff", "u", "v", "source", "comm_u", "comm_v", "y", "labeled"]


def feature_columns() -> List[str]:
    """Final model-input column names. Node-level features appear as ``<name>_min`` / ``<name>_max`` so the
    representation is symmetric in (u, v)."""
    cols = [f"{n}_{s}" for n in NODE_LEVEL for s in ("min", "max")]
    return cols + list(PAIR_LEVEL)


def feature_groups() -> Dict[str, List[str]]:
    """Group name -> final column names (used by ablations)."""
    out: Dict[str, List[str]] = {g: [] for g in GROUPS}
    for n, g in NODE_LEVEL.items():
        out[g] += [f"{n}_min", f"{n}_max"]
    for n, g in PAIR_LEVEL.items():
        out[g].append(n)
    return out


LOG_COLUMNS = {f"{n}_{s}" for n in NODE_LEVEL if n in _NONNEG_COUNTS for s in ("min", "max")} | \
              {n for n in PAIR_LEVEL if n in _NONNEG_COUNTS}


def prepare_matrix(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """Feature matrix with ``log1p`` on heavy-tailed non-negative counts (stateless: no fitting, no leakage)."""
    X = df[columns].to_numpy(dtype=float, copy=True)
    for j, c in enumerate(columns):
        if c in LOG_COLUMNS:
            X[:, j] = np.log1p(np.clip(X[:, j], 0.0, None))
    return X


# ----------------------------------------------------------------------------- snapshot container
@dataclass
class Snapshot:
    cutoff: int
    G: nx.Graph
    edges: pd.DataFrame
    membership: Dict[str, int]
    members: Dict[int, set]
    nodes: pd.DataFrame
    prev_nodes: Optional[pd.DataFrame]
    activity: pd.DataFrame
    extra: Dict[str, Any] = field(default_factory=dict)


class FeatureBuilder:
    """Builds snapshots, candidate pairs, features and labels from canonical tables.

    Parameters
    ----------
    tables : canonical tables (``works, authorships, work_topics, citations, trials, paper_trial_links``).
    pair_table : output of :func:`bridge_ct.graph.collab.build_pair_table`.
    cfg : validated :class:`Config`.
    """

    def __init__(self, tables: Dict[str, Any], pair_table: pd.DataFrame, cfg: Config) -> None:
        self.cfg = cfg
        self.works = (tables["works"][["work_id", "title", "abstract", "publication_year"]]
                      .drop_duplicates("work_id").sort_values("work_id").reset_index(drop=True))
        self.authorships = (tables["authorships"][["work_id", "author_id", "publication_year"]]
                            .dropna().drop_duplicates(["work_id", "author_id"]).reset_index(drop=True))
        self.authorships["author_id"] = self.authorships["author_id"].astype(str)
        self.work_topics = tables.get("work_topics", pd.DataFrame(columns=["work_id", "topic_id", "score"]))
        self.citations = tables.get("citations", pd.DataFrame(columns=["work_id", "cited_work_id"]))
        self.pair_table = pair_table
        self.first_year = (pair_table.groupby(["u", "v"])["year"].min() if len(pair_table)
                           else pd.Series(dtype="int64", index=pd.MultiIndex.from_arrays([[], []], names=["u", "v"])))
        links = tables.get("paper_trial_links", pd.DataFrame(columns=["work_id", "nct_id", "confidence"]))
        trials = tables.get("trials", pd.DataFrame(columns=["nct_id", "registration_year"]))
        links = links[links["confidence"] >= cfg.links.min_confidence]
        links = links.merge(trials[["nct_id", "registration_year"]], on="nct_id", how="inner")
        links = links.merge(self.works[["work_id", "publication_year"]], on="work_id", how="inner")
        self.links = links.rename(columns={"registration_year": "reg_year", "publication_year": "work_year"})

    # ------------------------------------------------------------------ snapshots
    def partition(self, cutoff: int) -> Tuple[Dict[str, int], Dict]:
        """Community partition of G_cutoff (used for the evolution pass)."""
        G = graph_from_edges(snapshot_nodes(self.authorships, cutoff),
                             snapshot_edges(self.pair_table, cutoff), cutoff)
        g = self.cfg.graph
        return detect_communities(G, g.community_algorithm, g.community_resolution,
                                  self.cfg.project.random_seed, g.min_community_size)

    def snapshot(self, cutoff: int, membership: Dict[str, int], prev_membership: Optional[Dict[str, int]]) -> Snapshot:
        """G_cutoff with node measures; ``prev_membership`` (cutoff-1) enables temporal-delta features."""
        seed = self.cfg.project.random_seed
        edges = snapshot_edges(self.pair_table, cutoff)
        G = graph_from_edges(snapshot_nodes(self.authorships, cutoff), edges, cutoff)
        nodes = node_table(G, membership, self.cfg.graph.betweenness_samples, seed)
        prev_nodes = None
        if prev_membership is not None:
            Gp = graph_from_edges(snapshot_nodes(self.authorships, cutoff - 1),
                                  snapshot_edges(self.pair_table, cutoff - 1), cutoff - 1)
            prev_nodes = node_table(Gp, prev_membership, 0, seed)  # betweenness skipped (not needed for deltas)
        act = author_activity(self.authorships, cutoff, self.cfg.candidates.active_window_years)
        return Snapshot(cutoff, G, edges, membership, community_members(membership), nodes, prev_nodes, act)

    # ------------------------------------------------------------------ semantic / citation / trial context
    def _semantic_context(self, snap: Snapshot, comm_ids: List[int]) -> Dict[str, Any]:
        """Sparse author x feature and community x feature matrices built ONLY from works <= cutoff."""
        t = snap.cutoff
        w = self.works[self.works["publication_year"] <= t]
        w_idx = {wid: i for i, wid in enumerate(w["work_id"])}
        authors = list(snap.G.nodes())
        a_idx = {a: i for i, a in enumerate(authors)}
        ships = self.authorships[self.authorships["publication_year"] <= t]
        rows = ships["author_id"].map(a_idx)
        cols = ships["work_id"].map(w_idx)
        ok = rows.notna() & cols.notna()
        A = sparse.csr_matrix((np.ones(int(ok.sum())), (rows[ok].astype(int), cols[ok].astype(int))),
                              shape=(len(authors), len(w_idx)))
        sem = self.cfg.semantic
        n_w = len(w_idx)
        W = None
        if sem.method == "topics" and len(self.work_topics):
            wt = self.work_topics[self.work_topics["work_id"].isin(w_idx)]
            vocab = {tid: j for j, tid in enumerate(sorted(wt["topic_id"].unique()))}
            if vocab:
                W = sparse.csr_matrix((wt["score"].to_numpy(float), (wt["work_id"].map(w_idx).astype(int),
                                       wt["topic_id"].map(vocab).astype(int))), shape=(n_w, len(vocab)))
        elif sem.method == "tfidf" and n_w:
            texts = (w["title"].fillna("") + " " + w["abstract"].fillna("")).tolist()
            try:  # vocabulary is fitted on works <= cutoff only (IDF statistics must not see the future)
                vec = TfidfVectorizer(max_features=sem.tfidf_max_features, stop_words="english",
                                      min_df=2 if n_w >= 200 else 1, sublinear_tf=True)
                W = vec.fit_transform(texts).tocsr()
            except ValueError:  # empty vocabulary
                W = None
        if W is None:
            W = sparse.csr_matrix((n_w, 1))
        W = normalize(W)
        AW = (A @ W).tocsr()
        AWn = normalize(AW)
        c_idx = {c: i for i, c in enumerate(comm_ids)}
        mr, mc = [], []
        for c in comm_ids:
            for a in snap.members.get(c, ()):
                mr.append(c_idx[c]); mc.append(a_idx[a])
        M = sparse.csr_matrix((np.ones(len(mr)), (mr, mc)), shape=(len(comm_ids), len(authors)))
        CWn = normalize((M @ AW).tocsr())
        sim = (CWn @ CWn.T).toarray() if len(comm_ids) else np.zeros((0, 0))
        np.fill_diagonal(sim, -1.0)  # a community is never its own semantic neighbour
        AWb = AW.copy(); AWb.data[:] = 1.0
        return {"A": A, "w_idx": w_idx, "a_idx": a_idx, "AWn": AWn, "AWb": AWb, "sim": sim, "c_idx": c_idx}

    def _citation_pair(self, ctx: Dict[str, Any], ui: np.ndarray, vi: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Citations between the work sets of u and v (either direction), among works <= cutoff.

        value(u,v) = A_u C A_v^T + A_v C A_u^T with C[i,j] = 1 if work i cites work j. Row-chunked to bound memory.
        """
        A, w_idx = ctx["A"], ctx["w_idx"]
        cit = self.citations
        r = cit["work_id"].map(w_idx)
        c = cit["cited_work_id"].map(w_idx)
        ok = r.notna() & c.notna()
        n_w = len(w_idx)
        if not ok.any() or len(ui) == 0:
            return np.zeros(len(ui))
        C = sparse.csr_matrix((np.ones(int(ok.sum())), (r[ok].astype(int), c[ok].astype(int))), shape=(n_w, n_w))
        C.sum_duplicates(); C.data[:] = 1.0
        out = np.zeros(len(ui))
        for s in range(0, len(ui), chunk):
            a, b = ui[s:s + chunk], vi[s:s + chunk]
            fwd = (A[a] @ C).multiply(A[b]).sum(axis=1)
            bwd = (A[b] @ C).multiply(A[a]).sum(axis=1)
            out[s:s + chunk] = np.asarray(fwd + bwd).ravel()
        return out

    def _trial_context(self, snap: Snapshot) -> Tuple[pd.Series, Dict[str, frozenset]]:
        """Per-author count of trial-linked papers and the set of linked NCT IDs (all dated <= cutoff)."""
        t = snap.cutoff
        lk = self.links[(self.links["work_year"] <= t) & (self.links["reg_year"] <= t)]
        if lk.empty:
            return pd.Series(dtype=float), {}
        ships = self.authorships[self.authorships["publication_year"] <= t][["work_id", "author_id"]]
        m = lk[["work_id", "nct_id"]].merge(ships, on="work_id")
        counts = m.groupby("author_id")["work_id"].nunique().astype(float)
        sets = {a: frozenset(g) for a, g in m.groupby("author_id")["nct_id"]}
        return counts, sets

    # ------------------------------------------------------------------ candidates
    def generate_candidates(self, snap: Snapshot, comm_ids: List[int], sim: np.ndarray) -> pd.DataFrame:
        """Candidate universe (documented rule; identical for every model being compared).

        A pair (u, v) is a candidate when u, v are *active* (paper within the last ``active_window_years``),
        belong to different communities of size >= ``min_community_size``, are not yet linked, and either
        (a) they are among the ``top_authors_per_community`` most connected active authors of two communities
            that are semantic neighbours (top ``neighbour_communities`` by cosine >= ``min_semantic_similarity``), or
        (b) they share a collaborator (two-hop pair via a broker; at most ``two_hop_per_broker`` per broker).
        """
        cc = self.cfg.candidates
        G, t = snap.G, snap.cutoff
        active = set(snap.activity.index[snap.activity["years_since_last"] < cc.active_window_years])
        elig = set(comm_ids)
        wdeg = snap.nodes["wdeg"]
        top: Dict[int, List[str]] = {}
        for c in comm_ids:
            mem = sorted(m for m in snap.members[c] if m in active)
            mem.sort(key=lambda a: (-wdeg[a], a))
            top[c] = mem[:cc.top_authors_per_community]
        pairs: Dict[Tuple[str, str], str] = {}

        def add(u: str, v: str, src: str) -> None:
            if u == v or snap.membership[u] == snap.membership[v] or G.has_edge(u, v):
                return
            key = (u, v) if u < v else (v, u)
            prev = pairs.get(key)
            pairs[key] = src if prev is None or prev == src else "both"

        done = set()
        for i, a in enumerate(comm_ids):  # generator (a): semantic neighbours
            order = np.lexsort((np.arange(len(comm_ids)), -sim[i]))
            taken = 0
            for j in order:
                if taken >= cc.neighbour_communities or sim[i, j] < cc.min_semantic_similarity:
                    break
                b = comm_ids[j]
                key = (min(a, b), max(a, b))
                taken += 1
                if key in done:
                    continue
                done.add(key)
                for u in top[a]:
                    for v in top[b]:
                        add(u, v, "semantic")
        rng = np.random.default_rng(self.cfg.project.random_seed + t)
        if cc.two_hop_per_broker > 0:  # generator (b): two-hop pairs through a broker
            for w in sorted(G.nodes()):
                nb = [x for x in sorted(G[w]) if x in active and snap.membership[x] in elig]
                if len(nb) < 2:
                    continue
                total = len(nb) * (len(nb) - 1) // 2
                if total <= cc.two_hop_per_broker * 2:
                    for i in range(len(nb)):
                        for j in range(i + 1, len(nb)):
                            add(nb[i], nb[j], "two_hop")
                else:
                    for _ in range(cc.two_hop_per_broker * 3):
                        i, j = rng.choice(len(nb), size=2, replace=False)
                        add(nb[i], nb[j], "two_hop")
        keys = sorted(pairs)
        if len(keys) > cc.max_candidates_per_snapshot:
            pick = np.sort(rng.choice(len(keys), cc.max_candidates_per_snapshot, replace=False))
            keys = [keys[i] for i in pick]
        return pd.DataFrame({"u": [k[0] for k in keys], "v": [k[1] for k in keys],
                             "source": [pairs[k] for k in keys]})

    # ------------------------------------------------------------------ features
    def build_features(self, snap: Snapshot, evo_t: pd.DataFrame) -> pd.DataFrame:
        """Candidates + features for one snapshot (no labels)."""
        cols = ID_COLS[:6] + feature_columns()
        comm_ids = sorted(int(c) for c in evo_t["community_id"])
        if len(comm_ids) < 2:
            return pd.DataFrame(columns=cols)
        ctx = self._semantic_context(snap, comm_ids)
        cand = self.generate_candidates(snap, comm_ids, ctx["sim"])
        if cand.empty:
            return pd.DataFrame(columns=cols)
        G = snap.G
        us, vs = cand["u"].to_numpy(), cand["v"].to_numpy()

        # ---- node-level table
        nt = snap.nodes.copy()
        nt["comm"] = pd.Series(snap.membership).reindex(nt.index).to_numpy()
        nt["comm_broker"] = nt.groupby("comm")["cross_ratio"].transform("mean")
        evo = evo_t.set_index("community_id")
        for col, src in (("comm_size", "size"), ("comm_age", "age"), ("comm_growth", "growth"), ("comm_turnover", "turnover")):
            nt[col] = nt["comm"].map(evo[src].to_dict())
        nt = nt.join(snap.activity[["papers_total", "papers_recent", "years_since_last"]])
        trial_counts, trial_sets = self._trial_context(snap)
        nt["trial_papers"] = trial_counts.reindex(nt.index).fillna(0.0)
        if snap.prev_nodes is not None:
            p = snap.prev_nodes.reindex(nt.index)
            nt["degree_delta"] = nt["degree"] - p["degree"].fillna(0.0)
            nt["cross_ratio_delta"] = (nt["cross_ratio"] - p["cross_ratio"]).fillna(0.0)
        else:
            nt["degree_delta"], nt["cross_ratio_delta"] = 0.0, 0.0

        # ---- pair-level graph features
        nbr = {n: set(G[n]) for n in set(us) | set(vs)}
        deg = dict(G.degree())
        cn, jac, aa, ra = (np.zeros(len(cand)) for _ in range(4))
        for k, (u, v) in enumerate(zip(us, vs)):
            common = nbr[u] & nbr[v]
            cn[k] = len(common)
            union = len(nbr[u] | nbr[v])
            jac[k] = len(common) / union if union else 0.0
            aa[k] = sum(1.0 / np.log(deg[w]) for w in common if deg[w] > 1)
            ra[k] = sum(1.0 / deg[w] for w in common)
        cu = np.array([snap.membership[u] for u in us]); cv = np.array([snap.membership[v] for v in vs])
        e = snap.edges
        cross_cnt: Dict[Tuple[int, int], int] = {}
        if len(e):
            ce = pd.DataFrame({"a": e["u"].map(snap.membership).to_numpy(), "b": e["v"].map(snap.membership).to_numpy()})
            ce = ce[ce["a"] != ce["b"]]
            ce = pd.DataFrame({"lo": np.minimum(ce["a"], ce["b"]), "hi": np.maximum(ce["a"], ce["b"])})
            cross_cnt = ce.groupby(["lo", "hi"]).size().to_dict()
        size = {c: len(m) for c, m in snap.members.items()}
        cec = np.array([cross_cnt.get((min(a, b), max(a, b)), 0) for a, b in zip(cu, cv)], dtype=float)
        ced = cec / np.array([size[a] * size[b] for a, b in zip(cu, cv)], dtype=float)

        # ---- semantic / citation / trial pair features
        ui = np.array([ctx["a_idx"][u] for u in us]); vi = np.array([ctx["a_idx"][v] for v in vs])
        AWn, AWb, sim = ctx["AWn"], ctx["AWb"], ctx["sim"]
        sem_author = np.asarray(AWn[ui].multiply(AWn[vi]).sum(axis=1)).ravel()
        shared = np.asarray(AWb[ui].multiply(AWb[vi]).sum(axis=1)).ravel()
        sem_comm = np.array([sim[ctx["c_idx"][a], ctx["c_idx"][b]] for a, b in zip(cu, cv)])
        cit = self._citation_pair(ctx, ui, vi)
        empty = frozenset()
        shared_tr = np.array([len(trial_sets.get(u, empty) & trial_sets.get(v, empty)) for u, v in zip(us, vs)], dtype=float)
        pair_vals = {"common_neighbors": cn, "jaccard": jac, "adamic_adar": aa, "resource_alloc": ra,
                     "pref_attach": np.array([deg[u] * deg[v] for u, v in zip(us, vs)], dtype=float),
                     "cross_edge_count": cec, "cross_edge_density": ced, "semantic_author": sem_author,
                     "semantic_comm": sem_comm, "shared_features": shared, "citation_pair": cit,
                     "shared_trials": shared_tr}

        out = {"cutoff": np.full(len(cand), snap.cutoff), "u": us, "v": vs, "source": cand["source"].to_numpy(),
               "comm_u": cu, "comm_v": cv}
        for name in NODE_LEVEL:
            a = nt.loc[list(us), name].to_numpy(dtype=float)
            b = nt.loc[list(vs), name].to_numpy(dtype=float)
            out[f"{name}_min"], out[f"{name}_max"] = np.fmin(a, b), np.fmax(a, b)
        out.update(pair_vals)
        return pd.DataFrame(out)[cols]

    # ------------------------------------------------------------------ labels (the ONLY place that looks forward)
    def label_candidates(self, cand: pd.DataFrame, cutoff: int) -> np.ndarray:
        """y = 1 if the pair first co-authors in (cutoff, cutoff + horizon]; otherwise 0.

        Candidates are unlinked at ``cutoff`` by construction. "Not observed" is treated as negative within the
        horizon - a documented approximation, since an unobserved link is not proof that none will ever form.
        """
        if cand.empty:
            return np.zeros(0, dtype=int)
        idx = pd.MultiIndex.from_arrays([cand["u"].to_numpy(), cand["v"].to_numpy()])
        fy = self.first_year.reindex(idx).to_numpy(dtype=float)
        return ((fy > cutoff) & (fy <= cutoff + self.cfg.horizon)).astype(int)
