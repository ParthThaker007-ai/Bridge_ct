"""Synthetic biomedical-like corpus for OFFLINE pipeline verification.

IMPORTANT: results on this data say nothing about science. It exists so every stage (graphs, communities,
features, leakage tests, models, simulation, dashboard) can be executed and unit-tested without network access.
The generator plants a learnable signal: topically adjacent communities collaborate increasingly over time,
and prolific authors collaborate more - exactly the structure the real features try to capture.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .cleaning.links import build_paper_trial_links
from .cleaning.normalize import WORK_COLS, normalize_trials

COMMON_WORDS = ["patients", "study", "treatment", "clinical", "outcome", "analysis", "risk", "trial", "effect", "data"]


def generate_synthetic(seed: int = 7, n_communities: int = 12, authors_per_community: int = 60,
                       start_year: int = 2006, end_year: int = 2022, papers_per_year: int = 400,
                       n_topics: int = 24, trials_per_year: int = 15) -> Dict[str, Any]:
    """Return canonical tables with the same schema as :func:`normalize_openalex`."""
    rng = np.random.default_rng(seed)
    C, A = n_communities, authors_per_community
    authors = [(f"A{c:02d}_{i:03d}", c) for c in range(C) for i in range(A)]
    weights = {c: rng.pareto(1.5, A) + 1.0 for c in range(C)}  # heavy-tailed productivity
    weights = {c: w / w.sum() for c, w in weights.items()}
    comm_topics = {c: [(2 * c + j) % n_topics for j in range(3)] for c in range(C)}  # neighbours share a topic
    topic_words = {t: [f"term{t}x{j}" for j in range(6)] for t in range(n_topics)}

    works, ships, wtopics, cites = [], [], [], []
    comm_works: Dict[int, List[str]] = {c: [] for c in range(C)}
    trials_by_year: Dict[int, List[str]] = {}
    trial_rows = []
    nct_counter = 10_000_000
    for y in range(start_year - 2, end_year + 1):
        trials_by_year[y] = []
        for _ in range(trials_per_year):
            nct = f"NCT{nct_counter:08d}"
            nct_counter += 1
            trials_by_year[y].append(nct)
            trial_rows.append({"nct_id": nct, "title": f"Trial {nct}", "conditions": "synthetic condition",
                               "interventions": "drug", "study_start": f"{y}-06-01",
                               "registration_date": f"{y}-03-15", "phase": "PHASE2", "status": "COMPLETED"})

    idx = 0
    for y in range(start_year, end_year + 1):
        p_cross = 0.05 + 0.012 * (y - start_year)
        for _ in range(papers_per_year):
            c = int(rng.integers(C))
            k = int(rng.integers(2, 7))
            partner = None
            if rng.random() < p_cross:
                partner = (c + int(rng.choice([-1, 1]))) % C
                if rng.random() < 0.15:  # noise: unrelated community
                    partner = int(rng.integers(C))
                if partner == c:
                    partner = None
            n_p = max(1, k // 2) if partner is not None else 0
            chosen = [(c, int(i)) for i in rng.choice(A, size=k - n_p, replace=False, p=weights[c])]
            if partner is not None:
                chosen += [(partner, int(i)) for i in rng.choice(A, size=n_p, replace=False, p=weights[partner])]
            wid = f"W{idx}"
            tops = list(comm_topics[c]) + (list(comm_topics[partner]) if partner is not None else [])
            words = [w for t in tops for w in rng.choice(topic_words[t], 3)] + list(rng.choice(COMMON_WORDS, 8))
            month, day = int(rng.integers(1, 13)), int(rng.integers(1, 28))
            linked = ""
            if rng.random() < 0.08:
                pool = trials_by_year.get(y, []) + trials_by_year.get(y - 1, [])
                linked = str(rng.choice(pool))
            works.append({"work_id": wid, "title": " ".join(words[:5]), "abstract": " ".join(words),
                          "publication_date": f"{y}-{month:02d}-{day:02d}", "publication_year": y,
                          "doi": f"10.0000/syn.{idx}", "pmid": str(100000 + idx), "source": "synthetic",
                          "retrieved_at": "synthetic", "mesh_terms": "", "pubmed_nct_ids": linked})
            for pos, (cc, ii) in enumerate(chosen):
                ships.append({"work_id": wid, "author_id": f"A{cc:02d}_{ii:03d}", "institution_id": f"I{cc:02d}_{ii % 2}",
                              "author_position": str(pos), "publication_year": y})
            for t in dict.fromkeys(tops):
                wtopics.append({"work_id": wid, "topic_id": f"T{t}", "score": float(rng.uniform(0.5, 1.0))})
            pool = list(comm_works[c][-200:]) + (list(comm_works[partner][-100:]) if partner is not None else [])
            if pool:
                for ref in rng.choice(pool, size=min(4, len(pool)), replace=False):
                    cites.append({"work_id": wid, "cited_work_id": str(ref)})
            for cc in {c, partner} - {None}:
                comm_works[cc].append(wid)
            idx += 1

    works_df = pd.DataFrame(works, columns=WORK_COLS)
    trials_df = normalize_trials(trial_rows)
    links, link_stats = build_paper_trial_links(works_df, trials_df)
    return {
        "works": works_df, "authorships": pd.DataFrame(ships),
        "authors": pd.DataFrame([{"author_id": a, "display_name": f"Author {a}"} for a, _ in authors]),
        "institutions": pd.DataFrame([{"institution_id": f"I{c:02d}_{j}", "display_name": f"Institute {c}-{j}",
                                       "country_code": ["IN", "US", "GB", "DE"][c % 4]} for c in range(C) for j in range(2)]),
        "topics": pd.DataFrame([{"topic_id": f"T{t}", "label": f"Synthetic topic {t}", "source": "synthetic"}
                                for t in range(n_topics)]),
        "work_topics": pd.DataFrame(wtopics), "citations": pd.DataFrame(cites),
        "trials": trials_df, "paper_trial_links": links,
        "quality": {"n_works": len(works_df), "synthetic": True, **link_stats},
    }
