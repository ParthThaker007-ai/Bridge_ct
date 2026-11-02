"""Paper -> trial links (identifier-based only) and the manual-review sample.

Blueprint rules: preserve provenance and confidence; never fabricate a link; semantic matches would only ever
be *candidates* needing manual audit. This implementation deliberately builds only high-precision
identifier links:

* ``pubmed_databank``  confidence 1.00 - the PubMed record itself lists the NCT ID in its DataBank section;
* ``abstract_nct_id``  confidence 0.95 - an NCT ID is written in the title/abstract text.

A link whose NCT ID is not in the downloaded trial table is dropped (and counted) because it has no
registration date and therefore cannot be placed on the timeline safely.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .normalize import NCT_RE


def _s(value) -> str:
    """Missing-safe string: NaN/None -> '' (pandas marks missing strings as NaN, which is truthy)."""
    return "" if value is None or (isinstance(value, float) and value != value) else str(value)

LINK_COLS = ["work_id", "nct_id", "method", "confidence", "evidence_uri", "reviewed"]


def build_paper_trial_links(works: pd.DataFrame, trials: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Return ``(links, stats)``; ``links`` has one row per (work, trial) with the best method kept."""
    known = set(trials["nct_id"]) if len(trials) else set()
    rows, n_unknown = [], 0
    for _, w in works.iterrows():
        found: Dict[str, Tuple[str, float, str]] = {}
        pmid = _s(w["pmid"])
        pm_uri = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        for nct in [n for n in _s(w["pubmed_nct_ids"]).split("|") if n]:
            found[nct.upper()] = ("pubmed_databank", 1.0, pm_uri)
        text = f"{_s(w['title'])} {_s(w['abstract'])}"
        for nct in {m.upper() for m in NCT_RE.findall(text)}:
            found.setdefault(nct, ("abstract_nct_id", 0.95, pm_uri or f"https://openalex.org/{w['work_id']}"))
        for nct, (method, conf, uri) in found.items():
            if nct in known:
                rows.append({"work_id": w["work_id"], "nct_id": nct, "method": method, "confidence": conf,
                             "evidence_uri": uri, "reviewed": False})
            else:
                n_unknown += 1
    links = pd.DataFrame(rows, columns=LINK_COLS)
    stats = {"n_links": len(links), "n_dropped_unknown_trial": n_unknown,
             **{f"n_{m}": int((links["method"] == m).sum()) for m in ("pubmed_databank", "abstract_nct_id")}}
    return links, stats


def sample_links_for_review(links: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Random sample (stratified by method) for manual precision review; fill ``correct`` with 1/0 by hand."""
    if links.empty:
        return links.assign(correct=pd.Series(dtype="float"))
    parts = [g.sample(min(len(g), max(1, n // links["method"].nunique())), random_state=seed)
             for _, g in links.groupby("method")]
    return pd.concat(parts).assign(correct=float("nan")).reset_index(drop=True)


def link_precision(reviewed: pd.DataFrame) -> Dict[str, float]:
    """Precision per method from a manually reviewed sample (rows with a non-null ``correct``)."""
    done = reviewed.dropna(subset=["correct"])
    if done.empty:
        return {}
    return done.groupby("method")["correct"].mean().to_dict()
