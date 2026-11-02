"""Phase-0 data audit: coverage, ID completeness, date quality, topic coverage, linkage and bias."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def data_audit(tables: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise data quality. ``tables`` is the dict produced by normalisation (plus ``trials`` / ``paper_trial_links``).

    Returns a JSON-serialisable dict; ``per_year`` is a list of rows (year, works, trials, linked_works).
    """
    works, ships = tables["works"], tables["authorships"]
    trials = tables.get("trials", pd.DataFrame())
    links = tables.get("paper_trial_links", pd.DataFrame())
    per_year = works.groupby("publication_year").size().rename("works").to_frame()
    if len(trials):
        per_year = per_year.join(trials.groupby("registration_year").size().rename("trials"), how="outer")
    if len(links):
        lw = links.merge(works[["work_id", "publication_year"]], on="work_id")
        per_year = per_year.join(lw.groupby("publication_year")["work_id"].nunique().rename("linked_works"), how="outer")
    per_year = per_year.fillna(0).astype(int).sort_index()
    per_year.index.name = "year"

    n_auth = ships["author_id"].nunique() if len(ships) else 0
    authors_per_work = ships.groupby("work_id").size() if len(ships) else pd.Series(dtype=int)
    inst = tables.get("institutions", pd.DataFrame())
    top_inst_share = None
    if len(ships) and ships["institution_id"].notna().any():
        counts = ships["institution_id"].value_counts(normalize=True)
        top_inst_share = float(counts.head(10).sum())
    country_top = {}
    if len(inst) and len(ships):
        merged = ships.merge(inst[["institution_id", "country_code"]], on="institution_id", how="left")
        country_top = merged["country_code"].value_counts(normalize=True).head(10).round(4).to_dict()
    return {
        **tables.get("quality", {}),
        "n_authors": int(n_auth),
        "n_authorships": int(len(ships)),
        "frac_authorships_with_institution": float(ships["institution_id"].notna().mean()) if len(ships) else 0.0,
        "authors_per_work_median": float(authors_per_work.median()) if len(authors_per_work) else 0.0,
        "authors_per_work_p99": float(authors_per_work.quantile(0.99)) if len(authors_per_work) else 0.0,
        "works_with_over_50_authors": int((authors_per_work > 50).sum()),
        "n_trials": int(len(trials)),
        "n_links": int(len(links)),
        "link_methods": links["method"].value_counts().to_dict() if len(links) else {},
        "top10_institution_share": top_inst_share,
        "top_country_shares": country_top,
        "per_year": per_year.reset_index().to_dict(orient="records"),
        "warnings": _warnings(per_year, links, ships),
    }


def _warnings(per_year: pd.DataFrame, links: pd.DataFrame, ships: pd.DataFrame) -> list:
    """Human-readable red flags that should stop or reshape the project (Phase-0 exit criterion)."""
    out = []
    covered = per_year[per_year["works"] > 0]  # ignore years that only have trials (outside the works window)
    if len(covered):  # gaps INSIDE the works window count as zero-work years
        covered = covered.reindex(range(int(covered.index.min()), int(covered.index.max()) + 1), fill_value=0)
    if len(covered) and covered["works"].min() < 50:
        out.append("Some years have fewer than 50 works: narrow the window or broaden the disease query.")
    if len(links) < 100:
        out.append("Fewer than 100 paper-trial links: keep future co-authorship as the ONLY target.")
    if len(ships) and ships["institution_id"].notna().mean() < 0.5:
        out.append("Less than half of authorships have an institution: institution-based features are unreliable.")
    return out