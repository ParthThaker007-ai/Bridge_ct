"""Secondary target: will a research community become linked to a NEWLY REGISTERED clinical trial?

Gated by a feasibility check: unless enough high-confidence paper->trial links exist the target is skipped
(the blueprint says future co-authorship stays the only target in that case). Labels use identifier-based links
only; right-censoring (papers about late-registered trials appear after the data window) adds label noise near
the end of the window - treat results as exploratory.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureBuilder, prepare_matrix
from .models.bayesian import BayesianLogisticRegression
from .models.evaluate import evaluate_scores
from .models.splits import make_splits

TRIAL_FEATURES = ["size", "age", "growth", "turnover", "prior_trial_papers", "prior_trial_ncts", "recent_papers"]


def build_trial_target(fb: FeatureBuilder, evo: pd.DataFrame, memberships: Dict[int, Dict[str, int]], cfg: Config,
                       min_links: int = 100) -> Optional[pd.DataFrame]:
    """One row per (cutoff, community). Returns ``None`` when the link audit fails the feasibility gate."""
    if len(fb.links) < min_links:
        return None
    H, W = cfg.horizon, cfg.candidates.active_window_years
    author_nct = fb.links[["work_id", "nct_id", "reg_year", "work_year"]].merge(
        fb.authorships[["work_id", "author_id"]], on="work_id")
    rows = []
    for t in cfg.labeled_cutoffs:
        evo_t = evo[evo["year"] == t]
        if evo_t.empty or t not in memberships:
            continue
        comm_of = pd.Series(memberships[t])
        eligible = set(evo_t["community_id"])
        prior = author_nct[(author_nct["work_year"] <= t) & (author_nct["reg_year"] <= t)].copy()
        prior["comm"] = prior["author_id"].map(comm_of)
        prior = prior[prior["comm"].isin(eligible)]
        n_papers = prior.drop_duplicates(["comm", "work_id"]).groupby("comm").size()
        n_ncts = prior.drop_duplicates(["comm", "nct_id"]).groupby("comm").size()
        fut = author_nct[(author_nct["reg_year"] > t) & (author_nct["reg_year"] <= t + H)].copy()
        fut["comm"] = fut["author_id"].map(comm_of)
        positive = set(fut["comm"].dropna().astype(int))
        ships = fb.authorships[(fb.authorships["publication_year"] <= t) & (fb.authorships["publication_year"] > t - W)].copy()
        ships["comm"] = ships["author_id"].map(comm_of)
        recent = ships[ships["comm"].isin(eligible)].drop_duplicates(["comm", "work_id"]).groupby("comm").size()
        for r in evo_t.itertuples(index=False):
            c = r.community_id
            rows.append({"cutoff": t, "community_id": c, "lineage_id": r.lineage_id, "size": r.size, "age": r.age,
                         "growth": r.growth, "turnover": r.turnover,
                         "prior_trial_papers": float(n_papers.get(c, 0)), "prior_trial_ncts": float(n_ncts.get(c, 0)),
                         "recent_papers": float(recent.get(c, 0)), "y": int(c in positive)})
    return pd.DataFrame(rows) if rows else None


def evaluate_trial_target(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Bayesian model vs a 'prior trial activity' baseline on the same chronological test split."""
    ev = cfg.evaluation
    sp = make_splits(sorted(int(c) for c in df["cutoff"].unique()), cfg.horizon, ev.val_first, ev.test_first, ev.purge)
    tr, te = df[df["cutoff"].isin(sp["refit"])], df[df["cutoff"].isin(sp["test"])]
    if tr["y"].nunique() < 2 or te["y"].sum() == 0:
        return {"status": "skipped: single-class train or test labels", "n_train": len(tr), "n_test": len(te)}
    model = BayesianLogisticRegression(prior_scale=1.0).fit(np.log1p(tr[TRIAL_FEATURES].clip(lower=0).to_numpy()), tr["y"].to_numpy(), TRIAL_FEATURES)
    p = model.predict_proba(np.log1p(te[TRIAL_FEATURES].clip(lower=0).to_numpy()))
    ks = [min(k, len(te)) for k in ev.top_k]
    return {"status": "ok", "n_train": len(tr), "n_test": len(te),
            "bayesian": evaluate_scores(te["y"].to_numpy(), p, ks, ev.n_bootstrap, is_probability=True),
            "baseline_prior_trial_papers": evaluate_scores(te["y"].to_numpy(), te["prior_trial_papers"].to_numpy(), ks, ev.n_bootstrap)}
