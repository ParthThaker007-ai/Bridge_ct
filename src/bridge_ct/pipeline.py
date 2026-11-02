"""Pipeline stages: fetch -> normalize -> build -> train -> shortlist (+ report, simulation, model card).

Every stage reads/writes files under ``data/`` and ``artifacts/`` so it can be re-run independently.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cleaning.audit import data_audit
from .cleaning.links import build_paper_trial_links, sample_links_for_review
from .cleaning.normalize import merge_pubmed, normalize_openalex, normalize_pmid, normalize_trials
from .config import Config
from .features import (GROUPS, FeatureBuilder, feature_columns, feature_groups, prepare_matrix)
from .graph.collab import build_pair_table, graph_from_edges, snapshot_edges, snapshot_nodes
from .graph.communities import community_members, community_stability
from .graph.evolution import track_evolution
from .ingestion.clinicaltrials import fetch_trials
from .ingestion.http import CachedClient
from .ingestion.openalex import fetch_works
from .ingestion.pubmed import fetch_pubmed_records
from .models.baselines import baseline_scores
from .models.bayesian import BayesianLogisticRegression
from .models.evaluate import calibration_table, evaluate_scores, paired_bootstrap_diff
from .models.logistic import LogisticModel
from .models.splits import make_splits
from .report import EvidenceReporter, to_markdown
from .simulation import GraphSimulator, summarise_simulation
from .sna.metrics import graph_summary
from .trial_target import build_trial_target, evaluate_trial_target
from .utils import ensure_dir, get_logger, read_json, set_seed, sha256_file, write_json

log = get_logger()
TABLE_NAMES = ["works", "authorships", "authors", "institutions", "topics", "work_topics", "citations",
               "trials", "paper_trial_links"]


# ============================================================================= I/O helpers
def _write_jsonl(records: List[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_tables(tables: Dict[str, Any], cfg: Config) -> None:
    out = ensure_dir(cfg.data_dir / "processed")
    for name in TABLE_NAMES:
        tables[name].to_parquet(out / f"{name}.parquet", index=False)
    write_json(tables.get("quality", {}), out / "quality.json")


def load_tables(cfg: Config) -> Dict[str, Any]:
    src = cfg.data_dir / "processed"
    missing = [n for n in TABLE_NAMES if not (src / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"missing processed tables {missing}; run the 'normalize' stage first")
    t: Dict[str, Any] = {n: pd.read_parquet(src / f"{n}.parquet") for n in TABLE_NAMES}
    t["quality"] = read_json(src / "quality.json") if (src / "quality.json").exists() else {}
    return t


# ============================================================================= stage 1: fetch
def stage_fetch(cfg: Config) -> None:
    """Download raw records (cached) and write a retrieval manifest with checksums."""
    raw = ensure_dir(cfg.data_dir / "raw")
    cache = raw / "cache"
    p, s = cfg.project, cfg.sources
    ncbi_gap = 0.11 if os.environ.get("NCBI_API_KEY") else 0.34
    oa, ncbi, ct = CachedClient(cache, 0.1), CachedClient(cache, ncbi_gap), CachedClient(cache, 0.2)
    files: Dict[str, Dict[str, Any]] = {}
    if s.openalex:
        works = fetch_works(oa, p.disease, p.start_year, p.end_year, s.max_works, s.work_types)
        _write_jsonl(works, raw / "openalex_works.jsonl")
        if s.pubmed:
            pmids = [normalize_pmid((w.get("ids") or {}).get("pmid")) for w in works]
            recs = fetch_pubmed_records(ncbi, [x for x in pmids if x])
            _write_jsonl(recs, raw / "pubmed_records.jsonl")
    if s.clinicaltrials:
        _write_jsonl(fetch_trials(ct, p.disease, p.start_year, p.end_year), raw / "trials.jsonl")
    for f in sorted(raw.glob("*.jsonl")):
        files[f.name] = {"sha256": sha256_file(f), "n_records": sum(1 for _ in open(f, encoding="utf-8"))}
    write_json({"created_at": datetime.now(timezone.utc).isoformat(), "disease": p.disease,
                "window": [p.start_year, p.end_year], "files": files,
                "calls": oa.manifest + ncbi.manifest + ct.manifest,
                "notes": "Check each source's current terms/licensing before redistributing data."},
               raw / "manifest.json")
    log.info("fetch done: %s", {k: v["n_records"] for k, v in files.items()})


# ============================================================================= stage 2: normalize + audit
def stage_normalize(cfg: Config) -> Dict[str, Any]:
    raw = cfg.data_dir / "raw"
    if not (raw / "openalex_works.jsonl").exists():
        raise FileNotFoundError("run the 'fetch' stage first (data/raw/openalex_works.jsonl not found)")
    manifest = read_json(raw / "manifest.json") if (raw / "manifest.json").exists() else {}
    tables = normalize_openalex(_read_jsonl(raw / "openalex_works.jsonl"), manifest.get("created_at"))
    if (raw / "pubmed_records.jsonl").exists():
        tables["works"] = merge_pubmed(tables["works"], _read_jsonl(raw / "pubmed_records.jsonl"))
    trials = normalize_trials(_read_jsonl(raw / "trials.jsonl")) if (raw / "trials.jsonl").exists() else normalize_trials([])
    tables["trials"] = trials
    tables["paper_trial_links"], link_stats = build_paper_trial_links(tables["works"], trials)
    tables["quality"] = {**tables["quality"], **link_stats}
    audit = data_audit(tables)
    save_tables(tables, cfg)
    art = ensure_dir(cfg.artifacts_dir / "metrics")
    write_json(audit, art / "data_audit.json")
    sample_links_for_review(tables["paper_trial_links"], 100, cfg.project.random_seed).to_csv(
        art / "link_review_sample.csv", index=False)
    for w in audit["warnings"]:
        log.warning("AUDIT: %s", w)
    return tables


# ============================================================================= stage 3: build (graphs, communities, features)
def stage_build(cfg: Config, tables: Dict[str, Any]) -> Dict[str, Any]:
    """Yearly snapshots -> communities -> evolution -> candidate features + labels. Saves parquet artifacts."""
    set_seed(cfg.project.random_seed)
    art = ensure_dir(cfg.artifacts_dir / "features")
    metrics_dir = ensure_dir(cfg.artifacts_dir / "metrics")
    pair_table, skipped = build_pair_table(tables["authorships"], cfg.graph.max_authors_per_paper,
                                           cfg.graph.collaboration_weight)
    log.info("pair table: %d rows (%d works skipped for > %d authors)", len(pair_table), skipped, cfg.graph.max_authors_per_paper)
    fb = FeatureBuilder(tables, pair_table, cfg)
    first = max(int(tables["authorships"]["publication_year"].min()), cfg.project.start_year)
    years = list(range(first, cfg.project.end_year + 1))
    memberships: Dict[int, Dict[str, int]] = {}
    members_by_year: Dict[int, Dict[int, set]] = {}
    part_diag: Dict[int, Dict] = {}
    for y in years:
        memberships[y], part_diag[y] = fb.partition(y)
        members_by_year[y] = community_members(memberships[y])
    evo, lineage = track_evolution(members_by_year, cfg.graph.min_community_size)
    labeled = set(cfg.labeled_cutoffs)
    frames, summary, cedge_rows, mem_rows = [], [], [], []
    for t in cfg.labeled_cutoffs + [cfg.prospective_cutoff]:
        if t not in memberships:
            continue
        snap = fb.snapshot(t, memberships[t], memberships.get(t - 1))
        evo_t = evo[evo["year"] == t]
        df = fb.build_features(snap, evo_t)
        if t in labeled:
            df["y"], df["labeled"] = fb.label_candidates(df, t).astype(float), True
        else:
            df["y"], df["labeled"] = np.nan, False
        frames.append(df)
        summary.append({"year": t, **graph_summary(snap.G), **{f"part_{k}": v for k, v in part_diag[t].items()},
                        "n_candidates": len(df), "n_positives": int(np.nansum(df["y"])) if len(df) else 0})
        e = snap.edges
        if len(e):
            ce = pd.DataFrame({"a": e["u"].map(snap.membership), "b": e["v"].map(snap.membership), "w": e["weight"]})
            ce = ce[ce["a"] != ce["b"]]
            ce = pd.DataFrame({"year": t, "comm_a": np.minimum(ce["a"], ce["b"]), "comm_b": np.maximum(ce["a"], ce["b"]), "w": ce["w"]})
            cedge_rows.append(ce.groupby(["year", "comm_a", "comm_b"]).agg(n_edges=("w", "size"), weight=("w", "sum")).reset_index())
        lin = lineage.get(t, {})
        for c in evo_t["community_id"]:
            for a in snap.members[c]:
                mem_rows.append((t, a, int(c), int(lin[c])))
        log.info("cutoff %d: %d candidates, %d positives", t, len(df), summary[-1]["n_positives"])
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError("no candidates were generated; check the data audit and candidate settings")
    features = pd.concat(frames, ignore_index=True)
    features.to_parquet(art / "features.parquet", index=False)
    evo.to_parquet(art / "evolution.parquet", index=False)
    mem_df = pd.DataFrame(mem_rows, columns=["year", "author_id", "community_id", "lineage_id"])
    mem_df.to_parquet(art / "memberships.parquet", index=False)
    (pd.concat(cedge_rows, ignore_index=True) if cedge_rows else pd.DataFrame(columns=["year", "comm_a", "comm_b", "n_edges", "weight"])
     ).to_parquet(art / "community_edges.parquet", index=False)
    pd.DataFrame(summary).to_csv(metrics_dir / "snapshot_summary.csv", index=False)
    last = max(memberships)
    Gl = graph_from_edges(snapshot_nodes(fb.authorships, last), snapshot_edges(pair_table, last), last)
    write_json(community_stability(Gl, (0, 1, 2), cfg.graph.community_algorithm, cfg.graph.community_resolution),
               metrics_dir / "community_stability.json")
    return {"features": features, "evolution": evo, "memberships": memberships, "pair_table": pair_table, "fb": fb}


def load_build(cfg: Config) -> Dict[str, pd.DataFrame]:
    d = cfg.artifacts_dir / "features"
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in ("features", "evolution", "memberships", "community_edges")}


# ============================================================================= stage 4: train + evaluate
def _fit_bayes(df: pd.DataFrame, cols: List[str], prior: float) -> BayesianLogisticRegression:
    return BayesianLogisticRegression(prior_scale=prior).fit(prepare_matrix(df, cols), df["y"].to_numpy(dtype=int), cols)


def _fit_lr(df: pd.DataFrame, cols: List[str], C: float, seed: int) -> LogisticModel:
    return LogisticModel(C=C, seed=seed).fit(prepare_matrix(df, cols), df["y"].to_numpy(dtype=int), cols)


def _row(df: pd.DataFrame, score: np.ndarray, model: str, phase: str, cfg: Config, prob: bool, boots: int) -> Dict[str, Any]:
    ks = list(cfg.evaluation.top_k)
    return {"phase": phase, "model": model,
            **evaluate_scores(df["y"].to_numpy(dtype=int), score, ks, boots, cfg.project.random_seed, prob, df["cutoff"].to_numpy())}


def stage_train(cfg: Config, features: pd.DataFrame, extra: Optional[Dict[str, Any]] = None
                ) -> Tuple[Dict[str, Any], BayesianLogisticRegression, List[str]]:
    """Chronological protocol: tune on validation, refit on all labels known at test start, report test.

    Returns ``(results, final_model, columns)``; ``final_model`` is refit on ALL labelled cutoffs (labels known at
    the prospective cutoff) for the real shortlist.
    """
    ev, seed = cfg.evaluation, cfg.project.random_seed
    lab = features[features["labeled"]].copy()
    lab["y"] = lab["y"].astype(int)
    sp = make_splits(sorted(int(c) for c in lab["cutoff"].unique()), cfg.horizon, ev.val_first, ev.test_first, ev.purge)
    part = {k: lab[lab["cutoff"].isin(v)] for k, v in sp.items()}
    for k in ("train", "refit"):
        if part[k]["y"].nunique() < 2:
            raise RuntimeError(f"'{k}' split has a single label class; widen data window / horizon / candidates")
    cols, groups = feature_columns(), feature_groups()
    log.info("splits: %s | base rates train=%.4f val=%.4f test=%.4f", sp,
             part["train"]["y"].mean(), part["val"]["y"].mean(), part["test"]["y"].mean())

    # ---- hyper-parameter selection on validation (AUPRC)
    def val_ap(model_fn) -> float:
        m = model_fn(); s = m.predict_proba(prepare_matrix(part["val"], cols))
        return evaluate_scores(part["val"]["y"].to_numpy(), s, [10], 10, seed)["auprc"]
    prior_scores = {pr: val_ap(lambda pr=pr: _fit_bayes(part["train"], cols, pr)) for pr in ev.prior_scale_grid}
    C_scores = {C: val_ap(lambda C=C: _fit_lr(part["train"], cols, C, seed)) for C in ev.logistic_C_grid}
    prior_star = max(prior_scores, key=lambda k: (np.nan_to_num(prior_scores[k], nan=-1), -k))
    C_star = max(C_scores, key=lambda k: (np.nan_to_num(C_scores[k], nan=-1), -k))
    log.info("selected prior_scale=%s (val AP %s), C=%s", prior_star, prior_scores, C_star)

    # ---- model variants (ablations share the selected prior)
    variants: Dict[str, List[str]] = {"bayes_full": cols}
    for g in GROUPS:
        rest = [c for c in cols if c not in groups[g]]
        if rest:
            variants[f"bayes_minus_{g}"] = rest
    variants["bayes_minus_temporal_evolution"] = [c for c in cols if c not in groups["temporal"] + groups["evolution"]]
    variants["bayes_semantic_only"] = groups["semantic"]

    def evaluate_phase(phase: str, fit_df: pd.DataFrame, eval_df: pd.DataFrame, boots: int, ci_names=None):
        """Bootstrap CIs are the expensive part, so only models in ``ci_names`` get them (None = all)."""
        rows, scores = [], {}
        nb = lambda name: boots if (ci_names is None or name in ci_names) else 0
        for name, s in baseline_scores(eval_df, seed).items():
            rows.append(_row(eval_df, s, name, phase, cfg, False, nb(name))); scores[name] = s
        lr = _fit_lr(fit_df, cols, C_star, seed)
        scores["logistic_balanced"] = lr.predict_proba(prepare_matrix(eval_df, cols))
        rows.append(_row(eval_df, scores["logistic_balanced"], "logistic_balanced", phase, cfg, True, nb("logistic_balanced")))
        models = {}
        for name, vc in variants.items():
            if not vc:
                continue
            m = _fit_bayes(fit_df, vc, prior_star)
            models[name] = m
            scores[name] = m.predict_proba(prepare_matrix(eval_df, vc))
            rows.append(_row(eval_df, scores[name], name, phase, cfg, True, nb(name)))
        return rows, scores, models

    # validation is for model selection only: no bootstrap there
    val_rows, _, _ = evaluate_phase("val", part["train"], part["val"], 0)
    base_names = ["preferential_attachment", "common_neighbours", "adamic_adar", "jaccard", "semantic_only"]
    val_ap_by_name = {r["model"]: r["auprc"] for r in val_rows if r["model"] in base_names}
    best_base = max(base_names, key=lambda n: np.nan_to_num(val_ap_by_name.get(n, -1.0), nan=-1.0))  # chosen on VAL, not test
    ci_names = {best_base, "semantic_only", "random", "logistic_balanced", "bayes_full", "bayes_semantic_only",
                "bayes_minus_temporal_evolution"}
    test_rows, test_scores, test_models = evaluate_phase("test", part["refit"], part["test"], ev.n_bootstrap, ci_names)
    ytest = part["test"]["y"].to_numpy()

    # ---- hypothesis comparisons on the test period (paired bootstrap on AUPRC)
    pairs = {"H1_full_vs_best_baseline(" + best_base + ")": ("bayes_full", best_base),
             "H1_full_vs_semantic_only_baseline": ("bayes_full", "semantic_only"),
             "H1_full_vs_semantic_only_model": ("bayes_full", "bayes_semantic_only"),
             "H1_full_vs_minus_collaboration": ("bayes_full", "bayes_minus_collaboration"),
             "H2_full_vs_minus_temporal_evolution": ("bayes_full", "bayes_minus_temporal_evolution"),
             "H2_full_vs_minus_evolution": ("bayes_full", "bayes_minus_evolution"),
             "H2_full_vs_minus_temporal": ("bayes_full", "bayes_minus_temporal"),
             "full_vs_logistic_balanced": ("bayes_full", "logistic_balanced")}
    comparisons = {k: paired_bootstrap_diff(ytest, test_scores[a], test_scores[b], ev.n_bootstrap, seed)
                   for k, (a, b) in pairs.items() if a in test_scores and b in test_scores}
    cal = calibration_table(ytest, test_scores["bayes_full"]).assign(model="bayes_full")
    cal_lr = calibration_table(ytest, test_scores["logistic_balanced"]).assign(model="logistic_balanced")
    coefs = test_models["bayes_full"].coef_table()

    results: Dict[str, Any] = {
        "horizon_years": cfg.horizon, "splits": sp, "selected": {"prior_scale": prior_star, "logistic_C": C_star},
        "val_auprc_by_prior": {str(k): v for k, v in prior_scores.items()}, "val_auprc_by_C": {str(k): v for k, v in C_scores.items()},
        "base_rates": {k: float(v["y"].mean()) for k, v in part.items()},
        "n_rows": {k: int(len(v)) for k, v in part.items()}, "metrics_val": val_rows, "metrics_test": test_rows,
        "comparisons": comparisons, "best_baseline_on_val": best_base,
        "calibration_note": "Bayesian model is NOT re-weighted; the balanced logistic model's probabilities are intentionally distorted.",
        "convergence": {n: bool(m.converged_) for n, m in test_models.items()}}
    if extra and extra.get("trial_target") is not None:
        results["secondary_trial_target"] = extra["trial_target"]
    out = ensure_dir(cfg.artifacts_dir / "metrics")
    write_json(results, out / "results.json")
    pd.DataFrame(test_rows).to_csv(out / "metrics_test.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "metrics_val.csv", index=False)
    pd.concat([cal, cal_lr]).to_csv(out / "calibration_test.csv", index=False)
    coefs.to_csv(out / "coefficients.csv", index=False)

    final = _fit_bayes(lab, cols, prior_star)  # all labels known at the prospective cutoff
    ensure_dir(cfg.artifacts_dir / "model")
    with open(cfg.artifacts_dir / "model" / "bayes_final.pkl", "wb") as fh:
        pickle.dump({"model": final, "columns": cols, "prior_scale": prior_star}, fh)
    return results, final, cols


# ============================================================================= stage 5: shortlist, evidence, simulation, model card
def stage_shortlist(cfg: Config, tables: Dict[str, Any], build: Dict[str, Any], model: BayesianLogisticRegression,
                    cols: List[str]) -> pd.DataFrame:
    """Rank prospective candidates, write evidence reports and the counterfactual simulation."""
    feats = build["features"]
    pro = feats[~feats["labeled"]].reset_index(drop=True)
    if pro.empty:
        raise RuntimeError("no prospective candidates (cutoff = end_year) were generated")
    X = prepare_matrix(pro, cols)
    p = model.predict_proba(X)
    lo, hi = model.predict_interval(X, 0.9)
    order = np.lexsort((pro["v"].to_numpy(), pro["u"].to_numpy(), -p))
    top = order[:cfg.evaluation.shortlist_size]
    contrib = model.contributions(X[top])
    summary = []
    for i in range(len(top)):
        c = contrib.iloc[i]
        best = c.reindex(c.abs().sort_values(ascending=False).index)[:5]
        summary.append("; ".join(f"{k}={v:+.2f}" for k, v in best.items()))
    names = dict(zip(tables["authors"]["author_id"], tables["authors"]["display_name"]))
    sl = pro.iloc[top].copy()
    sl["prob"], sl["prob_lo"], sl["prob_hi"] = p[top], lo[top], hi[top]
    sl["rank"] = np.arange(1, len(sl) + 1)
    sl["name_u"], sl["name_v"] = sl["u"].map(names), sl["v"].map(names)
    sl["top_contributions"] = summary
    keep = ["rank", "cutoff", "u", "v", "name_u", "name_v", "comm_u", "comm_v", "source", "prob", "prob_lo", "prob_hi",
            "top_contributions", "semantic_author", "semantic_comm", "common_neighbors", "citation_pair", "shared_trials"]
    sl = sl[keep].reset_index(drop=True)
    out = ensure_dir(cfg.artifacts_dir / "shortlist")
    sl.to_parquet(out / "shortlist.parquet", index=False)
    sl.to_csv(out / "shortlist.csv", index=False)

    reporter = EvidenceReporter(tables, build["pair_table"])
    reports = [reporter.report(r) for _, r in sl.head(25).iterrows()]
    write_json(reports, out / "evidence_reports.json")
    ensure_dir(out / "markdown")
    for rep in reports[:10]:
        (out / "markdown" / f"rank_{rep['rank']:03d}.md").write_text(to_markdown(rep), encoding="utf-8")

    # ---- counterfactual simulation on the prospective snapshot
    t = cfg.prospective_cutoff
    G = graph_from_edges(snapshot_nodes(build["fb"].authorships, t), snapshot_edges(build["pair_table"], t), t)
    membership = build["memberships"][t]
    sim = GraphSimulator(G, membership, cfg.simulation.max_nodes, cfg.project.random_seed)
    s = cfg.simulation
    runs = sim.run(list(zip(sl["u"], sl["v"])), list(zip(pro["u"], pro["v"])), s.k_edges, s.n_replicates,
                   s.ic_probability, s.n_seed_nodes, s.n_cascades)
    runs.to_csv(ensure_dir(cfg.artifacts_dir / "simulation") / "simulation_runs.csv", index=False)
    summarise_simulation(runs).to_csv(cfg.artifacts_dir / "simulation" / "simulation_summary.csv", index=False)
    return sl


def write_model_card(cfg: Config, results: Dict[str, Any], audit: Dict[str, Any]) -> Path:
    test = pd.DataFrame(results["metrics_test"]).set_index("model")
    cols = [c for c in ("n", "n_pos", "base_rate", "auprc", "roc_auc", "precision_at_10", "precision_at_50", "brier", "ece") if c in test.columns]
    comps = "\n".join(f"- **{k}**: dAUPRC = {v['diff_auprc']:+.4f} (95% CI {v['ci_lo']:+.4f} to {v['ci_hi']:+.4f})"
                      for k, v in results["comparisons"].items())
    text = f"""# BRIDGE-CT model card

**Task.** Rank author pairs from different collaboration communities by the probability of a first co-authorship
within {cfg.horizon} years. Disease scope: *{cfg.project.disease}*; data window {cfg.project.start_year}-{cfg.project.end_year}.

**Model.** Bayesian logistic regression (Laplace approximation), prior scale {results['selected']['prior_scale']},
standardised features computed from snapshots that use only records available at each cutoff.

**Evaluation.** Chronological split with label-horizon purging: {json.dumps(results['splits'])}.
Test base rate {results['base_rates']['test']:.4f}.

## Test metrics
{test[cols].round(4).to_markdown() if hasattr(test, 'to_markdown') and _has_tabulate() else test[cols].round(4).to_string()}

## Paired comparisons (AUPRC difference, bootstrap)
{comps}

## Data audit warnings
{chr(10).join('- ' + w for w in audit.get('warnings', [])) or '- none'}

## Intended use and limits
Research analysis of publicly recorded collaboration structure. NOT clinical decision support. Scores indicate
statistical association with future co-authorship, not scientific merit, clinical promise or causal benefit of a
collaboration. Candidate generation, "unobserved = negative" labels, database coverage bias (large institutions,
English-language, high-volume topics) and author-ID errors all limit validity. Bootstrap intervals resample
candidate rows and understate dependence within a snapshot.
"""
    path = ensure_dir(cfg.artifacts_dir / "model_cards") / "model_card.md"
    path.write_text(text, encoding="utf-8")
    return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================= orchestration
def stage_all_after_normalize(cfg: Config, tables: Dict[str, Any]) -> Dict[str, Any]:
    """build -> (secondary target) -> train -> shortlist -> model card."""
    build = stage_build(cfg, tables)
    extra: Dict[str, Any] = {}
    tt = build_trial_target(build["fb"], build["evolution"], build["memberships"], cfg)
    extra["trial_target"] = evaluate_trial_target(tt, cfg) if tt is not None else {"status": "skipped: too few identifier-based links"}
    results, model, cols = stage_train(cfg, build["features"], extra)
    sl = stage_shortlist(cfg, tables, build, model, cols)
    audit = read_json(cfg.artifacts_dir / "metrics" / "data_audit.json") if (cfg.artifacts_dir / "metrics" / "data_audit.json").exists() else {}
    write_model_card(cfg, results, audit)
    return {"results": results, "shortlist": sl}


def run_demo(cfg: Config, seed: int = 7, **synthetic_kwargs: Any) -> Dict[str, Any]:
    """End-to-end run on SYNTHETIC data (offline). Verifies the code path, not the science."""
    from .synthetic import generate_synthetic
    ev = cfg.evaluation.model_copy(update={"n_bootstrap": min(cfg.evaluation.n_bootstrap, 100)})
    cfg = cfg.model_copy(update={"evaluation": ev})
    tables = generate_synthetic(seed=seed, start_year=cfg.project.start_year, end_year=cfg.project.end_year, **synthetic_kwargs)
    save_tables(tables, cfg)
    ensure_dir(cfg.artifacts_dir / "metrics")
    write_json(data_audit(tables), cfg.artifacts_dir / "metrics" / "data_audit.json")
    return stage_all_after_normalize(cfg, tables)