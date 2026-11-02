"""Plain-language interpretation of BRIDGE-CT results.

Everything here is RULE-BASED and grounded in numbers that the pipeline wrote to disk: no language model is
involved, so every sentence can be traced to a metric. Verdict rules are deliberately conservative and are
documented next to each finding. This module has no Streamlit dependency (it is unit-tested).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASELINE_NAMES = {
    "common_neighbours": "the \"friends-of-friends\" rule (share many collaborators)",
    "preferential_attachment": "the \"popular authors attract more links\" rule",
    "adamic_adar": "the weighted friends-of-friends rule",
    "jaccard": "the \"overlapping circles\" rule",
    "semantic_only": "the \"similar topics\" rule",
}

_BASE_HELP: Dict[str, Tuple[str, str]] = {
    "degree": ("Collaborators", "how many different people the author has published with"),
    "wdeg": ("Collaboration volume", "total strength of the author's co-authorships"),
    "betweenness": ("Broker role", "how often the author sits on shortest paths between other researchers"),
    "cross_ratio": ("Outward ties", "share of the author's collaborations that reach into other communities"),
    "comm_broker": ("Community openness", "how outward-looking the author's whole community is"),
    "degree_delta": ("Collaborator growth", "new collaborators gained since last year"),
    "cross_ratio_delta": ("Opening up", "change in the share of cross-community ties since last year"),
    "comm_size": ("Community size", "number of authors in the author's community"),
    "comm_age": ("Community age", "how many years the community lineage has existed"),
    "comm_growth": ("Community growth", "how fast the community grew versus last year"),
    "comm_turnover": ("Community turnover", "how much the community membership changed versus last year"),
    "papers_total": ("Career output", "papers published up to the cutoff"),
    "papers_recent": ("Recent output", "papers published in the last few years"),
    "years_since_last": ("Time since last paper", "years since the author last published"),
    "trial_papers": ("Trial-linked papers", "papers linked to registered clinical trials"),
    "common_neighbors": ("Shared collaborators", "people both authors have already published with"),
    "jaccard": ("Circle overlap", "overlap between the two authors' collaborator circles"),
    "adamic_adar": ("Weighted shared collaborators", "shared collaborators, weighting rarer (less-connected) ones more"),
    "resource_alloc": ("Resource allocation", "shared collaborators weighted by how busy they are"),
    "pref_attach": ("Popularity product", "number of collaborators of one author times the other's"),
    "cross_edge_count": ("Ties between the two communities", "existing collaborations linking the two communities"),
    "cross_edge_density": ("Tie density between communities", "existing cross links relative to community sizes"),
    "semantic_author": ("Topic similarity (authors)", "how similar the content of the two authors' papers is"),
    "semantic_comm": ("Topic similarity (communities)", "how similar the two communities' research content is"),
    "shared_features": ("Shared terms/topics", "number of topic or vocabulary features both authors use"),
    "citation_pair": ("Mutual citations", "citations between the two authors' papers"),
    "shared_trials": ("Shared trials", "clinical trials both authors' papers are linked to"),
}


def describe_feature(col: str) -> Tuple[str, str]:
    """``(short label, plain meaning)`` for a model column such as ``degree_min`` or ``semantic_author``."""
    side = ""
    base = col
    if col.endswith("_min"):
        base, side = col[:-4], " (lower of the two)"
    elif col.endswith("_max"):
        base, side = col[:-4], " (higher of the two)"
    label, meaning = _BASE_HELP.get(base, (base.replace("_", " ").title(), "model feature"))
    return label + side, meaning


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _finding(key: str, title: str, verdict: str, headline: str, plain: str, technical: str, caveat: str) -> Dict[str, str]:
    return {"key": key, "title": title, "verdict": verdict, "headline": headline, "plain": plain,
            "technical": technical, "caveat": caveat}


def _cmp(results: Dict[str, Any], prefix: str) -> Optional[Tuple[str, Dict[str, float]]]:
    for k, v in results.get("comparisons", {}).items():
        if k.startswith(prefix):
            return k, v
    return None


def _ci_verdict(c: Dict[str, float]) -> str:
    """supported: the 95% interval of the AUPRC gain is entirely above 0; not_supported: entirely below."""
    lo, hi = c.get("ci_lo", np.nan), c.get("ci_hi", np.nan)
    if np.isnan(lo) or np.isnan(hi):
        return "inconclusive"
    return "supported" if lo > 0 else ("not_supported" if hi < 0 else "inconclusive")


def _metric(results: Dict[str, Any], model: str, key: str, phase: str = "metrics_test") -> float:
    for r in results.get(phase, []):
        if r["model"] == model:
            return float(r.get(key, np.nan))
    return float("nan")


def explain_results(results: Dict[str, Any], audit: Optional[Dict[str, Any]] = None,
                    sim_runs: Optional[pd.DataFrame] = None) -> List[Dict[str, str]]:
    """Turn ``results.json`` (+ audit, simulation runs) into a list of findings with verdicts.

    Verdict vocabulary: ``supported``, ``not_supported``, ``inconclusive``, ``warning``, ``info``.
    """
    out: List[Dict[str, str]] = []
    H = results.get("horizon_years")
    h = f"{H} years" if H else "the prediction horizon"
    base_rate = results.get("base_rates", {}).get("test", float("nan"))

    # ---------- H1
    c1 = _cmp(results, "H1_full_vs_best_baseline")
    if c1:
        key, c = c1
        name = key[key.index("(") + 1:-1]
        nice = BASELINE_NAMES.get(name, name)
        v = _ci_verdict(c)
        gain = f"{c['diff_auprc']:+.3f}"
        txt = {"supported": f"The model finds future collaborations better than {nice}.",
               "not_supported": f"The model is WORSE than {nice}.",
               "inconclusive": f"The model is not clearly better or worse than {nice}."}[v]
        out.append(_finding(
            "H1", "H1 - Do network features beat simple rules?", v, txt,
            f"We compared the full model with the best simple rule chosen on validation data ({nice}). "
            f"The accuracy-for-rare-events score (AUPRC) changed by {gain}. "
            + ("That gain is large enough that it is unlikely to be luck." if v == "supported" else
               "The gap is too small or too noisy to call." if v == "inconclusive" else "A simple rule did better."),
            f"AUPRC difference {c['diff_auprc']:+.4f}, 95% bootstrap CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}], "
            f"P(model better) = {c['p_a_better']:.2f}. Baseline picked by validation AUPRC, not test.",
            "Row-level bootstrap ignores dependence inside a snapshot, so the interval is optimistic. Check that the gain "
            "holds in each test year."))
    c1b = _cmp(results, "H1_full_vs_semantic_only_baseline")
    if c1b:
        _, c = c1b
        v = _ci_verdict(c)
        out.append(_finding(
            "H1b", "H1 - Does it beat \"similar topics\" alone?", v,
            {"supported": "Yes: network structure adds information beyond text similarity.",
             "not_supported": "No: text similarity alone did better.",
             "inconclusive": "Unclear: text similarity alone is about as good."}[v],
            "This is the key novelty question: do collaboration-network clues tell us more than just \"they work on similar things\"?",
            f"AUPRC diff vs semantic-only {c['diff_auprc']:+.4f}, CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}].",
            "Semantic similarity here is TF-IDF/topic cosine; a stronger text model might narrow the gap."))

    # ---------- H2
    c2 = _cmp(results, "H2_full_vs_minus_temporal_evolution")
    if c2:
        _, c = c2
        v = _ci_verdict(c)
        out.append(_finding(
            "H2", "H2 - Does change over time matter?", v,
            {"supported": "Yes: how communities and brokers change over time improves predictions.",
             "not_supported": "No: dropping the time-change features made predictions better.",
             "inconclusive": "Not clearly: time-change features add little beyond static measures."}[v],
            "We removed the features that describe how communities grew, split, merged and how authors' roles shifted year to year, "
            "and checked whether predictions got worse.",
            f"AUPRC diff (full - without temporal/evolution) {c['diff_auprc']:+.4f}, CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}].",
            "Ablations reuse the prior chosen for the full model; synthetic or short windows give weak evolution signals."))

    # ---------- H3
    brier, brier_base = _metric(results, "bayes_full", "brier"), _metric(results, "bayes_full", "brier_base_rate")
    ece, ece_lr = _metric(results, "bayes_full", "ece"), _metric(results, "logistic_balanced", "ece")
    if not np.isnan(brier):
        good = brier < brier_base and ece < 0.02
        v = "supported" if good else ("inconclusive" if brier < brier_base else "not_supported")
        out.append(_finding(
            "H3", "H3 - Can the probabilities be trusted?", v,
            "The probabilities are well calibrated." if good else "Probabilities are only partly reliable.",
            f"When the model says \"{_pct(0.1)}\", roughly that share of such pairs should really collaborate. "
            f"Average calibration error is {_pct(ece, 2)}"
            + (f" (the re-weighted logistic model is off by {_pct(ece_lr, 1)}, which is why we show the Bayesian one)." if not np.isnan(ece_lr) else "."),
            f"Brier {brier:.4f} vs constant-base-rate Brier {brier_base:.4f}; ECE {ece:.4f}. Rule used: supported if Brier beats the "
            f"base-rate predictor and ECE < 0.02 (rule of thumb).",
            "Calibration is measured on the test period; if the base rate drifts between periods probabilities drift too "
            f"(train base rate {results.get('base_rates', {}).get('train', float('nan')):.4f}, test {base_rate:.4f})."))

    # ---------- practical value
    p10 = _metric(results, "bayes_full", "precision_at_10_per_cutoff")
    if np.isnan(p10):
        p10 = _metric(results, "bayes_full", "precision_at_10")
    if not np.isnan(p10) and base_rate and not np.isnan(base_rate):
        lift = p10 / base_rate if base_rate > 0 else float("nan")
        out.append(_finding(
            "VALUE", "What would a reviewer gain?", "info",
            f"About {p10 * 10:.1f} of the top 10 suggested pairs are expected to start collaborating within {h}.",
            f"A random candidate has a {_pct(base_rate)} chance. The top of the list has about {_pct(p10)}, "
            f"roughly {lift:.0f} times better. That is the value of reading only the top of the shortlist.",
            f"Precision@10 (per-snapshot mean) {p10:.3f}; test base rate {base_rate:.4f}; lift {lift:.1f}x.",
            "A pair that collaborates is not automatically a good or beneficial collaboration."))

    # ---------- H4
    if sim_runs is not None and len(sim_runs):
        piv = sim_runs.pivot(index="replicate", columns="strategy", values="coverage")
        base_cov = float(piv["baseline"].mean()) if "baseline" in piv else float("nan")
        if base_cov > 0.95:
            out.append(_finding("H4", "H4 - Do predicted bridges help the network?", "inconclusive",
                                "Test not informative: the network is already almost fully reachable.",
                                "Information could already reach nearly everyone, so extra links cannot show a visible gain.",
                                f"Baseline diffusion coverage {base_cov:.3f} (> 0.95 saturation).",
                                "Lower ic_probability or use a sparser snapshot. Graph simulation only; not a clinical effect."))
        elif "model" in piv:
            res = {}
            for other in ("random", "degree", "betweenness"):
                if other in piv:
                    d = (piv["model"] - piv[other]).to_numpy()
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                    res[other] = (d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se)
            beats = [k for k, (m, lo, hi) in res.items() if lo > 0]
            v = "supported" if len(beats) == len(res) and res else ("inconclusive" if any(m > 0 for m, _, _ in res.values()) else "not_supported")
            detail = "; ".join(f"vs {k}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for k, (m, lo, hi) in res.items())
            out.append(_finding(
                "H4", "H4 - Do predicted bridges help the network?", v,
                {"supported": "In simulation, the predicted links spread information further than random, popular-author or broker-based links.",
                 "inconclusive": "In simulation, predicted links help somewhat but not clearly more than every alternative.",
                 "not_supported": "In simulation, predicted links did not beat simple alternatives."}[v],
                "We pretended the top predicted collaborations existed and simulated news spreading through the network, then compared with "
                "adding the same number of random links, links between very popular authors, and links between brokers.",
                f"Mean paired coverage difference (model minus alternative) with 95% CI: {detail}.",
                "This is a graph simulation under stated assumptions (independent cascade). It says nothing about clinical benefit."))

    # ---------- reliability and data quality
    n_pos = _metric(results, "bayes_full", "n_pos")
    if not np.isnan(n_pos) and n_pos < 30:
        out.append(_finding("N", "Sample size", "warning", f"Only {int(n_pos)} positive test cases.",
                            "Too few real future collaborations in the test period to trust fine differences between models.",
                            f"n_pos (test) = {int(n_pos)}.", "Widen the window or add a second disease area."))
    for w in (audit or {}).get("warnings", []):
        out.append(_finding("DATA", "Data warning", "warning", w, w, w, "Resolve before drawing conclusions."))
    tt = results.get("secondary_trial_target")
    if tt and str(tt.get("status", "")).startswith("skipped"):
        out.append(_finding("TRIAL", "Trial-link target", "info", "Secondary trial target was skipped.",
                            "There were not enough reliable paper-to-trial links (or only one outcome class), so only co-authorship is predicted.",
                            str(tt.get("status")), "As the blueprint recommends: do not rely on trial links without a feasibility audit."))
    return out


def overall_summary(findings: List[Dict[str, str]]) -> str:
    """One-sentence headline across all hypothesis findings."""
    hyp = [f for f in findings if f["key"] in ("H1", "H1b", "H2", "H3", "H4")]
    n_ok = sum(f["verdict"] == "supported" for f in hyp)
    n_bad = sum(f["verdict"] == "not_supported" for f in hyp)
    if not hyp:
        return "No hypothesis results available yet."
    return (f"{n_ok} of {len(hyp)} hypothesis checks are supported, {n_bad} contradicted, "
            f"{len(hyp) - n_ok - n_bad} inconclusive. Negative and inconclusive results are valid findings.")


_CONTRIB = re.compile(r"([A-Za-z_0-9]+)=([+-]\d+(?:\.\d+)?)")


def explain_pair(row: Dict[str, Any], base_rate: Optional[float] = None, horizon: Optional[int] = None) -> Dict[str, Any]:
    """Explain one shortlisted pair: headline, reasons for, reasons against, concrete facts, caution."""
    prob, lo, hi = float(row["prob"]), float(row["prob_lo"]), float(row["prob_hi"])
    h = f"{horizon} years" if horizon else "the prediction horizon"
    head = (f"Estimated {_pct(prob)} chance (90% interval {_pct(lo)} to {_pct(hi)}) that these two authors publish together "
            f"for the first time within {h}")
    if base_rate and base_rate > 0:
        head += f", about {prob / base_rate:.1f}x the average candidate."
    else:
        head += "."
    contribs = [(c, float(v)) for c, v in _CONTRIB.findall(str(row.get("top_contributions", "")))]
    f_for = [{"label": describe_feature(c)[0], "meaning": describe_feature(c)[1], "effect": v} for c, v in contribs if v > 0]
    f_against = [{"label": describe_feature(c)[0], "meaning": describe_feature(c)[1], "effect": v} for c, v in contribs if v < 0]
    facts = []
    if row.get("common_neighbors", 0) and float(row["common_neighbors"]) > 0:
        facts.append(f"They already share {int(row['common_neighbors'])} collaborator(s).")
    else:
        facts.append("They share no collaborators yet, so any link would be a genuine new bridge.")
    sa = float(row.get("semantic_author", 0) or 0)
    facts.append(f"Topic similarity of their papers is {sa:.2f} on a 0-1 scale" + (" (high)." if sa >= 0.4 else " (moderate)." if sa >= 0.15 else " (low)."))
    if row.get("citation_pair", 0) and float(row["citation_pair"]) > 0:
        facts.append(f"Their papers cite each other {int(row['citation_pair'])} time(s).")
    if row.get("shared_trials", 0) and float(row["shared_trials"]) > 0:
        facts.append(f"Both are linked to {int(row['shared_trials'])} common clinical trial(s).")
    width = hi - lo
    caution = ("The interval is wide: treat the ranking as uncertain." if prob > 0 and width / max(prob, 1e-9) > 1.0 else
               "The interval is fairly tight, but it only reflects parameter uncertainty, not data bias.")
    return {"headline": head, "reasons_for": f_for, "reasons_against": f_against, "facts": facts,
            "caution": caution + " This is a structural statistical estimate, not a recommendation to collaborate."}