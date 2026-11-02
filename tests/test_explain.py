import pandas as pd

from bridge_ct.explain import describe_feature, explain_pair, explain_results, overall_summary


def _results(lo, hi, diff=0.05):
    comp = {"diff_auprc": diff, "ci_lo": lo, "ci_hi": hi, "p_a_better": 0.97}
    row = {"model": "bayes_full", "brier": 0.019, "brier_base_rate": 0.0205, "ece": 0.003, "n_pos": 200,
           "precision_at_10_per_cutoff": 0.4}
    return {"horizon_years": 3, "base_rates": {"train": 0.02, "test": 0.02},
            "comparisons": {"H1_full_vs_best_baseline(common_neighbours)": comp, "H2_full_vs_minus_temporal_evolution": comp},
            "metrics_test": [row, {"model": "logistic_balanced", "ece": 0.2}]}


def test_verdict_rules():
    f = {x["key"]: x for x in explain_results(_results(0.01, 0.09))}
    assert f["H1"]["verdict"] == "supported" and f["H3"]["verdict"] == "supported"
    f = {x["key"]: x for x in explain_results(_results(-0.02, 0.09))}
    assert f["H1"]["verdict"] == "inconclusive"
    f = {x["key"]: x for x in explain_results(_results(-0.09, -0.01, -0.05))}
    assert f["H1"]["verdict"] == "not_supported" and "WORSE" in f["H1"]["headline"]


def test_h4_saturation_and_win():
    rows = []
    for r in range(10):
        for s, cov in (("baseline", 0.99), ("model", 0.995), ("random", 0.994), ("degree", 0.994), ("betweenness", 0.994)):
            rows.append({"replicate": r, "strategy": s, "coverage": cov})
    f = {x["key"]: x for x in explain_results(_results(0.01, 0.09), sim_runs=pd.DataFrame(rows))}
    assert f["H4"]["verdict"] == "inconclusive" and "already" in f["H4"]["headline"]
    rows = [{"replicate": r, "strategy": s, "coverage": c + 0.001 * (r % 3)} for r in range(10)
            for s, c in (("baseline", 0.3), ("model", 0.5), ("random", 0.35), ("degree", 0.4), ("betweenness", 0.42))]
    f = {x["key"]: x for x in explain_results(_results(0.01, 0.09), sim_runs=pd.DataFrame(rows))}
    assert f["H4"]["verdict"] == "supported"


def test_overall_and_warnings():
    fs = explain_results(_results(0.01, 0.09), audit={"warnings": ["few links"]})
    assert any(x["key"] == "DATA" for x in fs) and "supported" in overall_summary(fs)


def test_pair_explanation_and_feature_help():
    assert describe_feature("degree_min")[0] == "Collaborators (lower of the two)"
    e = explain_pair({"prob": 0.2, "prob_lo": 0.1, "prob_hi": 0.3, "top_contributions": "semantic_author=+0.80; comm_age_max=-0.30",
                      "common_neighbors": 0, "semantic_author": 0.5}, base_rate=0.02, horizon=3)
    assert "10.0x" in e["headline"] and e["reasons_for"][0]["label"].startswith("Topic similarity")
    assert e["reasons_against"] and any("no collaborators" in f for f in e["facts"])