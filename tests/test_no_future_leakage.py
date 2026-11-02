"""Strongest leakage test: features for cutoff t must be IDENTICAL whether or not post-t data exists."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bridge_ct.features import FeatureBuilder, feature_columns
from bridge_ct.graph.collab import build_pair_table
from bridge_ct.graph.communities import community_members
from bridge_ct.graph.evolution import track_evolution

T = 2010


def truncate(tables, t):
    out = dict(tables)
    ok = tables["works"]["publication_year"] <= t
    ids = set(tables["works"].loc[ok, "work_id"])
    out["works"] = tables["works"][ok]
    out["authorships"] = tables["authorships"][tables["authorships"]["work_id"].isin(ids)]
    out["work_topics"] = tables["work_topics"][tables["work_topics"]["work_id"].isin(ids)]
    out["citations"] = tables["citations"][tables["citations"]["work_id"].isin(ids)]
    out["trials"] = tables["trials"][tables["trials"]["registration_year"] <= t]
    out["paper_trial_links"] = tables["paper_trial_links"][tables["paper_trial_links"]["work_id"].isin(ids)]
    return out


def features_at(tables, cfg, t):
    pt, _ = build_pair_table(tables["authorships"])
    fb = FeatureBuilder(tables, pt, cfg)
    mem, by = {}, {}
    for y in range(2006, t + 1):
        mem[y], _ = fb.partition(y); by[y] = community_members(mem[y])
    evo, _ = track_evolution(by, cfg.graph.min_community_size)
    snap = fb.snapshot(t, mem[t], mem[t - 1])
    return fb, snap, fb.build_features(snap, evo[evo["year"] == t])


def test_features_identical_without_future_data(tiny_cfg, tiny_tables):
    _, _, full = features_at(tiny_tables, tiny_cfg, T)
    _, _, cut = features_at(truncate(tiny_tables, T), tiny_cfg, T)
    assert len(full) > 0
    assert_frame_equal(full.reset_index(drop=True), cut.reset_index(drop=True))


def test_candidates_unlinked_and_cross_community(tiny_cfg, tiny_tables):
    fb, snap, df = features_at(tiny_tables, tiny_cfg, T)
    assert all(not snap.G.has_edge(u, v) for u, v in zip(df["u"], df["v"]))
    assert (df["comm_u"] != df["comm_v"]).all() and (df["u"] < df["v"]).all()
    assert set(feature_columns()) <= set(df.columns) and np.isfinite(df[feature_columns()].to_numpy(float)).all()


def test_labels_only_count_future_window(tiny_cfg, tiny_tables):
    fb, _, df = features_at(tiny_tables, tiny_cfg, T)
    y = fb.label_candidates(df, T)
    first = fb.first_year.reindex(pd.MultiIndex.from_arrays([df["u"], df["v"]])).to_numpy(float)
    assert ((first[y == 1] > T) & (first[y == 1] <= T + tiny_cfg.horizon)).all()
    assert (np.isnan(first[y == 0]) | (first[y == 0] > T + tiny_cfg.horizon)).all()
