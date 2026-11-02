import pandas as pd

from bridge_ct.graph.collab import build_collaboration_graph, build_pair_table, snapshot_edges


def ships(rows):
    return pd.DataFrame(rows, columns=["work_id", "author_id", "publication_year"])


def test_two_author_paper_makes_one_edge():
    G = build_collaboration_graph(ships([("W1", "A", 2010), ("W1", "B", 2010)]), 2010)
    assert G.number_of_edges() == 1 and G["A"]["B"]["weight"] == 1.0


def test_fractional_weight_and_repeats():
    a = ships([("W1", "A", 2010), ("W1", "B", 2010), ("W1", "C", 2010), ("W2", "A", 2011), ("W2", "B", 2011)])
    G = build_collaboration_graph(a, 2011)
    assert abs(G["A"]["B"]["weight"] - 1.5) < 1e-12      # 1/(3-1) + 1/(2-1)
    assert abs(G["A"]["C"]["weight"] - 0.5) < 1e-12
    assert G["A"]["B"]["n_papers"] == 2


def test_snapshot_has_no_future_papers():
    a = ships([("W1", "A", 2010), ("W1", "B", 2010), ("W2", "A", 2012), ("W2", "C", 2012)])
    G = build_collaboration_graph(a, 2011)
    assert set(G.nodes) == {"A", "B"} and not G.has_edge("A", "C")


def test_binary_mode_and_max_authors_skip():
    a = ships([("W1", x, 2010) for x in "ABCDEF"])
    pt, skipped = build_pair_table(a, max_authors=5)
    assert skipped == 1 and pt.empty
    pt, _ = build_pair_table(a, max_authors=10, weight_mode="binary")
    assert (pt["weight"] == 1.0).all() and len(pt) == 15


def test_edge_first_year():
    a = ships([("W1", "A", 2010), ("W1", "B", 2010), ("W2", "A", 2012), ("W2", "B", 2012)])
    pt, _ = build_pair_table(a)
    e = snapshot_edges(pt, 2012)
    assert int(e["first_year"].iloc[0]) == 2010 and int(e["last_year"].iloc[0]) == 2012
