import pandas as pd

from bridge_ct.cleaning.links import build_paper_trial_links
from bridge_ct.cleaning.normalize import normalize_openalex, normalize_trials, reconstruct_abstract, short_id


def rec(i, authors):
    return {"id": f"https://openalex.org/W{i}", "publication_date": "2015-01-01", "publication_year": 2015,
            "authorships": [{"author": {"id": f"https://openalex.org/{a}", "display_name": "Same Name"}} for a in authors]}


def test_authors_not_merged_by_name():
    t = normalize_openalex([rec(1, ["A1", "A2"])])
    assert t["authorships"]["author_id"].nunique() == 2 and len(t["authors"]) == 2


def test_helpers():
    assert short_id("https://openalex.org/W9") == "W9"
    assert reconstruct_abstract({"b": [1], "a": [0]}) == "a b"


def test_links_keep_provenance_and_never_invent():
    works = pd.DataFrame([{"work_id": "W1", "title": "t", "abstract": "see NCT00000001 and NCT00000009", "pmid": "5",
                           "pubmed_nct_ids": "NCT00000002"}])
    trials = normalize_trials([{"nct_id": "NCT00000001", "registration_date": "2012-01-01"},
                               {"nct_id": "NCT00000002", "registration_date": "2013-01-01"}])
    links, stats = build_paper_trial_links(works, trials)
    assert set(links["nct_id"]) == {"NCT00000001", "NCT00000002"}      # NCT00000009 unknown -> dropped
    assert stats["n_dropped_unknown_trial"] == 1
    assert links["evidence_uri"].str.len().min() > 0 and set(links["method"]) == {"pubmed_databank", "abstract_nct_id"}
