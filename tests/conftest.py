import pytest

from bridge_ct.config import Config
from bridge_ct.synthetic import generate_synthetic


@pytest.fixture(scope="session")
def tiny_cfg():
    """Small, fast configuration (2006-2014, horizon 2)."""
    return Config.model_validate({
        "project": {"start_year": 2006, "end_year": 2014, "prediction_horizon_years": 2, "random_seed": 1},
        "graph": {"min_community_size": 5, "betweenness_samples": None},
        "candidates": {"top_authors_per_community": 8, "neighbour_communities": 3, "two_hop_per_broker": 10},
        "evaluation": {"cutoff_first": 2008, "val_first": 2010, "test_first": 2012, "n_bootstrap": 20,
                       "logistic_C_grid": [0.1, 1.0], "prior_scale_grid": [0.3, 1.0], "top_k": [10, 50]},
        "simulation": {"k_edges": 5, "n_replicates": 2, "n_cascades": 3},
    })


@pytest.fixture(scope="session")
def tiny_tables():
    return generate_synthetic(seed=3, n_communities=6, authors_per_community=25, start_year=2006, end_year=2014,
                              papers_per_year=120, n_topics=12, trials_per_year=6)
